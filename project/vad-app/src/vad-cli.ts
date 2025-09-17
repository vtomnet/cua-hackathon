/*
  Bun CLI for Voice Activity Detection (VAD)

  - Captures microphone audio via SoX (`sox -d`)
  - Processes 16kHz, mono, 16-bit PCM frames
  - Uses Silero VAD v6 (ONNX) via onnxruntime-node with the same logic/hysteresis as the web app
  - Saves detected speech segments to WAV files in ./segments

  Usage:
    bun run src/vad-cli.ts [--rate 16000] [--out segments] [--model public/silero_vad_v6.onnx]

  Requirements:
    - macOS: install SoX (brew install sox) to get `rec`/`sox`
    - Linux: `arecord` (alsa-utils) or SoX
*/

type Spawned = ReturnType<typeof Bun.spawn>;

interface Options {
  rate: number;
  outDir: string;
  modelPath: string;
  // Silero thresholds/logic preserved from web app
  speechThreshold: number;   // probability threshold to start speech
  silenceThreshold: number;  // probability threshold to end speech
  requiredSpeechFrames: number;   // consecutive frames above speechThreshold to start
  requiredSilenceFrames: number;  // consecutive frames below silenceThreshold to end
}

function parseArgs(argv: string[]): Options {
  const args = new Map<string, string>();
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a?.startsWith('--')) {
      const key = a.slice(2);
      const val = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : 'true';
      args.set(key, val);
    }
  }
  const rate = parseInt(args.get('rate') || '16000', 10);
  const outDir = args.get('out') || 'segments';
  const modelPath = args.get('model') || 'public/silero_vad_v6.onnx';
  // Preserve web app thresholds
  const speechThreshold = parseFloat(args.get('speech') || '0.35');
  const silenceThreshold = parseFloat(args.get('silence') || '0.05');
  const requiredSpeechFrames = parseInt(args.get('speechFrames') || '2', 10);
  const requiredSilenceFrames = parseInt(args.get('silenceFrames') || '20', 10);

  return {
    rate,
    outDir,
    modelPath,
    speechThreshold,
    silenceThreshold,
    requiredSpeechFrames,
    requiredSilenceFrames,
  };
}

function bytesToInt16Array(bytes: Uint8Array): Int16Array {
  // Ensure even length
  const evenLen = bytes.length & ~1;
  return new Int16Array(bytes.buffer, bytes.byteOffset, evenLen / 2);
}

// No fallback: we fail fast if ONNX model isn't available

async function ensureDir(path: string) {
  const { mkdir } = await import('node:fs/promises');
  await mkdir(path, { recursive: true });
}

function writeWav(int16: Int16Array, sampleRate: number): Uint8Array {
  const numChannels = 1;
  const bytesPerSample = 2;
  const dataSize = int16.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  function writeString(offset: number, str: string) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  // RIFF header
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, 'WAVE');

  // fmt chunk
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);        // PCM chunk size
  view.setUint16(20, 1, true);         // audio format = 1 (PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true);
  view.setUint16(32, numChannels * bytesPerSample, true);
  view.setUint16(34, 8 * bytesPerSample, true);

  // data chunk
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);

  // PCM data
  const out = new Uint8Array(buffer);
  out.set(new Uint8Array(int16.buffer, int16.byteOffset, int16.byteLength), 44);
  return out;
}

function nowTag() {
  const d = new Date();
  const pad = (n: number) => n.toString().padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}_${pad(d.getHours())}-${pad(d.getMinutes())}-${pad(d.getSeconds())}`;
}

function pickMicCommands(rate: number): Array<{ cmd: string; args: string[] }> {
  return [
    { cmd: 'sox', args: ['-q', '-d', '-b', '16', '-r', String(rate), '-c', '1', '-e', 'signed-integer', '-t', 'raw', '-'] },
  ];
}

async function main() {
  const opts = parseArgs(Bun.argv);
  // Silero expects chunkSize=512 at 16kHz
  const chunkSize = 512;
  if (opts.rate !== 16000) {
    console.warn('Silero VAD expects 16kHz. Forcing mic to 16k via recorder args.');
  }

  const candidates = pickMicCommands(opts.rate);
  if (candidates.length === 0) {
    console.error('Unsupported OS. Use macOS (SoX) or Linux (arecord/SoX).');
    process.exit(1);
  }

  console.log(`VAD: rate=${opts.rate}Hz, silero chunk=${chunkSize} samples (~32ms)`);
  console.log(`VAD thresholds: speech>${opts.speechThreshold}, silence<${opts.silenceThreshold}`);

  await ensureDir(opts.outDir);

  let proc: Spawned | null = null;
  for (const mic of candidates) {
    try {
      console.log(`Trying mic: ${mic.cmd} ${mic.args.join(' ')}`);
      proc = Bun.spawn({ cmd: [mic.cmd, ...mic.args], stdout: 'pipe', stderr: 'inherit' });
      break;
    } catch (e) {
      console.error(`Failed to start '${mic.cmd}': ${(e as Error).message}`);
      continue;
    }
  }
  if (!proc) {
    console.error('Unable to start microphone capture with sox.');
    console.error('Hint: install SoX and ensure `sox` is in PATH (macOS: brew install sox, Linux: sudo apt-get install sox)');
    process.exit(1);
  }

  let leftover = new Uint8Array(0);
  let sampleQueue = new Int16Array(0);

  let isRecording = false;
  let speechFrames = 0;
  let silenceFrames = 0;
  let segmentBuffers: Int16Array[] = [];
  let segmentStartTime: number | null = null;
  let segmentIndex = 0;

  // --- Initialize Silero VAD (ONNX) ---
  let ort: typeof import('onnxruntime-node') | null = null;
  let session: import('onnxruntime-node').InferenceSession | null = null;
  let stateTensor: import('onnxruntime-node').Tensor | null = null;
  const probBuffer: number[] = [];
  const probAvgWindow = 5;
  try {
    // Dynamically import to avoid hard failure if not installed
    // @ts-ignore
    ort = await import('onnxruntime-node');
    session = await ort.InferenceSession.create(opts.modelPath, {
      executionProviders: ['cpu'],
      graphOptimizationLevel: 'disabled',
    } as any);
    // State shape [2,1,128]
    stateTensor = new ort.Tensor('float32', new Float32Array(2 * 1 * 128).fill(0), [2, 1, 128]);
    console.log(`Loaded Silero model: ${opts.modelPath}`);
  } catch (e) {
    console.error('Failed to initialize onnxruntime-node or load Silero model.');
    console.error(String(e));
    process.exit(1);
  }

  const stopRecording = async () => {
    if (!isRecording) return;
    isRecording = false;
    const totalSamples = segmentBuffers.reduce((n, a) => n + a.length, 0);
    const merged = new Int16Array(totalSamples);
    let off = 0;
    for (const buf of segmentBuffers) { merged.set(buf, off); off += buf.length; }
    segmentBuffers = [];

    const duration = totalSamples / opts.rate;
    const wav = writeWav(merged, opts.rate);
    segmentIndex += 1;
    const tag = nowTag();
    const path = `${opts.outDir}/segment_${tag}_${segmentIndex}.wav`;
    await Bun.write(path, wav);
    const startedAt = segmentStartTime ? new Date(segmentStartTime).toLocaleTimeString() : 'n/a';
    segmentStartTime = null;
    console.log(`Saved segment -> ${path} (${duration.toFixed(2)}s, started ${startedAt})`);
  };

  const startRecording = () => {
    if (isRecording) return;
    isRecording = true;
    segmentBuffers = [];
    segmentStartTime = Date.now();
    console.log('Speech start');
  };

  const processFrame = async (frame: Int16Array) => {
    if (!session || !stateTensor || !ort) throw new Error('ONNX session not initialized');
    // Convert int16 -> float32 normalized [-1,1]
    const f32 = new Float32Array(frame.length);
    for (let i = 0; i < frame.length; i++) f32[i] = frame[i] / 32768;

    const input = new ort.Tensor('float32', f32, [1, frame.length]);
    // Some ORT builds expect int64 as BigInt64Array
    let srTensor: import('onnxruntime-node').Tensor;
    try {
      // @ts-ignore
      srTensor = new ort.Tensor('int64', new BigInt64Array([BigInt(16000)]), [1]);
    } catch {
      // Fallback (some builds accept number[])
      srTensor = new ort.Tensor('int64', [16000] as any, [1]);
    }

    const feeds: Record<string, any> = { input, state: stateTensor, sr: srTensor };
    const results = await session.run(feeds);
    if (results.state) stateTensor = results.state as any;
    const rawProb = results.output ? (results.output.data as Float32Array | number[])[0] : 0;

    probBuffer.push(Number(rawProb));
    if (probBuffer.length > probAvgWindow) probBuffer.shift();
    const prob = probBuffer.reduce((s, v) => s + v, 0) / probBuffer.length;

    if (!isRecording) {
      if (prob > opts.speechThreshold) {
        speechFrames += 1;
        if (speechFrames >= opts.requiredSpeechFrames) {
          startRecording();
          speechFrames = 0;
        }
      } else {
        speechFrames = 0;
      }
    }

    if (isRecording) {
      segmentBuffers.push(frame.slice());
      if (prob < opts.silenceThreshold) {
        silenceFrames += 1;
        if (silenceFrames >= opts.requiredSilenceFrames) {
          silenceFrames = 0;
          await stopRecording();
          console.log('Speech end');
        }
      } else {
        silenceFrames = 0;
      }
    }
  };

  // Graceful shutdown
  const cleanup = async () => {
    try { await stopRecording(); } catch {}
    try { proc?.kill(); } catch {}
    process.exit(0);
  };
  process.on('SIGINT', cleanup);
  process.on('SIGTERM', cleanup);

  if (!proc.stdout) {
    console.error('Mic process has no stdout to read.');
    await cleanup();
    return;
  }

  for await (const chunk of proc.stdout) {
    // chunk is Uint8Array
    const u8 = chunk as Uint8Array;
    // prepend leftover bytes
    const combined = new Uint8Array(leftover.length + u8.length);
    combined.set(leftover, 0);
    combined.set(u8, leftover.length);

    const evenLen = combined.length & ~1; // drop last odd byte if any
    const usable = combined.subarray(0, evenLen);
    leftover = combined.subarray(evenLen);

    const samples = bytesToInt16Array(usable);
    // append to queue
    if (sampleQueue.length === 0) {
      sampleQueue = samples;
    } else {
      const tmp = new Int16Array(sampleQueue.length + samples.length);
      tmp.set(sampleQueue, 0);
      tmp.set(samples, sampleQueue.length);
      sampleQueue = tmp;
    }

    // process in 512-sample chunks (Silero)
    while (sampleQueue.length >= chunkSize) {
      const frame = sampleQueue.subarray(0, chunkSize);
      await processFrame(frame);
      sampleQueue = sampleQueue.subarray(chunkSize);
    }
  }
}

main().catch((e) => {
  console.error('Fatal error:', e);
  process.exit(1);
});
