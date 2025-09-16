type Spawned = ReturnType<typeof Bun.spawn>;

export interface VADOptions {
  rate: number;
  outDir: string;
  modelPath: string;
  speechThreshold: number;
  silenceThreshold: number;
  requiredSpeechFrames: number;
  requiredSilenceFrames: number;
}

export interface VADStatus {
  running: boolean;
  segmentsSaved: number;
  lastSegmentPath: string | null;
}

export function defaultVADOptions(): VADOptions {
  return {
    rate: 16000,
    outDir: 'segments',
    modelPath: 'rec/public/silero_vad_v6.onnx',
    speechThreshold: 0.35,
    silenceThreshold: 0.05,
    requiredSpeechFrames: 2,
    requiredSilenceFrames: 20,
  };
}

function bytesToInt16Array(bytes: Uint8Array): Int16Array {
  const evenLen = bytes.length & ~1;
  return new Int16Array(bytes.buffer, bytes.byteOffset, evenLen / 2);
}

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

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, 'WAVE');

  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true);
  view.setUint16(32, numChannels * bytesPerSample, true);
  view.setUint16(34, 8 * bytesPerSample, true);

  writeString(36, 'data');
  view.setUint32(40, dataSize, true);

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

export type VADController = ReturnType<typeof createVADController>;

export function createVADController(initial?: Partial<VADOptions>) {
  const opts: VADOptions = { ...defaultVADOptions(), ...(initial || {}) };
  let running = false;
  let proc: Spawned | null = null;
  let loopPromise: Promise<void> | null = null;
  let abort = false;

  // status
  let segmentsSaved = 0;
  let lastSegmentPath: string | null = null;

  // VAD runtime state
  let ort: typeof import('onnxruntime-node') | null = null;
  let session: import('onnxruntime-node').InferenceSession | null = null;
  let stateTensor: import('onnxruntime-node').Tensor | null = null;

  async function initOnnx() {
    if (session) return;
    ort = await import('onnxruntime-node');
    const file = Bun.file(opts.modelPath);
    if (!(await file.exists())) throw new Error(`ONNX model not found at ${opts.modelPath}`);
    const buf = await file.arrayBuffer();
    session = await ort.InferenceSession.create(Buffer.from(buf));
    // @ts-ignore
    stateTensor = new ort!.Tensor('float32', new Float32Array(2 * 128), [1, 2, 128]);
  }

  async function start() {
    if (running) throw new Error('VAD already running');
    await ensureDir(opts.outDir);
    await initOnnx();

    const chunkSize = 512;
    const probAvgWindow = 5;
    let probBuffer: number[] = [];

    let leftover = new Uint8Array(0);
    let sampleQueue = new Int16Array(0);
    let isRecording = false;
    let segmentBuffers: Int16Array[] = [];
    let segmentStartTime: number | null = null;
    let speechFrames = 0;
    let silenceFrames = 0;
    let segmentIndex = 0;

    const cmds = pickMicCommands(opts.rate);
    let lastErr: unknown = null;
    abort = false;

    const tryStart = (): Spawned => {
      const { cmd, args } = cmds[0];
      return Bun.spawn({ cmd: [cmd, ...args], stdout: 'pipe', stderr: 'pipe' });
    };
    proc = tryStart();
    running = true;

    const stopRecording = async () => {
      if (!isRecording) return;
      isRecording = false;
      const totalSamples = segmentBuffers.reduce((s, b) => s + b.length, 0);
      const merged = new Int16Array(totalSamples);
      let off = 0;
      for (const buf of segmentBuffers) { merged.set(buf, off); off += buf.length; }
      segmentBuffers = [];
      const wav = writeWav(merged, opts.rate);
      segmentIndex += 1;
      const tag = nowTag();
      const path = `${opts.outDir}/segment_${tag}_${segmentIndex}.wav`;
      await Bun.write(path, wav);
      segmentsSaved += 1;
      lastSegmentPath = path;
    };

    const startRecording = () => {
      if (isRecording) return;
      isRecording = true;
      segmentBuffers = [];
      segmentStartTime = Date.now();
    };

    const processFrame = async (frame: Int16Array) => {
      if (!session || !stateTensor || !ort) throw new Error('ONNX session not initialized');
      const f32 = new Float32Array(frame.length);
      for (let i = 0; i < frame.length; i++) f32[i] = frame[i] / 32768;

      const input = new ort.Tensor('float32', f32, [1, frame.length]);
      let srTensor: import('onnxruntime-node').Tensor;
      try {
        // @ts-ignore
        srTensor = new ort.Tensor('int64', new BigInt64Array([BigInt(16000)]), [1]);
      } catch {
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
          }
        } else {
          silenceFrames = 0;
        }
      }
    };

    const loop = async () => {
      try {
        if (!proc || !proc.stdout) throw new Error('Mic process has no stdout');
        for await (const chunk of proc.stdout) {
          if (abort) break;
          const u8 = chunk as Uint8Array;
          const combined = new Uint8Array(leftover.length + u8.length);
          combined.set(leftover, 0);
          combined.set(u8, leftover.length);
          const evenLen = combined.length & ~1;
          const usable = combined.subarray(0, evenLen);
          leftover = combined.subarray(evenLen);
          const samples = bytesToInt16Array(usable);
          if (sampleQueue.length === 0) {
            sampleQueue = samples;
          } else {
            const tmp = new Int16Array(sampleQueue.length + samples.length);
            tmp.set(sampleQueue, 0);
            tmp.set(samples, sampleQueue.length);
            sampleQueue = tmp;
          }
          while (sampleQueue.length >= 512) {
            const frame = sampleQueue.subarray(0, 512);
            await processFrame(frame);
            sampleQueue = sampleQueue.subarray(512);
          }
        }
      } catch (e) {
        lastErr = e;
      } finally {
        running = false;
        try { await stopRecording(); } catch {}
      }
    };

    loopPromise = loop();
  }

  async function stop() {
    if (!running) return;
    abort = true;
    try { proc?.kill(); } catch {}
    await loopPromise?.catch(() => {});
    running = false;
  }

  function status(): VADStatus {
    return { running, segmentsSaved, lastSegmentPath };
  }

  function update(partial: Partial<VADOptions>) {
    if (running) throw new Error('Cannot update options while running');
    Object.assign(opts, partial);
  }

  return { start, stop, status, update };
}
