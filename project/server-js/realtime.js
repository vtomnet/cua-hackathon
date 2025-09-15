import record from 'node-record-lpcm16';
import Speaker from '@mastra/node-speaker';

// Helper to construct a fresh speaker so we can stop playback immediately on barge-in
const speakerConfig = {
  channels: 1,
  bitDepth: 16,
  sampleRate: 24000,
  signed: true,
};
let speaker = new Speaker(speakerConfig);

const model = 'gpt-realtime';
const url = `wss://api.openai.com/v1/realtime?model=${model}`;
const ws = new WebSocket(url, {
  headers: {
    Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
  },
});

let isAssistantSpeaking = false;
let currentResponseId = null;

function handleTruncation() {
  if (isAssistantSpeaking && currentResponseId) {
    console.log('User started speaking, truncating current response');

    // Cancel the specific in-flight response so the server stops sending audio
    ws.send(JSON.stringify({
      type: 'response.cancel',
      response_id: currentResponseId,
    }));

    // Immediately stop any buffered/ongoing playback by resetting the speaker
    try { speaker.end(); } catch (e) { }
    try { speaker = new Speaker(speakerConfig); } catch (e) { }

    // Flip the flag so we ignore any stray deltas that arrive after cancellation
    isAssistantSpeaking = false;
    currentResponseId = null;
  }
}

function handleMessage(data) {
  console.log("JSON:", data.data);
  const event = JSON.parse(data.data);
  if (!event) return;

  switch (event.type) {
    case 'session.created':
      ws.send(JSON.stringify({
        type: 'session.update',
        session: {
          type: 'realtime',
          instructions: SYSTEM_PROMPT,
          audio: {
            output: {
              voice: 'coral',
            }
          },
          tools: [
            {
              "type": "function",
              "name": "wait",
              "description": "Wait; no-op.",
            },
            {
              "type": "function",
              "name": "click_button",
              "description": "Click the button.",
            }
          ],
          tool_choice: 'auto',
        },
      }));

      const recording = record.record({
        sampleRate: 24000,
      });

      // Continuously stream mic audio to enable server-side VAD barge-in
      recording.stream().on('data', (chunk) => {
        const base64Chunk = chunk.toString('base64');
        ws.send(JSON.stringify({
          type: 'input_audio_buffer.append',
          audio: base64Chunk,
        }));
      });
      break;

    case 'input_audio_buffer.speech_started':
      handleTruncation();
      break;

    case 'response.created':
      currentResponseId = event.response.id;
      isAssistantSpeaking = true; // Set immediately when response starts
      console.log('Response created:', currentResponseId);
      break;

    case 'response.output_item.added':
      if (event.item.type === 'audio') {
        console.log('Audio output item added');
      }
      break;

    case 'response.output_audio.delta':
      // If we've been interrupted, ignore any further output audio deltas
      if (!isAssistantSpeaking) break;

      const base64 = event.delta;
      const pcm = Buffer.from(base64, 'base64');
      if (pcm.length % 2 !== 0) {
        console.warn('pcm chunk length is odd; expected even length for 16-bit pcm');
      }

      const ok = speaker.write(pcm);
      if (!ok) {
        if (ws._socket && ws._socket.pause) {
          ws._socket.pause();
          speaker.once('drain', () => {
            try { ws._socket.resume(); } catch (e) { }
          });
        } else {
          console.log('speaker backpressure: chunk queued');
        }
      }
      break;

    case 'response.output_item.done':
      const { item } = event;
      console.log("ITEM:", item);
      if (item.type === 'function_call') {
        process.stderr.write(`${item.name}\n`);
        console.log('got function call');
        // call function, send response.create
      } else if (item.type === 'audio') {
        isAssistantSpeaking = false;
        console.log('Assistant finished speaking');
      }
      break;

    case 'response.done':
      isAssistantSpeaking = false;
      currentResponseId = null;
      console.log('Response completed');
      break;

    case 'session.updated':
      console.log("Session updated:", event);
      break;
  }
}

const SYSTEM_PROMPT = `You are a digital assistant.
Respond in English unless told otherwise.
Respond concisely: your job is to aid the user when they ask for it, NOT to engage in a conversation with them.
Unless the user explicitly tells you to call a tool, you should assume the user is not addressing you, and you should call the 'wait' tool (NO OTHER OUTPUT).
If the user does tell you to call a tool, then do that, and respond with ONLY the tool call (no speech).`;

ws.addEventListener("open", async () => {
  console.log("Opened websocket");
});

ws.addEventListener('close', (event) => {
  console.log(`Websocket closed: ${event}`);
  try { speaker.end(); } catch (e) { }
});

ws.addEventListener('error', (err) => {
  console.error(`Websocket error: ${err}`);
  try { speaker.end(); } catch (e) { }
});

ws.addEventListener("message", handleMessage);
