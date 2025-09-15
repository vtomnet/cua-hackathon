import { useState, useRef, useEffect } from 'react';
import * as ort from 'onnxruntime-web';

// Configure ONNX Runtime Web environment before any usage
ort.env.wasm.wasmPaths = {
  'ort-wasm-simd-threaded.wasm': '/ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.jsep.wasm': '/ort-wasm-simd-threaded.jsep.wasm'
};
ort.env.wasm.numThreads = 1; // Disable threading to avoid issues
ort.env.logLevel = 'verbose'; // Enable verbose logging

// WAV conversion utility
function convertToWAV(audioBuffer, sampleRate = 16000) {
  const length = audioBuffer.length;
  const arrayBuffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(arrayBuffer);

  // WAV header
  const writeString = (offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, length * 2, true);

  // Convert float32 to int16
  let offset = 44;
  for (let i = 0; i < length; i++) {
    const sample = Math.max(-1, Math.min(1, audioBuffer[i]));
    view.setInt16(offset, sample * 0x7FFF, true);
    offset += 2;
  }

  return new Blob([arrayBuffer], { type: 'audio/wav' });
}

export default function App() {
  const [listening, setListening] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [recording, setRecording] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [recordedSegments, setRecordedSegments] = useState([]);
  const vadRef = useRef(null);
  const audioContextRef = useRef(null);
  const modelRef = useRef(null);
  const stateRef = useRef(null);
  const recordingBufferRef = useRef([]);
  const isRecordingRef = useRef(false);

  // Upload WAV to server
  const uploadWAV = async (wavBlob, metadata) => {
    try {
      setUploadStatus('Uploading...');
      const formData = new FormData();
      formData.append('audio', wavBlob, `speech_${Date.now()}.wav`);
      formData.append('metadata', JSON.stringify(metadata));

      // Replace with your server endpoint
      const response = await fetch('http://localhost:8002/process', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        setUploadStatus('Upload successful');
        setRecordedSegments(prev => [...prev, {
          id: Date.now(),
          timestamp: new Date().toLocaleTimeString(),
          duration: metadata.duration,
          status: 'uploaded'
        }]);
        console.log('Upload successful:', result);
      } else {
        setUploadStatus('Upload failed');
        console.error('Upload failed:', response.statusText);
      }
    } catch (error) {
      setUploadStatus('Upload error');
      console.error('Upload error:', error);
    }

    setTimeout(() => setUploadStatus(''), 3000);
  };

  const startListening = async () => {
    try {
      setError('');
      setLoading(true);

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContextRef.current = new AudioContext({ sampleRate: 16000 });
      const source = audioContextRef.current.createMediaStreamSource(stream);

      if (!modelRef.current) {
        try {
          // Try local model first, then CDN
          let modelUrl = '/silero_vad_v6.onnx';
          try {
            modelRef.current = await ort.InferenceSession.create(
              modelUrl,
              {
                executionProviders: ['cpu'], // Use CPU backend to avoid WASM issues
                graphOptimizationLevel: 'disabled',
                executionMode: 'sequential'
              }
            );
            console.log('Local Silero VAD v6 model loaded successfully');
            // Initialize state tensor for RNN model (shape: [2, 1, 128])
            stateRef.current = new ort.Tensor('float32', new Float32Array(2 * 1 * 128).fill(0), [2, 1, 128]);
          } catch (localError) {
            console.log('Local model failed, trying CDN:', localError.message);
            modelUrl = 'https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx?download=true';
            modelRef.current = await ort.InferenceSession.create(
              modelUrl,
              {
                executionProviders: ['cpu'], // Use CPU backend to avoid WASM issues
                graphOptimizationLevel: 'disabled',
                executionMode: 'sequential'
              }
            );
            console.log('CDN (Hugging Face) Silero VAD v6 model loaded successfully');
            // Initialize state tensor for RNN model (shape: [2, 1, 128])
            stateRef.current = new ort.Tensor('float32', new Float32Array(2 * 1 * 128).fill(0), [2, 1, 128]);
          }
        } catch (e) {
          console.error('VAD model load failed:', e.message, e.stack);
          console.warn('Using volume-based detection as fallback');
        }
      }

      const processor = audioContextRef.current.createScriptProcessor(4096, 1, 1);
      let isSpeech = false;
      let speechFrameCount = 0; // Count consecutive speech detections
      let silenceFrameCount = 0; // Count consecutive silence detections
      let probabilityBuffer = []; // Buffer to smooth probability scores
      const bufferSize = 5; // Average over 5 recent predictions

      processor.onaudioprocess = async (e) => {
        const inputData = e.inputBuffer.getChannelData(0);

        // Always collect audio data if we're in a recording session
        if (isRecordingRef.current) {
          recordingBufferRef.current.push(new Float32Array(inputData));
        }

        if (modelRef.current && stateRef.current) {
          // Process with VAD if model is available
          const chunkSize = 512; // Model requirement - cannot change this
          for (let i = 0; i < inputData.length; i += chunkSize) {
            const chunk = inputData.slice(i, i + chunkSize);
            if (chunk.length !== chunkSize) continue;
            try {
              // Create input tensor with correct shape [1, chunkSize]
              const inputTensor = new ort.Tensor('float32', chunk, [1, chunkSize]);

              // Run inference with input, state, and sample rate
              const feeds = {
                input: inputTensor,
                state: stateRef.current,
                sr: new ort.Tensor('int64', [16000], [1])
              };
              const results = await modelRef.current.run(feeds);

              // Update state for next iteration
              if (results.state) {
                stateRef.current = results.state;
              }

              // Get speech probability and smooth it
              const rawProb = results.output ? results.output.data[0] : 0;

              // Add to probability buffer and keep only recent values
              probabilityBuffer.push(rawProb);
              if (probabilityBuffer.length > bufferSize) {
                probabilityBuffer.shift();
              }

              // Calculate smoothed probability (average of recent predictions)
              const prob = probabilityBuffer.reduce((sum, p) => sum + p, 0) / probabilityBuffer.length;

              // Use hysteresis and debouncing for stable detection
              const speechThreshold = 0.35;  // Higher threshold for speech start
              const silenceThreshold = 0.05; // Much lower threshold for speech end (avoid breath cutoffs)
              const requiredSpeechFrames = 2; // Require 3 consecutive detections for speech start
              const requiredSilenceFrames = 20; // Require 5 consecutive detections for speech end

              if (prob > speechThreshold) {
                speechFrameCount++;
                silenceFrameCount = 0;

                // Trigger speech start after consecutive detections
                if (!isSpeech && speechFrameCount >= requiredSpeechFrames) {
                  console.log('speech start');
                  setSpeaking(true);
                  isSpeech = true;

                  // Start recording
                  isRecordingRef.current = true;
                  recordingBufferRef.current = [];
                  setRecording(true);
                  console.log('Recording started');
                }
              } else if (prob < silenceThreshold) {
                silenceFrameCount++;
                speechFrameCount = 0;

                // Trigger speech end after consecutive silence detections
                if (isSpeech && silenceFrameCount >= requiredSilenceFrames) {
                  console.log('speech end');
                  setSpeaking(false);
                  isSpeech = false;

                  // Stop recording and process
                  if (isRecordingRef.current) {
                    isRecordingRef.current = false;
                    setRecording(false);
                    console.log('Recording stopped');

                    // Convert recorded audio to WAV
                    const recordedChunks = recordingBufferRef.current;
                    if (recordedChunks.length > 0) {
                      // Concatenate all audio chunks
                      const totalLength = recordedChunks.reduce((sum, chunk) => sum + chunk.length, 0);
                      const concatenated = new Float32Array(totalLength);
                      let offset = 0;

                      for (const chunk of recordedChunks) {
                        concatenated.set(chunk, offset);
                        offset += chunk.length;
                      }

                      // Convert to WAV and upload
                      const wavBlob = convertToWAV(concatenated, 16000);
                      const metadata = {
                        duration: totalLength / 16000,
                        timestamp: new Date().toISOString(),
                        sampleRate: 16000,
                        channels: 1
                      };

                      uploadWAV(wavBlob, metadata);
                    }
                  }
                }
              } else {
                // In between thresholds - maintain current state but reset counters
                speechFrameCount = 0;
                silenceFrameCount = 0;
              }
            } catch (e) {
              setError(`VAD processing error: ${e.message}`);
            }
          }
        } else {
          // Volume-based detection as fallback
          const volume = Math.sqrt(inputData.reduce((sum, sample) => sum + sample * sample, 0) / inputData.length);
          if (volume > 0.01 && !isSpeech) {
            console.log('speech start (volume-based)');
            setSpeaking(true);
            isSpeech = true;
          } else if (volume < 0.005 && isSpeech) {
            console.log('speech end (volume-based)');
            setSpeaking(false);
            isSpeech = false;
          }
        }
      };

      source.connect(processor);
      processor.connect(audioContextRef.current.destination);

      vadRef.current = { stream, processor, source };
      setListening(true);
      setLoading(false);
    } catch (err) {
      setError(`Microphone error: ${err.message}`);
      setLoading(false);
    }
  };

  const stopListening = () => {
    if (vadRef.current?.stream) {
      vadRef.current.stream.getTracks().forEach(track => track.stop());
    }
    if (vadRef.current?.processor) {
      vadRef.current.processor.disconnect();
    }
    if (vadRef.current?.source) {
      vadRef.current.source.disconnect();
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    setListening(false);
    setSpeaking(false);
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>Voice Activity Detection</h1>
      <p style={{ fontSize: '14px', color: '#666' }}>
        Using Silero VAD v6 with ONNX Runtime Web
      </p>

      {error && (
        <p style={{ color: 'red', marginBottom: '10px' }}>{error}</p>
      )}
      {loading && (
        <p style={{ color: '#1976d2' }}>Loading VAD model...</p>
      )}
      {uploadStatus && (
        <p style={{
          color: uploadStatus.includes('successful') ? 'green' :
                 uploadStatus.includes('Uploading') ? '#1976d2' : 'red',
          marginBottom: '10px'
        }}>{uploadStatus}</p>
      )}

      {!listening ? (
        <button
          onClick={startListening}
          disabled={loading}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            backgroundColor: loading ? '#cccccc' : '#44ff44',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: loading ? 'not-allowed' : 'pointer',
            marginBottom: '20px'
          }}
        >
          {loading ? 'Loading...' : 'Turn Mic On'}
        </button>
      ) : (
        <button
          onClick={stopListening}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            backgroundColor: '#ff4444',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            marginBottom: '20px'
          }}
        >
          Turn Mic Off
        </button>
      )}

      <p style={{ fontSize: '18px', margin: '10px 0' }}>
        Status: <strong>{listening ? 'Listening...' : 'Not listening'}</strong>
      </p>
      <p style={{ fontSize: '18px', margin: '10px 0' }}>
        Voice: <strong>{speaking ? 'Speaking...' : 'Silent'}</strong>
      </p>
      <p style={{ fontSize: '18px', margin: '10px 0' }}>
        Recording: <strong style={{ color: recording ? 'red' : 'gray' }}>
          {recording ? '🔴 Recording' : '⚫ Not Recording'}
        </strong>
      </p>

      {recordedSegments.length > 0 && (
        <div style={{ marginTop: '30px' }}>
          <h3>Recorded Segments</h3>
          <div style={{ maxHeight: '200px', overflowY: 'auto', border: '1px solid #ccc', padding: '10px', borderRadius: '5px' }}>
            {recordedSegments.map(segment => (
              <div key={segment.id} style={{
                marginBottom: '8px',
                padding: '5px',
                backgroundColor: '#f5f5f5',
                borderRadius: '3px',
                fontSize: '14px'
              }}>
                <strong>{segment.timestamp}</strong> -
                Duration: {segment.duration.toFixed(1)}s -
                Status: <span style={{ color: segment.status === 'uploaded' ? 'green' : 'orange' }}>
                  {segment.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <p style={{ fontSize: '14px', color: '#666', marginTop: '20px' }}>
        Check the browser console (F12) for speech start/end logs.
        <br />
        Speech segments are automatically recorded and uploaded as WAV files.
      </p>
    </div>
  );
}
