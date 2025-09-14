import { useState, useRef } from 'react';
import * as ort from 'onnxruntime-web';

export default function App() {
  const [listening, setListening] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const vadRef = useRef(null);
  const audioContextRef = useRef(null);
  const modelRef = useRef(null);

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
          // Configure WASM paths for onnxruntime-web
          ort.env.wasm.wasmPaths = {
            'ort-wasm.wasm': '/ort-wasm.wasm',
            'ort-wasm-threaded.wasm': '/ort-wasm-threaded.wasm',
            'ort-wasm-simd.wasm': '/ort-wasm-simd.wasm',
            'ort-wasm-simd-threaded.wasm': '/ort-wasm-simd-threaded.wasm'
          };
          
          // Try local model first, then CDN
          let modelUrl = '/silero_vad_v6.onnx';
          try {
            modelRef.current = await ort.InferenceSession.create(
              modelUrl,
              { executionProviders: ['wasm', 'cpu'] }
            );
            console.log('Local Silero VAD v6 model loaded successfully');
          } catch (localError) {
            console.log('Local model failed, trying CDN:', localError.message);
            modelUrl = 'https://cdn.jsdelivr.net/gh/snakers4/silero-vad@master/files/silero_vad_v6.onnx';
            modelRef.current = await ort.InferenceSession.create(
              modelUrl,
              { executionProviders: ['wasm', 'cpu'] }
            );
            console.log('CDN Silero VAD v6 model loaded successfully');
          }
        } catch (e) {
          console.error('VAD model load failed:', e.message, e.stack);
          console.warn('Using volume-based detection as fallback');
        }
      }

      const processor = audioContextRef.current.createScriptProcessor(4096, 1, 1);
      let isSpeech = false;

      processor.onaudioprocess = async (e) => {
        const inputData = e.inputBuffer.getChannelData(0);

        if (modelRef.current) {
          // Process with VAD if model is available
          const chunkSize = 512;
          for (let i = 0; i < inputData.length; i += chunkSize) {
            const chunk = inputData.slice(i, i + chunkSize);
            if (chunk.length !== chunkSize) continue;
            try {
              const inputTensor = new ort.Tensor('float32', chunk, [1, chunkSize]);
              const { output } = await modelRef.current.run({ input: inputTensor });
              const prob = output.data[0];
              if (prob > 0.5 && !isSpeech) {
                console.log('speech start');
                setSpeaking(true);
                isSpeech = true;
              } else if (prob < 0.35 && isSpeech) {
                console.log('speech end');
                setSpeaking(false);
                isSpeech = false;
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

      <p style={{ fontSize: '14px', color: '#666', marginTop: '20px' }}>
        Check the browser console (F12) for speech start/end logs.
      </p>
    </div>
  );
}