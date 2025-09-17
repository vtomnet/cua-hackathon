1. `bun i`
2. `cp node_modules/onnxruntime-web/dist/*.wasm ./public/`
3. `curl -Lo ./public/silero_vad_v6.onnx https://github.com/snakers4/silero-vad/raw/refs/heads/master/src/silero_vad/data/silero_vad.onnx`
4. `bun run dev`
