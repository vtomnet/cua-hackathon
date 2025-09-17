## Init

```sh
bun i
brew install sox  # or apt install sox
curl -Lo public/silero_vad_v6.onnx https://github.com/snakers4/silero-vad/raw/refs/heads/master/src/silero_vad/data/silero_vad.onnx
```

## Run

```sh
bun run vad
```

## Etc

Optional flags:

- `--rate 16000`               Sample rate (Hz). Silero expects 16k.
- `--out segments`             Output directory for WAV segments
- `--model public/silero_vad_v6.onnx`  Path to the Silero ONNX model
- `--speech 0.35`              Probability threshold to start speech
- `--silence 0.05`             Probability threshold to end speech
- `--speechFrames 2`           Consecutive frames to confirm start
- `--silenceFrames 20`         Consecutive frames to confirm end

Segments are saved to `segments/segment_YYYY-MM-DD_HH-MM-SS_N.wav`.
