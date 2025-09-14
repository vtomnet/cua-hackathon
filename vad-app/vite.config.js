import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  assetsInclude: ['**/*.wasm', '**/*.onnx'],
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp'
    },
    fs: {
      allow: ['.']
    }
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  },
  build: {
    rollupOptions: {
      external: (id) => id.endsWith('.wasm')
    }
  }
})
