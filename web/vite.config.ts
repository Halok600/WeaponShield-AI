import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { readFileSync } from 'node:fs';
import { defaultClientConditions } from 'vite';
import { defineConfig } from 'vitest/config';

const ortVersion = (JSON.parse(readFileSync('./node_modules/onnxruntime-web/package.json', 'utf8')) as { version: string }).version;

// Cross-origin isolation enables SharedArrayBuffer, which ONNX Runtime Web needs for WASM threads.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  // Use ORT's non-bundled entry: the worker loads its wasm loader + binary from /ort/<version>/ (see
  // detector.worker.ts) instead of Vite emitting a second 26 MB copy of the binary into dist/assets.
  resolve: { conditions: [...defaultClientConditions, 'onnxruntime-web-use-extern-wasm'] },
  define: { __ORT_VERSION__: JSON.stringify(ortVersion) },
  plugins: [react(), tailwindcss()],
  server: { headers: isolation },
  preview: { headers: isolation },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  worker: { format: 'es' },
  test: {
    include: ['src/**/*.test.{ts,tsx}'],
    environment: 'node',
    testTimeout: 30_000,
  },
});
