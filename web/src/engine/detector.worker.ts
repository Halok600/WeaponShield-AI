import * as ort from 'onnxruntime-web/webgpu';
import type { ModelConfig } from '../config/models';
import { letterboxGeometry } from '../core/geometry';
import { decodeEndToEnd, uniformThresholds } from '../core/postprocess';
import { rgbaToTensor } from '../core/preprocess';
import type { Backend, FromWorker, ToWorker } from './protocol';

const scope = self as unknown as DedicatedWorkerGlobalScope;

// WASM binaries are copied to /ort/ by scripts/prepare-assets.mjs and served same-origin.
ort.env.wasm.wasmPaths = new URL('/ort/', scope.location.origin).href;
ort.env.wasm.numThreads = scope.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

let session: ort.InferenceSession | null = null;
let model: ModelConfig | null = null;
let ctx: OffscreenCanvasRenderingContext2D | null = null;
let tensorData: Float32Array | null = null;

const post = (msg: FromWorker) => scope.postMessage(msg);

async function download(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`Model download failed: HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length') ?? 0);
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    post({ type: 'progress', loaded, total });
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) {
    bytes.set(c, offset);
    offset += c.length;
  }
  return bytes;
}

async function warmUp(s: ort.InferenceSession, data: Float32Array, size: number): Promise<void> {
  // Compiles WebGPU shaders / allocates WASM memory before the first real frame.
  await s.run({ [s.inputNames[0]!]: new ort.Tensor('float32', data, [1, 3, size, size]) });
}

async function createSession(
  bytes: Uint8Array,
  preferWebGPU: boolean,
  data: Float32Array,
  size: number,
): Promise<{ s: ort.InferenceSession; backend: Backend }> {
  if (preferWebGPU && 'gpu' in navigator) {
    let gpuSession: ort.InferenceSession | null = null;
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        gpuSession = await ort.InferenceSession.create(bytes, { executionProviders: ['webgpu'] });
        await warmUp(gpuSession, data, size);
        return { s: gpuSession, backend: 'webgpu' };
      }
    } catch (err) {
      console.warn('WebGPU session failed, falling back to WASM', err);
      try {
        await gpuSession?.release();
      } catch {
        // ignore release errors
      }
    }
  }
  const s = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
  await warmUp(s, data, size);
  return { s, backend: 'wasm' };
}

async function init(m: ModelConfig, preferWebGPU: boolean): Promise<void> {
  const t0 = performance.now();
  const bytes = await download(m.url);
  const size = m.inputSize;
  const data = new Float32Array(3 * size * size);
  const { s, backend } = await createSession(bytes, preferWebGPU, data, size);
  ctx = new OffscreenCanvas(size, size).getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('OffscreenCanvas 2D context unavailable');
  tensorData = data;
  session = s;
  model = m;
  post({ type: 'ready', backend, loadMs: Math.round(performance.now() - t0) });
}

async function detect(id: number, bitmap: ImageBitmap, lowConf: number): Promise<void> {
  if (!session || !model || !ctx || !tensorData) {
    bitmap.close();
    throw new Error('Detector not initialised');
  }
  const t0 = performance.now();
  const size = model.inputSize;
  const { width, height } = bitmap;
  const lb = letterboxGeometry(width, height, size);
  try {
    ctx.fillStyle = 'rgb(114,114,114)';
    ctx.fillRect(0, 0, size, size);
    ctx.drawImage(bitmap, lb.padX, lb.padY, lb.newW, lb.newH);
  } finally {
    bitmap.close();
  }
  rgbaToTensor(ctx.getImageData(0, 0, size, size).data, size, tensorData);

  const out = await session.run({ [session.inputNames[0]!]: new ort.Tensor('float32', tensorData, [1, 3, size, size]) });
  const o = out[session.outputNames[0]!]!;
  const detections = decodeEndToEnd(o.data as Float32Array, o.dims, model.labels, uniformThresholds(lowConf), lb, width, height);
  post({ type: 'result', id, detections, inferMs: performance.now() - t0, width, height });
}

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

scope.onmessage = (ev: MessageEvent<ToWorker>) => {
  const m = ev.data;
  if (m.type === 'init') init(m.model, m.preferWebGPU).catch((e) => post({ type: 'error', message: message(e) }));
  else detect(m.id, m.bitmap, m.lowConf).catch((e) => post({ type: 'error', id: m.id, message: message(e) }));
};
