import * as ort from 'onnxruntime-web/webgpu';
import type { ModelConfig } from '../config/models';
import { letterboxGeometry } from '../core/geometry';
import { decodeEndToEnd, uniformThresholds } from '../core/postprocess';
import { rgbaToTensor } from '../core/preprocess';
import type { AdapterInfo, Backend, FromWorker, ToWorker } from './protocol';

const scope = self as unknown as DedicatedWorkerGlobalScope;

// The WASM runtime is copied to ort/<version>/ by scripts/prepare-assets.mjs and served same-origin.
ort.env.wasm.wasmPaths = new URL(`${import.meta.env.BASE_URL}ort/${__ORT_VERSION__}/`, scope.location.origin).href;
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

interface CreatedSession {
  s: ort.InferenceSession;
  backend: Backend;
  adapterInfo: AdapterInfo | null;
}

async function createSession(bytes: Uint8Array, preferWebGPU: boolean, data: Float32Array, size: number): Promise<CreatedSession> {
  if (preferWebGPU && 'gpu' in navigator) {
    let gpuSession: ort.InferenceSession | null = null;
    try {
      // Ask for the discrete GPU on hybrid-graphics laptops (Chrome otherwise tends to pick the integrated one).
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (adapter) {
        // Deprecated in ORT 1.30 but the only way to hand ORT a specific adapter; it must be set before the first session.
        ort.env.webgpu.adapter = adapter;
        gpuSession = await ort.InferenceSession.create(bytes, { executionProviders: ['webgpu'] });
        await warmUp(gpuSession, data, size);
        const { vendor, architecture, description } = adapter.info ?? {};
        return { s: gpuSession, backend: 'webgpu', adapterInfo: { vendor: vendor ?? '', architecture: architecture ?? '', description: description ?? '' } };
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
  return { s, backend: 'wasm', adapterInfo: null };
}

async function init(m: ModelConfig, preferWebGPU: boolean): Promise<void> {
  const t0 = performance.now();
  const bytes = await download(m.url);
  const size = m.inputSize;
  const data = new Float32Array(3 * size * size);
  const { s, backend, adapterInfo } = await createSession(bytes, preferWebGPU, data, size);
  ctx = new OffscreenCanvas(size, size).getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('OffscreenCanvas 2D context unavailable');
  tensorData = data;
  session = s;
  model = m;
  post({ type: 'ready', backend, loadMs: Math.round(performance.now() - t0), adapterInfo, threads: ort.env.wasm.numThreads ?? 1 });
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
  switch (m.type) {
    case 'init':
      init(m.model, m.preferWebGPU).catch((e) => post({ type: 'error', message: message(e) }));
      return;
    case 'detect':
      detect(m.id, m.bitmap, m.lowConf).catch((e) => post({ type: 'error', id: m.id, message: message(e) }));
      return;
    default: {
      const unhandled: never = m;
      console.error('Detector worker: unknown message', unhandled);
    }
  }
};
