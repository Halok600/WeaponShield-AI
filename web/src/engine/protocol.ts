import type { ModelConfig } from '../config/models';
import type { Detection } from '../core/types';

export type Backend = 'webgpu' | 'wasm';

export interface ReadyInfo {
  backend: Backend;
  loadMs: number;
}

export interface DetectResult {
  detections: Detection[];
  /** Worker-side time for letterbox + inference + decode. */
  inferMs: number;
  width: number;
  height: number;
}

export type ToWorker =
  | { type: 'init'; model: ModelConfig; preferWebGPU: boolean }
  | { type: 'detect'; id: number; bitmap: ImageBitmap; lowConf: number };

export type FromWorker =
  | { type: 'ready'; backend: Backend; loadMs: number }
  | { type: 'progress'; loaded: number; total: number }
  | ({ type: 'result'; id: number } & DetectResult)
  | { type: 'error'; id?: number; message: string };
