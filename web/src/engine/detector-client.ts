import type { ModelConfig } from '../config/models';
import type { DetectResult, FromWorker, ReadyInfo, ToWorker } from './protocol';

export interface WorkerLike {
  postMessage(message: ToWorker, transfer: Transferable[]): void;
  onmessage: ((ev: MessageEvent<FromWorker>) => void) | null;
  onerror: ((ev: ErrorEvent) => void) | null;
  terminate(): void;
}

export class BusyError extends Error {
  constructor() {
    super('Detector is busy with another frame');
    this.name = 'BusyError';
  }
}

interface Pending<T> {
  resolve: (v: T) => void;
  reject: (e: Error) => void;
}

export function createDetectorWorker(): Worker {
  return new Worker(new URL('./detector.worker.ts', import.meta.url), { type: 'module' });
}

function closeQuietly(bitmap: ImageBitmap): void {
  try {
    bitmap.close();
  } catch {
    // already closed or transferred
  }
}

/** Main-thread handle on the detector worker. At most one frame is in flight; extra frames are refused. */
export class DetectorClient {
  onProgress: ((loaded: number, total: number) => void) | null = null;
  private readonly worker: WorkerLike;
  private nextId = 1;
  private inFlight: { id: number; pending: Pending<DetectResult> } | null = null;
  private initPending: Pending<ReadyInfo> | null = null;
  private disposed = false;

  constructor(worker: WorkerLike) {
    this.worker = worker;
    worker.onmessage = (ev) => this.handle(ev.data);
    worker.onerror = (ev) => this.failAll(new Error(`Detector worker failed: ${ev.message || 'unknown error'}`));
  }

  get busy(): boolean {
    return this.inFlight !== null;
  }

  init(model: ModelConfig, preferWebGPU = true): Promise<ReadyInfo> {
    if (this.disposed) return Promise.reject(new Error('Detector disposed'));
    return new Promise<ReadyInfo>((resolve, reject) => {
      this.initPending = { resolve, reject };
      try {
        this.worker.postMessage({ type: 'init', model, preferWebGPU }, []);
      } catch (err) {
        this.initPending = null;
        reject(err instanceof Error ? err : new Error(String(err)));
      }
    });
  }

  detect(bitmap: ImageBitmap, lowConf: number): Promise<DetectResult> {
    if (this.disposed) {
      closeQuietly(bitmap);
      return Promise.reject(new Error('Detector disposed'));
    }
    if (this.inFlight) {
      closeQuietly(bitmap);
      return Promise.reject(new BusyError());
    }
    const id = this.nextId++;
    return new Promise<DetectResult>((resolve, reject) => {
      this.inFlight = { id, pending: { resolve, reject } };
      try {
        this.worker.postMessage({ type: 'detect', id, bitmap, lowConf }, [bitmap]);
      } catch (err) {
        this.inFlight = null;
        closeQuietly(bitmap);
        reject(err instanceof Error ? err : new Error(String(err)));
      }
    });
  }

  dispose(): void {
    this.disposed = true;
    this.worker.terminate();
    this.failAll(new Error('Detector disposed'));
  }

  private failAll(err: Error): void {
    const frame = this.inFlight;
    const init = this.initPending;
    this.inFlight = null;
    this.initPending = null;
    frame?.pending.reject(err);
    init?.reject(err);
  }

  private handle(msg: FromWorker): void {
    switch (msg.type) {
      case 'progress':
        this.onProgress?.(msg.loaded, msg.total);
        return;
      case 'ready':
        this.initPending?.resolve({ backend: msg.backend, loadMs: msg.loadMs });
        this.initPending = null;
        return;
      case 'result': {
        if (this.inFlight?.id !== msg.id) return;
        const { pending } = this.inFlight;
        this.inFlight = null;
        pending.resolve({ detections: msg.detections, inferMs: msg.inferMs, width: msg.width, height: msg.height });
        return;
      }
      case 'error': {
        const err = new Error(msg.message);
        if (msg.id === undefined) {
          this.initPending?.reject(err);
          this.initPending = null;
        } else if (this.inFlight?.id === msg.id) {
          const { pending } = this.inFlight;
          this.inFlight = null;
          pending.reject(err);
        }
        return;
      }
    }
  }
}
