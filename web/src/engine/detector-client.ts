import type { ModelConfig } from '../config/models';
import type { DetectResult, FromWorker, ReadyInfo, ToWorker } from './protocol';

export interface WorkerLike {
  postMessage(message: ToWorker, transfer: Transferable[]): void;
  onmessage: ((ev: MessageEvent<FromWorker>) => void) | null;
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

/** Main-thread handle on the detector worker. At most one frame is in flight; extra frames are refused. */
export class DetectorClient {
  onProgress: ((loaded: number, total: number) => void) | null = null;
  private readonly worker: WorkerLike;
  private nextId = 1;
  private inFlight: { id: number; pending: Pending<DetectResult> } | null = null;
  private initPending: Pending<ReadyInfo> | null = null;

  constructor(worker: WorkerLike) {
    this.worker = worker;
    worker.onmessage = (ev) => this.handle(ev.data);
  }

  get busy(): boolean {
    return this.inFlight !== null;
  }

  init(model: ModelConfig, preferWebGPU = true): Promise<ReadyInfo> {
    return new Promise<ReadyInfo>((resolve, reject) => {
      this.initPending = { resolve, reject };
      this.worker.postMessage({ type: 'init', model, preferWebGPU }, []);
    });
  }

  detect(bitmap: ImageBitmap, lowConf: number): Promise<DetectResult> {
    if (this.inFlight) {
      bitmap.close();
      return Promise.reject(new BusyError());
    }
    const id = this.nextId++;
    return new Promise<DetectResult>((resolve, reject) => {
      this.inFlight = { id, pending: { resolve, reject } };
      this.worker.postMessage({ type: 'detect', id, bitmap, lowConf }, [bitmap]);
    });
  }

  dispose(): void {
    this.worker.terminate();
    const err = new Error('Detector disposed');
    this.inFlight?.pending.reject(err);
    this.initPending?.reject(err);
    this.inFlight = null;
    this.initPending = null;
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
