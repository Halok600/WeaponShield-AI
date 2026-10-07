import { describe, expect, it } from 'vitest';
import { STOCK_COCO_MODEL } from '../config/models';
import { BusyError, DetectorClient, type WorkerLike } from './detector-client';
import type { FromWorker, ToWorker } from './protocol';

class FakeWorker implements WorkerLike {
  sent: ToWorker[] = [];
  onmessage: ((ev: MessageEvent<FromWorker>) => void) | null = null;
  terminated = false;
  postMessage(message: ToWorker): void {
    this.sent.push(message);
  }
  terminate(): void {
    this.terminated = true;
  }
  reply(msg: FromWorker): void {
    this.onmessage?.({ data: msg } as MessageEvent<FromWorker>);
  }
}

const fakeBitmap = () => {
  const b: { closed: boolean; close(): void } = {
    closed: false,
    close() {
      b.closed = true;
    },
  };
  return b as unknown as ImageBitmap & { closed: boolean };
};

describe('DetectorClient', () => {
  it('resolves init when the worker reports ready and forwards progress', async () => {
    const w = new FakeWorker();
    const c = new DetectorClient(w);
    const progress: number[] = [];
    c.onProgress = (loaded) => progress.push(loaded);
    const ready = c.init(STOCK_COCO_MODEL);
    expect(w.sent[0]).toEqual({ type: 'init', model: STOCK_COCO_MODEL, preferWebGPU: true });
    w.reply({ type: 'progress', loaded: 10, total: 100 });
    w.reply({ type: 'ready', backend: 'wasm', loadMs: 42 });
    await expect(ready).resolves.toEqual({ backend: 'wasm', loadMs: 42 });
    expect(progress).toEqual([10]);
  });

  it('rejects init on a worker error without an id', async () => {
    const w = new FakeWorker();
    const c = new DetectorClient(w);
    const ready = c.init(STOCK_COCO_MODEL);
    w.reply({ type: 'error', message: 'HTTP 404' });
    await expect(ready).rejects.toThrow('HTTP 404');
  });

  it('allows one frame in flight and rejects extra frames with BusyError', async () => {
    const w = new FakeWorker();
    const c = new DetectorClient(w);
    const first = c.detect(fakeBitmap(), 0.1);
    expect(c.busy).toBe(true);
    const extra = fakeBitmap();
    await expect(c.detect(extra, 0.1)).rejects.toBeInstanceOf(BusyError);
    expect(extra.closed).toBe(true);
    const sent = w.sent[0] as Extract<ToWorker, { type: 'detect' }>;
    w.reply({ type: 'result', id: sent.id, detections: [], inferMs: 5, width: 640, height: 480 });
    await expect(first).resolves.toEqual({ detections: [], inferMs: 5, width: 640, height: 480 });
    expect(c.busy).toBe(false);
  });

  it('rejects a frame when the worker reports an error for its id', async () => {
    const w = new FakeWorker();
    const c = new DetectorClient(w);
    const p = c.detect(fakeBitmap(), 0.1);
    const sent = w.sent[0] as Extract<ToWorker, { type: 'detect' }>;
    w.reply({ type: 'error', id: sent.id, message: 'boom' });
    await expect(p).rejects.toThrow('boom');
    expect(c.busy).toBe(false);
  });

  it('dispose terminates the worker and rejects pending work', async () => {
    const w = new FakeWorker();
    const c = new DetectorClient(w);
    const p = c.detect(fakeBitmap(), 0.1);
    c.dispose();
    await expect(p).rejects.toThrow(/disposed/);
    expect(w.terminated).toBe(true);
  });
});
