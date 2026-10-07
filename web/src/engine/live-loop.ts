import { BusyError, type DetectorClient } from './detector-client';
import type { DetectResult } from './protocol';

export interface LiveLoopOptions {
  video: HTMLVideoElement;
  client: DetectorClient;
  lowConf: () => number;
  /** Timestamp for a frame: performance.now() for live input, video.currentTime * 1000 for files. */
  clock: () => number;
  onResult: (r: DetectResult, tMs: number) => void;
  onError: (e: unknown) => void;
}

/** Feeds video frames to the detector, sending a new frame only when the previous one has finished. */
export function startLiveLoop(o: LiveLoopOptions): () => void {
  let stopped = false;
  let inFlight = false;
  let handle = 0;
  const usesVfc = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;

  const schedule = () => {
    handle = usesVfc ? o.video.requestVideoFrameCallback(tick) : requestAnimationFrame(tick);
  };

  const tick = () => {
    if (stopped) return;
    if (!inFlight && o.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && o.video.videoWidth > 0) {
      inFlight = true;
      const t = o.clock();
      createImageBitmap(o.video)
        .then((bitmap) => {
          if (stopped) {
            bitmap.close();
            return undefined;
          }
          return o.client.detect(bitmap, o.lowConf());
        })
        .then((r) => {
          if (r && !stopped) o.onResult(r, t);
        })
        .catch((e) => {
          if (!(e instanceof BusyError) && !stopped) o.onError(e);
        })
        .finally(() => {
          inFlight = false;
        });
    }
    schedule();
  };

  schedule();
  return () => {
    stopped = true;
    if (usesVfc) o.video.cancelVideoFrameCallback(handle);
    else cancelAnimationFrame(handle);
  };
}
