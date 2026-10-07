import { useEffect, useRef, useState } from 'react';
import { ACTIVE_MODEL } from '../config/models';
import { Pipeline } from '../core/pipeline';
import { filterForImage, PRESETS } from '../core/presets';
import { type Detection, isWeapon } from '../core/types';
import { createDetectorWorker, DetectorClient } from '../engine/detector-client';
import { startLiveLoop } from '../engine/live-loop';
import type { Backend } from '../engine/protocol';
import { buildDetectionPrimitives, buildTrackPrimitives, paint } from '../render/overlay';

type Status = 'loading' | 'ready' | 'error';
const preset = PRESETS.balanced;

function counts(dets: readonly { cls: Detection['cls'] }[]) {
  return { person: dets.filter((d) => d.cls === 'person').length, weapon: dets.filter((d) => isWeapon(d.cls)).length };
}

/** Temporary development harness for the detection core. Replaced by the real console in Plan 3. */
export function LabPage() {
  const [status, setStatus] = useState<Status>('loading');
  const [backend, setBackend] = useState<Backend | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [count, setCount] = useState({ person: 0, weapon: 0 });
  const [inferMs, setInferMs] = useState(0);
  const [frames, setFrames] = useState(0);
  const [live, setLive] = useState(false);
  const clientRef = useRef<DetectorClient | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const client = new DetectorClient(createDetectorWorker());
    clientRef.current = client;
    // React StrictMode (dev) mounts, cleans up and re-mounts: ignore results of a disposed client.
    let stale = false;
    client.onProgress = (loaded, total) => {
      if (!stale) setProgress(total > 0 ? loaded / total : 0);
    };
    client
      .init(ACTIVE_MODEL)
      .then((info) => {
        if (stale) return;
        setBackend(info.backend);
        setStatus('ready');
      })
      .catch((e: Error) => {
        if (stale) return;
        setError(e.message);
        setStatus('error');
      });
    return () => {
      stale = true;
      client.dispose();
    };
  }, []);

  useEffect(() => {
    const client = clientRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!live || !client || !video || !canvas) return;
    let stopLoop: (() => void) | null = null;
    let stream: MediaStream | null = null;
    const pipeline = new Pipeline(preset);
    let cancelled = false;

    navigator.mediaDevices
      .getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false })
      .then(async (s) => {
        if (cancelled) return s.getTracks().forEach((t) => t.stop());
        stream = s;
        video.srcObject = s;
        await video.play();
        stopLoop = startLiveLoop({
          video,
          client,
          lowConf: () => preset.lowConf,
          clock: () => performance.now(),
          onResult: (r, t) => {
            const frame = pipeline.process(r.detections, t);
            canvas.width = r.width;
            canvas.height = r.height;
            const ctx = canvas.getContext('2d')!;
            ctx.drawImage(video, 0, 0, r.width, r.height);
            paint(ctx, buildTrackPrimitives(frame), 1, 1);
            setCount(counts(frame.tracks));
            setInferMs(r.inferMs);
            setFrames((f) => f + 1);
          },
          onError: (e) => setError(e instanceof Error ? e.message : String(e)),
        });
      })
      .catch((e: Error) => setError(e.message));

    return () => {
      cancelled = true;
      stopLoop?.();
      stream?.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    };
  }, [live]);

  async function onImage(file: File) {
    const client = clientRef.current;
    const canvas = canvasRef.current;
    if (!client || !canvas) return;
    setLive(false);
    const shown = await createImageBitmap(file);
    const sent = await createImageBitmap(file);
    canvas.width = shown.width;
    canvas.height = shown.height;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(shown, 0, 0);
    shown.close();
    const r = await client.detect(sent, preset.lowConf);
    const dets = filterForImage(r.detections, preset.imageConf);
    paint(ctx, buildDetectionPrimitives(dets), 1, 1);
    setCount(counts(dets));
    setInferMs(r.inferMs);
  }

  return (
    <main className="mx-auto max-w-5xl space-y-4 p-4 font-mono text-sm">
      <h1 className="text-lg font-semibold">WeaponShield v2 — detection lab</h1>
      <dl className="grid grid-cols-2 gap-2 sm:grid-cols-6">
        <div>status: <span data-testid="status">{status}</span></div>
        <div>backend: <span data-testid="backend">{backend ?? '…'}</span></div>
        <div>download: {Math.round(progress * 100)}%</div>
        <div>person: <span data-testid="count-person">{count.person}</span></div>
        <div>weapon: <span data-testid="count-weapon">{count.weapon}</span></div>
        <div>frames: <span data-testid="frames">{frames}</span> · {inferMs.toFixed(0)} ms</div>
      </dl>
      {error && <p className="text-red-400">{error}</p>}
      <div className="flex flex-wrap gap-3">
        <label className="cursor-pointer rounded border border-slate-600 px-3 py-1">
          Upload image
          <input
            data-testid="image-input"
            type="file"
            accept="image/*"
            className="sr-only"
            disabled={status !== 'ready'}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void onImage(file);
            }}
          />
        </label>
        <button
          data-testid="webcam-toggle"
          type="button"
          className="rounded border border-slate-600 px-3 py-1 disabled:opacity-50"
          disabled={status !== 'ready'}
          onClick={() => setLive((v) => !v)}
        >
          {live ? 'Stop webcam' : 'Start webcam'}
        </button>
      </div>
      <video ref={videoRef} muted playsInline className="hidden" />
      <canvas ref={canvasRef} className="w-full rounded border border-slate-700" />
    </main>
  );
}
