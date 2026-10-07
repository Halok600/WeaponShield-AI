import { type Letterbox, unletterbox } from './geometry';
import { CLASS_KEYS, type ClassKey, type Detection } from './types';

export type ClassThresholds = Readonly<Record<ClassKey, number>>;

export const uniformThresholds = (conf: number): ClassThresholds =>
  Object.fromEntries(CLASS_KEYS.map((k) => [k, conf])) as Record<ClassKey, number>;

/** Decode an end-to-end YOLO26 output ([1, N, 6] = x1,y1,x2,y2,conf,class in letterboxed pixels). */
export function decodeEndToEnd(
  data: ArrayLike<number>,
  dims: readonly number[],
  labels: Readonly<Record<number, ClassKey>>,
  thresholds: ClassThresholds,
  lb: Letterbox,
  srcW: number,
  srcH: number,
): Detection[] {
  if (dims.length !== 3 || dims[0] !== 1 || dims[2] !== 6) {
    throw new Error(`Unexpected model output shape [${dims.join(',')}], expected [1,N,6]`);
  }
  const n = dims[1]!;
  const dets: Detection[] = [];
  for (let i = 0; i < n; i++) {
    const o = i * 6;
    const conf = data[o + 4]!;
    const cls = labels[Math.round(data[o + 5]!)];
    if (cls === undefined || conf < thresholds[cls]) continue;
    const box = unletterbox({ x1: data[o]!, y1: data[o + 1]!, x2: data[o + 2]!, y2: data[o + 3]! }, lb, srcW, srcH);
    if (box.x2 - box.x1 < 1 || box.y2 - box.y1 < 1) continue;
    dets.push({ cls, conf, box });
  }
  return dets.sort((a, b) => b.conf - a.conf);
}
