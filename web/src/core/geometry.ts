import type { Box, Point } from './types';

export interface Letterbox {
  size: number;
  scale: number;
  newW: number;
  newH: number;
  padX: number;
  padY: number;
}

/** Same geometry as Ultralytics LetterBox(auto=False): centred, odd padding pixel on the bottom/right. */
export function letterboxGeometry(srcW: number, srcH: number, size: number): Letterbox {
  const scale = Math.min(size / srcW, size / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  return { size, scale, newW, newH, padX: Math.floor((size - newW) / 2), padY: Math.floor((size - newH) / 2) };
}

const clamp = (v: number, max: number) => Math.min(max, Math.max(0, v));

export function unletterbox(b: Box, lb: Letterbox, srcW: number, srcH: number): Box {
  return {
    x1: clamp((b.x1 - lb.padX) / lb.scale, srcW),
    y1: clamp((b.y1 - lb.padY) / lb.scale, srcH),
    x2: clamp((b.x2 - lb.padX) / lb.scale, srcW),
    y2: clamp((b.y2 - lb.padY) / lb.scale, srcH),
  };
}

export const area = (b: Box): number => Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);

export const center = (b: Box): Point => ({ x: (b.x1 + b.x2) / 2, y: (b.y1 + b.y2) / 2 });

export function iou(a: Box, b: Box): number {
  const inter = area({ x1: Math.max(a.x1, b.x1), y1: Math.max(a.y1, b.y1), x2: Math.min(a.x2, b.x2), y2: Math.min(a.y2, b.y2) });
  const union = area(a) + area(b) - inter;
  return union > 0 ? inter / union : 0;
}
