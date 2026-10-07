import { describe, expect, it } from 'vitest';
import { area, center, iou, letterboxGeometry, unletterbox } from './geometry';

describe('letterboxGeometry', () => {
  it('fits a landscape frame by width and pads top/bottom', () => {
    expect(letterboxGeometry(1280, 720, 640)).toEqual({ size: 640, scale: 0.5, newW: 640, newH: 360, padX: 0, padY: 140 });
  });

  it('fits a portrait frame by height and pads left/right', () => {
    const lb = letterboxGeometry(810, 1080, 640);
    expect(lb.newH).toBe(640);
    expect(lb.newW).toBe(480);
    expect(lb.padX).toBe(80);
    expect(lb.padY).toBe(0);
  });

  it('puts the odd pixel of padding on the bottom/right like Ultralytics', () => {
    const lb = letterboxGeometry(1000, 999, 640);
    expect(lb.newH).toBe(639);
    expect(lb.padY).toBe(0);
  });

  it('scales small frames up', () => {
    expect(letterboxGeometry(320, 320, 640).scale).toBe(2);
  });
});

describe('unletterbox', () => {
  it('maps a letterboxed box back to source pixels', () => {
    const lb = letterboxGeometry(1280, 720, 640);
    expect(unletterbox({ x1: 100, y1: 190, x2: 300, y2: 290 }, lb, 1280, 720)).toEqual({ x1: 200, y1: 100, x2: 600, y2: 300 });
  });

  it('clamps boxes that spill into the padding', () => {
    const lb = letterboxGeometry(1280, 720, 640);
    const b = unletterbox({ x1: -10, y1: 100, x2: 700, y2: 600 }, lb, 1280, 720);
    expect(b).toEqual({ x1: 0, y1: 0, x2: 1280, y2: 720 });
  });
});

describe('box helpers', () => {
  it('computes IoU', () => {
    expect(iou({ x1: 0, y1: 0, x2: 10, y2: 10 }, { x1: 0, y1: 0, x2: 10, y2: 10 })).toBe(1);
    expect(iou({ x1: 0, y1: 0, x2: 10, y2: 10 }, { x1: 20, y1: 20, x2: 30, y2: 30 })).toBe(0);
    expect(iou({ x1: 0, y1: 0, x2: 10, y2: 10 }, { x1: 5, y1: 0, x2: 15, y2: 10 })).toBeCloseTo(50 / 150);
  });

  it('computes centre and area', () => {
    expect(center({ x1: 0, y1: 10, x2: 20, y2: 30 })).toEqual({ x: 10, y: 20 });
    expect(area({ x1: 0, y1: 0, x2: 4, y2: 5 })).toBe(20);
    expect(area({ x1: 5, y1: 5, x2: 1, y2: 1 })).toBe(0);
  });
});
