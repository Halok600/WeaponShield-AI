import { describe, expect, it } from 'vitest';
import { letterboxGeometry } from './geometry';
import { decodeEndToEnd, uniformThresholds } from './postprocess';

const lb = letterboxGeometry(1280, 720, 640); // scale 0.5, padY 140
const labels = { 0: 'person', 43: 'knife' } as const;

function rows(...r: number[][]): Float32Array {
  const out = new Float32Array(300 * 6);
  r.forEach((row, i) => out.set(row, i * 6));
  return out;
}

describe('decodeEndToEnd', () => {
  it('keeps mapped classes above threshold, maps boxes to source pixels, sorts by confidence', () => {
    const data = rows([100, 190, 300, 290, 0.6, 0], [10, 150, 20, 160, 0.9, 43]);
    const dets = decodeEndToEnd(data, [1, 300, 6], labels, uniformThresholds(0.25), lb, 1280, 720);
    expect(dets).toEqual([
      { cls: 'knife', conf: expect.closeTo(0.9, 5), box: { x1: 20, y1: 20, x2: 40, y2: 40 } },
      { cls: 'person', conf: expect.closeTo(0.6, 5), box: { x1: 200, y1: 100, x2: 600, y2: 300 } },
    ]);
  });

  it('drops unmapped classes and low-confidence rows', () => {
    const data = rows([100, 190, 300, 290, 0.9, 2], [100, 190, 300, 290, 0.2, 0]);
    expect(decodeEndToEnd(data, [1, 300, 6], labels, uniformThresholds(0.25), lb, 1280, 720)).toEqual([]);
  });

  it('applies per-class thresholds', () => {
    const data = rows([100, 190, 300, 290, 0.5, 0], [100, 190, 300, 290, 0.5, 43]);
    const thresholds = { ...uniformThresholds(0.25), knife: 0.6 };
    const dets = decodeEndToEnd(data, [1, 300, 6], labels, thresholds, lb, 1280, 720);
    expect(dets.map((d) => d.cls)).toEqual(['person']);
  });

  it('drops boxes that collapse to under a pixel after clamping', () => {
    const data = rows([0, 0, 640, 100, 0.9, 0]); // entirely inside the top padding
    expect(decodeEndToEnd(data, [1, 300, 6], labels, uniformThresholds(0.25), lb, 1280, 720)).toEqual([]);
  });

  it('rejects unexpected output shapes', () => {
    expect(() => decodeEndToEnd(new Float32Array(84 * 8400), [1, 84, 8400], labels, uniformThresholds(0.25), lb, 1280, 720)).toThrow(
      /\[1,84,8400\]/,
    );
  });
});
