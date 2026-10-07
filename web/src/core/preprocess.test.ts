import { describe, expect, it } from 'vitest';
import { rgbaToTensor } from './preprocess';

describe('rgbaToTensor', () => {
  it('converts RGBA bytes to planar RGB floats in [0,1] and drops alpha', () => {
    // 2x2 image: red, green, blue, white
    const rgba = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255]);
    const t = rgbaToTensor(rgba, 2);
    expect(Array.from(t)).toEqual([1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]);
  });

  it('reuses the provided output buffer', () => {
    const out = new Float32Array(3);
    expect(rgbaToTensor(new Uint8ClampedArray([51, 102, 204, 255]), 1, out)).toBe(out);
    expect(out[0]).toBeCloseTo(0.2);
  });

  it('rejects input of the wrong length', () => {
    expect(() => rgbaToTensor(new Uint8ClampedArray(8), 2)).toThrow(/expected 16/);
  });
});
