import { describe, expect, it } from 'vitest';
import { BoxKalman, Kalman1D } from './kalman';

describe('Kalman1D', () => {
  it('learns a constant velocity', () => {
    const k = new Kalman1D(0, 1, 1);
    for (let t = 1; t <= 20; t++) {
      k.predict(0.01, 0.01);
      k.update(t * 3, 0.1);
    }
    expect(Math.abs(k.v - 3)).toBeLessThan(0.2);
    expect(Math.abs(k.predict(0.01, 0.01) - 63)).toBeLessThan(1);
  });
});

describe('BoxKalman', () => {
  it('predicts where a moving box goes next', () => {
    const kf = new BoxKalman({ x1: 0, y1: 0, x2: 50, y2: 100 });
    for (let t = 1; t <= 10; t++) {
      kf.predict();
      kf.update({ x1: t * 5, y1: 0, x2: 50 + t * 5, y2: 100 });
    }
    const p = kf.predict();
    expect(Math.abs(p.x1 - 55)).toBeLessThan(2);
    expect(Math.abs(p.x2 - 105)).toBeLessThan(2);
    expect(Math.abs(p.y2 - p.y1 - 100)).toBeLessThan(1);
  });
});
