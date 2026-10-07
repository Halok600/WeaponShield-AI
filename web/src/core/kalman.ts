import type { Box } from './types';

/** Constant-velocity Kalman filter for one coordinate. State [x, v]; F = [[1,1],[0,1]], H = [1,0]. */
export class Kalman1D {
  x: number;
  v = 0;
  private p00: number;
  private p01 = 0;
  private p10 = 0;
  private p11: number;

  constructor(x0: number, posVar: number, velVar: number) {
    this.x = x0;
    this.p00 = posVar;
    this.p11 = velVar;
  }

  predict(qPos: number, qVel: number): number {
    this.x += this.v;
    // P = F P F^T + Q
    const p00 = this.p00 + this.p01 + this.p10 + this.p11 + qPos;
    const p01 = this.p01 + this.p11;
    const p10 = this.p10 + this.p11;
    const p11 = this.p11 + qVel;
    this.p00 = p00;
    this.p01 = p01;
    this.p10 = p10;
    this.p11 = p11;
    return this.x;
  }

  update(z: number, r: number): void {
    const s = this.p00 + r;
    const k0 = this.p00 / s;
    const k1 = this.p10 / s;
    const y = z - this.x;
    this.x += k0 * y;
    this.v += k1 * y;
    const p00 = (1 - k0) * this.p00;
    const p01 = (1 - k0) * this.p01;
    const p10 = this.p10 - k1 * this.p00;
    const p11 = this.p11 - k1 * this.p01;
    this.p00 = p00;
    this.p01 = p01;
    this.p10 = p10;
    this.p11 = p11;
  }
}

// Noise weights from ByteTrack's KalmanFilter, scaled by box height.
const POS_W = 1 / 20;
const VEL_W = 1 / 160;

/** Four independent constant-velocity filters over (cx, cy, w, h). */
export class BoxKalman {
  private readonly f: Kalman1D[];

  constructor(b: Box) {
    const s = Math.max(b.y2 - b.y1, 1);
    const posVar = (2 * POS_W * s) ** 2;
    const velVar = (10 * VEL_W * s) ** 2;
    this.f = toCxcywh(b).map((v) => new Kalman1D(v, posVar, velVar));
  }

  predict(): Box {
    const h = Math.max(this.f[3]!.x, 1);
    const q = (POS_W * h) ** 2;
    const qv = (VEL_W * h) ** 2;
    for (const k of this.f) k.predict(q, qv);
    return this.box();
  }

  update(b: Box): void {
    const z = toCxcywh(b);
    const r = (POS_W * Math.max(z[3]!, 1)) ** 2;
    this.f.forEach((k, i) => k.update(z[i]!, r));
  }

  box(): Box {
    const [cx, cy, w, h] = this.f.map((k) => k.x) as [number, number, number, number];
    const hw = Math.max(w, 1) / 2;
    const hh = Math.max(h, 1) / 2;
    return { x1: cx - hw, y1: cy - hh, x2: cx + hw, y2: cy + hh };
  }
}

function toCxcywh(b: Box): number[] {
  return [(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2, b.x2 - b.x1, b.y2 - b.y1];
}
