import type { PipelineFrame } from '../core/pipeline';
import { type Box, type Detection, isWeapon, type Point } from '../core/types';

export const OVERLAY_COLORS = { person: '#22c55e', weapon: '#ef4444', armed: '#f59e0b' } as const;

export type Primitive =
  | { kind: 'box'; box: Box; color: string; label: string; solid: boolean }
  | { kind: 'trail'; points: readonly Point[]; color: string };

const pct = (c: number) => `${Math.round(c * 100)}%`;

export function buildTrackPrimitives(frame: PipelineFrame): Primitive[] {
  const confirmed = new Set(frame.threat.confirmedWeaponIds);
  const armed = new Set(frame.threat.armedPersonIds);
  const prims: Primitive[] = [];
  for (const tr of frame.tracks) {
    let color: string;
    let label: string;
    let solid = true;
    if (isWeapon(tr.cls)) {
      color = OVERLAY_COLORS.weapon;
      label = `#${tr.id} ${tr.cls} ${pct(tr.conf)}`;
      solid = confirmed.has(tr.id);
    } else if (armed.has(tr.id)) {
      color = OVERLAY_COLORS.armed;
      label = `#${tr.id} person · ARMED`;
    } else {
      color = OVERLAY_COLORS.person;
      label = `#${tr.id} person`;
    }
    if (tr.trail.length >= 2) prims.push({ kind: 'trail', points: tr.trail, color });
    prims.push({ kind: 'box', box: tr.box, color, label, solid });
  }
  // Boxes after trails so labels sit on top; keep track order among boxes.
  return [...prims.filter((p) => p.kind === 'trail'), ...prims.filter((p) => p.kind === 'box')];
}

export const buildDetectionPrimitives = (dets: readonly Detection[]): Primitive[] =>
  dets.map((d) => ({
    kind: 'box',
    box: d.box,
    color: isWeapon(d.cls) ? OVERLAY_COLORS.weapon : OVERLAY_COLORS.person,
    label: `${d.cls} ${pct(d.conf)}`,
    solid: true,
  }));

export function paint(ctx: CanvasRenderingContext2D, prims: readonly Primitive[], scaleX: number, scaleY: number): void {
  const lw = Math.max(2, Math.round(2 * Math.max(scaleX, scaleY)));
  ctx.lineJoin = 'round';
  for (const p of prims) {
    ctx.strokeStyle = p.color;
    if (p.kind === 'trail') {
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = lw;
      ctx.setLineDash([]);
      ctx.beginPath();
      p.points.forEach((pt, i) => (i === 0 ? ctx.moveTo(pt.x * scaleX, pt.y * scaleY) : ctx.lineTo(pt.x * scaleX, pt.y * scaleY)));
      ctx.stroke();
      ctx.globalAlpha = 1;
      continue;
    }
    const x = p.box.x1 * scaleX;
    const y = p.box.y1 * scaleY;
    const w = (p.box.x2 - p.box.x1) * scaleX;
    const h = (p.box.y2 - p.box.y1) * scaleY;
    ctx.lineWidth = lw;
    ctx.setLineDash(p.solid ? [] : [6, 4]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
    const fontPx = Math.max(12, Math.round(13 * Math.max(scaleX, scaleY)));
    ctx.font = `600 ${fontPx}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    const tw = ctx.measureText(p.label).width + 8;
    const th = fontPx + 6;
    const ly = y - th < 0 ? y : y - th;
    ctx.fillStyle = p.color;
    ctx.fillRect(x, ly, tw, th);
    ctx.fillStyle = '#0b0f14';
    ctx.fillText(p.label, x + 4, ly + fontPx);
  }
}
