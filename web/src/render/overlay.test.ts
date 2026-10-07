import { describe, expect, it } from 'vitest';
import type { PipelineFrame } from '../core/pipeline';
import type { Track } from '../core/types';
import { buildDetectionPrimitives, buildTrackPrimitives, OVERLAY_COLORS, type Primitive } from './overlay';

type BoxPrim = Extract<Primitive, { kind: 'box' }>;
const isBox = (p: Primitive): p is BoxPrim => p.kind === 'box';

const t = (id: number, cls: Track['cls'], conf = 0.87): Track => ({
  id,
  cls,
  conf,
  box: { x1: 0, y1: 0, x2: 10, y2: 10 },
  hits: 3,
  missed: 0,
  history: [true, true, true],
  trail: [{ x: 1, y: 1 }, { x: 2, y: 2 }],
});

const frame = (tracks: Track[], confirmed: number[], armed: number[]): PipelineFrame => ({
  tracks,
  threat: { confirmedWeaponIds: confirmed, armedPersonIds: armed, weaponOwner: new Map(), active: [], started: [], ended: [] },
});

describe('buildTrackPrimitives', () => {
  it('colours people green, armed people amber, weapons red', () => {
    const prims = buildTrackPrimitives(frame([t(1, 'person'), t(2, 'person'), t(3, 'pistol')], [3], [2]));
    const boxes = prims.filter(isBox);
    expect(boxes.map((b) => [b.color, b.label, b.solid])).toEqual([
      [OVERLAY_COLORS.person, '#1 person', true],
      [OVERLAY_COLORS.armed, '#2 person · ARMED', true],
      [OVERLAY_COLORS.weapon, '#3 pistol 87%', true],
    ]);
  });

  it('draws unconfirmed weapons dashed', () => {
    const [box] = buildTrackPrimitives(frame([t(3, 'knife')], [], [])).filter(isBox);
    expect(box).toMatchObject({ solid: false });
  });

  it('adds a trail for each track with at least two points', () => {
    expect(buildTrackPrimitives(frame([t(1, 'person')], [], [])).filter((p) => p.kind === 'trail')).toHaveLength(1);
  });
});

describe('buildDetectionPrimitives', () => {
  it('labels raw detections with class and confidence', () => {
    const prims = buildDetectionPrimitives([{ cls: 'rifle', conf: 0.5, box: { x1: 0, y1: 0, x2: 5, y2: 5 } }]);
    expect(prims).toEqual([{ kind: 'box', box: { x1: 0, y1: 0, x2: 5, y2: 5 }, color: OVERLAY_COLORS.weapon, label: 'rifle 50%', solid: true }]);
  });
});
