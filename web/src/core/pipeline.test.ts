import { describe, expect, it } from 'vitest';
import { Pipeline } from './pipeline';
import { filterForImage, PRESETS } from './presets';
import type { Detection } from './types';

const personDet: Detection = { cls: 'person', conf: 0.9, box: { x1: 0, y1: 0, x2: 100, y2: 200 } };
const gunDet: Detection = { cls: 'pistol', conf: 0.9, box: { x1: 80, y1: 90, x2: 110, y2: 110 } };

describe('Pipeline', () => {
  it('raises an ARMED event after the balanced preset confirms the weapon', () => {
    const p = new Pipeline(PRESETS.balanced);
    let startedAt = -1;
    for (let i = 0; i < 10; i++) {
      const f = p.process([personDet, gunDet], i * 100);
      if (f.threat.started.length > 0 && startedAt < 0) startedAt = i;
    }
    // Track becomes visible on frame 1 (minHits 2); history needs confirmK true entries.
    expect(startedAt).toBe(PRESETS.balanced.confirmK - 1);
    const last = p.process([personDet, gunDet], 1000);
    expect(last.threat.armedPersonIds).toHaveLength(1);
  });

  it('reset clears tracks and events', () => {
    const p = new Pipeline(PRESETS.balanced);
    for (let i = 0; i < 6; i++) p.process([gunDet], i * 100);
    p.reset();
    expect(p.process([gunDet], 0).tracks).toEqual([]);
  });
});

describe('filterForImage', () => {
  it('applies per-class confidence floors', () => {
    const dets: Detection[] = [
      { ...personDet, conf: 0.4 },
      { ...gunDet, conf: 0.7 },
    ];
    expect(filterForImage(dets, PRESETS.balanced.imageConf).map((d) => d.cls)).toEqual(['pistol']);
  });
});
