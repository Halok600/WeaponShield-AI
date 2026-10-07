import { describe, expect, it } from 'vitest';
import { Tracker } from './tracker';
import type { Box, ClassKey, Detection } from './types';

const det = (cls: ClassKey, conf: number, box: Box): Detection => ({ cls, conf, box });
const box = (x: number, y: number, w = 50, h = 100): Box => ({ x1: x, y1: y, x2: x + w, y2: y + h });

describe('Tracker', () => {
  it('waits for minHits before reporting a new track', () => {
    const tr = new Tracker();
    expect(tr.update([det('person', 0.9, box(0, 0))])).toEqual([]);
    expect(tr.update([det('person', 0.9, box(2, 0))])).toHaveLength(1);
  });

  it('keeps one id for an object moving steadily', () => {
    const tr = new Tracker();
    const ids = new Set<number>();
    for (let t = 0; t < 15; t++) for (const track of tr.update([det('person', 0.9, box(t * 6, 0))])) ids.add(track.id);
    expect([...ids]).toEqual([1]);
  });

  it('keeps ids apart for two people walking past each other', () => {
    const tr = new Tracker();
    let last: { id: number; x: number }[] = [];
    for (let t = 0; t < 20; t++) {
      const out = tr.update([det('person', 0.9, box(t * 10, 0)), det('person', 0.9, box(200 - t * 10, 40))]);
      last = out.map((o) => ({ id: o.id, x: o.box.x1 }));
    }
    const left = last.find((l) => l.x > 150)!; // the one that started on the left is now on the right
    expect(left.id).toBe(1);
  });

  it('re-acquires the same id after a short occlusion', () => {
    const tr = new Tracker();
    for (let t = 0; t < 5; t++) tr.update([det('person', 0.9, box(t * 5, 0))]);
    for (let t = 5; t < 10; t++) expect(tr.update([])).toEqual([]);
    const out = tr.update([det('person', 0.9, box(50, 0))]);
    expect(out.map((o) => o.id)).toEqual([1]);
    expect(out[0]!.history.slice(-6)).toEqual([false, false, false, false, false, true]);
  });

  it('uses low-confidence detections to continue a track but never to start one', () => {
    const tr = new Tracker();
    tr.update([det('pistol', 0.9, box(0, 0, 20, 20))]);
    tr.update([det('pistol', 0.9, box(1, 0, 20, 20))]);
    expect(tr.update([det('pistol', 0.2, box(2, 0, 20, 20))]).map((t) => t.id)).toEqual([1]);
    expect(tr.update([det('pistol', 0.2, box(300, 300, 20, 20))])).toEqual([]);
    expect(tr.update([det('pistol', 0.2, box(301, 300, 20, 20))])).toEqual([]);
  });

  it('never matches a person detection to a weapon track', () => {
    const tr = new Tracker();
    tr.update([det('knife', 0.9, box(0, 0))]);
    tr.update([det('knife', 0.9, box(0, 0))]);
    const out = tr.update([det('person', 0.9, box(0, 0))]);
    expect(out).toEqual([]); // knife track missed, person track is new (1 hit)
  });

  it('chooses the class by confidence-weighted vote within weapons', () => {
    const tr = new Tracker();
    const seq: ClassKey[] = ['pistol', 'pistol', 'rifle', 'pistol'];
    let out = tr.update([]);
    for (const cls of seq) out = tr.update([det(cls, 0.8, box(0, 0, 30, 30))]);
    expect(out[0]!.cls).toBe('pistol');
  });

  it('drops tracks missed for more than maxMissed frames', () => {
    const tr = new Tracker({ maxMissed: 3 });
    tr.update([det('person', 0.9, box(0, 0))]);
    tr.update([det('person', 0.9, box(0, 0))]);
    for (let i = 0; i < 4; i++) tr.update([]);
    tr.update([det('person', 0.9, box(0, 0))]);
    expect(tr.update([det('person', 0.9, box(0, 0))])[0]!.id).toBe(2);
  });

  it('records a trail of centres', () => {
    const tr = new Tracker();
    let out = tr.update([]);
    for (let t = 0; t < 4; t++) out = tr.update([det('person', 0.9, box(t * 10, 0))]);
    expect(out[0]!.trail.length).toBe(4);
  });
});
