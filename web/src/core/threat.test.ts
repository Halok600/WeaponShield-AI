import { describe, expect, it } from 'vitest';
import { findOwner, isConfirmed, ThreatEngine } from './threat';
import type { Box, ClassKey, Track } from './types';

const track = (id: number, cls: ClassKey, box: Box, history: boolean[] = [true], conf = 0.8): Track => ({
  id,
  cls,
  conf,
  box,
  hits: history.filter(Boolean).length,
  missed: 0,
  history,
  trail: [],
});
const person = (id: number, x = 0) => track(id, 'person', { x1: x, y1: 0, x2: x + 100, y2: 200 });
const gun = (id: number, history: boolean[], x = 80, conf = 0.8) => track(id, 'pistol', { x1: x, y1: 90, x2: x + 30, y2: 110 }, history, conf);
const confirmedHistory = [true, true, true, true];

describe('isConfirmed', () => {
  it('needs K hits within the last N frames', () => {
    expect(isConfirmed([true, true, true, true], 4, 10)).toBe(true);
    expect(isConfirmed([true, false, true, false, true], 4, 10)).toBe(false);
    expect(isConfirmed([true, true, true, true, false, false, false, false, false, false, false], 4, 10)).toBe(false);
  });
});

describe('findOwner', () => {
  it('returns the person whose expanded box contains the weapon centre', () => {
    expect(findOwner(gun(9, [true], 105), [person(1), person(2, 300)], 0.2)).toBe(1);
  });
  it('picks the nearest person when several qualify', () => {
    // weapon centre x=105 lies inside both expanded boxes; person 2's centre (110) is nearer than person 1's (50)
    expect(findOwner(gun(9, [true], 90), [person(1, 0), person(2, 60)], 0.2)).toBe(2);
  });
  it('returns null when nobody holds it', () => {
    expect(findOwner(gun(9, [true], 600), [person(1)], 0.2)).toBeNull();
  });
});

describe('ThreatEngine', () => {
  it('does not raise an event before confirmation', () => {
    const te = new ThreatEngine();
    const f = te.update([gun(5, [true, true])], 0);
    expect(f.started).toEqual([]);
    expect(f.confirmedWeaponIds).toEqual([]);
  });

  it('starts one event on confirmation and marks the holder ARMED', () => {
    const te = new ThreatEngine();
    const f = te.update([person(1), gun(5, confirmedHistory)], 1000);
    expect(f.confirmedWeaponIds).toEqual([5]);
    expect(f.armedPersonIds).toEqual([1]);
    expect(f.weaponOwner.get(5)).toBe(1);
    expect(f.started).toEqual([
      { id: 1, weaponTrackId: 5, cls: 'pistol', personTrackId: 1, startMs: 1000, endMs: 1000, maxConf: 0.8, ended: false },
    ]);
    expect(te.update([person(1), gun(5, confirmedHistory)], 1100).started).toEqual([]);
  });

  it('extends the event while visible and tracks the peak confidence', () => {
    const te = new ThreatEngine();
    te.update([gun(5, confirmedHistory, 80, 0.7)], 0);
    const f = te.update([gun(5, confirmedHistory, 80, 0.95)], 500);
    expect(f.active[0]).toMatchObject({ endMs: 500, maxConf: 0.95 });
  });

  it('ends the event after the weapon is gone for longer than eventGapMs', () => {
    const te = new ThreatEngine({ eventGapMs: 2000 });
    te.update([gun(5, confirmedHistory)], 0);
    expect(te.update([], 1500).ended).toEqual([]);
    const f = te.update([], 2500);
    expect(f.ended).toHaveLength(1);
    expect(f.ended[0]).toMatchObject({ id: 1, ended: true, endMs: 0 });
    expect(f.active).toEqual([]);
    expect(te.events()).toHaveLength(1);
  });

  it('returns copies so callers cannot mutate engine state', () => {
    const te = new ThreatEngine();
    const f = te.update([gun(5, confirmedHistory)], 0);
    f.started[0]!.maxConf = 0;
    expect(te.events()[0]!.maxConf).toBe(0.8);
  });
});
