import { center } from './geometry';
import { type ClassKey, isWeapon, type Track } from './types';

export interface ThreatOptions {
  /** A weapon track is confirmed once matched in at least confirmK of its last confirmN frames. */
  confirmK: number;
  confirmN: number;
  /** A person box is grown by this fraction of its width/height on each side when testing who holds a weapon. */
  armedExpand: number;
  /** An event ends after its weapon has been unseen for this long. */
  eventGapMs: number;
}

export const DEFAULT_THREAT_OPTIONS: ThreatOptions = { confirmK: 4, confirmN: 10, armedExpand: 0.2, eventGapMs: 2000 };

export interface ThreatEvent {
  id: number;
  weaponTrackId: number;
  cls: ClassKey;
  personTrackId: number | null;
  startMs: number;
  endMs: number;
  maxConf: number;
  ended: boolean;
}

export interface ThreatFrame {
  confirmedWeaponIds: number[];
  armedPersonIds: number[];
  weaponOwner: ReadonlyMap<number, number>;
  active: ThreatEvent[];
  started: ThreatEvent[];
  ended: ThreatEvent[];
}

export function isConfirmed(history: readonly boolean[], k: number, n: number): boolean {
  let hits = 0;
  for (let i = Math.max(0, history.length - n); i < history.length; i++) if (history[i]) hits++;
  return hits >= k;
}

export function findOwner(weapon: Track, persons: readonly Track[], expand: number): number | null {
  const c = center(weapon.box);
  let best: number | null = null;
  let bestDist = Infinity;
  for (const p of persons) {
    const w = p.box.x2 - p.box.x1;
    const h = p.box.y2 - p.box.y1;
    const inside =
      c.x >= p.box.x1 - expand * w && c.x <= p.box.x2 + expand * w && c.y >= p.box.y1 - expand * h && c.y <= p.box.y2 + expand * h;
    if (!inside) continue;
    const pc = center(p.box);
    const d = Math.hypot(pc.x - c.x, pc.y - c.y);
    if (d < bestDist) {
      bestDist = d;
      best = p.id;
    }
  }
  return best;
}

const copy = (e: ThreatEvent): ThreatEvent => ({ ...e });

export class ThreatEngine {
  private opts: ThreatOptions;
  private confirmed = new Set<number>();
  private open = new Map<number, ThreatEvent>(); // weapon track id -> event
  private all: ThreatEvent[] = [];
  private nextId = 1;

  constructor(opts: Partial<ThreatOptions> = {}) {
    this.opts = { ...DEFAULT_THREAT_OPTIONS, ...opts };
  }

  setOptions(opts: Partial<ThreatOptions>): void {
    this.opts = { ...this.opts, ...opts };
  }

  reset(): void {
    this.confirmed.clear();
    this.open.clear();
    this.all = [];
    this.nextId = 1;
  }

  events(): ThreatEvent[] {
    return this.all.map(copy);
  }

  update(tracks: readonly Track[], tMs: number): ThreatFrame {
    const { confirmK, confirmN, armedExpand, eventGapMs } = this.opts;
    const persons = tracks.filter((t) => t.cls === 'person');
    const weapons = tracks.filter((t) => isWeapon(t.cls));
    const owner = new Map<number, number>();
    const armed = new Set<number>();
    const started: ThreatEvent[] = [];
    const ended: ThreatEvent[] = [];

    for (const w of weapons) {
      if (!this.confirmed.has(w.id) && isConfirmed(w.history, confirmK, confirmN)) this.confirmed.add(w.id);
      if (!this.confirmed.has(w.id)) continue;

      const pid = findOwner(w, persons, armedExpand);
      if (pid !== null) {
        owner.set(w.id, pid);
        armed.add(pid);
      }

      const ev = this.open.get(w.id);
      if (ev === undefined) {
        const created: ThreatEvent = {
          id: this.nextId++,
          weaponTrackId: w.id,
          cls: w.cls,
          personTrackId: pid,
          startMs: tMs,
          endMs: tMs,
          maxConf: w.conf,
          ended: false,
        };
        this.open.set(w.id, created);
        this.all.push(created);
        started.push(copy(created));
      } else {
        ev.endMs = tMs;
        ev.cls = w.cls;
        ev.maxConf = Math.max(ev.maxConf, w.conf);
        if (ev.personTrackId === null) ev.personTrackId = pid;
      }
    }

    const visible = new Set(weapons.map((w) => w.id));
    for (const [wid, ev] of this.open) {
      if (visible.has(wid) || tMs - ev.endMs <= eventGapMs) continue;
      ev.ended = true;
      ended.push(copy(ev));
      this.open.delete(wid);
      this.confirmed.delete(wid);
    }

    return {
      confirmedWeaponIds: weapons.filter((w) => this.confirmed.has(w.id)).map((w) => w.id),
      armedPersonIds: [...armed],
      weaponOwner: owner,
      active: [...this.open.values()].map(copy),
      started,
      ended,
    };
  }
}
