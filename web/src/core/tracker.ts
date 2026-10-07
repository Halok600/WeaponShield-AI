import { center, iou } from './geometry';
import { BoxKalman } from './kalman';
import { type Box, type ClassKey, type Detection, isWeapon, type Point, type Track } from './types';

export interface TrackerOptions {
  /** A detection at or above this starts tracks and joins the first association pass. */
  highConf: Readonly<Record<ClassKey, number>>;
  /** Detections between lowConf and highConf can only continue tracks seen last frame. */
  lowConf: number;
  firstIou: number;
  secondIou: number;
  minHits: number;
  maxMissed: number;
  historyLength: number;
  trailLength: number;
}

export const DEFAULT_TRACKER_OPTIONS: TrackerOptions = {
  highConf: { person: 0.5, pistol: 0.45, rifle: 0.45, knife: 0.45 },
  lowConf: 0.1,
  firstIou: 0.2,
  secondIou: 0.4,
  minHits: 2,
  maxMissed: 30,
  historyLength: 30,
  trailLength: 30,
};

interface InternalTrack {
  id: number;
  kf: BoxKalman;
  predicted: Box;
  votes: Map<ClassKey, number>;
  cls: ClassKey;
  conf: number;
  hits: number;
  missed: number;
  history: boolean[];
  trail: Point[];
}

interface Matching {
  pairs: [number, number][];
  unmatchedTracks: number[];
  unmatchedDets: number[];
}

const sameGroup = (a: ClassKey, b: ClassKey) => isWeapon(a) === isWeapon(b);

function pushCapped<T>(arr: T[], v: T, cap: number): void {
  arr.push(v);
  if (arr.length > cap) arr.splice(0, arr.length - cap);
}

/** Greedy IoU matching (highest IoU first). Persons and weapons never match each other. */
function greedyMatch(tracks: readonly InternalTrack[], dets: readonly Detection[], minIou: number): Matching {
  const candidates: { t: number; d: number; v: number }[] = [];
  tracks.forEach((t, ti) =>
    dets.forEach((d, di) => {
      if (!sameGroup(t.cls, d.cls)) return;
      const v = iou(t.predicted, d.box);
      if (v >= minIou) candidates.push({ t: ti, d: di, v });
    }),
  );
  candidates.sort((a, b) => b.v - a.v);
  const usedT = new Set<number>();
  const usedD = new Set<number>();
  const pairs: [number, number][] = [];
  for (const c of candidates) {
    if (usedT.has(c.t) || usedD.has(c.d)) continue;
    usedT.add(c.t);
    usedD.add(c.d);
    pairs.push([c.t, c.d]);
  }
  return {
    pairs,
    unmatchedTracks: tracks.map((_, i) => i).filter((i) => !usedT.has(i)),
    unmatchedDets: dets.map((_, i) => i).filter((i) => !usedD.has(i)),
  };
}

export class Tracker {
  private opts: TrackerOptions;
  private tracks: InternalTrack[] = [];
  private nextId = 1;

  constructor(opts: Partial<TrackerOptions> = {}) {
    this.opts = { ...DEFAULT_TRACKER_OPTIONS, ...opts };
  }

  setOptions(opts: Partial<TrackerOptions>): void {
    this.opts = { ...this.opts, ...opts };
  }

  reset(): void {
    this.tracks = [];
    this.nextId = 1;
  }

  update(dets: readonly Detection[]): Track[] {
    const o = this.opts;
    for (const t of this.tracks) t.predicted = t.kf.predict();

    const high = dets.filter((d) => d.conf >= o.highConf[d.cls]);
    const low = dets.filter((d) => d.conf >= o.lowConf && d.conf < o.highConf[d.cls]);

    const first = greedyMatch(this.tracks, high, o.firstIou);
    for (const [ti, di] of first.pairs) this.apply(this.tracks[ti]!, high[di]!);

    const remaining = first.unmatchedTracks.map((i) => this.tracks[i]!);
    const seenLastFrame = remaining.filter((t) => t.missed === 0 && t.hits >= o.minHits);
    const second = greedyMatch(seenLastFrame, low, o.secondIou);
    const rescued = new Set<InternalTrack>();
    for (const [ti, di] of second.pairs) {
      this.apply(seenLastFrame[ti]!, low[di]!);
      rescued.add(seenLastFrame[ti]!);
    }

    for (const t of remaining) {
      if (rescued.has(t)) continue;
      t.missed += 1;
      pushCapped(t.history, false, o.historyLength);
    }

    // Confirmed tracks survive maxMissed frames; tentative ones die on their first miss.
    this.tracks = this.tracks.filter((t) => (t.hits >= o.minHits ? t.missed <= o.maxMissed : t.missed === 0));

    for (const di of first.unmatchedDets) this.spawn(high[di]!);

    return this.tracks
      .filter((t) => t.missed === 0 && t.hits >= o.minHits)
      .sort((a, b) => a.id - b.id)
      .map((t) => ({
        id: t.id,
        cls: t.cls,
        conf: t.conf,
        box: t.kf.box(),
        hits: t.hits,
        missed: t.missed,
        history: [...t.history],
        trail: [...t.trail],
      }));
  }

  private apply(t: InternalTrack, d: Detection): void {
    t.kf.update(d.box);
    t.hits += 1;
    t.missed = 0;
    t.conf = 0.6 * t.conf + 0.4 * d.conf;
    t.votes.set(d.cls, (t.votes.get(d.cls) ?? 0) + d.conf);
    let best = t.cls;
    for (const [cls, score] of t.votes) if (score > (t.votes.get(best) ?? 0)) best = cls;
    t.cls = best;
    pushCapped(t.history, true, this.opts.historyLength);
    pushCapped(t.trail, center(t.kf.box()), this.opts.trailLength);
  }

  private spawn(d: Detection): void {
    this.tracks.push({
      id: this.nextId++,
      kf: new BoxKalman(d.box),
      predicted: d.box,
      votes: new Map([[d.cls, d.conf]]),
      cls: d.cls,
      conf: d.conf,
      hits: 1,
      missed: 0,
      history: [true],
      trail: [center(d.box)],
    });
  }
}
