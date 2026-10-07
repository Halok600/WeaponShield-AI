import type { Preset } from './presets';
import { type ThreatFrame, ThreatEngine } from './threat';
import { Tracker } from './tracker';
import type { Detection, Track } from './types';

export interface PipelineFrame {
  tracks: Track[];
  threat: ThreatFrame;
}

export class Pipeline {
  private readonly tracker = new Tracker();
  private readonly threat = new ThreatEngine();

  constructor(preset: Preset) {
    this.setPreset(preset);
  }

  setPreset(p: Preset): void {
    this.tracker.setOptions({ highConf: p.highConf, lowConf: p.lowConf });
    this.threat.setOptions({ confirmK: p.confirmK, confirmN: p.confirmN });
  }

  process(dets: readonly Detection[], tMs: number): PipelineFrame {
    const tracks = this.tracker.update(dets);
    return { tracks, threat: this.threat.update(tracks, tMs) };
  }

  reset(): void {
    this.tracker.reset();
    this.threat.reset();
  }
}
