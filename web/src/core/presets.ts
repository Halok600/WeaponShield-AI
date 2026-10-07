import type { ClassKey, Detection } from './types';

export type SensitivityName = 'low' | 'balanced' | 'high';

export interface Preset {
  /** Floor passed to the decoder; anything below is discarded before tracking. */
  lowConf: number;
  /** Per-class confidence needed to start a track (tracker's high-confidence pass). */
  highConf: Record<ClassKey, number>;
  /** Per-class confidence for single-image mode, where there is no temporal confirmation. */
  imageConf: Record<ClassKey, number>;
  confirmK: number;
  confirmN: number;
}

// Provisional values for the stock model. Plan 5 retunes them from the trained model's validation PR curves.
export const PRESETS: Record<SensitivityName, Preset> = {
  low: {
    lowConf: 0.15,
    highConf: { person: 0.5, pistol: 0.6, rifle: 0.6, knife: 0.6 },
    imageConf: { person: 0.5, pistol: 0.7, rifle: 0.7, knife: 0.7 },
    confirmK: 6,
    confirmN: 10,
  },
  balanced: {
    lowConf: 0.1,
    highConf: { person: 0.45, pistol: 0.45, rifle: 0.45, knife: 0.45 },
    imageConf: { person: 0.5, pistol: 0.55, rifle: 0.55, knife: 0.55 },
    confirmK: 4,
    confirmN: 10,
  },
  high: {
    lowConf: 0.05,
    highConf: { person: 0.4, pistol: 0.3, rifle: 0.3, knife: 0.3 },
    imageConf: { person: 0.4, pistol: 0.4, rifle: 0.4, knife: 0.4 },
    confirmK: 3,
    confirmN: 10,
  },
};

export const filterForImage = (dets: readonly Detection[], conf: Readonly<Record<ClassKey, number>>): Detection[] =>
  dets.filter((d) => d.conf >= conf[d.cls]);
