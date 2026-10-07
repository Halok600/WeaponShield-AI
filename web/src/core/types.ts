export type ClassKey = 'person' | 'pistol' | 'rifle' | 'knife';

export const CLASS_KEYS: readonly ClassKey[] = ['person', 'pistol', 'rifle', 'knife'];

export const isWeapon = (cls: ClassKey): boolean => cls !== 'person';

export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Point {
  x: number;
  y: number;
}

export interface Detection {
  cls: ClassKey;
  conf: number;
  box: Box;
}

/** A tracked object as seen by consumers. `history` holds one entry per processed frame (true = matched). */
export interface Track {
  id: number;
  cls: ClassKey;
  conf: number;
  box: Box;
  hits: number;
  missed: number;
  history: readonly boolean[];
  trail: readonly Point[];
}
