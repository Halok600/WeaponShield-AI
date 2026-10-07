import type { ClassKey } from '../core/types';

export interface ModelConfig {
  id: string;
  /** Same-origin URL; files are placed in public/models by scripts/prepare-assets.mjs. */
  url: string;
  inputSize: number;
  /** Model class id -> app class. Ids not listed are ignored. */
  labels: Readonly<Record<number, ClassKey>>;
}

/** Stock Ultralytics YOLO26n trained on COCO: only person (0) and knife (43) are relevant. Development only. */
export const STOCK_COCO_MODEL: ModelConfig = {
  id: 'yolo26n-coco',
  url: '/models/yolo26n-coco.onnx',
  inputSize: 640,
  labels: { 0: 'person', 43: 'knife' },
};

export const ACTIVE_MODEL: ModelConfig = STOCK_COCO_MODEL;
