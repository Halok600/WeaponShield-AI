import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import * as ort from 'onnxruntime-node';
import { beforeAll, describe, expect, it } from 'vitest';
import { iou, letterboxGeometry } from './geometry';
import { decodeEndToEnd, uniformThresholds } from './postprocess';
import { rgbaToTensor } from './preprocess';

const root = fileURLToPath(new URL('../../../', import.meta.url));
const golden = (f: string) => `${root}fixtures/golden/${f}`;
const MODEL = `${root}web/public/models/yolo26n-coco.onnx`;

interface Meta {
  srcW: number;
  srcH: number;
  size: number;
  ultralytics: { classId: number; conf: number; box: [number, number, number, number] }[];
}

let session: ort.InferenceSession;
beforeAll(async () => {
  session = await ort.InferenceSession.create(MODEL);
});

describe.each(['bus', 'zidane'])('golden parity: %s', (name) => {
  const meta = JSON.parse(readFileSync(golden(`${name}.json`), 'utf8')) as Meta;
  const rgba = new Uint8Array(readFileSync(golden(`${name}.lb.rgba`)));
  const pyRaw = new Float32Array(new Uint8Array(readFileSync(golden(`${name}.out.f32`))).buffer);

  async function run() {
    const tensor = new ort.Tensor('float32', rgbaToTensor(rgba, meta.size), [1, 3, meta.size, meta.size]);
    const out = await session.run({ [session.inputNames[0]!]: tensor });
    return out[session.outputNames[0]!]!;
  }

  it('raw output matches Python onnxruntime', async () => {
    const out = await run();
    expect(out.dims).toEqual([1, 300, 6]);
    const data = out.data as Float32Array;
    for (let i = 0; i < 300; i++) {
      if (pyRaw[i * 6 + 4]! < 0.25) continue;
      for (let j = 0; j < 6; j++) expect(Math.abs(data[i * 6 + j]! - pyRaw[i * 6 + j]!)).toBeLessThan(j < 4 ? 1e-2 : 1e-3);
    }
  });

  it('decoded person boxes match Ultralytics predictions (IoU >= 0.95, |dconf| <= 0.02)', async () => {
    const out = await run();
    const lb = letterboxGeometry(meta.srcW, meta.srcH, meta.size);
    const ours = decodeEndToEnd(out.data as Float32Array, out.dims, { 0: 'person' }, uniformThresholds(0.25), lb, meta.srcW, meta.srcH);
    const theirs = meta.ultralytics.filter((d) => d.classId === 0);
    expect(theirs.length).toBeGreaterThan(0);
    expect(ours).toHaveLength(theirs.length);
    for (const t of theirs) {
      const [x1, y1, x2, y2] = t.box;
      const best = ours.reduce((a, b) => (iou(b.box, { x1, y1, x2, y2 }) > iou(a.box, { x1, y1, x2, y2 }) ? b : a));
      expect(iou(best.box, { x1, y1, x2, y2 })).toBeGreaterThanOrEqual(0.95);
      expect(Math.abs(best.conf - t.conf)).toBeLessThanOrEqual(0.02);
    }
  });
});
