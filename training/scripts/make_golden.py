"""Write golden fixtures that pin the browser pipeline to Ultralytics' own results.

For each reference image this saves:
  {name}.lb.rgba  640x640 RGBA after Ultralytics' LetterBox (exact model input)
  {name}.out.f32  raw [1,300,6] float32 output from Python onnxruntime
  {name}.json     source size + Ultralytics predictions (conf >= 0.25, source pixels)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import ultralytics
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

NAMES = ("bus", "zidane")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()

    images_dir = args.out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    assets = Path(ultralytics.__file__).parent / "assets"

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    model = YOLO(str(args.model), task="detect")

    for name in NAMES:
        src = images_dir / f"{name}.jpg"
        shutil.copyfile(assets / f"{name}.jpg", src)
        bgr = cv2.imread(str(src))
        h, w = bgr.shape[:2]

        letterboxed = LetterBox(new_shape=(args.size, args.size), auto=False)(image=bgr)
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
        rgba.astype(np.uint8).tofile(args.out / f"{name}.lb.rgba")

        tensor = np.ascontiguousarray((rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
        raw = session.run(None, {input_name: tensor})[0].astype("<f4")
        raw.tofile(args.out / f"{name}.out.f32")

        result = model.predict(bgr, imgsz=args.size, conf=0.25, verbose=False)[0]
        dets = [
            {"classId": int(c), "conf": float(s), "box": [float(v) for v in b]}
            for b, s, c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist())
        ]
        meta = {"srcW": w, "srcH": h, "size": args.size, "ultralytics": dets}
        (args.out / f"{name}.json").write_text(json.dumps(meta, indent=2))
        print(f"{name}: {w}x{h}, {len(dets)} detections")


if __name__ == "__main__":
    main()
