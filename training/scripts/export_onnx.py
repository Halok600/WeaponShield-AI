"""Export a YOLO26 checkpoint to end-to-end ONNX for the web app and edge tool.

The web app expects a single output shaped [1, 300, 6] holding
[x1, y1, x2, y2, conf, class_id] in letterboxed input pixels. This script
refuses to produce anything else.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import onnxruntime as ort
from ultralytics import YOLO

EXPECTED_SHAPE = [1, 300, 6]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(weights: str, out: Path, imgsz: int = 640) -> str:
    model = YOLO(weights)
    exported = Path(model.export(format="onnx", imgsz=imgsz, nms=False, opset=17, simplify=True, dynamic=False))
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), out)

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    shape = list(session.get_outputs()[0].shape)
    if shape != EXPECTED_SHAPE:
        raise SystemExit(f"Unexpected ONNX output shape {shape}, expected {EXPECTED_SHAPE}")
    return sha256(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="e.g. yolo26n.pt or runs/.../best.pt")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    digest = export(args.weights, args.out, args.imgsz)
    print(f"{args.out} sha256={digest} bytes={args.out.stat().st_size}")


if __name__ == "__main__":
    main()
