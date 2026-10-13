"""Evaluate a trained model honestly: held-out test metrics, val-tuned thresholds, hard negatives, failures.

python scripts/evaluate.py --weights runs/ws-yolo26s/weights/best.pt --data-root datasets/ws --out eval/ws-yolo26s
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weaponshield_ml.classes import CLASS_NAMES, WEAPON_IDS  # noqa: E402
from weaponshield_ml.evaluation import (  # noqa: E402
    PRESET_NAMES,
    choose_thresholds,
    downsample,
    hardneg_fp_rate,
    match_detections,
    weapon_summary,
)
from weaponshield_ml.labels import Box, read_labels  # noqa: E402
from weaponshield_ml.training import resolve_device, write_data_yaml  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GALLERY_PER_KIND = 12


def images_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def predictions(result) -> list[tuple[Box, float]]:
    return [
        (Box(int(c), *map(float, xyxy)), float(s))
        for xyxy, s, c in zip(
            result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist(), strict=True
        )
    ]


def predict_stream(model, images: list[Path], out: Path, name: str, conf: float, device):
    """Stream predictions without letting Ultralytics write outside `out` (its global runs_dir is not ours)."""
    return model.predict(
        [str(p) for p in images],
        conf=conf,
        device=device,
        stream=True,
        verbose=False,
        save=False,
        project=str(out),
        name=name,
        exist_ok=True,
    )


def save_gallery(model, data_root: Path, out: Path, balanced: dict[str, float], device) -> list[dict]:
    import cv2

    misses: list[tuple[float, Path, Box]] = []
    false_alarms: list[tuple[float, Path, Box]] = []
    floor = min(balanced[CLASS_NAMES[i]] for i in WEAPON_IDS)
    test_images = images_in(data_root / "images" / "test")
    # Ultralytics cannot predict on an empty source, so an empty test split just gives an empty gallery.
    for r in predict_stream(model, test_images, out, "gallery-predict", floor, device) if test_images else ():
        path = Path(r.path)
        h, w = r.orig_shape
        label_file = data_root / "labels" / "test" / f"{path.stem}.txt"
        gt = [b for b in read_labels(label_file, w, h) if b.cls in WEAPON_IDS] if label_file.exists() else []
        pred = [(b, c) for b, c in predictions(r) if b.cls in WEAPON_IDS and c >= balanced[CLASS_NAMES[b.cls]]]
        _, fn, fp = match_detections(gt, pred)
        misses += [(b.area, path, b) for b in fn]
        false_alarms += [(c, path, b) for b, c in fp]

    gallery_dir = out / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    picks = [
        ("miss", sorted(misses, key=lambda t: -t[0])[:GALLERY_PER_KIND]),
        ("false-alarm", sorted(false_alarms, key=lambda t: -t[0])[:GALLERY_PER_KIND]),
    ]
    for kind, items in picks:
        for k, (score, path, b) in enumerate(items):
            im = cv2.imread(str(path))
            colour = (0, 200, 80) if kind == "miss" else (40, 40, 230)
            cv2.rectangle(im, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), colour, 3)
            name = f"{kind}-{k:02d}.jpg"
            cv2.imwrite(str(gallery_dir / name), im)
            entry = {"file": f"gallery/{name}", "kind": kind, "class": CLASS_NAMES[b.cls], "source_image": path.name}
            if kind == "false-alarm":
                entry["conf"] = round(score, 3)
            entries.append(entry)
    return entries


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    import ultralytics
    from ultralytics import YOLO

    device = resolve_device(args.device)
    if isinstance(device, list):
        device = device[0]  # validation and prediction run on one GPU
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_yaml = write_data_yaml(args.data_root, out / "data.yaml")
    model = YOLO(str(args.weights))
    common = dict(
        data=str(data_yaml),
        imgsz=640,
        batch=16,
        conf=0.001,
        iou=0.7,
        device=device,
        project=str(out),
        exist_ok=True,
        verbose=False,
    )

    val = model.val(split="val", name="val", plots=False, **common)
    val_rows = [CLASS_NAMES[int(c)] for c in val.box.ap_class_index]
    thresholds = choose_thresholds(val.box.px, val.box.p_curve, val.box.r_curve, val.box.f1_curve, val_rows)

    test = model.val(split="test", name="test", plots=True, **common)
    per_class: dict[str, dict[str, float]] = {}
    pr_curves: dict[str, list[list[float]]] = {}
    for i, c in enumerate(test.box.ap_class_index):
        name = CLASS_NAMES[int(c)]
        prec, rec, ap50, ap = test.box.class_result(i)
        per_class[name] = {"precision": float(prec), "recall": float(rec), "map50": float(ap50), "map50_95": float(ap)}
        pr_curves[name] = downsample(test.box.px, test.box.prec_values[i])

    hard_images = images_in(args.data_root / "hardneg" / "images")
    hard_preds: list[list[tuple[int, float]]] = []
    if hard_images:  # an empty hard-negative set skips predict entirely and reports fp_rate 0.0
        floor = min(min(t.values()) for t in thresholds.values())
        hard_preds = [
            [(int(c), float(s)) for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), strict=True)]
            for r in predict_stream(model, hard_images, out, "hardneg-predict", floor, device)
        ]

    metrics = {
        "model": args.weights.name,
        "ultralytics": ultralytics.__version__,
        "test": {
            "per_class": per_class,
            "weapons": weapon_summary(per_class),
            "map50_all": float(test.box.map50),
            "map50_95_all": float(test.box.map),
        },
        "thresholds": thresholds,
        "hardneg": {
            "images": len(hard_images),
            "fp_rate": {k: hardneg_fp_rate(hard_preds, thresholds[k]) for k in PRESET_NAMES},
        },
        "pr_curves": pr_curves,
        "confusion": {"labels": [*CLASS_NAMES, "background"], "matrix": test.confusion_matrix.matrix.tolist()},
    }
    gallery = save_gallery(model, args.data_root, out, thresholds["balanced"], device)
    (out / "gallery.json").write_text(json.dumps(gallery, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics["test"]["weapons"] | {"hardneg_fp_balanced": metrics["hardneg"]["fp_rate"]["balanced"]}))


if __name__ == "__main__":
    main()
