"""Roboflow Universe 'weapon-detection' by yolov7test (CC BY 4.0), exported as YOLOv8 detection.

Needs a Roboflow API key (Kaggle secret ROBOFLOW_API_KEY). The export is read straight from disk, so everything
except the download itself is unit-testable offline.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Literal

import yaml
from PIL import Image

from ..classes import DROP_IMAGE, ROBOFLOW_LABELS, resolve
from ..labels import Box, from_yolo_line
from ..sample import Sample

RF_WORKSPACE = "yolov7test-u13vc"
RF_PROJECT = "weapon-detection-m7qso"
RF_VERSION = 16
SOURCE = "roboflow-weapons"

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_RF_SUFFIX = re.compile(r"_(jpg|jpeg|png|webp)\.rf\.[0-9a-f]+$", re.IGNORECASE)
_FRAME_MARKER = re.compile(r"_(mp4|avi|mov)-\d+$", re.IGNORECASE)
_MAX_STEM = 120


def rf_group(stem: str) -> str:
    """Group hint so augmented copies of one image, and frames of one video, land in the same split."""
    base = _FRAME_MARKER.sub("", _RF_SUFFIX.sub("", stem))
    return f"rf-{base}"


def check_names(names: Sequence[str]) -> None:
    unknown = sorted({n for n in names if n not in ROBOFLOW_LABELS})
    if unknown:
        raise KeyError(f"Roboflow class names missing from ROBOFLOW_LABELS: {unknown}")


def rf_labels_to_boxes(
    lines: Iterable[str], names: Sequence[str], w: int, h: int, dropped: Counter[str]
) -> list[Box] | Literal["drop-image"]:
    boxes: list[Box] = []
    for line in lines:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(
                f"expected 5 fields (class cx cy w h), got {len(fields)}: polygon exports are not supported"
            )
        r = resolve(names[int(fields[0])], ROBOFLOW_LABELS, strict=True)
        if r == DROP_IMAGE:
            return DROP_IMAGE
        if r is None:
            continue
        b = from_yolo_line(line, w, h)
        boxes.append(Box(r, b.x1, b.y1, b.x2, b.y2))
    return boxes


def _safe_stem(stem: str) -> str:
    if len(stem) <= _MAX_STEM:
        return stem
    return f"{stem[:100]}-{hashlib.sha1(stem.encode('utf-8')).hexdigest()[:10]}"


def _names_from_yaml(data_yaml: Path) -> list[str]:
    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    return list(names)


def load_roboflow(cache: Path, dropped: Counter[str], api_key: str) -> Iterator[Sample]:
    root = cache / "roboflow"
    if not (root / "data.yaml").exists():
        from roboflow import Roboflow

        version = Roboflow(api_key=api_key).workspace(RF_WORKSPACE).project(RF_PROJECT).version(RF_VERSION)
        version.download("yolov8", location=str(root), overwrite=False)
    names = _names_from_yaml(root / "data.yaml")
    check_names(names)
    for split in ("train", "valid", "test"):
        images = root / split / "images"
        if not images.is_dir():
            continue
        for path in sorted(images.iterdir()):
            if path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            with Image.open(path) as im:
                w, h = im.size
            label_file = root / split / "labels" / f"{path.stem}.txt"
            lines = label_file.read_text(encoding="utf-8").splitlines() if label_file.exists() else []
            boxes = rf_labels_to_boxes(lines, names, w, h, dropped)
            if boxes == DROP_IMAGE:
                dropped["roboflow_ambiguous_label"] += 1
                continue
            if not boxes:
                dropped["roboflow_no_weapon_boxes"] += 1
                continue
            yield Sample(
                key=f"rf-{split}-{_safe_stem(path.stem)}",
                source=SOURCE,
                image_path=path,
                width=w,
                height=h,
                boxes=list(boxes),
                group=rf_group(path.stem),
            )
