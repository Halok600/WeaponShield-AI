"""Pixel-space boxes and YOLO label files (class cx cy w h, normalised)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def clip(b: Box, w: int, h: int) -> Box:
    return Box(b.cls, min(max(b.x1, 0), w), min(max(b.y1, 0), h), min(max(b.x2, 0), w), min(max(b.y2, 0), h))


def iou(a: Box, b: Box) -> float:
    inter = Box(0, max(a.x1, b.x1), max(a.y1, b.y1), min(a.x2, b.x2), min(a.y2, b.y2)).area
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def to_yolo_line(b: Box, w: int, h: int) -> str | None:
    c = clip(b, w, h)
    if c.x2 - c.x1 < 1 or c.y2 - c.y1 < 1:
        return None
    cx, cy = (c.x1 + c.x2) / 2 / w, (c.y1 + c.y2) / 2 / h
    return f"{c.cls} {cx:.6f} {cy:.6f} {(c.x2 - c.x1) / w:.6f} {(c.y2 - c.y1) / h:.6f}"


def from_yolo_line(line: str, w: int, h: int) -> Box:
    cls, cx, cy, bw, bh = line.split()
    x, y, ww, hh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
    return Box(int(cls), x - ww / 2, y - hh / 2, x + ww / 2, y + hh / 2)


def write_labels(path: Path, boxes: Iterable[Box], w: int, h: int) -> int:
    lines = [line for b in boxes if (line := to_yolo_line(b, w, h)) is not None]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return len(lines)


def read_labels(path: Path, w: int, h: int) -> list[Box]:
    return [from_yolo_line(line, w, h) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
