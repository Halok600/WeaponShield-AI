"""Uniform person labels from a pretrained COCO detector (source person boxes are inconsistent or missing)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from .classes import PERSON
from .labels import Box
from .sample import Sample

COCO_PERSON = 0


class PersonLabeler(Protocol):
    def __call__(self, paths: Sequence[Path]) -> list[list[Box]]: ...


class YoloPersonLabeler:
    def __init__(
        self, weights: str = "yolo26x.pt", conf: float = 0.5, device: str | int | None = None, batch: int = 16
    ):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.device = device
        self.batch = batch

    def __call__(self, paths: Sequence[Path]) -> list[list[Box]]:
        results = self.model.predict(
            [str(p) for p in paths],
            conf=self.conf,
            classes=[COCO_PERSON],
            device=self.device,
            batch=self.batch,
            verbose=False,
        )
        return [[Box(PERSON, *map(float, xyxy)) for xyxy in r.boxes.xyxy.tolist()] for r in results]


def add_person_labels(samples: Sequence[Sample], labeler: PersonLabeler, chunk: int = 64) -> None:
    for start in range(0, len(samples), chunk):
        part = samples[start : start + chunk]
        for s, persons in zip(part, labeler([s.image_path for s in part]), strict=True):
            s.boxes = [b for b in s.boxes if b.cls != PERSON] + persons
