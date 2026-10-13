"""Open Images V7 via FiftyOne: weapon images and look-alike negatives. Boxes are relative [x, y, w, h]."""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Literal

from ..classes import DROP_IMAGE, OI_LABELS, resolve
from ..labels import Box
from ..sample import Sample

_dataset_counter = itertools.count()


def oi_detections_to_boxes(
    dets: Iterable[tuple[str, Sequence[float]]], w: int, h: int
) -> list[Box] | Literal["drop-image"]:
    boxes: list[Box] = []
    for label, (x, y, bw, bh) in dets:
        r = resolve(label, OI_LABELS, strict=False)
        if r == DROP_IMAGE:
            return DROP_IMAGE
        if r is None:
            continue
        boxes.append(Box(r, x * w, y * h, (x + bw) * w, (y + bh) * h))
    return boxes


def load_open_images(
    cache: Path,
    dropped: Counter[str],
    *,
    split: str,
    classes: Sequence[str],
    max_samples: int | None,
    role: Literal["weapons", "negatives"],
    seed: int = 51,
) -> Iterator[Sample]:
    import fiftyone as fo
    import fiftyone.zoo as foz

    # FiftyOne 1.x has no `dataset_dir` argument on load_zoo_dataset (it collides with the importer's own);
    # the download location is the global zoo directory.
    fo.config.dataset_zoo_dir = str((cache / "open-images-v7").resolve())
    ds = foz.load_zoo_dataset(
        "open-images-v7",
        split=split,
        label_types=["detections"],
        classes=list(classes),
        max_samples=max_samples,
        only_matching=False,  # keep every label on matched images, so weapons in negatives can be detected
        shuffle=True,
        seed=seed,
        dataset_name=f"ws-{role}-{split}-{max_samples}-{next(_dataset_counter)}",  # unique per call
        drop_existing_dataset=True,
    )
    ds.compute_metadata()
    field = "ground_truth" if ds.has_sample_field("ground_truth") else "detections"
    source = f"oi-{role}" if split == "train" else f"oi-{role}-{split}"
    for s in ds.iter_samples():
        w, h = s.metadata.width, s.metadata.height
        labels = s[field]
        dets = [(d.label, d.bounding_box) for d in (labels.detections if labels else [])]
        boxes = oi_detections_to_boxes(dets, w, h)
        if boxes == DROP_IMAGE:
            dropped[f"{source}_ambiguous_label"] += 1
            continue
        if role == "weapons" and not boxes:
            dropped[f"{source}_no_weapon_boxes"] += 1
            continue
        if role == "negatives" and boxes:
            dropped[f"{source}_contains_weapon"] += 1
            continue
        path = Path(s.filepath)
        yield Sample(key=f"oi-{role}-{path.stem}", source=source, image_path=path, width=w, height=h, boxes=list(boxes))
