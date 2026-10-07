"""Group-aware train/val/test split by image count."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from .sample import Sample

SPLITS = ("train", "val", "test")


def split_groups(
    samples: Sequence[Sample], ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 1234
) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        groups[s.group].append(s)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    total = len(samples)
    targets = [r * total for r in ratios]
    counts = [0, 0, 0]
    out: dict[str, list[Sample]] = {name: [] for name in SPLITS}
    for key in keys:
        # Give the next group to the split that is furthest below its target, relative to that target.
        i = min(range(3), key=lambda j: counts[j] / targets[j] if targets[j] > 0 else float("inf"))
        out[SPLITS[i]].extend(groups[key])
        counts[i] += len(groups[key])
    return out
