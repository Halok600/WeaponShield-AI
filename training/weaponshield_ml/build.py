"""Assemble the WeaponShield dataset.

python -m weaponshield_ml.build --out datasets/ws --cache datasets/cache [--smoke]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

from .autolabel import PersonLabeler, add_person_labels
from .grouping import compute_hashes, dedupe_and_group, near_duplicate_pairs
from .sample import Sample
from .split import split_groups
from .writer import write_dataset


def build(
    out: Path,
    sources: Iterable[Iterable[Sample]],
    hardneg: Iterable[Sample],
    labeler: PersonLabeler,
    *,
    dropped: Mapping[str, int] | None = None,
    seed: int = 1234,
    max_distance: int = 6,
) -> dict:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    samples = [s for source in sources for s in source]
    hard = list(hardneg)
    kept, exact = dedupe_and_group(samples, compute_hashes(samples), max_distance)
    kept_hashes = compute_hashes(kept)
    labelled = add_person_labels(kept, labeler)
    size_mismatch = len(kept) - len(labelled)
    hard_hashes = compute_hashes(hard)
    pairs = near_duplicate_pairs(kept_hashes + hard_hashes, max_distance)
    hardneg_to_drop = {j - len(kept) for i, j in pairs if i < len(kept) and j >= len(kept)}
    hard = [h for i, h in enumerate(hard) if i not in hardneg_to_drop]
    hardneg_overlaps = len(hardneg_to_drop)
    splits = split_groups(labelled, seed=seed)
    dropped_dict = {
        **(dropped or {}),
        "exact_duplicates": exact,
        "size_mismatch": size_mismatch,
        "hardneg_overlaps_train": hardneg_overlaps,
    }
    return write_dataset(out, splits, hard, dropped_dict)


def main() -> None:
    from .autolabel import YoloPersonLabeler
    from .classes import OI_NEGATIVE_CLASSES, OI_WEAPON_CLASSES
    from .sources.hf_weapons import load_hf_weapons
    from .sources.open_images import load_open_images

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--device", default=None, help="e.g. 0 or cpu; default lets Ultralytics choose")
    p.add_argument("--labeler-weights", default="yolo26x.pt")
    p.add_argument("--oi-negatives", type=int, default=6000)
    p.add_argument("--hardneg", type=int, default=500)
    p.add_argument("--smoke", action="store_true", help="tiny limits for a quick end-to-end check")
    args = p.parse_args()

    hf_limit = 30 if args.smoke else None
    oi_weapons_max = 30 if args.smoke else None
    oi_negatives = 20 if args.smoke else args.oi_negatives
    hardneg_max = 10 if args.smoke else args.hardneg
    weights = "yolo26n.pt" if args.smoke else args.labeler_weights
    # Smoke runs use the small validation split so they skip the multi-GB train annotation files.
    oi_split = "validation" if args.smoke else "train"

    dropped: Counter[str] = Counter()
    sources = [
        load_hf_weapons(args.cache, dropped, limit=hf_limit),
        load_open_images(
            args.cache, dropped, split=oi_split, classes=OI_WEAPON_CLASSES, max_samples=oi_weapons_max, role="weapons"
        ),
        load_open_images(
            args.cache, dropped, split=oi_split, classes=OI_NEGATIVE_CLASSES, max_samples=oi_negatives, role="negatives"
        ),
    ]
    # Over-fetch so that filtering out images that contain weapons still leaves enough hard negatives.
    hard = list(
        load_open_images(
            args.cache,
            dropped,
            split="validation",
            classes=OI_NEGATIVE_CLASSES,
            max_samples=hardneg_max * 2,
            role="negatives",
        )
    )[:hardneg_max]
    stats = build(args.out, sources, hard, YoloPersonLabeler(weights, device=args.device), dropped=dropped)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
