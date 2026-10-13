"""Pure evaluation helpers (no Ultralytics import), unit-tested."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .classes import CLASS_NAMES, WEAPON_IDS
from .labels import Box, iou

PRESET_NAMES = ("low", "balanced", "high")
WEAPON_NAMES = tuple(CLASS_NAMES[i] for i in sorted(WEAPON_IDS))
# Used for classes with no validation instances (matches the web app's provisional Balanced-ish values).
DEFAULT_THRESHOLDS = {"low": 0.6, "balanced": 0.45, "high": 0.3}
EPS = 1e-9  # curve values like 1.2 - 0.3 land a hair under 0.9; do not let float noise move a threshold a whole step


def choose_thresholds(
    px: Sequence[float],
    p: np.ndarray,
    r: np.ndarray,
    f1: np.ndarray,
    row_names: Sequence[str],
    *,
    low_precision: float = 0.9,
    high_recall: float = 0.9,
) -> dict[str, dict[str, float]]:
    """Per-class confidence thresholds for the Low / Balanced / High presets, from validation curves."""
    x = np.asarray(px, dtype=float)
    out: dict[str, dict[str, float]] = {k: {} for k in PRESET_NAMES}
    for i, name in enumerate(row_names):
        balanced = float(x[int(np.argmax(f1[i]))])
        precise = np.where((p[i] >= low_precision - EPS) & (x >= balanced))[0]
        low = float(x[precise[0]]) if len(precise) else min(balanced + 0.15, 0.9)
        sensitive = np.where((r[i] >= high_recall - EPS) & (x <= balanced))[0]
        high = float(x[sensitive[-1]]) if len(sensitive) else max(balanced - 0.15, 0.05)
        out["balanced"][name] = round(balanced, 2)
        out["low"][name] = round(low, 2)
        out["high"][name] = round(high, 2)
    for name in CLASS_NAMES:
        if name not in row_names:
            for preset in PRESET_NAMES:
                out[preset][name] = DEFAULT_THRESHOLDS[preset]
    return out


def downsample(x: Sequence[float], y: Sequence[float], n: int = 101) -> list[list[float]]:
    idx = np.linspace(0, len(x) - 1, n).round().astype(int)
    return [[round(float(x[i]), 4), round(float(y[i]), 4)] for i in idx]


def match_detections(
    gt: Sequence[Box], pred: Sequence[tuple[Box, float]], iou_thr: float = 0.5
) -> tuple[int, list[Box], list[tuple[Box, float]]]:
    """Greedy matching by confidence (same class only). Returns (true positives, missed gt, false positives)."""
    used: set[int] = set()
    tp = 0
    false_pos: list[tuple[Box, float]] = []
    for box, conf in sorted(pred, key=lambda bc: -bc[1]):
        best, best_iou = None, iou_thr
        for j, g in enumerate(gt):
            if j in used or g.cls != box.cls:
                continue
            v = iou(g, box)
            if v >= best_iou:
                best, best_iou = j, v
        if best is None:
            false_pos.append((box, conf))
        else:
            used.add(best)
            tp += 1
    return tp, [g for j, g in enumerate(gt) if j not in used], false_pos


def weapon_summary(per_class: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Mean metrics over the weapon classes present in `per_class` (0.0 when none has validation instances)."""
    keys = ("precision", "recall", "map50", "map50_95")
    rows = [per_class[n] for n in WEAPON_NAMES if n in per_class]
    return {k: float(np.mean([row[k] for row in rows])) if rows else 0.0 for k in keys}


def hardneg_fp_rate(per_image: Sequence[Sequence[tuple[int, float]]], thresholds: Mapping[str, float]) -> float:
    if not per_image:
        return 0.0
    flagged = sum(
        any(cls in WEAPON_IDS and conf >= thresholds[CLASS_NAMES[cls]] for cls, conf in preds) for preds in per_image
    )
    return flagged / len(per_image)
