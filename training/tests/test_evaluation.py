import numpy as np
import pytest

from weaponshield_ml.classes import KNIFE, PISTOL
from weaponshield_ml.evaluation import (
    DEFAULT_THRESHOLDS,
    choose_thresholds,
    downsample,
    hardneg_fp_rate,
    match_detections,
    weapon_summary,
)
from weaponshield_ml.labels import Box


def test_choose_thresholds_from_curves():
    px = np.linspace(0, 1, 101)
    p = np.clip(px + 0.2, 0, 1)[None, :]  # precision rises with confidence
    r = np.clip(1.2 - px, 0, 1)[None, :]  # recall falls with confidence
    f1 = 2 * p * r / (p + r)
    t = choose_thresholds(px, p, r, f1, ["pistol"])
    assert t["balanced"]["pistol"] == pytest.approx(0.5, abs=0.01)
    assert t["low"]["pistol"] == pytest.approx(0.7, abs=0.01)
    assert t["high"]["pistol"] == pytest.approx(0.3, abs=0.01)


def test_choose_thresholds_fallbacks():
    px = np.linspace(0, 1, 101)
    flat = np.full((1, 101), 0.5)
    f1 = np.exp(-((px - 0.4) ** 2) / 0.01)[None, :]
    t = choose_thresholds(px, flat, flat, f1, ["knife"])
    assert t["balanced"]["knife"] == pytest.approx(0.4)
    assert t["low"]["knife"] == pytest.approx(0.55)
    assert t["high"]["knife"] == pytest.approx(0.25)


def test_classes_without_validation_instances_get_defaults():
    px = np.linspace(0, 1, 101)
    flat = np.full((1, 101), 0.5)
    t = choose_thresholds(px, flat, flat, flat, ["person"])
    for preset, value in DEFAULT_THRESHOLDS.items():
        assert t[preset]["knife"] == value
        assert set(t[preset]) == {"person", "pistol", "rifle", "knife"}


def test_no_validation_rows_at_all_gives_all_defaults():
    px = np.linspace(0, 1, 101)
    empty = np.zeros((0, 101))
    t = choose_thresholds(px, empty, empty, empty, [])
    for preset, value in DEFAULT_THRESHOLDS.items():
        assert t[preset] == dict.fromkeys(("person", "pistol", "rifle", "knife"), value)


def test_downsample_keeps_endpoints():
    pts = downsample(np.linspace(0, 1, 1000), np.linspace(1, 0, 1000), n=5)
    assert pts[0] == [0.0, 1.0] and pts[-1] == [1.0, 0.0] and len(pts) == 5


def test_match_detections_greedy_by_confidence_same_class():
    gt = [Box(PISTOL, 0, 0, 10, 10), Box(KNIFE, 50, 50, 60, 60)]
    pred = [(Box(PISTOL, 1, 1, 10, 10), 0.6), (Box(PISTOL, 0, 0, 10, 10), 0.9), (Box(PISTOL, 50, 50, 60, 60), 0.8)]
    tp, fn, fp = match_detections(gt, pred)
    assert tp == 1
    assert fn == [gt[1]]  # knife missed: the pistol prediction on it is the wrong class
    assert sorted(c for _, c in fp) == [0.6, 0.8]


def test_weapon_summary_averages_weapon_classes_only():
    per = {
        "person": {"precision": 1, "recall": 1, "map50": 1, "map50_95": 1},
        "pistol": {"precision": 0.9, "recall": 0.8, "map50": 0.9, "map50_95": 0.6},
        "rifle": {"precision": 0.7, "recall": 0.6, "map50": 0.7, "map50_95": 0.4},
        "knife": {"precision": 0.5, "recall": 0.4, "map50": 0.5, "map50_95": 0.2},
    }
    assert weapon_summary(per) == pytest.approx({"precision": 0.7, "recall": 0.6, "map50": 0.7, "map50_95": 0.4})


def test_weapon_summary_without_weapon_rows_is_zero_not_nan():
    per = {"person": {"precision": 1, "recall": 1, "map50": 1, "map50_95": 1}}
    assert weapon_summary(per) == {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map50_95": 0.0}


def test_hardneg_fp_rate_counts_images_with_any_weapon_above_threshold():
    per_image = [[(0, 0.99)], [(1, 0.5)], [(3, 0.2)], []]
    assert hardneg_fp_rate(per_image, {"person": 0.5, "pistol": 0.45, "rifle": 0.45, "knife": 0.45}) == 0.25


def test_hardneg_fp_rate_empty_set_is_zero():
    assert hardneg_fp_rate([], dict(DEFAULT_THRESHOLDS)) == 0.0
