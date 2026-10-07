# Plan 2 — Training Pipeline (data, training, evaluation, Kaggle runs)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, tested pipeline that assembles the WeaponShield dataset (person / pistol / rifle / knife) from free sources, trains YOLO26n and YOLO26s on Kaggle's free GPUs with CCTV-style augmentation, and produces honest evaluation artefacts plus end-to-end ONNX models for the web app.

**Architecture:** A small Python package `training/weaponshield_ml` does all the logic: source loaders → person pseudo-labels → pHash de-duplication and grouping → group-aware split → YOLO dataset writer. Thin CLIs wrap it (`python -m weaponshield_ml.build`, `scripts/train.py`, `scripts/evaluate.py`, plus Plan 1's `scripts/export_onnx.py`). Kaggle kernels are tiny bootstraps: they clone this repo and run `training/kaggle/run_*.py`, so everything that matters is versioned and unit-tested here.

**Tech Stack:** Python 3.12 (3.10+ compatible), Ultralytics 8.4.174 (YOLO26), Albumentations 2.0.8, Hugging Face `datasets` 5.1.0, FiftyOne 1.22.1 (Open Images V7), ImageHash 4.3.2, pytest 9, ruff 0.16, Kaggle CLI 2.2.4.

**Spec:** `docs/superpowers/specs/2026-10-07-weaponshield-v2-design.md` (section 2)

## Global Constraints

- Branch `v2` only; commits under the repo's configured identity (`Priyanshu Kumar Tiwari <143282195+Halok600@users.noreply.github.com>`). **No `Co-Authored-By` trailers, no "Generated with" lines.** Before every push: `git log origin/v2..HEAD --format=%B | grep -i -E "co-authored|claude|generated with"` must print nothing.
- Class ids are exactly `0 person · 1 pistol · 2 rifle · 3 knife` (`CLASS_NAMES = ("person", "pistol", "rifle", "knife")`), matching the web app's `ClassKey`.
- Source-class mapping: Handgun/Pistol → pistol; Rifle/Shotgun/Machine gun/long guns → rifle; Knife/Kitchen knife/Dagger → knife; Person → person. Labels that do not say which weapon it is ("Guns", "weapon", Open Images "Weapon"/"Sword") drop the whole image.
- People are labelled uniformly by a pretrained YOLO26x (COCO class 0) at conf ≥ 0.5 on every training image; source person boxes are discarded.
- Exact pHash duplicates are dropped (first occurrence wins); near-duplicates (Hamming distance ≤ 6) and samples sharing a group hint always land in the same split. Split ratios train 80 / val 10 / test 10 by image count, seed 1234.
- The test split is never used for model selection, threshold tuning or early stopping. Thresholds are chosen on the **val** split.
- Hard-negative eval set: Open Images **validation** images of look-alike objects with no weapon label, 500 images, kept outside the splits.
- Training: imgsz 640; YOLO26n (time budget 4 h, batch 64) and YOLO26s (time budget 10 h, batch 32); patience 30; CCTV augmentation list exactly as in Task 5.
- Export: `scripts/export_onnx.py` (Plan 1) → output `[1, 300, 6]`.
- Heavy libraries (ultralytics, torch, fiftyone, datasets, cv2) are imported inside functions, never at module top level, so unit tests and CI run with only `requirements-dev.txt`.
- Never commit datasets, model weights or Kaggle outputs (`training/datasets/`, `training/runs/`, `*.pt`, `*.onnx` are git-ignored already).
- No paid services. The Kaggle API token already exists at `~/.kaggle/access_token`; never print or commit it.

---

## File Structure

```
training/pyproject.toml                     package metadata + ruff + pytest config
training/requirements.txt                   full runtime deps (Kaggle / local data build)
training/requirements-dev.txt               light deps for tests, lint and Kaggle CLI
training/weaponshield_ml/__init__.py
training/weaponshield_ml/classes.py         class schema + source label tables + resolve()
training/weaponshield_ml/labels.py          Box, IoU, YOLO label read/write
training/weaponshield_ml/sample.py          Sample dataclass
training/weaponshield_ml/grouping.py        pHash, near-duplicate pairs, dedupe + group
training/weaponshield_ml/split.py           group-aware split
training/weaponshield_ml/autolabel.py       PersonLabeler protocol + YOLO26x implementation
training/weaponshield_ml/writer.py          dataset writer + stats
training/weaponshield_ml/build.py           orchestration + CLI (python -m weaponshield_ml.build)
training/weaponshield_ml/sources/__init__.py
training/weaponshield_ml/sources/hf_weapons.py    Subh775/WeaponDetection loader
training/weaponshield_ml/sources/open_images.py   Open Images V7 loader (FiftyOne)
training/weaponshield_ml/training.py        CCTV augmentations, data.yaml rewrite, device choice
training/weaponshield_ml/evaluation.py      threshold choice, matching, summaries
training/scripts/train.py                   training CLI
training/scripts/evaluate.py                evaluation CLI (metrics.json, curves, gallery)
training/kaggle/run_dataset.py              Kaggle job: build dataset, tar it
training/kaggle/run_train.py                Kaggle job: train + evaluate + export one variant
training/kaggle/{dataset,train-n,train-s}/kernel-metadata.json + bootstrap.py
training/tests/test_*.py
.github/workflows/training.yml
docs/training-runs.md                       results log (Task 8)
```

---

### Task 1: Package scaffold, class schema and YOLO labels

**Files:**
- Create: `training/pyproject.toml`, `training/requirements.txt`, `training/requirements-dev.txt`, `training/weaponshield_ml/__init__.py`, `training/weaponshield_ml/classes.py`, `training/weaponshield_ml/labels.py`, `training/weaponshield_ml/sample.py`
- Test: `training/tests/test_classes.py`, `training/tests/test_labels.py`

**Interfaces:**
- Produces:
  - `CLASS_NAMES: tuple[str, ...]`, `PERSON, PISTOL, RIFLE, KNIFE = 0, 1, 2, 3`, `WEAPON_IDS: frozenset[int]`
  - `DROP_IMAGE = "drop-image"`; `Resolved = int | None | Literal["drop-image"]`
  - `HF_LABELS: dict[str, Resolved]`, `OI_LABELS: dict[str, Resolved]`, `OI_WEAPON_CLASSES: tuple[str, ...]`, `OI_NEGATIVE_CLASSES: tuple[str, ...]`
  - `resolve(name: str, table: Mapping[str, Resolved], *, strict: bool) -> Resolved`
  - `@dataclass(frozen=True) Box(cls: int, x1: float, y1: float, x2: float, y2: float)` with `.area`
  - `clip(b: Box, w: int, h: int) -> Box`, `iou(a: Box, b: Box) -> float`
  - `to_yolo_line(b: Box, w: int, h: int) -> str | None`, `from_yolo_line(line: str, w: int, h: int) -> Box`
  - `write_labels(path: Path, boxes: Iterable[Box], w: int, h: int) -> int`, `read_labels(path: Path, w: int, h: int) -> list[Box]`
  - `@dataclass Sample(key: str, source: str, image_path: Path, width: int, height: int, boxes: list[Box] = [], group: str = "")` — empty `group` defaults to `key`; property `weapon_classes: set[int]`

- [ ] **Step 1: Create packaging and requirement files**

`training/pyproject.toml`:
```toml
[project]
name = "weaponshield-ml"
version = "2.0.0"
description = "Dataset, training and evaluation pipeline for WeaponShield AI"
requires-python = ">=3.10"
license = "AGPL-3.0-only"

[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["weaponshield_ml*"]

[tool.ruff]
line-length = 120
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

`training/requirements.txt`:
```text
-r requirements-export.txt
albumentations==2.0.8
datasets==5.1.0
fiftyone==1.22.1
imagehash==4.3.2
pillow>=10
pyyaml>=6
```

`training/requirements-dev.txt`:
```text
pytest==9.1.1
ruff==0.16.10
numpy>=1.26
pillow>=10
imagehash==4.3.2
pyyaml>=6
albumentations==2.0.8
kaggle==2.2.4
```

`training/weaponshield_ml/__init__.py`:
```python
"""WeaponShield AI dataset, training and evaluation pipeline."""
```

- [ ] **Step 2: Install dev requirements into the existing venv**

```bash
cd /d/Projects/WeaponShield-AI/training
.venv/Scripts/python -m pip install -q -r requirements-dev.txt
.venv/Scripts/python -m pytest --version
```
Expected: `pytest 9.1.1`. (The venv was created in Plan 1 Task 3; if `training/.venv` is missing, create it with `python -m venv .venv` first.)

- [ ] **Step 3: Write the failing tests**

`training/tests/test_classes.py`:
```python
import pytest

from weaponshield_ml.classes import (
    CLASS_NAMES,
    DROP_IMAGE,
    HF_LABELS,
    KNIFE,
    OI_LABELS,
    OI_NEGATIVE_CLASSES,
    OI_WEAPON_CLASSES,
    PERSON,
    PISTOL,
    RIFLE,
    WEAPON_IDS,
    resolve,
)

# Exact category names published by Subh775/WeaponDetection (datasets-server /info, 2026-10-07).
HF_NAMES = [
    "weapons", "Aggressor", "Blood", "Guns", "Guns perspective", "Hand", "Heavy Gun", "Knife", "Knife_Deploy",
    "Knife_Weapon", "Long guns", "Person", "Pistol", "Rifle", "Shotgun", "Stabbing", "Victim", "al", "guns",
    "handgun", "heavyweapon", "larga", "person", "pistol", "pistols", "rifle", "shotgun", "violence", "weapon",
]


def test_schema_matches_web_app():
    assert CLASS_NAMES == ("person", "pistol", "rifle", "knife")
    assert (PERSON, PISTOL, RIFLE, KNIFE) == (0, 1, 2, 3)
    assert WEAPON_IDS == {PISTOL, RIFLE, KNIFE}


def test_every_hf_label_has_an_explicit_rule():
    assert sorted(HF_LABELS) == sorted(HF_NAMES)


@pytest.mark.parametrize("name", ["Pistol", "pistol", "pistols", "handgun"])
def test_handguns_are_pistols(name):
    assert resolve(name, HF_LABELS, strict=True) == PISTOL


@pytest.mark.parametrize("name", ["Rifle", "rifle", "Shotgun", "shotgun", "Long guns", "larga", "Heavy Gun", "heavyweapon"])
def test_long_guns_are_rifles(name):
    assert resolve(name, HF_LABELS, strict=True) == RIFLE


@pytest.mark.parametrize("name", ["Knife", "Knife_Deploy", "Knife_Weapon"])
def test_knives(name):
    assert resolve(name, HF_LABELS, strict=True) == KNIFE


@pytest.mark.parametrize("name", ["weapons", "weapon", "Guns", "guns", "Guns perspective", "al"])
def test_generic_weapon_labels_drop_the_image(name):
    assert resolve(name, HF_LABELS, strict=True) == DROP_IMAGE


@pytest.mark.parametrize("name", ["Person", "person", "Aggressor", "Victim", "Blood", "Hand", "Stabbing", "violence"])
def test_people_and_context_boxes_are_discarded(name):
    assert resolve(name, HF_LABELS, strict=True) is None


def test_unknown_labels():
    with pytest.raises(KeyError, match="Revolver"):
        resolve("Revolver", HF_LABELS, strict=True)
    assert resolve("Mobile phone", OI_LABELS, strict=False) is None


def test_open_images_tables():
    assert {resolve(c, OI_LABELS, strict=False) for c in OI_WEAPON_CLASSES} == {PISTOL, RIFLE, KNIFE}
    assert resolve("Kitchen knife", OI_LABELS, strict=False) == KNIFE
    assert resolve("Sword", OI_LABELS, strict=False) == DROP_IMAGE
    assert resolve("Weapon", OI_LABELS, strict=False) == DROP_IMAGE
    assert not set(OI_NEGATIVE_CLASSES) & set(OI_LABELS)
```

`training/tests/test_labels.py`:
```python
from weaponshield_ml.labels import Box, clip, from_yolo_line, iou, read_labels, to_yolo_line, write_labels
from weaponshield_ml.sample import Sample


def test_yolo_line_round_trip():
    b = Box(2, 10, 20, 110, 70)
    line = to_yolo_line(b, 200, 100)
    assert line == "2 0.300000 0.450000 0.500000 0.500000"
    back = from_yolo_line(line, 200, 100)
    assert back.cls == 2
    assert (round(back.x1), round(back.y1), round(back.x2), round(back.y2)) == (10, 20, 110, 70)


def test_boxes_are_clipped_and_slivers_dropped():
    assert clip(Box(1, -5, -5, 50, 300), 100, 100) == Box(1, 0, 0, 50, 100)
    assert to_yolo_line(Box(1, 99.5, 0, 140, 10), 100, 100) is None


def test_iou_and_area():
    a = Box(0, 0, 0, 10, 10)
    assert a.area == 100
    assert iou(a, Box(0, 5, 0, 15, 10)) == 50 / 150
    assert iou(a, Box(0, 20, 20, 30, 30)) == 0


def test_label_files(tmp_path):
    path = tmp_path / "x.txt"
    assert write_labels(path, [Box(0, 0, 0, 50, 50), Box(3, 200, 200, 300, 300)], 100, 100) == 1
    assert [b.cls for b in read_labels(path, 100, 100)] == [0]
    assert write_labels(tmp_path / "empty.txt", [], 100, 100) == 0
    assert (tmp_path / "empty.txt").read_text() == ""


def test_sample_defaults(tmp_path):
    s = Sample(key="hf-1", source="hf-weapons", image_path=tmp_path / "a.jpg", width=10, height=10, boxes=[Box(0, 0, 0, 5, 5), Box(3, 0, 0, 5, 5)])
    assert s.group == "hf-1"
    assert s.weapon_classes == {3}
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `cd /d/Projects/WeaponShield-AI/training && .venv/Scripts/python -m pytest tests/test_classes.py tests/test_labels.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'weaponshield_ml.classes'`.

- [ ] **Step 5: Implement**

`training/weaponshield_ml/classes.py`:
```python
"""Class schema shared by every data source, the trainer, the web app and the edge tool."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal

CLASS_NAMES: Final = ("person", "pistol", "rifle", "knife")
PERSON, PISTOL, RIFLE, KNIFE = 0, 1, 2, 3
WEAPON_IDS: Final = frozenset({PISTOL, RIFLE, KNIFE})

DROP_IMAGE: Final = "drop-image"
Resolved = int | None | Literal["drop-image"]
"""int: keep the box as that class; None: discard the box; DROP_IMAGE: discard the whole image."""

# Subh775/WeaponDetection: Hugging Face mirror of Roboflow Universe "weapon-detection" (yolov7test), CC BY 4.0.
HF_LABELS: Final[dict[str, Resolved]] = {
    "Pistol": PISTOL,
    "pistol": PISTOL,
    "pistols": PISTOL,
    "handgun": PISTOL,
    "Rifle": RIFLE,
    "rifle": RIFLE,
    "Shotgun": RIFLE,
    "shotgun": RIFLE,
    "Long guns": RIFLE,
    "larga": RIFLE,
    "Heavy Gun": RIFLE,
    "heavyweapon": RIFLE,
    "Knife": KNIFE,
    "Knife_Deploy": KNIFE,
    "Knife_Weapon": KNIFE,
    # Generic labels that do not say which weapon it is: the image cannot be labelled correctly.
    "weapons": DROP_IMAGE,
    "weapon": DROP_IMAGE,
    "Guns": DROP_IMAGE,
    "guns": DROP_IMAGE,
    "Guns perspective": DROP_IMAGE,
    "al": DROP_IMAGE,
    # People and scene context: boxes are discarded (people are re-labelled uniformly by autolabel).
    "Person": None,
    "person": None,
    "Aggressor": None,
    "Victim": None,
    "Blood": None,
    "Hand": None,
    "Stabbing": None,
    "violence": None,
}

OI_WEAPON_CLASSES: Final = ("Handgun", "Rifle", "Shotgun", "Knife", "Kitchen knife", "Dagger")

OI_LABELS: Final[dict[str, Resolved]] = {
    "Handgun": PISTOL,
    "Rifle": RIFLE,
    "Shotgun": RIFLE,
    "Knife": KNIFE,
    "Kitchen knife": KNIFE,
    "Dagger": KNIFE,
    "Weapon": DROP_IMAGE,
    "Sword": DROP_IMAGE,
}

# Look-alike objects people hold: phones, tablets, tools. Used for training negatives and the hard-negative set.
OI_NEGATIVE_CLASSES: Final = (
    "Mobile phone",
    "Tablet computer",
    "Remote control",
    "Drill (Tool)",
    "Hair dryer",
    "Umbrella",
    "Camera",
    "Flashlight",
    "Hammer",
    "Screwdriver",
    "Wrench",
    "Scissors",
    "Microphone",
    "Baseball bat",
)


def resolve(name: str, table: Mapping[str, Resolved], *, strict: bool) -> Resolved:
    if name in table:
        return table[name]
    if strict:
        raise KeyError(f"Unmapped source label {name!r}")
    return None
```

`training/weaponshield_ml/labels.py`:
```python
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
```

`training/weaponshield_ml/sample.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .classes import WEAPON_IDS
from .labels import Box


@dataclass
class Sample:
    """One image on its way into the dataset. `group` ties together images that must share a split."""

    key: str
    source: str
    image_path: Path
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)
    group: str = ""

    def __post_init__(self) -> None:
        if not self.group:
            self.group = self.key

    @property
    def weapon_classes(self) -> set[int]:
        return {b.cls for b in self.boxes if b.cls in WEAPON_IDS}
```

- [ ] **Step 6: Run the tests and lint**

Run: `cd /d/Projects/WeaponShield-AI/training && .venv/Scripts/python -m pytest tests/test_classes.py tests/test_labels.py -q && .venv/Scripts/python -m ruff format . && .venv/Scripts/python -m ruff check .`
Expected: all tests pass; ruff prints `All checks passed!` (the formatter may reflow long lines — that is expected).

- [ ] **Step 7: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add training/pyproject.toml training/requirements.txt training/requirements-dev.txt training/weaponshield_ml training/tests
git commit -m "feat(training): package scaffold, class schema with explicit source mappings, YOLO labels"
```

---

### Task 2: De-duplication, grouping and group-aware split

**Files:**
- Create: `training/weaponshield_ml/grouping.py`, `training/weaponshield_ml/split.py`
- Test: `training/tests/test_grouping.py`, `training/tests/test_split.py`

**Interfaces:**
- Consumes: `Sample` (Task 1)
- Produces:
  - `phash_int(path: Path) -> int`, `compute_hashes(samples: Sequence[Sample]) -> list[int]`
  - `near_duplicate_pairs(hashes: Sequence[int], max_distance: int = 6) -> list[tuple[int, int]]` (requires `max_distance < 8`)
  - `dedupe_and_group(samples: list[Sample], hashes: list[int], max_distance: int = 6) -> tuple[list[Sample], int]` — returns kept samples and number of exact duplicates dropped; rewrites `Sample.group` to `"g<root>"`
  - `SPLITS = ("train", "val", "test")`; `split_groups(samples: Sequence[Sample], ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 1234) -> dict[str, list[Sample]]`

- [ ] **Step 1: Write the failing tests**

`training/tests/test_grouping.py`:
```python
import numpy as np
from PIL import Image

from weaponshield_ml.grouping import dedupe_and_group, near_duplicate_pairs, phash_int
from weaponshield_ml.sample import Sample


def sample(key, group=""):
    return Sample(key=key, source="t", image_path=None, width=1, height=1, group=group)  # type: ignore[arg-type]


def test_near_duplicate_pairs_respects_distance():
    h0, h1, h2 = 0, 0b111111, 0b1111111  # d(h0,h1)=6, d(h1,h2)=1, d(h0,h2)=7
    assert near_duplicate_pairs([h0, h1, h2], max_distance=6) == [(0, 1), (1, 2)]


def test_near_duplicate_pairs_far_apart():
    assert near_duplicate_pairs([0, (1 << 64) - 1], max_distance=6) == []


def test_dedupe_drops_exact_copies_first_wins():
    kept, dropped = dedupe_and_group([sample("a"), sample("b"), sample("c")], [5, 5, 1 << 40])
    assert [s.key for s in kept] == ["a", "c"]
    assert dropped == 1


def test_groups_merge_near_duplicates_and_shared_hints():
    s = [sample("a"), sample("b"), sample("c", group="vid1"), sample("d", group="vid1"), sample("e")]
    # a/b are 2 bits apart; c, d and e are >= 32 bits from everything else
    hashes = [0, 0b11, 0xFFFF_FFFF_0000_0000, 0x0F0F_0F0F_0F0F_0F0F, 0xFFFF_FFFF_FFFF_FFFF]
    kept, _ = dedupe_and_group(s, hashes)
    g = {x.key: x.group for x in kept}
    assert g["a"] == g["b"]
    assert g["c"] == g["d"]
    assert len({g["a"], g["c"], g["e"]}) == 3


def test_phash_is_stable_and_distinguishes_images(tmp_path):
    rng = np.random.default_rng(0)
    a = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    b = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    a.save(tmp_path / "a.png")
    a.save(tmp_path / "a2.png")
    b.save(tmp_path / "b.png")
    assert phash_int(tmp_path / "a.png") == phash_int(tmp_path / "a2.png")
    assert phash_int(tmp_path / "a.png") != phash_int(tmp_path / "b.png")
```

`training/tests/test_split.py`:
```python
from weaponshield_ml.sample import Sample
from weaponshield_ml.split import SPLITS, split_groups


def make(n_singletons, big_group=0):
    out = [Sample(key=f"s{i}", source="t", image_path=None, width=1, height=1) for i in range(n_singletons)]  # type: ignore[arg-type]
    out += [Sample(key=f"b{i}", source="t", image_path=None, width=1, height=1, group="big") for i in range(big_group)]  # type: ignore[arg-type]
    return out


def test_ratios_are_close_to_target():
    splits = split_groups(make(1000))
    sizes = {k: len(v) for k, v in splits.items()}
    assert sum(sizes.values()) == 1000
    assert abs(sizes["train"] - 800) <= 20
    assert abs(sizes["val"] - 100) <= 20
    assert abs(sizes["test"] - 100) <= 20


def test_no_group_is_split_across_splits():
    splits = split_groups(make(500, big_group=50))
    owner = {}
    for name in SPLITS:
        for s in splits[name]:
            assert owner.setdefault(s.group, name) == name


def test_split_is_deterministic():
    a = split_groups(make(300), seed=7)
    b = split_groups(make(300), seed=7)
    assert {k: [s.key for s in v] for k, v in a.items()} == {k: [s.key for s in v] for k, v in b.items()}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /d/Projects/WeaponShield-AI/training && .venv/Scripts/python -m pytest tests/test_grouping.py tests/test_split.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'weaponshield_ml.grouping'`.

- [ ] **Step 3: Implement**

`training/weaponshield_ml/grouping.py`:
```python
"""Perceptual-hash de-duplication and grouping, so near-identical frames never straddle train and test."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from .sample import Sample


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def phash_int(path: Path) -> int:
    import imagehash
    from PIL import Image

    with Image.open(path) as im:
        return int(str(imagehash.phash(im.convert("RGB"))), 16)


def compute_hashes(samples: Sequence[Sample]) -> list[int]:
    return [phash_int(s.image_path) for s in samples]


def near_duplicate_pairs(hashes: Sequence[int], max_distance: int = 6) -> list[tuple[int, int]]:
    """All index pairs whose 64-bit hashes differ in at most `max_distance` bits.

    Multi-index hashing: split each hash into 8 bytes. Two hashes within distance < 8 must agree on at least
    one byte, so only hashes sharing a (position, byte) bucket are compared.
    """
    if not 0 <= max_distance < 8:
        raise ValueError("max_distance must be in [0, 8)")
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, h in enumerate(hashes):
        for k in range(8):
            buckets[(k, (h >> (8 * k)) & 0xFF)].append(i)
    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a, b = members[x], members[y]
                if (hashes[a] ^ hashes[b]).bit_count() <= max_distance:
                    pairs.add((a, b) if a < b else (b, a))
    return sorted(pairs)


def dedupe_and_group(samples: list[Sample], hashes: list[int], max_distance: int = 6) -> tuple[list[Sample], int]:
    """Drop exact pHash duplicates (first wins), then merge near-duplicates and shared group hints into groups."""
    if len(samples) != len(hashes):
        raise ValueError("samples and hashes must have the same length")
    seen: set[int] = set()
    kept: list[Sample] = []
    kept_hashes: list[int] = []
    for s, h in zip(samples, hashes, strict=True):
        if h in seen:
            continue
        seen.add(h)
        kept.append(s)
        kept_hashes.append(h)

    uf = UnionFind(len(kept))
    first_with_hint: dict[str, int] = {}
    for i, s in enumerate(kept):
        uf.union(i, first_with_hint.setdefault(s.group, i))
    for a, b in near_duplicate_pairs(kept_hashes, max_distance):
        uf.union(a, b)
    for i, s in enumerate(kept):
        s.group = f"g{uf.find(i)}"
    return kept, len(samples) - len(kept)
```

`training/weaponshield_ml/split.py`:
```python
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
```

- [ ] **Step 4: Run the tests and lint**

Run: `cd /d/Projects/WeaponShield-AI/training && .venv/Scripts/python -m pytest -q && .venv/Scripts/python -m ruff format . && .venv/Scripts/python -m ruff check .`
Expected: all tests pass, ruff clean.

- [ ] **Step 5: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add training/weaponshield_ml/grouping.py training/weaponshield_ml/split.py training/tests/test_grouping.py training/tests/test_split.py
git commit -m "feat(training): pHash de-duplication, near-duplicate grouping and group-aware split"
```

---

### Task 3: Person pseudo-labels, dataset writer and build orchestration

**Files:**
- Create: `training/weaponshield_ml/autolabel.py`, `training/weaponshield_ml/writer.py`, `training/weaponshield_ml/build.py`, `training/weaponshield_ml/sources/__init__.py`
- Test: `training/tests/test_build.py`

**Interfaces:**
- Consumes: `Sample`, `Box`, `write_labels`, `CLASS_NAMES`, `PERSON` (Task 1); `compute_hashes`, `dedupe_and_group`, `split_groups`, `SPLITS` (Task 2)
- Produces:
  - `class PersonLabeler(Protocol): def __call__(self, paths: Sequence[Path]) -> list[list[Box]]`
  - `class YoloPersonLabeler(weights: str = "yolo26x.pt", conf: float = 0.5, device: str | int | None = None, batch: int = 16)`
  - `add_person_labels(samples: Sequence[Sample], labeler: PersonLabeler, chunk: int = 64) -> None`
  - `compute_stats(splits: Mapping[str, Sequence[Sample]], hardneg: Sequence[Sample], dropped: Mapping[str, int]) -> dict`
  - `write_dataset(out: Path, splits: Mapping[str, Sequence[Sample]], hardneg: Sequence[Sample], dropped: Mapping[str, int]) -> dict`
  - `build(out: Path, sources: Iterable[Iterable[Sample]], hardneg: Iterable[Sample], labeler: PersonLabeler, *, dropped: Mapping[str, int] | None = None, seed: int = 1234, max_distance: int = 6) -> dict`
  - Dataset layout: `data.yaml`, `images/{train,val,test}/<key><ext>`, `labels/{train,val,test}/<key>.txt`, `hardneg/images/<key><ext>`, `manifest.jsonl`, `stats.json`, `LICENSES.md`
  - `stats.json` = `{"splits": {split: {"images", "with_weapon", "without_weapon", "instances": {class: n}}}, "sources": {source: n}, "hardneg_images": n, "dropped": {reason: n}}`

- [ ] **Step 1: Write the failing test**

`training/tests/test_build.py`:
```python
import json

import numpy as np
import pytest
import yaml
from PIL import Image

from weaponshield_ml.build import build
from weaponshield_ml.classes import KNIFE, PERSON, PISTOL
from weaponshield_ml.labels import Box, read_labels
from weaponshield_ml.sample import Sample


class FakeLabeler:
    def __init__(self):
        self.calls = 0

    def __call__(self, paths):
        self.calls += 1
        return [[Box(PERSON, 1, 1, 20, 40)] for _ in paths]


def make_samples(tmp_path, n, source, weapon_cls):
    rng = np.random.default_rng(abs(hash(source)) % 2**32)
    out = []
    for i in range(n):
        p = tmp_path / f"{source}-{i}.png"
        Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(p)
        boxes = [Box(weapon_cls, 5, 5, 30, 30), Box(PERSON, 0, 0, 2, 2)] if weapon_cls is not None else []
        out.append(Sample(key=f"{source}-{i}", source=source, image_path=p, width=64, height=48, boxes=boxes))
    return out


@pytest.fixture
def dataset(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    weapons = make_samples(src, 40, "hf-weapons", PISTOL)
    knives = make_samples(src, 20, "oi-weapons", KNIFE)
    negatives = make_samples(src, 20, "oi-negatives", None)
    dup = Sample(key="dup", source="oi-negatives", image_path=weapons[0].image_path, width=64, height=48)
    hardneg = make_samples(src, 5, "oi-negatives-validation", None)
    out = tmp_path / "ds"
    labeler = FakeLabeler()
    stats = build(out, [weapons, knives, negatives + [dup]], hardneg, labeler, dropped={"hf_ambiguous_label": 3})
    return out, stats, labeler


def test_layout_and_yaml(dataset):
    out, _, _ = dataset
    cfg = yaml.safe_load((out / "data.yaml").read_text())
    assert cfg["names"] == {0: "person", 1: "pistol", 2: "rifle", 3: "knife"}
    assert (cfg["train"], cfg["val"], cfg["test"]) == ("images/train", "images/val", "images/test")
    for split in ("train", "val", "test"):
        images = sorted(p.stem for p in (out / "images" / split).iterdir())
        labels = sorted(p.stem for p in (out / "labels" / split).iterdir())
        assert images == labels
    assert len(list((out / "hardneg" / "images").iterdir())) == 5
    assert (out / "LICENSES.md").read_text().startswith("# Dataset sources and licences")


def test_people_come_from_the_labeler_only(dataset):
    out, _, labeler = dataset
    assert labeler.calls >= 1
    for f in (out / "labels").rglob("*.txt"):
        persons = [b for b in read_labels(f, 64, 48) if b.cls == PERSON]
        assert len(persons) == 1
        assert round(persons[0].x2) == 20  # the source's tiny person box was replaced


def test_stats_and_duplicates(dataset):
    out, stats, _ = dataset
    assert stats == json.loads((out / "stats.json").read_text())
    assert sum(v["images"] for v in stats["splits"].values()) == 80  # 81 samples minus 1 exact duplicate
    assert stats["dropped"] == {"hf_ambiguous_label": 3, "exact_duplicates": 1}
    assert stats["sources"] == {"hf-weapons": 40, "oi-weapons": 20, "oi-negatives": 20}
    assert stats["hardneg_images"] == 5
    total_knives = sum(v["instances"]["knife"] for v in stats["splits"].values())
    assert total_knives == 20


def test_manifest_has_no_group_in_two_splits(dataset):
    out, _, _ = dataset
    owner = {}
    for line in (out / "manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        assert owner.setdefault(row["group"], row["split"]) == row["split"]


def test_refuses_to_overwrite(tmp_path):
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "x").write_text("x")
    with pytest.raises(FileExistsError):
        build(tmp_path / "ds", [[]], [], FakeLabeler())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /d/Projects/WeaponShield-AI/training && .venv/Scripts/python -m pytest tests/test_build.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'weaponshield_ml.build'`.

- [ ] **Step 3: Implement `training/weaponshield_ml/autolabel.py`**

```python
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
    def __init__(self, weights: str = "yolo26x.pt", conf: float = 0.5, device: str | int | None = None, batch: int = 16):
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
```

- [ ] **Step 4: Implement `training/weaponshield_ml/writer.py`**

```python
"""Writes the YOLO dataset directory, manifest, statistics and licence notes."""
from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from .classes import CLASS_NAMES
from .labels import write_labels
from .sample import Sample

LICENSES = """# Dataset sources and licences

- **Weapon images (hf-weapons):** "weapon-detection" by yolov7test on Roboflow Universe, via the Hugging Face mirror
  `Subh775/WeaponDetection`. Licence: CC BY 4.0.
- **Open Images V7 (oi-*):** annotations (c) Google LLC, CC BY 4.0. Images are listed by Open Images as CC BY 2.0;
  see https://storage.googleapis.com/openimages/web/factsfigures_v7.html for per-image authors.
- **Person boxes:** generated by Ultralytics YOLO26x (COCO, AGPL-3.0) at confidence >= 0.5.
"""


def compute_stats(
    splits: Mapping[str, Sequence[Sample]], hardneg: Sequence[Sample], dropped: Mapping[str, int]
) -> dict:
    stats: dict = {"splits": {}, "sources": {}, "hardneg_images": len(hardneg), "dropped": dict(dropped)}
    sources: Counter[str] = Counter()
    for split, items in splits.items():
        instances: Counter[str] = Counter()
        with_weapon = 0
        for s in items:
            sources[s.source] += 1
            with_weapon += bool(s.weapon_classes)
            for b in s.boxes:
                instances[CLASS_NAMES[b.cls]] += 1
        stats["splits"][split] = {
            "images": len(items),
            "with_weapon": with_weapon,
            "without_weapon": len(items) - with_weapon,
            "instances": {name: instances[name] for name in CLASS_NAMES},
        }
    stats["sources"] = dict(sorted(sources.items()))
    return stats


def _image_name(s: Sample) -> str:
    return f"{s.key}{s.image_path.suffix.lower() or '.jpg'}"


def write_dataset(
    out: Path, splits: Mapping[str, Sequence[Sample]], hardneg: Sequence[Sample], dropped: Mapping[str, int]
) -> dict:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for split, items in splits.items():
            img_dir, lbl_dir = out / "images" / split, out / "labels" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            for s in items:
                name = _image_name(s)
                shutil.copyfile(s.image_path, img_dir / name)
                write_labels(lbl_dir / f"{s.key}.txt", s.boxes, s.width, s.height)
                row = {
                    "file": f"images/{split}/{name}",
                    "split": split,
                    "source": s.source,
                    "group": s.group,
                    "classes": sorted({CLASS_NAMES[b.cls] for b in s.boxes}),
                }
                manifest.write(json.dumps(row) + "\n")

    hard_dir = out / "hardneg" / "images"
    hard_dir.mkdir(parents=True, exist_ok=True)
    for s in hardneg:
        shutil.copyfile(s.image_path, hard_dir / _image_name(s))

    data = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(CLASS_NAMES)),
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    (out / "LICENSES.md").write_text(LICENSES, encoding="utf-8")
    stats = compute_stats(splits, hardneg, dropped)
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
```

- [ ] **Step 5: Implement `training/weaponshield_ml/build.py` and the sources package marker**

`training/weaponshield_ml/sources/__init__.py`:
```python
"""Data source loaders. Each yields Sample objects with weapon boxes only (people are added by autolabel)."""
```

`training/weaponshield_ml/build.py`:
```python
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
from .grouping import compute_hashes, dedupe_and_group
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
    add_person_labels(samples, labeler)
    kept, exact = dedupe_and_group(samples, compute_hashes(samples), max_distance)
    splits = split_groups(kept, seed=seed)
    return write_dataset(out, splits, hard, {**(dropped or {}), "exact_duplicates": exact})


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
            args.cache, dropped, split="validation", classes=OI_NEGATIVE_CLASSES, max_samples=hardneg_max * 2, role="negatives"
        )
    )[:hardneg_max]
    stats = build(args.out, sources, hard, YoloPersonLabeler(weights, device=args.device), dropped=dropped)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the tests and lint**

Run: `cd /d/Projects/WeaponShield-AI/training && .venv/Scripts/python -m pytest -q && .venv/Scripts/python -m ruff format . && .venv/Scripts/python -m ruff check .`
Expected: all tests pass (the `build` imports of `sources.*` happen inside `main`, so they are not needed yet); ruff clean.

- [ ] **Step 7: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add training/weaponshield_ml/autolabel.py training/weaponshield_ml/writer.py training/weaponshield_ml/build.py training/weaponshield_ml/sources/__init__.py training/tests/test_build.py
git commit -m "feat(training): person pseudo-labels, dataset writer with stats and licences, build orchestration"
```

---

### Task 4: Real data sources and a local smoke build

**Files:**
- Create: `training/weaponshield_ml/sources/hf_weapons.py`, `training/weaponshield_ml/sources/open_images.py`
- Test: `training/tests/test_sources.py`

**Interfaces:**
- Consumes: `resolve`, `HF_LABELS`, `OI_LABELS`, `DROP_IMAGE` (Task 1); `Sample`, `Box`; `build.main` (Task 3)
- Produces:
  - `HF_REPO = "Subh775/WeaponDetection"`
  - `hf_objects_to_sample(objects: Mapping[str, Sequence], names: Sequence[str], w: int, h: int, image_path: Path, key: str, dropped: Counter[str]) -> Sample | None`
  - `load_hf_weapons(cache: Path, dropped: Counter[str], limit: int | None = None) -> Iterator[Sample]`
  - `oi_detections_to_boxes(dets: Iterable[tuple[str, Sequence[float]]], w: int, h: int) -> list[Box] | Literal["drop-image"]` (Open Images boxes are relative `[x, y, w, h]`)
  - `load_open_images(cache: Path, dropped: Counter[str], *, split: str, classes: Sequence[str], max_samples: int | None, role: Literal["weapons", "negatives"], seed: int = 51) -> Iterator[Sample]`
  - Sample sources: `hf-weapons`, `oi-weapons`, `oi-negatives` (train) and `oi-negatives-validation` (hard negatives)

- [ ] **Step 1: Write the failing test**

`training/tests/test_sources.py`:
```python
from collections import Counter
from pathlib import Path

import pytest

from weaponshield_ml.classes import DROP_IMAGE, KNIFE, PISTOL, RIFLE
from weaponshield_ml.labels import Box
from weaponshield_ml.sources.hf_weapons import hf_objects_to_sample
from weaponshield_ml.sources.open_images import oi_detections_to_boxes

NAMES = ["weapons", "Aggressor", "Pistol", "Rifle", "Knife", "person"]


def test_hf_converts_coco_pixel_boxes():
    dropped = Counter()
    s = hf_objects_to_sample(
        {"bbox": [[10, 20, 30, 40], [0, 0, 5, 5]], "category": [2, 5]}, NAMES, 100, 100, Path("x.jpg"), "hf-train-1", dropped
    )
    assert s is not None
    assert s.boxes == [Box(PISTOL, 10, 20, 40, 60)]
    assert s.source == "hf-weapons"


def test_hf_drops_images_with_generic_labels():
    dropped = Counter()
    s = hf_objects_to_sample({"bbox": [[1, 1, 5, 5], [1, 1, 5, 5]], "category": [3, 0]}, NAMES, 50, 50, Path("x.jpg"), "k", dropped)
    assert s is None
    assert dropped == {"hf_ambiguous_label": 1}


def test_hf_drops_images_without_weapons():
    dropped = Counter()
    assert hf_objects_to_sample({"bbox": [[1, 1, 5, 5]], "category": [1]}, NAMES, 50, 50, Path("x.jpg"), "k", dropped) is None
    assert dropped == {"hf_no_weapon_boxes": 1}


def test_hf_rejects_boxes_that_are_not_coco_pixels():
    with pytest.raises(ValueError, match="normalised"):
        hf_objects_to_sample({"bbox": [[0.1, 0.1, 0.5, 0.5]], "category": [4]}, NAMES, 100, 100, Path("x"), "k", Counter())
    with pytest.raises(ValueError, match="outside"):
        hf_objects_to_sample({"bbox": [[90, 90, 50, 50]], "category": [4]}, NAMES, 100, 100, Path("x"), "k", Counter())


def test_open_images_relative_boxes():
    boxes = oi_detections_to_boxes(
        [("Handgun", [0.1, 0.2, 0.3, 0.4]), ("Shotgun", [0, 0, 1, 1]), ("Dagger", [0.5, 0.5, 0.1, 0.1]), ("Man", [0, 0, 1, 1])],
        200,
        100,
    )
    assert isinstance(boxes, list)
    assert [(b.cls, *(round(v, 6) for v in (b.x1, b.y1, b.x2, b.y2))) for b in boxes] == [
        (PISTOL, 20, 20, 80, 60),
        (RIFLE, 0, 0, 200, 100),
        (KNIFE, 100, 50, 120, 60),
    ]


def test_open_images_sword_drops_the_image():
    assert oi_detections_to_boxes([("Knife", [0, 0, 0.1, 0.1]), ("Sword", [0, 0, 0.1, 0.1])], 10, 10) == DROP_IMAGE
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /d/Projects/WeaponShield-AI/training && .venv/Scripts/python -m pytest tests/test_sources.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'weaponshield_ml.sources.hf_weapons'`.

- [ ] **Step 3: Implement `training/weaponshield_ml/sources/hf_weapons.py`**

```python
"""Subh775/WeaponDetection: Hugging Face mirror of Roboflow Universe "weapon-detection" (CC BY 4.0).

Boxes are COCO pixel [x, y, w, h]. No API key is needed.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from ..classes import DROP_IMAGE, HF_LABELS, resolve
from ..labels import Box
from ..sample import Sample

HF_REPO = "Subh775/WeaponDetection"
SOURCE = "hf-weapons"


def hf_objects_to_sample(
    objects: Mapping[str, Sequence],
    names: Sequence[str],
    w: int,
    h: int,
    image_path: Path,
    key: str,
    dropped: Counter[str],
) -> Sample | None:
    boxes: list[Box] = []
    for bbox, cat in zip(objects["bbox"], objects["category"], strict=True):
        r = resolve(names[cat], HF_LABELS, strict=True)
        if r == DROP_IMAGE:
            dropped["hf_ambiguous_label"] += 1
            return None
        x, y, bw, bh = (float(v) for v in bbox)
        if max(x, y, bw, bh) <= 1.0 and min(w, h) > 32:
            raise ValueError(f"{key}: bbox {list(bbox)} looks normalised; expected COCO pixel [x, y, w, h]")
        if x + bw > w * 1.05 + 1 or y + bh > h * 1.05 + 1:
            raise ValueError(f"{key}: bbox {list(bbox)} outside {w}x{h}; expected COCO pixel [x, y, w, h]")
        if r is None:
            continue
        boxes.append(Box(r, x, y, x + bw, y + bh))
    if not boxes:
        dropped["hf_no_weapon_boxes"] += 1
        return None
    return Sample(key=key, source=SOURCE, image_path=image_path, width=w, height=h, boxes=boxes)


def load_hf_weapons(cache: Path, dropped: Counter[str], limit: int | None = None) -> Iterator[Sample]:
    from datasets import load_dataset

    img_dir = cache / "hf-images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        ds = load_dataset(HF_REPO, split=split if limit is None else f"{split}[:{limit}]", cache_dir=str(cache / "hf"))
        names = ds.features["objects"]["category"].feature.names
        for row in ds:
            key = f"hf-{split}-{row['image_id']}"
            image = row["image"]
            w, h = image.size
            s = hf_objects_to_sample(row["objects"], names, w, h, img_dir / f"{key}.jpg", key, dropped)
            if s is None:
                continue
            if not s.image_path.exists():
                image.convert("RGB").save(s.image_path, quality=95)
            yield s
```

- [ ] **Step 4: Implement `training/weaponshield_ml/sources/open_images.py`**

```python
"""Open Images V7 via FiftyOne: weapon images and look-alike negatives. Boxes are relative [x, y, w, h]."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Literal

from ..classes import DROP_IMAGE, OI_LABELS, resolve
from ..labels import Box
from ..sample import Sample


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
    import fiftyone.zoo as foz

    ds = foz.load_zoo_dataset(
        "open-images-v7",
        split=split,
        label_types=["detections"],
        classes=list(classes),
        max_samples=max_samples,
        only_matching=False,  # keep every label on matched images, so weapons in negatives can be detected
        shuffle=True,
        seed=seed,
        dataset_dir=str(cache / "open-images-v7"),
        dataset_name=f"ws-{role}-{split}",
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
```

- [ ] **Step 5: Run the unit tests and lint**

Run: `cd /d/Projects/WeaponShield-AI/training && .venv/Scripts/python -m pytest -q && .venv/Scripts/python -m ruff format . && .venv/Scripts/python -m ruff check .`
Expected: all pass, ruff clean.

- [ ] **Step 6: Install the full requirements and run a smoke build**

```bash
cd /d/Projects/WeaponShield-AI/training
.venv/Scripts/python -m pip install -q -r requirements.txt
.venv/Scripts/python -m weaponshield_ml.build --out datasets/smoke --cache datasets/cache --device cpu --smoke
```
Expected: JSON stats printed; `datasets/smoke/images/train` holds tens of images; `stats.json` shows non-zero `pistol`/`rifle`/`knife` instances and `hardneg_images` ≤ 10. If FiftyOne fails to start its database on Windows, record the error in the report, then re-run only the Hugging Face part by temporarily calling `build()` from a Python shell with `[load_hf_weapons(...)]` as the only source and `[]` as hardneg, to at least verify that path — the Open Images path will be verified on Kaggle (Linux) in Task 8's smoke run.

- [ ] **Step 7: Eyeball the labels**

```bash
cd /d/Projects/WeaponShield-AI/training
.venv/Scripts/python - <<'EOF'
from pathlib import Path
import cv2
from weaponshield_ml.classes import CLASS_NAMES
from weaponshield_ml.labels import read_labels
root = Path("datasets/smoke")
out = Path("datasets/smoke-preview"); out.mkdir(exist_ok=True)
for img in sorted((root / "images" / "train").iterdir())[:8]:
    im = cv2.imread(str(img)); h, w = im.shape[:2]
    for b in read_labels(root / "labels" / "train" / f"{img.stem}.txt", w, h):
        colour = (80, 200, 0) if b.cls == 0 else (40, 40, 230)
        cv2.rectangle(im, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), colour, 2)
        cv2.putText(im, CLASS_NAMES[b.cls], (int(b.x1), max(12, int(b.y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    cv2.imwrite(str(out / img.name.replace(".png", ".jpg")), im)
print("wrote", len(list(out.iterdir())))
EOF
```
Open 3–4 files in `training/datasets/smoke-preview/` with the Read tool and confirm the boxes sit on the weapons/people. Describe what you saw in the report (e.g. "hf-train-12: pistol box on handgun, person box on holder").

- [ ] **Step 8: Commit (code only — datasets are git-ignored)**

```bash
cd /d/Projects/WeaponShield-AI
git status --short training | grep -v "^?? training/datasets" || true
git add training/weaponshield_ml/sources training/tests/test_sources.py
git commit -m "feat(training): Hugging Face weapon and Open Images V7 loaders with strict label mapping"
```

---

### Task 5: Training CLI with CCTV-style augmentation

**Files:**
- Create: `training/weaponshield_ml/training.py`, `training/scripts/train.py`
- Test: `training/tests/test_training.py`

**Interfaces:**
- Consumes: dataset layout from Task 3 (`data.yaml` with `path: .`)
- Produces:
  - `cctv_augmentations() -> list` (Albumentations transforms, exact list below)
  - `write_data_yaml(data_root: Path, dest: Path) -> Path` — copy of `data.yaml` with absolute `path`
  - `resolve_device(device: str) -> str | int | list[int]` — `"auto"` → `"cpu"` / `0` / `[0, 1, …]`
  - CLI: `python scripts/train.py --data-root DIR --model yolo26s.pt --out DIR --name NAME [--epochs 300] [--hours H] [--batch 32] [--imgsz 640] [--device auto] [--workers 8]`; prints the path to `best.pt` as the last line

- [ ] **Step 1: Write the failing test**

`training/tests/test_training.py`:
```python
import numpy as np
import yaml

from weaponshield_ml.training import cctv_augmentations, resolve_device, write_data_yaml


def test_cctv_augmentations_are_image_only_and_shape_preserving():
    import albumentations as A

    transforms = cctv_augmentations()
    assert [type(t).__name__ for t in transforms] == [
        "MotionBlur",
        "ImageCompression",
        "RandomBrightnessContrast",
        "RandomGamma",
        "GaussNoise",
        "Downscale",
        "ToGray",
        "CLAHE",
    ]
    assert all(isinstance(t, A.ImageOnlyTransform) for t in transforms)
    img = np.random.default_rng(0).integers(0, 255, (96, 128, 3), dtype=np.uint8)
    for t in transforms:
        t.p = 1.0
        assert t(image=img)["image"].shape == img.shape


def test_write_data_yaml_makes_path_absolute(tmp_path):
    root = tmp_path / "ds"
    root.mkdir()
    (root / "data.yaml").write_text(yaml.safe_dump({"path": ".", "train": "images/train", "names": {0: "person"}}))
    dest = write_data_yaml(root, tmp_path / "run" / "data.yaml")
    cfg = yaml.safe_load(dest.read_text())
    assert cfg["path"] == str(root.resolve()).replace("\\", "/")
    assert cfg["train"] == "images/train"


def test_resolve_device_passthrough():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("0") == "0"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /d/Projects/WeaponShield-AI/training && .venv/Scripts/python -m pytest tests/test_training.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'weaponshield_ml.training'`.

- [ ] **Step 3: Implement `training/weaponshield_ml/training.py`**

```python
"""Training helpers: CCTV-style augmentation, dataset yaml rewrite, device selection."""
from __future__ import annotations

from pathlib import Path

import yaml


def cctv_augmentations() -> list:
    """Image-only transforms that imitate surveillance footage (applied on top of Ultralytics' defaults)."""
    import albumentations as A

    return [
        A.MotionBlur(blur_limit=(3, 9), p=0.2),
        A.ImageCompression(quality_range=(30, 90), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.1), contrast_limit=(-0.2, 0.2), p=0.3),
        A.RandomGamma(gamma_limit=(60, 140), p=0.2),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
        A.Downscale(scale_range=(0.35, 0.7), p=0.15),
        A.ToGray(p=0.05),
        A.CLAHE(p=0.02),
    ]


def write_data_yaml(data_root: Path, dest: Path) -> Path:
    cfg = yaml.safe_load((data_root / "data.yaml").read_text(encoding="utf-8"))
    cfg["path"] = str(data_root.resolve()).replace("\\", "/")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return dest


def resolve_device(device: str) -> str | int | list[int]:
    if device != "auto":
        return device
    import torch

    n = torch.cuda.device_count()
    return "cpu" if n == 0 else 0 if n == 1 else list(range(n))
```

- [ ] **Step 4: Implement `training/scripts/train.py`**

```python
"""Fine-tune YOLO26 on the WeaponShield dataset.

    python scripts/train.py --data-root datasets/ws --model yolo26s.pt --out runs --name ws-yolo26s --hours 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weaponshield_ml.training import cctv_augmentations, resolve_device, write_data_yaml  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--hours", type=float, default=None, help="time budget; overrides --epochs when set")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    from ultralytics import YOLO

    data_yaml = write_data_yaml(args.data_root, args.out / f"{args.name}-data.yaml")
    extra = {"time": args.hours} if args.hours else {}
    YOLO(args.model).train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=resolve_device(args.device),
        workers=args.workers,
        patience=30,
        cos_lr=True,
        close_mosaic=10,
        augmentations=cctv_augmentations(),
        project=str(args.out.resolve()),
        name=args.name,
        exist_ok=True,
        plots=True,
        seed=0,
        **extra,
    )
    best = args.out.resolve() / args.name / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"Training finished without {best}")
    print(best)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the unit tests and lint**

Run: `cd /d/Projects/WeaponShield-AI/training && .venv/Scripts/python -m pytest -q && .venv/Scripts/python -m ruff format . && .venv/Scripts/python -m ruff check .`
Expected: all pass, ruff clean.

- [ ] **Step 6: Smoke-train one epoch on CPU**

```bash
cd /d/Projects/WeaponShield-AI/training
.venv/Scripts/python scripts/train.py --data-root datasets/smoke --model yolo26n.pt --out runs --name smoke --epochs 1 --batch 4 --device cpu --workers 0
```
Expected: one epoch completes; the log line `albumentations:` lists `MotionBlur`, `ImageCompression`, … (proof the custom list was used); the last printed line is a path ending in `runs\smoke\weights\best.pt`.

- [ ] **Step 7: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add training/weaponshield_ml/training.py training/scripts/train.py training/tests/test_training.py
git commit -m "feat(training): training CLI with CCTV-style Albumentations and time budget"
```

---

### Task 6: Evaluation CLI (metrics, thresholds, hard negatives, failure gallery)

**Files:**
- Create: `training/weaponshield_ml/evaluation.py`, `training/scripts/evaluate.py`
- Test: `training/tests/test_evaluation.py`

**Interfaces:**
- Consumes: `Box`, `iou`, `read_labels`, `CLASS_NAMES`, `WEAPON_IDS` (Task 1); `write_data_yaml` (Task 5); dataset layout (Task 3)
- Produces:
  - `PRESET_NAMES = ("low", "balanced", "high")`
  - `DEFAULT_THRESHOLDS = {"low": 0.6, "balanced": 0.45, "high": 0.3}`
  - `choose_thresholds(px, p, r, f1, row_names, *, low_precision=0.9, high_recall=0.9) -> dict[str, dict[str, float]]` — balanced = argmax F1; low = smallest conf ≥ balanced with precision ≥ 0.9 (else min(balanced + 0.15, 0.9)); high = largest conf ≤ balanced with recall ≥ 0.9 (else max(balanced − 0.15, 0.05)); rounded to 2 dp; every class in `CLASS_NAMES` missing from `row_names` (no validation instances) gets `DEFAULT_THRESHOLDS`
  - `downsample(x, y, n=101) -> list[list[float]]`
  - `match_detections(gt: Sequence[Box], pred: Sequence[tuple[Box, float]], iou_thr=0.5) -> tuple[int, list[Box], list[tuple[Box, float]]]` (greedy by confidence, same class)
  - `weapon_summary(per_class: Mapping[str, Mapping[str, float]]) -> dict[str, float]` (mean over pistol/rifle/knife)
  - `hardneg_fp_rate(per_image: Sequence[Sequence[tuple[int, float]]], thresholds: Mapping[str, float]) -> float`
  - CLI: `python scripts/evaluate.py --weights best.pt --data-root DIR --out DIR [--device auto]` writes `metrics.json`, `gallery/*.jpg`, `gallery.json`, and Ultralytics plots under `test/`
  - `metrics.json` keys: `model`, `ultralytics`, `test.per_class{class: {precision, recall, map50, map50_95}}`, `test.weapons`, `test.map50_all`, `test.map50_95_all`, `thresholds{preset: {class: conf}}`, `hardneg{images, fp_rate{preset: rate}}`, `pr_curves{class: [[recall, precision], …]}`, `confusion{labels, matrix}`

- [ ] **Step 1: Write the failing test**

`training/tests/test_evaluation.py`:
```python
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
    f1 = (2 * p * r / (p + r))
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


def test_hardneg_fp_rate_counts_images_with_any_weapon_above_threshold():
    per_image = [[(0, 0.99)], [(1, 0.5)], [(3, 0.2)], []]
    assert hardneg_fp_rate(per_image, {"person": 0.5, "pistol": 0.45, "rifle": 0.45, "knife": 0.45}) == 0.25
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /d/Projects/WeaponShield-AI/training && .venv/Scripts/python -m pytest tests/test_evaluation.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'weaponshield_ml.evaluation'`.

- [ ] **Step 3: Implement `training/weaponshield_ml/evaluation.py`**

```python
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
        precise = np.where((p[i] >= low_precision) & (x >= balanced))[0]
        low = float(x[precise[0]]) if len(precise) else min(balanced + 0.15, 0.9)
        sensitive = np.where((r[i] >= high_recall) & (x <= balanced))[0]
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
    keys = ("precision", "recall", "map50", "map50_95")
    rows = [per_class[n] for n in WEAPON_NAMES if n in per_class]
    return {k: float(np.mean([row[k] for row in rows])) for k in keys}


def hardneg_fp_rate(per_image: Sequence[Sequence[tuple[int, float]]], thresholds: Mapping[str, float]) -> float:
    if not per_image:
        return 0.0
    flagged = sum(
        any(cls in WEAPON_IDS and conf >= thresholds[CLASS_NAMES[cls]] for cls, conf in preds) for preds in per_image
    )
    return flagged / len(per_image)
```

- [ ] **Step 4: Implement `training/scripts/evaluate.py`**

```python
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
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def predictions(result) -> list[tuple[Box, float]]:
    return [
        (Box(int(c), *map(float, xyxy)), float(s))
        for xyxy, s, c in zip(
            result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist(), strict=True
        )
    ]


def save_gallery(model, data_root: Path, out: Path, balanced: dict[str, float], device) -> list[dict]:
    import cv2

    misses: list[tuple[float, Path, Box]] = []
    false_alarms: list[tuple[float, Path, Box]] = []
    floor = min(balanced[CLASS_NAMES[i]] for i in WEAPON_IDS)
    for r in model.predict([str(p) for p in images_in(data_root / "images" / "test")], conf=floor, device=device, stream=True, verbose=False):
        path = Path(r.path)
        h, w = r.orig_shape
        gt = [b for b in read_labels(data_root / "labels" / "test" / f"{path.stem}.txt", w, h) if b.cls in WEAPON_IDS]
        pred = [(b, c) for b, c in predictions(r) if b.cls in WEAPON_IDS and c >= balanced[CLASS_NAMES[b.cls]]]
        _, fn, fp = match_detections(gt, pred)
        misses += [(b.area, path, b) for b in fn]
        false_alarms += [(c, path, b) for b, c in fp]

    gallery_dir = out / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    picks = [("miss", sorted(misses, key=lambda t: -t[0])[:GALLERY_PER_KIND]), ("false-alarm", sorted(false_alarms, key=lambda t: -t[0])[:GALLERY_PER_KIND])]
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
    common = dict(data=str(data_yaml), imgsz=640, batch=16, conf=0.001, iou=0.7, device=device, project=str(out), exist_ok=True, verbose=False)

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
    floor = min(min(t.values()) for t in thresholds.values())
    hard_preds = [
        [(int(c), float(s)) for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), strict=True)]
        for r in model.predict([str(x) for x in hard_images], conf=floor, device=device, stream=True, verbose=False)
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
        "hardneg": {"images": len(hard_images), "fp_rate": {k: hardneg_fp_rate(hard_preds, thresholds[k]) for k in PRESET_NAMES}},
        "pr_curves": pr_curves,
        "confusion": {"labels": [*CLASS_NAMES, "background"], "matrix": test.confusion_matrix.matrix.tolist()},
    }
    gallery = save_gallery(model, args.data_root, out, thresholds["balanced"], device)
    (out / "gallery.json").write_text(json.dumps(gallery, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics["test"]["weapons"] | {"hardneg_fp_balanced": metrics["hardneg"]["fp_rate"]["balanced"]}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the unit tests and lint**

Run: `cd /d/Projects/WeaponShield-AI/training && .venv/Scripts/python -m pytest -q && .venv/Scripts/python -m ruff format . && .venv/Scripts/python -m ruff check .`
Expected: all pass, ruff clean.

- [ ] **Step 6: Smoke-evaluate and export the smoke model**

```bash
cd /d/Projects/WeaponShield-AI/training
.venv/Scripts/python scripts/evaluate.py --weights runs/smoke/weights/best.pt --data-root datasets/smoke --out runs/smoke-eval --device cpu
.venv/Scripts/python scripts/export_onnx.py --weights runs/smoke/weights/best.pt --out runs/smoke.onnx
.venv/Scripts/python -c "import json; m=json.load(open('runs/smoke-eval/metrics.json')); print(sorted(m), sorted(m['thresholds']), m['hardneg'])"
```
Expected: `metrics.json` has keys `confusion, hardneg, model, pr_curves, test, thresholds, ultralytics`; thresholds have `balanced, high, low`; the export prints a sha256. Metric values from a 1-epoch smoke model are meaningless — only the structure is being checked.

- [ ] **Step 7: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add training/weaponshield_ml/evaluation.py training/scripts/evaluate.py training/tests/test_evaluation.py
git commit -m "feat(training): evaluation CLI (test metrics, val-tuned thresholds, hard negatives, failure gallery)"
```

---

### Task 7: Kaggle jobs and training CI

**Files:**
- Create: `training/kaggle/run_dataset.py`, `training/kaggle/run_train.py`
- Create: `training/kaggle/dataset/kernel-metadata.json`, `training/kaggle/dataset/bootstrap.py`, `training/kaggle/train-n/kernel-metadata.json`, `training/kaggle/train-n/bootstrap.py`, `training/kaggle/train-s/kernel-metadata.json`, `training/kaggle/train-s/bootstrap.py`
- Create: `.github/workflows/training.yml`

**Interfaces:**
- Consumes: `python -m weaponshield_ml.build` (Task 3/4), `scripts/train.py` (Task 5), `scripts/evaluate.py` (Task 6), `scripts/export_onnx.py` (Plan 1)
- Produces:
  - Kaggle kernel `<user>/weaponshield-dataset` output: `ws-dataset.tar`, `stats.json`, `source-commit.txt`
  - Kaggle kernels `<user>/weaponshield-train-n` and `<user>/weaponshield-train-s` outputs: `ws-yolo26{n,s}.pt`, `ws-yolo26{n,s}.onnx`, `ws-yolo26{n,s}.onnx.sha256`, `eval-yolo26{n,s}/` (metrics.json, gallery), `runs/` plots, `source-commit.txt`

- [ ] **Step 1: Find the Kaggle username**

```bash
cd /d/Projects/WeaponShield-AI/training
.venv/Scripts/kaggle config view 2>&1 | grep -i -E "username|user" || true
.venv/Scripts/kaggle kernels list --mine --page-size 5 2>&1 | head -5
```
Expected: a username line, or kernel refs of the form `<username>/<slug>`. If neither reveals the username, stop and report NEEDS_CONTEXT asking the controller for the Kaggle username. Never print the token file.

- [ ] **Step 2: Write `training/kaggle/run_dataset.py`**

```python
"""Kaggle job: build the WeaponShield dataset and save it as one tar (Kaggle outputs prefer few large files)."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1]
SRC = TRAINING.parent
WORK = Path("/kaggle/working")
DATASET = Path("/tmp/ws-dataset")


def run(*cmd: str, cwd: Path = TRAINING) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> None:
    smoke = "--smoke" in sys.argv
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=SRC, capture_output=True, text=True, check=True).stdout.strip()
    run(sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt")
    build = [sys.executable, "-m", "weaponshield_ml.build", "--out", str(DATASET), "--cache", "/tmp/ws-cache", "--device", "0"]
    run(*build, *(["--smoke"] if smoke else []))
    with tarfile.open(WORK / "ws-dataset.tar", "w") as tar:
        tar.add(DATASET, arcname="ws-dataset")
    shutil.copyfile(DATASET / "stats.json", WORK / "stats.json")
    (WORK / "source-commit.txt").write_text(commit + "\n")
    shutil.rmtree(SRC)  # keep the Kaggle output to the dataset artefacts only


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write `training/kaggle/run_train.py`**

```python
"""Kaggle job: train, evaluate and export one YOLO26 variant on the WeaponShield dataset.

    python run_train.py --variant s
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1]
SRC = TRAINING.parent
WORK = Path("/kaggle/working")
DATA_PARENT = Path("/tmp")

VARIANTS = {
    "n": {"model": "yolo26n.pt", "hours": "4", "batch": "64"},
    "s": {"model": "yolo26s.pt", "hours": "10", "batch": "32"},
}


def run(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=TRAINING)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    p.add_argument("--smoke", action="store_true", help="one short epoch to test the job end to end")
    args = p.parse_args()
    v = VARIANTS[args.variant]
    name = f"ws-yolo26{args.variant}"

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=SRC, capture_output=True, text=True, check=True).stdout.strip()
    run(sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt")

    tar_path = next(Path("/kaggle/input").rglob("ws-dataset.tar"))
    safe = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
    with tarfile.open(tar_path) as tar:
        tar.extractall(DATA_PARENT, **safe)
    data_root = DATA_PARENT / "ws-dataset"

    budget = ["--epochs", "1"] if args.smoke else ["--hours", v["hours"]]
    run(sys.executable, "scripts/train.py", "--data-root", str(data_root), "--model", v["model"], "--out", str(WORK / "runs"),
        "--name", name, "--batch", v["batch"], "--device", "auto", *budget)
    best = WORK / "runs" / name / "weights" / "best.pt"
    run(sys.executable, "scripts/evaluate.py", "--weights", str(best), "--data-root", str(data_root), "--out", str(WORK / f"eval-yolo26{args.variant}"))
    onnx = WORK / f"{name}.onnx"
    run(sys.executable, "scripts/export_onnx.py", "--weights", str(best), "--out", str(onnx))

    shutil.copyfile(best, WORK / f"{name}.pt")
    (WORK / f"{name}.onnx.sha256").write_text(hashlib.sha256(onnx.read_bytes()).hexdigest() + "\n")
    (WORK / "source-commit.txt").write_text(commit + "\n")
    (WORK / "runs" / name / "weights" / "last.pt").unlink(missing_ok=True)
    shutil.rmtree(SRC)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the kernel bootstraps and metadata** (replace `<KAGGLE_USER>` with the username from Step 1 in all three metadata files)

`training/kaggle/dataset/bootstrap.py`:
```python
# Kaggle bootstrap. The real job lives in the repo: training/kaggle/run_dataset.py. Add "--smoke" to ARGS for a quick test.
import subprocess
import sys

ARGS: list[str] = []
subprocess.run(["git", "clone", "--depth", "1", "--branch", "v2", "https://github.com/Halok600/WeaponShield-AI.git", "/kaggle/working/src"], check=True)
subprocess.run([sys.executable, "/kaggle/working/src/training/kaggle/run_dataset.py", *ARGS], check=True)
```

`training/kaggle/dataset/kernel-metadata.json`:
```json
{
  "id": "<KAGGLE_USER>/weaponshield-dataset",
  "title": "weaponshield-dataset",
  "code_file": "bootstrap.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
```

`training/kaggle/train-n/bootstrap.py`:
```python
# Kaggle bootstrap. The real job lives in the repo: training/kaggle/run_train.py. Add "--smoke" to ARGS for a quick test.
import subprocess
import sys

ARGS: list[str] = ["--variant", "n"]
subprocess.run(["git", "clone", "--depth", "1", "--branch", "v2", "https://github.com/Halok600/WeaponShield-AI.git", "/kaggle/working/src"], check=True)
subprocess.run([sys.executable, "/kaggle/working/src/training/kaggle/run_train.py", *ARGS], check=True)
```

`training/kaggle/train-n/kernel-metadata.json`:
```json
{
  "id": "<KAGGLE_USER>/weaponshield-train-n",
  "title": "weaponshield-train-n",
  "code_file": "bootstrap.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": ["<KAGGLE_USER>/weaponshield-dataset"]
}
```

`training/kaggle/train-s/bootstrap.py`:
```python
# Kaggle bootstrap. The real job lives in the repo: training/kaggle/run_train.py. Add "--smoke" to ARGS for a quick test.
import subprocess
import sys

ARGS: list[str] = ["--variant", "s"]
subprocess.run(["git", "clone", "--depth", "1", "--branch", "v2", "https://github.com/Halok600/WeaponShield-AI.git", "/kaggle/working/src"], check=True)
subprocess.run([sys.executable, "/kaggle/working/src/training/kaggle/run_train.py", *ARGS], check=True)
```

`training/kaggle/train-s/kernel-metadata.json`:
```json
{
  "id": "<KAGGLE_USER>/weaponshield-train-s",
  "title": "weaponshield-train-s",
  "code_file": "bootstrap.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": ["<KAGGLE_USER>/weaponshield-dataset"]
}
```

- [ ] **Step 5: Write `.github/workflows/training.yml`**

```yaml
name: training

on:
  push:
    branches: [main, v2]
    paths: ["training/**", ".github/workflows/training.yml"]
  pull_request:
    paths: ["training/**"]

jobs:
  training:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: training
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: training/requirements-dev.txt
      - run: pip install -r requirements-dev.txt
      - run: ruff check .
      - run: ruff format --check .
      - run: pytest -q
```

- [ ] **Step 6: Lint and test**

Run: `cd /d/Projects/WeaponShield-AI/training && .venv/Scripts/python -m pytest -q && .venv/Scripts/python -m ruff format . && .venv/Scripts/python -m ruff check .`
Expected: all pass, ruff clean (the kaggle scripts are linted too).

- [ ] **Step 7: Commit, check attribution, push**

```bash
cd /d/Projects/WeaponShield-AI
git add training/kaggle .github/workflows/training.yml
git commit -m "feat(training): Kaggle dataset/train jobs and training CI"
git log origin/v2..HEAD --format=%B | grep -i -E "co-authored|claude|generated with" || echo "attribution check: clean"
git push origin v2
```
Expected: `attribution check: clean`; push succeeds; the `training` workflow passes (`gh run list --repo Halok600/WeaponShield-AI --workflow training --limit 1`).

---

### Task 8: Run on Kaggle and record results (controller-run, long-running)

This task waits on remote jobs for hours. The controller runs it directly with background polling instead of dispatching an implementer; each step's command and expected output are below.

**Files:**
- Create: `docs/training-runs.md`
- Local only (git-ignored): `training/runs/kaggle-dataset/`, `training/runs/kaggle-n/`, `training/runs/kaggle-s/`

**Interfaces:**
- Consumes: Kaggle kernels from Task 7
- Produces: `training/runs/kaggle-{n,s}/ws-yolo26{n,s}.onnx` + `eval-yolo26{n,s}/metrics.json` for Plan 5; `docs/training-runs.md` with the numbers

- [ ] **Step 1: Smoke-run the dataset job on Kaggle**

Set `ARGS: list[str] = ["--smoke"]` in `training/kaggle/dataset/bootstrap.py` (local edit, do not commit), then:
```bash
cd /d/Projects/WeaponShield-AI/training
.venv/Scripts/kaggle kernels push -p kaggle/dataset
```
Poll with a background loop (every 60 s) on `.venv/Scripts/kaggle kernels status <KAGGLE_USER>/weaponshield-dataset` until the status is `complete`, `error` or `cancelAcknowledged`. On `error`, fetch the log with `.venv/Scripts/kaggle kernels output <KAGGLE_USER>/weaponshield-dataset -p runs/kaggle-dataset-smoke` and fix the cause in the repo (commit + push), then re-push. If the push itself is rejected because of `machine_shape`, delete that line from the three metadata files, commit, push, and re-push the kernel. If Kaggle reports that internet or GPU needs phone verification, stop and ask the user to verify their Kaggle account (one-time, free).
Expected: `complete`; the downloaded `stats.json` shows non-zero weapon instances from all three sources.

- [ ] **Step 2: Full dataset run**

Restore `ARGS: list[str] = []`, `git diff --exit-code training/kaggle` (must be clean), then push again and poll as in Step 1 (expected wall-clock 1–3 h). Download outputs:
```bash
.venv/Scripts/kaggle kernels output <KAGGLE_USER>/weaponshield-dataset -p runs/kaggle-dataset --file-pattern "stats.json|source-commit.txt"
```
(If this CLI version does not accept `--file-pattern`, drop the flag only after checking the output size with `.venv/Scripts/kaggle kernels output --help`; never download the multi-GB tar locally — read `stats.json` from the job log instead with `.venv/Scripts/kaggle kernels logs <KAGGLE_USER>/weaponshield-dataset`.)
Expected: `stats.json` with roughly 15k–25k images; the train split has `without_weapon` between 15% and 40% of images; knife instances ≥ 2,000. If knife instances are < 2,000 or negatives fall outside 15–40%, record the numbers and adjust `--oi-negatives` in `build.py`'s defaults (commit + push) before training.

- [ ] **Step 3: Train YOLO26n (≈4.5 h)**

```bash
.venv/Scripts/kaggle kernels push -p kaggle/train-n
```
Poll until terminal status, then:
```bash
.venv/Scripts/kaggle kernels output <KAGGLE_USER>/weaponshield-train-n -p runs/kaggle-n
.venv/Scripts/python -c "import json; m=json.load(open('runs/kaggle-n/eval-yolo26n/metrics.json')); print(json.dumps(m['test']['weapons'], indent=1)); print(m['test']['per_class']['knife']); print(m['hardneg'])"
```
Expected: the files `ws-yolo26n.onnx`, `ws-yolo26n.onnx.sha256` and `eval-yolo26n/metrics.json` are present.

- [ ] **Step 4: Train YOLO26s (≈10.5 h)**

Same as Step 3 with `train-s` / `kaggle-s` / `eval-yolo26s`. The two training kernels can run back to back; the weekly free GPU quota (30 h) covers dataset + n + s.

- [ ] **Step 5: Write `docs/training-runs.md`**

Use this structure, filling every cell from the two `metrics.json` files and `stats.json` (numbers rounded to 3 dp):

```markdown
# Training runs

Source commit: `<from source-commit.txt>` · Ultralytics `<metrics.ultralytics>` · Kaggle T4 GPUs

## Dataset (`stats.json`)
| Split | Images | With weapon | Without weapon | person | pistol | rifle | knife |
|---|---|---|---|---|---|---|---|
| train | … | … | … | … | … | … | … |
| val | … |
| test | … |

Dropped: <each reason: count>. Hard-negative set: <n> images.

## Held-out test results
| Model | Weapons mAP50 | Weapons mAP50-95 | Pistol mAP50 | Rifle mAP50 | Knife mAP50 | Person mAP50 | Hard-neg FP (balanced) |
|---|---|---|---|---|---|---|---|
| v1 YOLOv8s (for reference) | 0.833 | 0.613 | 0.908 | 0.954 | 0.636 | — | — |
| YOLO26n | … |
| YOLO26s | … |

v1's numbers came from a random split that likely mixed Roboflow augmented copies of the same photo across train and test, so they are not strictly comparable; v2 uses de-duplicated, group-aware splits.

## Thresholds (validation split)
<table of thresholds.low/balanced/high per class for the chosen model>
```

- [ ] **Step 6: Commit and push**

```bash
cd /d/Projects/WeaponShield-AI
git add docs/training-runs.md
git commit -m "docs: record v2 dataset and training results"
git log origin/v2..HEAD --format=%B | grep -i -E "co-authored|claude|generated with" || echo "attribution check: clean"
git push origin v2
```

---

## Notes for later plans

- Plan 5 picks YOLO26s or YOLO26n by measured browser FPS (spec §2.4: largest model with ≥ 15 FPS WebGPU and ≥ 5 FPS WASM on the RTX 3050 laptop), publishes the chosen `.onnx` as release `models-v1`, updates `web/models.json` + `ACTIVE_MODEL` (labels `{0: 'person', 1: 'pistol', 2: 'rifle', 3: 'knife'}`), converts `metrics.json.thresholds` into `PRESETS`, and feeds `metrics.json` + `gallery/` into the model-card page.
- Plan 4 (edge) reuses the same `.onnx`.
