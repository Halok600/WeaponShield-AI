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
    def __init__(self, path_sizes=None):
        self.calls = 0
        self.path_sizes = path_sizes or {}

    def __call__(self, paths):
        self.calls += 1
        return [([Box(PERSON, 1, 1, 20, 40)], self.path_sizes.get(p, (48, 64))) for p in paths]


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
    weapons = make_samples(src, 40, "roboflow-weapons", PISTOL)
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
    assert "path" not in cfg
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
    assert stats["dropped"] == {
        "hf_ambiguous_label": 3,
        "exact_duplicates": 1,
        "size_mismatch": 0,
        "hardneg_overlaps_train": 0,
    }
    assert stats["sources"] == {"roboflow-weapons": 40, "oi-weapons": 20, "oi-negatives": 20}
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


def test_hardneg_leakage(tmp_path):
    """Hard negatives that duplicate training images are excluded."""
    src = tmp_path / "src"
    src.mkdir()
    weapons = make_samples(src, 10, "roboflow-weapons", PISTOL)
    # A hard negative that reuses the same image file as a training sample
    hardneg = [Sample(key="hn-dup", source="oi-neg", image_path=weapons[0].image_path, width=64, height=48)]
    out = tmp_path / "ds"
    labeler = FakeLabeler()

    # Suppress stdout
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        stats = build(out, [weapons], hardneg, labeler, dropped={})
    finally:
        sys.stdout = old_stdout

    # The duplicate hard negative should be excluded
    assert len(list((out / "hardneg" / "images").iterdir())) == 0
    assert stats["dropped"]["hardneg_overlaps_train"] == 1
    assert stats["hardneg_images"] == 0


def test_duplicate_key_guard(tmp_path):
    """Duplicate keys across splits and hardneg raise ValueError."""
    src = tmp_path / "src"
    src.mkdir()
    samples = make_samples(src, 3, "roboflow-weapons", PISTOL)
    # Create a hard negative with a key that might collide (though unlikely with make_samples)
    # Instead, we'll use the build.py write_dataset directly to test
    from weaponshield_ml.writer import write_dataset

    out = tmp_path / "ds"
    splits = {"train": samples[:2], "val": samples[2:]}
    hardneg = [Sample(key=samples[0].key, source="oi-neg", image_path=samples[0].image_path, width=64, height=48)]
    with pytest.raises(ValueError, match="duplicate sample key"):
        write_dataset(out, splits, hardneg, {})


def test_size_mismatch_filtering(tmp_path):
    """Samples with mismatched image sizes are excluded and counted."""
    src = tmp_path / "src"
    src.mkdir()
    samples = make_samples(src, 5, "roboflow-weapons", PISTOL)

    # FakeLabeler that reports wrong height for one sample
    path_sizes = {s.image_path: (48, 64) for s in samples}
    path_sizes[samples[0].image_path] = (100, 100)  # Wrong size for first sample
    labeler = FakeLabeler(path_sizes=path_sizes)

    hardneg = make_samples(src, 2, "oi-neg", None)
    out = tmp_path / "ds"

    # Suppress stdout during build
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        stats = build(out, [samples], hardneg, labeler, dropped={})
    finally:
        sys.stdout = old_stdout

    # The first sample should be excluded due to size mismatch
    assert sum(v["images"] for v in stats["splits"].values()) == 4  # 5 - 1 mismatch
    assert stats["dropped"]["size_mismatch"] == 1


def test_degenerate_box_filtering(tmp_path):
    """Boxes that become < 1 px after clipping are filtered from stats and written labels."""
    src = tmp_path / "src"
    src.mkdir()
    p = src / "img.png"
    Image.fromarray(np.ones((48, 64, 3), dtype=np.uint8)).save(p)

    # Sample with a degenerate weapon box (clipped to < 1 px width)
    samples = [
        Sample(
            key="degenerative",
            source="test",
            image_path=p,
            width=64,
            height=48,
            boxes=[
                Box(PISTOL, 0, 0, 0.5, 10),  # < 1 px width after clipping
                Box(PERSON, 10, 10, 30, 30),  # valid
            ],
        )
    ]

    labeler = FakeLabeler()
    hardneg = []
    out = tmp_path / "ds"

    # Suppress stdout
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        stats = build(out, [samples], hardneg, labeler, dropped={})
    finally:
        sys.stdout = old_stdout

    # The sample should be counted as without_weapon because the degenerate box is filtered
    assert stats["splits"]["train"]["images"] == 1
    assert stats["splits"]["train"]["with_weapon"] == 0
    assert stats["splits"]["train"]["without_weapon"] == 1
    # Verify the written label file only has the PERSON box
    labels = read_labels(out / "labels" / "train" / "degenerative.txt", 64, 48)
    assert len(labels) == 1
    assert labels[0].cls == PERSON
