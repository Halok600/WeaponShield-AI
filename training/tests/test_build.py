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
