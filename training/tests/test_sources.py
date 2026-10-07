import sys
import types
from collections import Counter
from pathlib import Path

import pytest
import yaml
from PIL import Image

from weaponshield_ml.classes import DROP_IMAGE, KNIFE, PISTOL, RIFLE
from weaponshield_ml.sources.open_images import oi_detections_to_boxes
from weaponshield_ml.sources.roboflow import check_names, load_roboflow, rf_group, rf_labels_to_boxes

NAMES = ["weapons", "Aggressor", "Pistol", "Rifle", "Knife", "person"]


def test_rf_group_strips_roboflow_suffix_and_video_frame_marker():
    assert rf_group("WIN_20240726_12_38_04_Pro_mp4-0027_jpg.rf.0e0c67ad") == "rf-WIN_20240726_12_38_04_Pro"
    assert rf_group("clip_avi-12_png.rf.ABC123") == "rf-clip"
    assert rf_group("clip_mov-3_jpeg.rf.ff") == "rf-clip"


def test_rf_group_shares_a_group_between_augmented_copies():
    assert rf_group("gun12_jpg.rf.ab12") == rf_group("gun12_jpg.rf.cd34") == "rf-gun12"
    assert rf_group("plain") == "rf-plain"


def test_rf_labels_map_classes_and_discard_people():
    # YOLO lines: class cx cy w h (normalised). 200x100 image.
    lines = ["2 0.5 0.5 0.5 0.5", "3 0.25 0.5 0.5 1.0", "4 0.5 0.5 0.1 0.2", "5 0.5 0.5 0.4 0.4", ""]
    boxes = rf_labels_to_boxes(lines, NAMES, 200, 100, Counter())
    assert isinstance(boxes, list)
    assert [(b.cls, round(b.x1), round(b.y1), round(b.x2), round(b.y2)) for b in boxes] == [
        (PISTOL, 50, 25, 150, 75),
        (RIFLE, 0, 0, 100, 100),
        (KNIFE, 90, 40, 110, 60),
    ]


def test_rf_generic_label_drops_the_image():
    assert rf_labels_to_boxes(["2 0.5 0.5 0.1 0.1", "0 0.5 0.5 0.1 0.1"], NAMES, 10, 10, Counter()) == DROP_IMAGE


def test_rf_rejects_polygon_exports():
    with pytest.raises(ValueError, match="polygon"):
        rf_labels_to_boxes(["2 0.1 0.1 0.2 0.1 0.2 0.2"], NAMES, 10, 10, Counter())


def test_rf_unmapped_label_is_a_hard_error():
    with pytest.raises(KeyError, match="Revolver"):
        rf_labels_to_boxes(["0 0.5 0.5 0.1 0.1"], ["Revolver"], 10, 10, Counter())


def test_check_names_lists_every_unknown_name():
    check_names(NAMES)
    with pytest.raises(KeyError) as e:
        check_names(["Pistol", "Zeta", "Alpha"])
    assert "['Alpha', 'Zeta']" in str(e.value)


def test_open_images_relative_boxes():
    boxes = oi_detections_to_boxes(
        [
            ("Handgun", [0.1, 0.2, 0.3, 0.4]),
            ("Shotgun", [0, 0, 1, 1]),
            ("Dagger", [0.5, 0.5, 0.1, 0.1]),
            ("Man", [0, 0, 1, 1]),
        ],
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


def test_load_roboflow_reads_an_existing_export_offline(tmp_path):
    root = tmp_path / "roboflow"
    root.mkdir()
    (root / "data.yaml").write_text(yaml.safe_dump({"names": NAMES}), encoding="utf-8")
    long_stem = "x" * 200
    cases = {
        ("train", "a_jpg.rf.01"): "2 0.5 0.5 0.5 0.5\n5 0.5 0.5 0.2 0.2\n",
        ("train", long_stem): "4 0.5 0.5 0.5 0.5\n",
        ("valid", "generic"): "0 0.5 0.5 0.5 0.5\n",
        ("valid", "people_only"): "5 0.5 0.5 0.5 0.5\n",
        ("test", "unlabelled"): None,
    }
    for (split, stem), label in cases.items():
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 20)).save(root / split / "images" / f"{stem}.jpg")
        if label is not None:
            (root / split / "labels" / f"{stem}.txt").write_text(label, encoding="utf-8")
    dropped = Counter()
    samples = list(load_roboflow(tmp_path, dropped, api_key="unused"))
    assert [s.source for s in samples] == ["roboflow-weapons"] * 2
    by_key = {s.key: s for s in samples}
    a = by_key["rf-train-a_jpg.rf.01"]
    assert a.group == "rf-a" and (a.width, a.height) == (40, 20)
    assert [b.cls for b in a.boxes] == [PISTOL]
    long_key = next(k for k in by_key if k.startswith("rf-train-xxxx"))
    assert len(long_key) < 140 and long_key != "rf-train-" + long_stem
    assert dropped == {"roboflow_ambiguous_label": 1, "roboflow_no_weapon_boxes": 2}


def test_rf_rejects_out_of_range_class_index():
    with pytest.raises(ValueError, match="class index 9 out of range for 6 names"):
        rf_labels_to_boxes(["9 0.5 0.5 0.1 0.1"], NAMES, 10, 10, Counter())
    with pytest.raises(ValueError, match="class index -1 out of range"):
        rf_labels_to_boxes(["-1 0.5 0.5 0.1 0.1"], NAMES, 10, 10, Counter())


def _fake_roboflow(download):
    class _Version:
        def download(self, *args, **kwargs):
            return download(*args, **kwargs)

    class _Project:
        def version(self, number):
            return _Version()

    class _Workspace:
        def project(self, name):
            return _Project()

    class Roboflow:
        def __init__(self, api_key):
            self.api_key = api_key

        def workspace(self, name):
            return _Workspace()

    return types.SimpleNamespace(Roboflow=Roboflow)


def test_download_error_never_leaks_the_api_key(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise ConnectionError("GET https://api.roboflow.com/x?api_key=SECRET123 failed")

    monkeypatch.setitem(sys.modules, "roboflow", _fake_roboflow(boom))
    with pytest.raises(RuntimeError) as e:
        list(load_roboflow(tmp_path, Counter(), api_key="SECRET123"))
    assert "***" in str(e.value) and "SECRET123" not in str(e.value)
    assert e.value.__cause__ is None and e.value.__suppress_context__


def test_download_overwrites_a_partial_cache_dir(tmp_path, monkeypatch):
    (tmp_path / "roboflow").mkdir()  # interrupted earlier run: directory exists, no data.yaml
    seen = {}

    def download(fmt, location, overwrite):
        seen.update(fmt=fmt, overwrite=overwrite)
        Path(location, "data.yaml").write_text(yaml.safe_dump({"names": NAMES}), encoding="utf-8")

    monkeypatch.setitem(sys.modules, "roboflow", _fake_roboflow(download))
    assert list(load_roboflow(tmp_path, Counter(), api_key="k")) == []
    assert seen == {"fmt": "yolov8", "overwrite": True}


def test_download_that_produces_no_data_yaml_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "roboflow", _fake_roboflow(lambda *a, **k: None))
    with pytest.raises(RuntimeError, match="did not produce data.yaml"):
        list(load_roboflow(tmp_path, Counter(), api_key="k"))
