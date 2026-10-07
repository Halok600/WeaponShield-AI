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
    s = Sample(
        key="hf-1",
        source="roboflow-weapons",
        image_path=tmp_path / "a.jpg",
        width=10,
        height=10,
        boxes=[Box(0, 0, 0, 5, 5), Box(3, 0, 0, 5, 5)],
    )
    assert s.group == "hf-1"
    assert s.weapon_classes == {3}
