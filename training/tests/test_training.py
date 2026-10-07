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
