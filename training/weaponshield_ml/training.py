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
