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
