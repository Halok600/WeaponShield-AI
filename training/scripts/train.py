"""Fine-tune YOLO26 on the WeaponShield dataset.

python scripts/train.py --data-root datasets/ws --model yolo26s.pt --out runs --name ws-yolo26s --hours 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weaponshield_ml.training import (  # noqa: E402
    cctv_augmentations,
    checkpoint_is_fresh,
    resolve_device,
    write_data_yaml,
)


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
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="data-loader workers PER GPU process; on Kaggle (4 vCPUs, 2 GPUs) use 2-3",
    )
    p.add_argument("--resume", action="store_true", help="continue from <out>/<name>/weights/last.pt")
    args = p.parse_args()

    from ultralytics import YOLO

    device = resolve_device(args.device)
    mode = f"time budget {args.hours} h" if args.hours else f"{args.epochs} epochs"
    print(f"mode: {mode}{' (resume)' if args.resume else ''}; device: {device}", flush=True)

    run_dir = args.out.resolve() / args.name
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    time_kw = {"time": args.hours} if args.hours else {}
    started = time.time()

    if args.resume:
        if not last.exists():
            raise SystemExit(f"--resume: no checkpoint at {last}")
        import torch

        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        if ckpt.get("optimizer") is None or not ckpt.get("train_args"):
            raise SystemExit(f"--resume: {last} is a finished run (optimizer state stripped); nothing to resume")
        del ckpt
        # Ultralytics restores the saved run args; only these overrides are honoured on resume.
        YOLO(str(last)).train(
            resume=True,
            device=device,
            batch=args.batch,
            workers=args.workers,
            augmentations=cctv_augmentations(),
            **time_kw,
        )
    else:
        data_yaml = write_data_yaml(args.data_root, args.out / f"{args.name}-data.yaml")
        YOLO(args.model).train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
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
            # time-budgeted runs favour throughput; reproducibility = fixed seed + stored config
            deterministic=False,
            **time_kw,
        )

    if not checkpoint_is_fresh(best, started):
        if args.resume and best.exists() and checkpoint_is_fresh(last, started):
            print(f"warning: no new best.pt this session (no improvement); reporting {best}", file=sys.stderr)
        else:
            raise SystemExit(f"Training did not produce a fresh {best} (missing or older than this run)")
    print(best)


if __name__ == "__main__":
    main()
