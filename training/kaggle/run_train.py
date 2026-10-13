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

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=SRC, capture_output=True, text=True, check=True
    ).stdout.strip()
    run(sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt")

    tar_path = next(Path("/kaggle/input").rglob("ws-dataset.tar"))
    safe = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
    with tarfile.open(tar_path) as tar:
        tar.extractall(DATA_PARENT, **safe)
    data_root = DATA_PARENT / "ws-dataset"

    budget = ["--epochs", "1"] if args.smoke else ["--hours", v["hours"]]
    run(
        sys.executable,
        "scripts/train.py",
        "--data-root",
        str(data_root),
        "--model",
        v["model"],
        "--out",
        str(WORK / "runs"),
        "--name",
        name,
        "--batch",
        v["batch"],
        "--device",
        "auto",
        "--workers",
        "3",  # Kaggle has few CPU cores
        *budget,
    )
    best = WORK / "runs" / name / "weights" / "best.pt"
    run(
        sys.executable,
        "scripts/evaluate.py",
        "--weights",
        str(best),
        "--data-root",
        str(data_root),
        "--out",
        str(WORK / f"eval-yolo26{args.variant}"),
    )
    onnx = WORK / f"{name}.onnx"
    run(sys.executable, "scripts/export_onnx.py", "--weights", str(best), "--out", str(onnx))

    shutil.copyfile(best, WORK / f"{name}.pt")
    (WORK / f"{name}.onnx.sha256").write_text(hashlib.sha256(onnx.read_bytes()).hexdigest() + "\n")
    (WORK / "source-commit.txt").write_text(commit + "\n")
    (WORK / "runs" / name / "weights" / "last.pt").unlink(missing_ok=True)
    shutil.rmtree(SRC)


if __name__ == "__main__":
    main()
