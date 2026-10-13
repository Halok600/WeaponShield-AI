"""Kaggle job: build the WeaponShield dataset and save it as one tar (Kaggle outputs prefer few large files).

Needs the Kaggle secret ROBOFLOW_API_KEY (or the same env var when run elsewhere). `--smoke` builds a tiny dataset
and falls back to `--no-roboflow` when no key is available.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1]
SRC = TRAINING.parent
WORK = Path("/kaggle/working")
DATASET = Path("/tmp/ws-dataset")


def run(*cmd: str, cwd: Path = TRAINING, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def roboflow_key() -> str | None:
    """Kaggle secret first, then an already-set env var. The value is never printed."""
    try:
        from kaggle_secrets import UserSecretsClient

        key = UserSecretsClient().get_secret("ROBOFLOW_API_KEY")
    except Exception:  # not on Kaggle, or the secret is not attached to this kernel
        key = None
    return key or os.environ.get("ROBOFLOW_API_KEY") or None


def main() -> None:
    smoke = "--smoke" in sys.argv
    key = roboflow_key()
    if not key and not smoke:
        sys.exit(
            "ROBOFLOW_API_KEY is not available: attach it as a Kaggle secret (Add-ons > Secrets) or set the env var"
        )
    env = {**os.environ, "ROBOFLOW_API_KEY": key} if key else dict(os.environ)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=SRC, capture_output=True, text=True, check=True
    ).stdout.strip()
    run(sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt")
    build = [sys.executable, "-m", "weaponshield_ml.build", "--out", str(DATASET), "--cache", "/tmp/ws-cache"]
    build += ["--device", "0"]
    if smoke:
        build.append("--smoke")
    if not key:
        build.append("--no-roboflow")
    run(*build, env=env)
    with tarfile.open(WORK / "ws-dataset.tar", "w") as tar:
        tar.add(DATASET, arcname="ws-dataset")
    shutil.copyfile(DATASET / "stats.json", WORK / "stats.json")
    (WORK / "source-commit.txt").write_text(commit + "\n")
    shutil.rmtree(SRC)  # keep the Kaggle output to the dataset artefacts only


if __name__ == "__main__":
    main()
