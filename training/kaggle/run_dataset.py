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


def _kaggle_secret(label: str, url_base: str | None = None) -> str | None:
    """`url_base` overrides KAGGLE_URL_BASE for this call only (the client reads it when constructed)."""
    from kaggle_secrets import UserSecretsClient

    saved = os.environ.get("KAGGLE_URL_BASE")
    if url_base:
        os.environ["KAGGLE_URL_BASE"] = url_base
    try:
        return UserSecretsClient().get_secret(label)
    finally:
        if saved is None:
            os.environ.pop("KAGGLE_URL_BASE", None)
        else:
            os.environ["KAGGLE_URL_BASE"] = saved


def roboflow_key() -> str | None:
    """Kaggle secret first, then an already-set env var. The value is never printed.

    Kaggle batch runs set KAGGLE_URL_BASE=https://api.kaggle.com, which answers HTTP 400 to every secret request
    (reported as a ConnectionError); the same request to https://www.kaggle.com works, so retry there.
    """
    key = None
    for url_base in (None, "https://www.kaggle.com"):
        try:
            key = _kaggle_secret("ROBOFLOW_API_KEY", url_base)
            break
        except Exception as exc:  # not on Kaggle, or the secret is not attached to this kernel
            where = url_base or os.environ.get("KAGGLE_URL_BASE") or "default URL"
            print(
                f"ROBOFLOW_API_KEY secret unavailable via {where}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
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
