# Kaggle bootstrap. The real job lives in the repo (see the run_*.py path below).
# Add "--smoke" to ARGS for a quick test.
import subprocess
import sys

ARGS: list[str] = []
subprocess.run(
    [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        "v2",
        "https://github.com/Halok600/WeaponShield-AI.git",
        "/kaggle/working/src",
    ],
    check=True,
)
subprocess.run([sys.executable, "/kaggle/working/src/training/kaggle/run_dataset.py", *ARGS], check=True)
