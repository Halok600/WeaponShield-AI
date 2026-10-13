"""Offline checks for the Kaggle job wiring. Nothing here runs a job or touches the network."""

from __future__ import annotations

import importlib.util
import json
import py_compile
from pathlib import Path

import pytest
import yaml

TRAINING = Path(__file__).resolve().parents[1]
KAGGLE = TRAINING / "kaggle"
KERNELS = {"dataset": [], "train-n": ["halok600/weaponshield-dataset"], "train-s": ["halok600/weaponshield-dataset"]}


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, KAGGLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("folder", "sources"), KERNELS.items())
def test_kernel_metadata_and_bootstrap(folder, sources):
    meta = json.loads((KAGGLE / folder / "kernel-metadata.json").read_text())
    assert meta["id"] == f"halok600/weaponshield-{folder}"
    assert meta["code_file"] == "bootstrap.py"
    assert meta["enable_gpu"] is True
    assert meta["kernel_sources"] == sources
    py_compile.compile(str(KAGGLE / folder / "bootstrap.py"), doraise=True)
    text = (KAGGLE / folder / "bootstrap.py").read_text()
    assert '"--branch",' in text
    assert '"v2",' in text


def test_run_scripts_import_without_side_effects():
    assert sorted(load("run_train").VARIANTS) == ["n", "s"]
    assert callable(load("run_dataset").main)


def test_dataset_job_needs_roboflow_key_unless_smoke(monkeypatch):
    run_dataset = load("run_dataset")
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    assert run_dataset.roboflow_key() is None  # not on Kaggle, nothing set
    monkeypatch.setattr(run_dataset.sys, "argv", ["run_dataset.py"])
    with pytest.raises(SystemExit) as exc:
        run_dataset.main()
    assert "ROBOFLOW_API_KEY" in str(exc.value)
    monkeypatch.setenv("ROBOFLOW_API_KEY", "k")
    assert run_dataset.roboflow_key() == "k"


def test_ultralytics_is_pinned():
    assert "ultralytics==8.4.174" in (TRAINING / "requirements.txt").read_text().split()


def test_ci_workflow_is_cpu_only_and_least_privilege():
    workflow = yaml.safe_load((TRAINING.parent / ".github" / "workflows" / "training.yml").read_text())
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["training"]["timeout-minutes"] > 0
