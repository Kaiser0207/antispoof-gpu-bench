"""Shared test helpers. All tests are CPU-only, offline and use tiny random models (SPEC 13.1)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

# Test hooks and settings that must never leak from the developer's shell into a test run.
_SCRUB = ("AGBENCH_SIMULATE_OOM_MB_GE", "AGBENCH_SIMULATE_FAULT", "AGBENCH_REGISTRY_OVERRIDE",
          "AGBENCH_RESULTS", "AGBENCH_CACHE", "AGBENCH_MODELS_DIR", "PYTORCH_CUDA_ALLOC_CONF")


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


def child_env(tmp_path: Path, extra: dict | None = None) -> dict:
    """Environment for `python -m agbench ...` subprocesses: offline, CPU only, results under tmp_path."""
    env = {k: v for k, v in os.environ.items() if k not in _SCRUB}
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update({
        "AGBENCH_RESULTS": str(tmp_path / "results"),
        "AGBENCH_CACHE": str(tmp_path / "cache"),
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
        "CUDA_VISIBLE_DEVICES": "",   # tests never touch a GPU
    })
    env.update(extra or {})
    return env


@pytest.fixture
def run_agbench(tmp_path):
    """Run `python -m agbench <args>` in a fresh process (cwd = tmp_path) and return CompletedProcess."""

    def _run(*args, env: dict | None = None, timeout: float = 600.0) -> subprocess.CompletedProcess:
        cmd = [sys.executable, "-m", "agbench", *[str(a) for a in args]]
        return subprocess.run(cmd, cwd=tmp_path, env=child_env(tmp_path, env), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)

    return _run


def scenario_records(path: Path) -> list[dict]:
    from agbench.schema import load_records

    return [r for r in load_records(path) if r.get("kind") == "scenario"]


def by_id(records: list[dict]) -> dict[str, dict]:
    """Last record per scenario id."""
    out: dict[str, dict] = {}
    for r in records:
        out[r["scenario_id"]] = r
    return out


def explain(proc: subprocess.CompletedProcess, out: Path | None = None) -> str:
    """Failure message: console output plus the results file, if any."""
    text = f"exit code {proc.returncode}\n--- stdout ---\n{proc.stdout[-6000:]}\n--- stderr ---\n{proc.stderr[-6000:]}"
    if out is not None and Path(out).exists():
        from agbench.schema import load_records

        text += "\n--- results ---\n" + "\n".join(
            json.dumps({k: r.get(k) for k in ("kind", "scenario_id", "status", "status_detail", "error")})
            for r in load_records(out))
    return text
