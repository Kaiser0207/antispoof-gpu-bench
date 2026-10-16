"""Workload key -> "module:function" (imported lazily, only inside child processes)."""

from __future__ import annotations

import importlib
import os
from typing import Callable

# key: (target, workload_version). Bump the version whenever what is measured changes.
WORKLOADS: dict[str, tuple[str, int]] = {
    "train": ("agbench.workloads.train:run", 1),
    "frozen": ("agbench.workloads.frozen:run", 1),
    "infer": ("agbench.workloads.infer:run", 1),
    "e2e": ("agbench.workloads.e2e:run", 2),  # 2: clips_per_s = wall rate; reference numerics
    "sustain": ("agbench.workloads.sustain:run", 1),
    "basics.matmul": ("agbench.workloads.basics:run_matmul", 1),
    "basics.membw": ("agbench.workloads.basics:run_membw", 1),
    "basics.pcie": ("agbench.workloads.basics:run_pcie", 1),
    "cpu.decode": ("agbench.workloads.cpu_pipeline:run_decode", 1),
    "cpu.rawboost": ("agbench.workloads.cpu_pipeline:run_rawboost", 1),
    "cpu.loader": ("agbench.workloads.cpu_pipeline:run_loader", 2),  # 2: steady-state window
}

# Test hook: AGBENCH_REGISTRY_OVERRIDE="train=pkg.mod:fn,cpu.decode=pkg.mod:fn2" replaces targets
# (used only by the harness's own tests to stub workloads through real subprocesses).
_OVERRIDE_ENV = "AGBENCH_REGISTRY_OVERRIDE"


def workload_version(key: str) -> int:
    return WORKLOADS[key][1]


def target(key: str) -> str:
    override = os.environ.get(_OVERRIDE_ENV, "")
    for item in filter(None, (p.strip() for p in override.split(","))):
        k, _, v = item.partition("=")
        if k.strip() == key and v.strip():
            return v.strip()
    return WORKLOADS[key][0]


def resolve(key: str) -> Callable:
    """Import and return the workload function for ``key``."""
    mod_name, _, fn_name = target(key).partition(":")
    module = importlib.import_module(mod_name)
    return getattr(module, fn_name)
