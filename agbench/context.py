"""RunContext and MeasureCfg: what a workload function receives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class MeasureCfg:
    """Warm-up and measurement window (SPEC 5.5).

    Warm-up runs until it has done ``warmup_steps`` and ``warmup_min_s`` (capped by
    ``warmup_max_s``). Measurement runs until ``measure_steps`` and ``measure_min_s``
    (capped by ``measure_max_s``, but never fewer than min(5, measure_steps) steps).
    """

    warmup_steps: int = 5
    warmup_min_s: float = 5.0
    warmup_max_s: float = 60.0
    measure_steps: int = 20
    measure_min_s: float = 10.0
    measure_max_s: float = 60.0

    @staticmethod
    def tiny() -> "MeasureCfg":
        return MeasureCfg(warmup_steps=1, warmup_min_s=0.0, warmup_max_s=60.0,
                          measure_steps=3, measure_min_s=0.0, measure_max_s=60.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "MeasureCfg":
        keys = MeasureCfg.__dataclass_fields__.keys()
        return MeasureCfg(**{k: d[k] for k in keys if k in d})


def _default_log(msg: str) -> None:
    print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


@dataclass
class RunContext:
    scenario_id: str
    workload: str
    params: dict            # scenario params + {"micro_batch": int, "accum": int} for this attempt
    device: str             # "cuda" or "cpu"
    tiny: bool
    measure: MeasureCfg
    seed: int
    cache_dir: Path         # synthetic corpus root (AGBENCH_CACHE or <repo>/cache)
    keep_series: bool       # sustain only
    log: Callable[[str], None] = field(default=_default_log)
    # Additions beyond SPEC 12 (defaults keep the spec's constructor valid):
    gpu_index: int = 0              # physical (NVML / nvidia-smi) index of the GPU in use
    gpu_uuid: str | None = None     # NVML-style "GPU-..." uuid of the GPU in use, if known
    attempt: int = 1

    @property
    def micro_batch(self) -> int:
        return int(self.params.get("micro_batch", self.params.get("batch", 1)))

    @property
    def accum(self) -> int:
        return int(self.params.get("accum", 1))
