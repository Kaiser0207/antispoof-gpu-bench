"""The scenario matrix (SPEC 6) as data, suite order, tiny-mode transform and scenario hashes.

Parameter keys that workloads read from ``ctx.params`` (all scenarios add ``micro_batch`` and
``accum`` for the attempt; model scenarios only):

* train / sustain: model, mode, seconds, batch, precision, head, grad_ckpt, deterministic,
  lr_encoder, lr_head, weight_decay, grad_clip, layerdrop, seed; the reference-numerics rows
  (``.det`` and the two plain ``e2e`` rows) add conv_tf32=False and flash_sdp
  (+ encoders for fusion; sustain adds duration_s, window_s; the .repeat row adds repeat_of)
* frozen: model, mode, seconds, batch, precision, head ("probe_set"), head_sets, lr_head,
  weight_decay, seed
* infer: model, mode, seconds, batch, precision, head, seed (+ encoders for fusion)
* e2e: the train keys + num_workers (int; "auto" in the matrix is resolved by the child),
  rawboost_p, pin_memory, persistent_workers, blas
* basics.matmul: sizes, dtypes, n_batches, batch_min_s; basics.membw: buffer_mib, ops;
  basics.pcie: buffer_mib, modes
* cpu.decode: seconds, processes; cpu.rawboost: algo, p, seconds, blas;
  cpu.loader: num_workers, batch, seconds, rawboost_p, algo, blas, pin_memory, persistent_workers,
  warmup_batches, measure_batches, cap_s
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace

from . import constants as C
from .context import MeasureCfg
from .registry import workload_version


@dataclass(frozen=True)
class Scenario:
    id: str
    category: str
    workload: str
    params: dict
    suites: tuple[str, ...]
    needs_gpu: bool
    models: tuple[str, ...]
    ladder: tuple[tuple[int, int], ...] | None
    est_s: float
    env: dict[str, str | None] = field(default_factory=dict)   # None = remove from the child env
    # Additions beyond SPEC 12 (with defaults):
    measure: MeasureCfg | None = None        # None = MeasureCfg() (or MeasureCfg.tiny() in tiny mode)
    mem_est_gib: float | None = None         # SPEC 6 memory column (informative, for `list`)

    def has_micro_batch(self) -> bool:
        return self.category in ("train", "frozen", "infer", "e2e", "sustain")


# ---------------------------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------------------------

_TRAIN_RECIPE = {"lr_encoder": 1e-5, "lr_head": 1e-4, "weight_decay": 0.01, "grad_clip": 1.0,
                 "layerdrop": 0.0, "seed": C.DEFAULT_SEED}
_BLAS1 = {"OPENBLAS_NUM_THREADS": "1"}
_BLAS_DEFAULT = {k: None for k in C.BLAS_THREAD_VARS}
_DET_ENV = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
# Numerics of our reference training runs (references/reference_gpus.json): strict deterministic
# algorithms, TF32 off for matmuls and cuDNN convolutions, and flash attention off for clips longer
# than 4 s. Used by the scenarios whose rate is compared with the training logs, and by ``.det``.
_REF_NUMERICS = {"deterministic": True, "conv_tf32": False}

_ALL: list[Scenario] = []


def _models_for(model: str) -> tuple[str, ...]:
    return ("xlsr", "wavlmb") if model == "fusion" else (model,)


def _ladder_for(batch: int, ladder: str | None):
    if ladder == "L16":
        return C.LADDER_16
    if ladder == "L8":
        return C.LADDER_8
    return None


def _add(s: Scenario) -> None:
    _ALL.append(s)


def _q(qf: str) -> tuple[str, ...]:
    return ("quick", "full") if qf == "QF" else ("full",)


def _train(model, seconds, batch, precision, qf, est, mem, *, head="pyramid", variant="",
           ladder="L16", grad_ckpt=False, ref_numerics=False, mode="train", extra=None, env=None):
    sid = f"{mode}.{model}.{seconds:g}s.bs{batch}.{precision}" + (f".{variant}" if variant else "")
    params = {"model": model, "mode": mode, "seconds": float(seconds), "batch": batch,
              "precision": precision, "head": "fusion" if model == "fusion" else head,
              "grad_ckpt": grad_ckpt, "deterministic": False, **_TRAIN_RECIPE}
    if ref_numerics:
        params.update(_REF_NUMERICS, flash_sdp=not float(seconds) > 4.0)
        env = {**_DET_ENV, **(env or {})}
    if model == "fusion":
        params["encoders"] = ["xlsr", "wavlmb"]
    params.update(extra or {})
    _add(Scenario(id=sid, category=mode, workload=mode, params=params, suites=_q(qf), needs_gpu=True,
                  models=_models_for(model), ladder=_ladder_for(batch, ladder), est_s=float(est),
                  env=dict(env or {}), mem_est_gib=mem))


def _frozen(model, qf, est, mem, *, head_sets=1, variant=""):
    sid = f"frozen.{model}.6s.bs16.bf16" + (f".{variant}" if variant else "")
    params = {"model": model, "mode": "frozen", "seconds": 6.0, "batch": 16, "precision": "bf16",
              "head": "probe_set", "head_sets": head_sets, "lr_head": 1e-3, "weight_decay": 0.01,
              "seed": C.DEFAULT_SEED}
    _add(Scenario(id=sid, category="frozen", workload="frozen", params=params, suites=_q(qf),
                  needs_gpu=True, models=_models_for(model), ladder=C.LADDER_16, est_s=float(est),
                  mem_est_gib=mem))


def _infer(model, seconds, batch, qf, est, mem, ladder="L16"):
    sid = f"infer.{model}.{seconds:g}s.bs{batch}.bf16"
    params = {"model": model, "mode": "infer", "seconds": float(seconds), "batch": batch,
              "precision": "bf16", "head": "fusion" if model == "fusion" else "pyramid",
              "seed": C.DEFAULT_SEED}
    if model == "fusion":
        params["encoders"] = ["xlsr", "wavlmb"]
    _add(Scenario(id=sid, category="infer", workload="infer", params=params, suites=_q(qf),
                  needs_gpu=True, models=_models_for(model), ladder=_ladder_for(batch, ladder),
                  est_s=float(est), mem_est_gib=mem))


def _cpu_loader(workers: int, blas: str, qf: str):
    sid = f"cpu.loader.w{workers}.blas{blas}"
    params = {"num_workers": workers, "batch": 16, "seconds": 6.0, "rawboost_p": 0.5, "algo": 4,
              "blas": blas, "pin_memory": False, "persistent_workers": False, "warmup_batches": 2,
              "measure_batches": 30, "cap_s": 60.0}
    measure = MeasureCfg(warmup_steps=2, warmup_min_s=0.0, warmup_max_s=60.0,
                         measure_steps=30, measure_min_s=0.0, measure_max_s=60.0)
    _add(Scenario(id=sid, category="cpu", workload="cpu.loader", params=params, suites=_q(qf),
                  needs_gpu=False, models=(), ladder=None, est_s=25.0,
                  env=dict(_BLAS1 if blas == "1" else _BLAS_DEFAULT), measure=measure))


# --- 6.1 GPU basics ---------------------------------------------------------------------------
_add(Scenario("gpu.matmul", "basics", "basics.matmul",
              {"sizes": [4096, 8192], "dtypes": ["bf16", "fp16", "tf32", "fp32"], "n_batches": 10,
               "batch_min_s": 0.2}, ("quick", "full"), True, (), None, 25.0, mem_est_gib=0.8))
_add(Scenario("gpu.membw", "basics", "basics.membw",
              {"buffer_mib": 1024, "ops": ["copy", "read", "triad"]},
              ("quick", "full"), True, (), None, 12.0, mem_est_gib=3.0))
_add(Scenario("gpu.pcie", "basics", "basics.pcie",
              {"buffer_mib": 256, "modes": ["h2d_pinned", "d2h_pinned", "h2d_pageable"]},
              ("quick", "full"), True, (), None, 12.0, mem_est_gib=0.6))

# --- 6.2 model scenarios ----------------------------------------------------------------------
_train("w2v2b", 6, 16, "bf16", "QF", 30, 5.3)                                  # anchor
_train("w2v2b", 10, 16, "bf16", "QF", 30, 7.8)
_train("w2v2b", 2, 16, "bf16", "F", 30, 2.7)
_train("w2v2b", 4, 16, "bf16", "F", 30, 4.0)
_train("w2v2b", 8, 16, "bf16", "F", 30, 6.5)
_train("w2v2b", 6, 16, "bf16", "F", 30, 5.3, head="gmean", variant="gmean")
_train("w2v2b", 6, 16, "tf32", "F", 31, 8.5)
_train("w2v2b", 6, 16, "fp16", "F", 30, 5.3)
_train("w2v2b", 6, 16, "bf16", "F", 32, 5.3, variant="det", ref_numerics=True)
_train("w2v2b", 6, 16, "bf16", "F", 35, 2.5, variant="ckpt", grad_ckpt=True)
_train("w2v2b", 6, 8, "bf16", "F", 30, 3.3, ladder="L8")
_train("w2v2b", 6, 32, "bf16", "F", 31, 9.1, ladder=None)
_train("xlsr", 6, 16, "bf16", "QF", 57, 14.5)
_train("xlsr", 6, 16, "tf32", "F", 94, 22.0)
_train("xlsr", 6, 16, "bf16", "F", 44, 7.0, variant="ckpt", grad_ckpt=True)
_train("w2v2l", 6, 16, "bf16", "QF", 56, 12.5)
_train("wavlmb", 6, 16, "bf16", "F", 30, 6.5)
_train("fusion", 4, 16, "bf16", "QF", 59, 13.6)
_frozen("xlsr", "QF", 33, 3.9)
_frozen("xlsr", "F", 33, 4.2, head_sets=10, variant="h40")
_frozen("w2v2b", "F", 30, 1.6)
_frozen("w2v2l", "F", 33, 3.9)
_infer("w2v2b", 6, 16, "QF", 30, 1.0)
_infer("xlsr", 6, 16, "QF", 33, 2.2)
_infer("w2v2b", 6, 4, "F", 30, 0.6, ladder=None)
_infer("xlsr", 6, 4, "F", 33, 1.5, ladder=None)
_infer("w2v2l", 6, 16, "F", 33, 2.2)
_infer("wavlmb", 6, 16, "F", 30, 1.1)
_infer("fusion", 4, 16, "F", 35, 2.6)
_train("w2v2b", 6, 16, "bf16", "F", 320, 5.3, mode="sustain",
       extra={"duration_s": 300.0, "window_s": 30.0})
_train("w2v2b", 6, 16, "bf16", "F", 30, 5.3, variant="repeat",
       extra={"repeat_of": "train.w2v2b.6s.bs16.bf16"})

# --- 6.1 end-to-end ---------------------------------------------------------------------------
# The two plain e2e rows are the ones compared with our training logs: same loader setup (2 spawn
# workers, FLAC decode + crop, no augmentation) and the same numerics (_REF_NUMERICS).
_E2E_REF = {"num_workers": 2, "rawboost_p": 0.0, "pin_memory": False, "persistent_workers": False,
            "blas": "inherit"}
_train("w2v2b", 6, 16, "bf16", "QF", 36, 5.3, mode="e2e", extra=_E2E_REF, ref_numerics=True)
_train("w2v2b", 6, 16, "bf16", "F", 40, 5.3, mode="e2e", variant="rawboost",
       extra={"num_workers": "auto", "rawboost_p": 0.5, "pin_memory": True,
              "persistent_workers": True, "blas": "1"}, env=_BLAS1)
_train("xlsr", 6, 16, "bf16", "F", 64, 14.5, mode="e2e", extra=_E2E_REF, ref_numerics=True)

# --- 6.1 CPU pipeline -------------------------------------------------------------------------
_add(Scenario("cpu.decode", "cpu", "cpu.decode", {"seconds": 6.0, "processes": 1},
              ("quick", "full"), False, (), None, 15.0))
_add(Scenario("cpu.rawboost.blas1", "cpu", "cpu.rawboost",
              {"algo": 4, "p": 1.0, "seconds": 6.0, "blas": "1"},
              ("quick", "full"), False, (), None, 20.0, env=dict(_BLAS1)))
_add(Scenario("cpu.rawboost.blasdefault", "cpu", "cpu.rawboost",
              {"algo": 4, "p": 1.0, "seconds": 6.0, "blas": "default"},
              ("quick", "full"), False, (), None, 20.0, env=dict(_BLAS_DEFAULT)))
_cpu_loader(4, "1", "QF")
_cpu_loader(8, "1", "QF")
_cpu_loader(4, "default", "QF")
for _w in (1, 2, 12, 16):
    _cpu_loader(_w, "1", "F")
_cpu_loader(1, "default", "F")
_cpu_loader(8, "default", "F")

SCENARIOS: dict[str, Scenario] = {s.id: s for s in _ALL}
assert len(SCENARIOS) == len(_ALL), "duplicate scenario ids"

# ---------------------------------------------------------------------------------------------
# Suites (SPEC 6.3)
# ---------------------------------------------------------------------------------------------

_QUICK = [
    "gpu.matmul", "gpu.membw", "gpu.pcie",
    "train.w2v2b.6s.bs16.bf16", "infer.w2v2b.6s.bs16.bf16",
    "train.xlsr.6s.bs16.bf16", "infer.xlsr.6s.bs16.bf16",
    "train.w2v2l.6s.bs16.bf16",
    "train.fusion.4s.bs16.bf16",
    "frozen.xlsr.6s.bs16.bf16",
    "train.w2v2b.10s.bs16.bf16",
    "e2e.w2v2b.6s.bs16.bf16",
    "cpu.decode", "cpu.rawboost.blas1", "cpu.rawboost.blasdefault", "cpu.loader.w4.blas1",
    "cpu.loader.w8.blas1", "cpu.loader.w4.blasdefault",
]
_FULL_REST = [
    # base variants
    "train.w2v2b.2s.bs16.bf16", "train.w2v2b.4s.bs16.bf16", "train.w2v2b.8s.bs16.bf16",
    "train.w2v2b.6s.bs16.bf16.gmean", "train.w2v2b.6s.bs16.tf32", "train.w2v2b.6s.bs16.fp16",
    "train.w2v2b.6s.bs16.bf16.det", "train.w2v2b.6s.bs16.bf16.ckpt", "train.w2v2b.6s.bs8.bf16",
    "train.w2v2b.6s.bs32.bf16",
    # xlsr variants
    "train.xlsr.6s.bs16.tf32", "train.xlsr.6s.bs16.bf16.ckpt",
    # wavlmb
    "train.wavlmb.6s.bs16.bf16",
    # frozen
    "frozen.xlsr.6s.bs16.bf16.h40", "frozen.w2v2b.6s.bs16.bf16", "frozen.w2v2l.6s.bs16.bf16",
    # infer
    "infer.w2v2b.6s.bs4.bf16", "infer.xlsr.6s.bs4.bf16", "infer.w2v2l.6s.bs16.bf16",
    "infer.wavlmb.6s.bs16.bf16", "infer.fusion.4s.bs16.bf16",
    # e2e
    "e2e.w2v2b.6s.bs16.bf16.rawboost", "e2e.xlsr.6s.bs16.bf16",
    # cpu
    "cpu.loader.w1.blas1", "cpu.loader.w2.blas1", "cpu.loader.w12.blas1", "cpu.loader.w16.blas1",
    "cpu.loader.w1.blasdefault", "cpu.loader.w8.blasdefault",
    # thermal soak, then the anchor again
    "sustain.w2v2b.6s.bs16.bf16",
    "train.w2v2b.6s.bs16.bf16.repeat",
]
SUITES: dict[str, list[str]] = {"quick": list(_QUICK), "full": _QUICK + _FULL_REST}


def suite_est_s(suite: str) -> float:
    return sum(SCENARIOS[i].est_s for i in SUITES[suite])


# ---------------------------------------------------------------------------------------------
# Effective (per-run) scenario: tiny mode and CPU device transforms (SPEC 12 "Tiny mode contract")
# ---------------------------------------------------------------------------------------------


SMOKE_BASICS = {
    "basics.matmul": {"sizes": [1024], "n_batches": 3, "batch_min_s": 0.02},
    "basics.membw": {"buffer_mib": 64, "n_batches": 3, "batch_min_s": 0.02},
    "basics.pcie": {"buffer_mib": 32, "n_batches": 3, "batch_min_s": 0.02},
}


def effective(s: Scenario, *, tiny: bool, device: str = "cuda", smoke: bool = False) -> Scenario:
    """The scenario as it will run: tiny mode shrinks batch/length/workers/loops; CPU shrinks basics.

    ``smoke`` (maintainers, shared GPUs) applies the same batch/length/loop shrink as tiny mode but
    keeps the real model weights, and makes the GPU-basics buffers small. Its numbers are not
    benchmark results.
    """
    params = dict(s.params)
    ladder = s.ladder
    measure = s.measure or MeasureCfg()
    if smoke and s.workload in SMOKE_BASICS:
        params.update(SMOKE_BASICS[s.workload])
    if tiny or smoke:
        if "batch" in params:
            params["batch"] = C.TINY_BATCH
        if "seconds" in params:
            params["seconds"] = min(float(params["seconds"]), C.TINY_MAX_SECONDS)
        if isinstance(params.get("num_workers"), int):
            params["num_workers"] = min(params["num_workers"], C.TINY_MAX_WORKERS)
        if s.category == "sustain":
            params["duration_s"], params["window_s"] = 3.0, 1.0
        ladder = C.LADDER_TINY if ladder else None
        measure = MeasureCfg.tiny()
    if device == "cpu":
        if s.workload == "basics.matmul":
            params["sizes"] = [256]
            params["cpu_size"] = 256
        elif s.workload == "basics.membw":
            params["buffer_mib"] = 64
    return replace(s, params=params, ladder=ladder, measure=measure)


def scenario_hash(s: Scenario, measure: MeasureCfg | None = None) -> str:
    """First 12 hex chars of sha256 over canonical {id, workload, workload_version, params, ladder, measure}."""
    m = measure or s.measure or MeasureCfg()
    payload = {
        "id": s.id,
        "workload": s.workload,
        "workload_version": workload_version(s.workload),
        "params": s.params,
        "ladder": [list(r) for r in s.ladder] if s.ladder else None,
        "measure": m.to_dict(),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def timeout_s(s: Scenario, timeout_scale: float = 1.0) -> float:
    """Per-attempt timeout (SPEC 8.3)."""
    if s.category == "sustain":
        base = float(s.params.get("duration_s", 300.0)) + C.SUSTAIN_TIMEOUT_EXTRA_S
    else:
        base = max(C.MIN_TIMEOUT_S, C.TIMEOUT_FACTOR * s.est_s)
    return base * float(timeout_scale)


def first_rung(s: Scenario) -> tuple[int, int] | None:
    if not s.has_micro_batch():
        return None
    if s.ladder:
        return tuple(s.ladder[0])
    return (int(s.params["batch"]), 1)
