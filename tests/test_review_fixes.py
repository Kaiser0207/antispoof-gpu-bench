"""Regression tests for the pre-release review: measurement definitions, reference numerics,
precision API, OOM classification, memory margins, Windows file locks, keep-awake, report notes.

CPU only, offline, no downloads.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from agbench import child as CH  # noqa: E402
from agbench import util as U  # noqa: E402
from agbench.scenarios import SCENARIOS, SUITES  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- cpu.loader window


def test_loader_plan_whole_bursts_and_backlog():
    assert CH is not None
    from agbench.workloads.cpu_pipeline import loader_plan

    assert loader_plan(30, 1, 2) == (32, 2)
    assert loader_plan(30, 4, 2) == (8 + 32, 8)
    assert loader_plan(30, 12, 2) == (24 + 72, 24)
    meas, backlog = loader_plan(30, 16, 2)
    assert backlog == 32 and (meas - backlog) % 16 == 0 and meas - backlog >= 96


def test_steady_rate_ignores_prefetched_backlog_and_partial_bursts():
    from agbench.workloads.cpu_pipeline import steady_rate

    workers, batch, period = 4, 16, 1.0   # each worker needs 1 s per batch; they deliver in lockstep
    backlog = 2 * workers
    arrivals = [0.0] + [0.001 * k for k in range(1, backlog + 1)]  # prefetched before timing: "free"
    t = 0.0
    for _ in range(9):  # 9 bursts of 4
        t += period
        arrivals += [t] * workers
    arrivals += [t + period] * 2  # a partial burst at the end
    rate, used = steady_rate(arrivals, batch, workers, backlog)
    assert used % workers == 0
    assert rate == pytest.approx(workers * batch / period, rel=0.13)
    naive = batch * (len(arrivals) - 1) / arrivals[-1]
    assert naive > 1.1 * rate                        # what the old definition over-reported


def test_steady_rate_falls_back_when_capped():
    from agbench.workloads.cpu_pipeline import steady_rate

    rate, used = steady_rate([0.0, 1.0, 2.0], 16, 4, 8)
    assert used == 2 and rate == pytest.approx(16.0)


# --------------------------------------------------------------------------- e2e rate = wall rate


def test_e2e_rate_counts_loader_stalls():
    from agbench.context import MeasureCfg
    from agbench.timing import measure_loop
    from agbench.workloads.e2e import with_wall_rate

    calls = {"n": 0}

    def step():
        calls["n"] += 1
        time.sleep(0.02 + (0.1 if calls["n"] % 4 == 0 else 0.0))  # a stall every 4th step

    cfg = MeasureCfg(warmup_steps=2, warmup_min_s=0.0, warmup_max_s=5.0, measure_steps=12,
                     measure_min_s=0.0, measure_max_s=10.0)
    m = with_wall_rate(measure_loop(step, device="cpu", items_per_step=16, cfg=cfg))
    assert m["clips_per_s"] == m["clips_per_s_wall"]
    assert m["clips_per_s_median_step"] > 1.5 * m["clips_per_s"]


# --------------------------------------------------------------------------- reference numerics


def test_reference_numerics_rows():
    for sid in ("e2e.w2v2b.6s.bs16.bf16", "e2e.xlsr.6s.bs16.bf16", "train.w2v2b.6s.bs16.bf16.det"):
        s = SCENARIOS[sid]
        assert s.params["deterministic"] is True and s.params["conv_tf32"] is False, sid
        assert s.params["flash_sdp"] is False, sid                    # 6 s > 4 s
        assert s.env.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", sid
    for sid in ("train.w2v2b.6s.bs16.bf16", "e2e.w2v2b.6s.bs16.bf16.rawboost", "train.fusion.4s.bs16.bf16"):
        s = SCENARIOS[sid]
        assert s.params["deterministic"] is False and "conv_tf32" not in s.params, sid
    assert "e2e.w2v2b.6s.bs16.bf16" in SUITES["quick"]


@pytest.fixture
def torch_flags():
    torch = pytest.importorskip("torch")
    yield torch
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.fp32_precision = "none"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.backends.cuda.enable_flash_sdp(True)


def test_determinism_strict_and_warn_only(torch_flags):
    from agbench.workloads import common as WC

    assert WC.apply_determinism(True)["deterministic_mode"] == "strict"
    assert WC.apply_determinism(True, warn_only=True)["deterministic_mode"] == "warn_only"
    torch_flags.use_deterministic_algorithms(False)
    assert WC.apply_determinism(False)["deterministic_mode"] == "off"
    sdp = WC.apply_sdp(False)
    assert sdp["sdp_flash_requested"] is False and sdp.get("sdp_flash_enabled") is False


def test_precision_uses_the_new_tf32_api_only(torch_flags):
    from agbench.workloads import common as WC

    torch = torch_flags
    d = WC.Precision("tf32", "cuda").describe()   # only sets flags; no CUDA call
    assert (d["matmul_fp32_precision"], d["cudnn_conv_fp32_precision"]) == ("tf32", "tf32")
    d = WC.Precision("bf16", "cuda", conv_tf32=False).describe()
    assert (d["matmul_fp32_precision"], d["cudnn_conv_fp32_precision"]) == ("ieee", "ieee")
    d = WC.Precision("fp32", "cuda").describe()
    assert (d["matmul_fp32_precision"], d["cudnn_conv_fp32_precision"]) == ("ieee", "ieee")
    d = WC.Precision("bf16", "cuda").describe()
    assert (d["matmul_fp32_precision"], d["cudnn_conv_fp32_precision"]) == ("ieee", "tf32")
    assert torch.backends.cuda.matmul.fp32_precision == "ieee"


def test_tiny_reference_numerics_row_runs_strict_on_cpu(tmp_path, torch_flags):
    """The .det row end to end on CPU in strict mode (no op without a deterministic CPU kernel)."""
    from agbench.context import MeasureCfg, RunContext
    from agbench.registry import resolve
    from agbench.scenarios import effective

    s = effective(SCENARIOS["train.w2v2b.6s.bs16.bf16.det"], tiny=True, device="cpu")
    params = {**s.params, "micro_batch": 2, "accum": 1}
    ctx = RunContext(scenario_id=s.id, workload=s.workload, params=params, device="cpu", tiny=True,
                     measure=MeasureCfg.tiny(), seed=1, cache_dir=tmp_path, keep_series=False, log=None)
    torch_flags.use_deterministic_algorithms(True)
    out = resolve(s.workload)(ctx)
    ex = out["extra"]
    assert ex["deterministic_mode"] == "strict" and ex["params_resolved"]["conv_tf32"] is False
    assert ex["sdp_flash_requested"] is False and out["metrics"]["loss_finite"] is True


_STRICT_STUB = '''
def run(ctx):
    if not ctx.params.get("deterministic_warn_only"):
        raise RuntimeError("index_add_cuda_ does not have a deterministic implementation, but you set "
                           "'torch.use_deterministic_algorithms(True)'.")
    return {"metrics": {"clips_per_s": 5.0, "step_ms_median": 100.0, "n_steps": 3}, "extra": {},
            "timeseries": None}
'''


def test_child_reruns_warn_only_when_strict_mode_has_no_kernel(tmp_path):
    from conftest import child_env

    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "agbench_strict_stub.py").write_text(_STRICT_STUB, encoding="utf-8")
    env = child_env(tmp_path, {"AGBENCH_REGISTRY_OVERRIDE": "train=agbench_strict_stub:run",
                               "AGBENCH_NO_KEEP_AWAKE": "1"})
    env["PYTHONPATH"] = os.pathsep.join([str(stub), env["PYTHONPATH"]])
    out = tmp_path / "results" / "cpu_x.jsonl"
    cmd = [sys.executable, "-m", "agbench", "run", "--tiny", "--device", "cpu", "--no-report", "--no-network",
           "--scenario", "train.w2v2b.6s.bs16.bf16.det", "--out", str(out)]
    p = subprocess.run(cmd, cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    from agbench.schema import load_records

    rec = [r for r in load_records(out) if r.get("kind") == "scenario"][0]
    assert rec["status"] == "ok"
    assert "does not have a deterministic implementation" in rec["extra"]["deterministic_strict_failed"]
    from agbench.report import build_report

    summary = build_report([out], tmp_path / "rep", references_path=REPO / "references" / "reference_gpus.json")
    assert "no deterministic kernel" in Path(summary).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- OOM and memory cap


@pytest.mark.parametrize("msg", [
    "CUDA out of memory. Tried to allocate 20.00 MiB",
    "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.",
    "Unable to find a valid cuDNN algorithm to run convolution",
    "FIND was unable to find an engine to execute this computation",
    "CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling `cublasCreate(handle)`",
])
def test_near_cap_errors_count_as_oom(msg):
    assert CH.is_oom(RuntimeError(msg))


def test_other_errors_are_not_oom():
    assert not CH.is_oom(RuntimeError("shape mismatch"))
    assert not CH.is_oom(ValueError("out of memory"))  # only RuntimeError / torch OOM


def test_memory_margin_by_os():
    assert CH.mem_cap_margin_gib("win32") == 0.5
    assert CH.mem_cap_margin_gib("linux") == 0.1


# --------------------------------------------------------------------------- Windows file locks


def test_sanitize_file_survives_a_locked_file(tmp_path, monkeypatch):
    log = tmp_path / "x.log"
    log.write_text(f"path {Path.home()}/secret\n", encoding="utf-8")

    def locked(*a, **k):
        raise PermissionError(13, "The process cannot access the file")

    monkeypatch.setattr(U, "atomic_write_text", locked)
    assert U.sanitize_file(log) is False      # no exception: the suite goes on


def test_replace_file_retries(tmp_path, monkeypatch):
    src, dst = tmp_path / "a", tmp_path / "b"
    src.write_text("x")
    real = os.replace
    fails = {"n": 2}

    def flaky(a, b):
        if fails["n"]:
            fails["n"] -= 1
            raise PermissionError(32, "in use")
        return real(a, b)

    monkeypatch.setattr(U.os, "replace", flaky)
    U.replace_file(src, dst, retries=3, wait_s=0.0)
    assert dst.read_text() == "x"
    src.write_text("y")
    fails["n"] = 5
    with pytest.raises(PermissionError):
        U.replace_file(src, dst, retries=2, wait_s=0.0)


# --------------------------------------------------------------------------- keep awake


def test_keep_awake_never_raises_and_cleans_up():
    from agbench.awake import KeepAwake

    with KeepAwake(max_s=60, enabled=False) as k:
        assert k.method is None
    with KeepAwake(max_s=60) as k:       # refused without a desktop session: then a no-op
        assert k.method in (None, "systemd-inhibit", "SetThreadExecutionState")
    assert k._proc is None


# --------------------------------------------------------------------------- report notes


def test_report_says_when_there_is_no_sustained_run(tmp_path, fixtures_dir):
    from agbench.report import build_report

    refs = REPO / "references" / "reference_gpus.json"
    quick_only = build_report([fixtures_dir / "rtx-3090-24gb_20261016.jsonl"], tmp_path / "a", references_path=refs)
    assert "no sustained run" in Path(quick_only).read_text(encoding="utf-8")
    full = build_report([fixtures_dir / "rtx-5060-ti-16gb_20261016.jsonl"], tmp_path / "b", references_path=refs)
    assert "no sustained run" not in Path(full).read_text(encoding="utf-8")


def test_runner_and_child_share_the_parent_variable():
    from agbench import runner

    assert runner.PARENT_ENV == CH.PARENT_ENV
    assert runner.parent_tag().startswith(f"{os.getpid()}:")
