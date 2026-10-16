"""Unit tests for the harness (builder A): CPU only, no network, no downloads.

Covers next_rung, scenario hashes, gpu_slug, resume selection, budget math, the path sanitizer,
JSONL append + truncated-line tolerance, process-tree kill on timeout, suite membership, timing
loops, schema builders, child helpers, the run lock, and one stubbed run through real subprocesses.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # never touch a GPU from these tests

from agbench import constants as C  # noqa: E402
from agbench import registry  # noqa: E402
from agbench.child import compute_estimates, downsample, is_oom, merge_windows, resolve_workers  # noqa: E402
from agbench.context import MeasureCfg, RunContext  # noqa: E402
from agbench.env import (arch_supported, driver_requirement, env_fingerprint, gpu_slug,  # noqa: E402
                         slug_from_name, vram_class_gb)
from agbench.runner import (calibration_factor, child_env, launch_process, next_rung,  # noqa: E402
                            over_budget, resume_decision, resume_index, summarize_attempts)
from agbench.scenarios import SCENARIOS, SUITES, effective, scenario_hash, suite_est_s, timeout_s  # noqa: E402
from agbench.schema import (load_records, make_run_end, make_run_start, make_scenario_record,  # noqa: E402
                            schema_path, validate_record)
from agbench.timing import cpu_loop, measure_for_duration, measure_loop  # noqa: E402
from agbench.util import (LockHeld, RunLock, append_jsonl, build_sanitize_rules, kill_tree,  # noqa: E402
                          percentile, pid_alive, sanitize_obj)

REPO = Path(__file__).resolve().parents[1]
L16 = C.LADDER_16


def att(mb, acc, status):
    return {"micro_batch": mb, "accum": acc, "status": status}


# ---------------------------------------------------------------------------------------------
# OOM ladder
# ---------------------------------------------------------------------------------------------


def test_next_rung_sequences():
    assert next_rung(L16, []) == (16, 1)
    assert next_rung(L16, [att(16, 1, "oom")]) == (8, 2)
    assert next_rung(L16, [att(16, 1, "oom"), att(8, 2, "oom")]) == (4, 4)
    assert next_rung(L16, [att(16, 1, "oom"), att(8, 2, "oom"), att(4, 4, "oom")]) == (2, 8)
    assert next_rung(L16, [att(16, 1, "oom"), att(8, 2, "oom"), att(4, 4, "oom"), att(2, 8, "oom")]) is None
    assert next_rung(L16, [att(16, 1, "ok")]) is None
    for st in ("error", "timeout", "crashed", "skipped"):
        assert next_rung(L16, [att(16, 1, "oom"), att(8, 2, st)]) is None
    assert next_rung(C.LADDER_8, [att(8, 1, "oom")]) == (4, 2)
    assert next_rung(C.LADDER_TINY, [att(2, 1, "oom")]) == (1, 2)
    assert next_rung(C.LADDER_TINY, [att(2, 1, "oom"), att(1, 2, "oom")]) is None
    assert next_rung(None, []) is None
    # every rung keeps the effective batch
    for lad in (C.LADDER_16, C.LADDER_8, C.LADDER_TINY):
        assert len({m * a for m, a in lad}) == 1


def test_summarize_attempts_details():
    st, detail, fb = summarize_attempts([att(16, 1, "oom"), att(8, 2, "ok")], L16)
    assert (st, detail) == ("ok", "8x2 after OOM at 16x1")
    assert fb["used"] == {"micro_batch": 8, "accum": 2} and fb["fits_requested"] is False
    assert fb["ladder"] == [[16, 1], [8, 2], [4, 4], [2, 8]] and len(fb["attempts"]) == 2
    st, detail, fb = summarize_attempts([att(16, 1, "ok")], L16)
    assert (st, detail, fb["fits_requested"]) == ("ok", "fits at 16x1", True)
    st, detail, fb = summarize_attempts([att(m, a, "oom") for m, a in L16], L16)
    assert st == "oom" and detail == "OOM at 16x1, 8x2, 4x4, 2x8" and fb["used"] is None
    st, detail, _ = summarize_attempts([att(16, 1, "oom"), {**att(8, 2, "crashed"),
                                                            "status_detail": "crashed (exit code 3)"}], L16)
    assert st == "crashed" and detail.startswith("crashed at 8x2 after OOM at 16x1")


# ---------------------------------------------------------------------------------------------
# Scenarios, suites, registry
# ---------------------------------------------------------------------------------------------


def test_suite_membership_and_estimates():
    quick, full = SUITES["quick"], SUITES["full"]
    assert len(quick) == 18 and len(full) == 49 and len(SCENARIOS) == 49
    assert len(set(full)) == len(full) and set(quick) <= set(full)
    assert full[:18] == quick                      # full starts with the quick list
    assert full[-2:] == ["sustain.w2v2b.6s.bs16.bf16", "train.w2v2b.6s.bs16.bf16.repeat"]
    assert set(full) == set(SCENARIOS)
    for sid in full:
        s = SCENARIOS[sid]
        assert s.id == sid and s.workload in registry.WORKLOADS
        assert ("quick" in s.suites) == (sid in quick)
    # SPEC 6.3: ~10 min (quick) and ~32 min (full) incl. ~45 s setup
    # SPEC 6.3 + 18 (reference-numerics rows are a little slower): ~10 min / ~32 min incl. setup
    assert abs(suite_est_s("quick") - 543) < 1 and abs(suite_est_s("full") - 1881) < 1
    assert 9.0 < (suite_est_s("quick") + C.SETUP_EST_S) / 60 < 10.5
    assert 31.0 < (suite_est_s("full") + C.SETUP_EST_S) / 60 < 33.0


def test_registry_targets_and_override(monkeypatch):
    for key, (target, version) in registry.WORKLOADS.items():
        mod, _, fn = target.partition(":")
        assert mod.startswith("agbench.workloads.") and fn and version >= 1
    monkeypatch.setenv("AGBENCH_REGISTRY_OVERRIDE", "train=json:dumps, cpu.decode = os.path:join")
    assert registry.resolve("train") is json.dumps
    assert registry.resolve("cpu.decode") is os.path.join
    assert registry.target("infer") == "agbench.workloads.infer:run"


def test_scenario_ids_follow_grammar():
    import re

    model = re.compile(r"^(train|frozen|infer|e2e|sustain)\.(w2v2b|xlsr|w2v2l|wavlmb|fusion)\.\d+s\.bs\d+\."
                       r"(bf16|fp16|tf32)(\.[a-z0-9]+)?$")
    for sid, s in SCENARIOS.items():
        if s.category in ("train", "frozen", "infer", "e2e", "sustain"):
            assert model.match(sid), sid
            assert s.params["mode"] == s.category and s.needs_gpu and s.models
        elif s.category == "basics":
            assert sid.startswith("gpu.")
        else:
            assert re.match(r"^cpu\.(decode|rawboost\.blas(1|default)|loader\.w\d+\.blas(1|default))$", sid)
    assert SCENARIOS["train.fusion.4s.bs16.bf16"].models == ("xlsr", "wavlmb")
    assert SCENARIOS["train.w2v2b.6s.bs32.bf16"].ladder is None
    assert SCENARIOS["train.w2v2b.6s.bs8.bf16"].ladder == C.LADDER_8
    assert SCENARIOS["train.w2v2b.6s.bs16.bf16.det"].env == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    assert SCENARIOS["cpu.loader.w4.blasdefault"].env == {k: None for k in C.BLAS_THREAD_VARS}
    assert SCENARIOS["cpu.rawboost.blas1"].env == {"OPENBLAS_NUM_THREADS": "1"}


def test_scenario_hash_stability():
    anchor = SCENARIOS["train.w2v2b.6s.bs16.bf16"]
    h = scenario_hash(anchor)
    assert h == scenario_hash(anchor, MeasureCfg()) and len(h) == 12
    # golden value: changing what the anchor measures must be a deliberate decision
    assert h == "f9fe24f1e016", h
    from dataclasses import replace

    reordered = replace(anchor, params=dict(reversed(list(anchor.params.items()))))
    assert scenario_hash(reordered) == h
    assert scenario_hash(replace(anchor, params={**anchor.params, "seconds": 4.0})) != h
    assert scenario_hash(anchor, MeasureCfg.tiny()) != h
    assert scenario_hash(replace(anchor, ladder=C.LADDER_8)) != h
    assert scenario_hash(effective(anchor, tiny=True, device="cpu")) != h
    assert len({scenario_hash(s) for s in SCENARIOS.values()}) == len(SCENARIOS)


def test_effective_tiny_transform():
    s = effective(SCENARIOS["train.w2v2b.10s.bs16.bf16"], tiny=True, device="cpu")
    assert s.params["batch"] == 2 and s.params["seconds"] == 2.0
    assert s.ladder == C.LADDER_TINY and s.measure == MeasureCfg.tiny()
    assert effective(SCENARIOS["train.w2v2b.6s.bs32.bf16"], tiny=True).ladder is None
    su = effective(SCENARIOS["sustain.w2v2b.6s.bs16.bf16"], tiny=True)
    assert (su.params["duration_s"], su.params["window_s"]) == (3.0, 1.0)
    lo = effective(SCENARIOS["cpu.loader.w16.blas1"], tiny=True)
    assert lo.params["num_workers"] == 2 and lo.params["batch"] == 2
    assert effective(SCENARIOS["e2e.w2v2b.6s.bs16.bf16.rawboost"], tiny=True).params["num_workers"] == "auto"
    mm = effective(SCENARIOS["gpu.matmul"], tiny=False, device="cpu")
    assert mm.params["sizes"] == [256]
    assert effective(SCENARIOS["gpu.membw"], tiny=True, device="cpu").params["buffer_mib"] == 64
    full = effective(SCENARIOS["train.xlsr.6s.bs16.bf16"], tiny=False)
    assert full.params == SCENARIOS["train.xlsr.6s.bs16.bf16"].params and full.ladder == L16
    assert full.measure == MeasureCfg()
    # the original matrix is never mutated
    assert SCENARIOS["train.w2v2b.10s.bs16.bf16"].params["batch"] == 16


def test_effective_smoke_transform():
    """--smoke: tiny shapes and loops with real weights (the child keeps ctx.tiny False)."""
    s = effective(SCENARIOS["train.w2v2b.6s.bs16.bf16"], tiny=False, device="cuda", smoke=True)
    assert s.params["batch"] == 2 and s.params["seconds"] == 2.0
    assert s.ladder == C.LADDER_TINY and s.measure == MeasureCfg.tiny()
    mm = effective(SCENARIOS["gpu.matmul"], tiny=False, device="cuda", smoke=True)
    assert mm.params["sizes"] == [1024] and mm.params["n_batches"] == 3
    assert effective(SCENARIOS["gpu.membw"], tiny=False, smoke=True).params["buffer_mib"] == 64
    assert effective(SCENARIOS["gpu.pcie"], tiny=False, smoke=True).params["buffer_mib"] == 32
    anchor = SCENARIOS["train.w2v2b.6s.bs16.bf16"]
    assert scenario_hash(s) not in (scenario_hash(anchor), scenario_hash(effective(anchor, tiny=False)))
    assert SCENARIOS["gpu.matmul"].params["sizes"] == [4096, 8192]   # matrix not mutated


def test_run_slug_separates_test_runs():
    from agbench.runner import run_slug

    assert run_slug("rtx-3090-24gb") == "rtx-3090-24gb"
    assert run_slug("rtx-3090-24gb", tiny=True) == "rtx-3090-24gb-tiny"
    assert run_slug("rtx-a4500-20gb", smoke=True) == "rtx-a4500-20gb-smoke"
    assert run_slug("cpu", tiny=True) == "cpu"


def test_timeout_formula():
    assert timeout_s(SCENARIOS["gpu.membw"]) == 120.0                 # max(120, 4*12)
    assert timeout_s(SCENARIOS["train.xlsr.6s.bs16.tf32"]) == 376.0   # 4*94
    assert timeout_s(SCENARIOS["sustain.w2v2b.6s.bs16.bf16"]) == 480.0  # 300 + 180
    assert timeout_s(SCENARIOS["gpu.membw"], 0.05) == pytest.approx(6.0)
    assert timeout_s(effective(SCENARIOS["sustain.w2v2b.6s.bs16.bf16"], tiny=True)) == 183.0


# ---------------------------------------------------------------------------------------------
# GPU slug, fingerprint, doctor helpers
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name,gib,slug", [
    ("NVIDIA GeForce RTX 5060 Ti", 15.93, "rtx-5060-ti-16gb"),
    ("NVIDIA GeForce RTX 5060 Ti", 7.96, "rtx-5060-ti-8gb"),
    ("NVIDIA GeForce RTX 3090", 23.69, "rtx-3090-24gb"),
    ("NVIDIA RTX A4500", 19.67, "rtx-a4500-20gb"),
    ("NVIDIA A100-SXM4-40GB", 39.39, "a100-sxm4-40gb"),
    ("NVIDIA GeForce GTX 1080 Ti", 10.91, "gtx-1080-ti-11gb"),
])
def test_gpu_slug_examples(name, gib, slug):
    assert slug_from_name(name, gib) == slug
    assert gpu_slug({"gpu_name": name, "vram_total_gib": gib, "device": "cuda"}) == slug


def test_gpu_slug_cpu_and_vram_class():
    assert gpu_slug({"gpu_name": None}) == "cpu"
    assert gpu_slug({"gpu_name": "NVIDIA GeForce RTX 3090", "vram_total_gib": 23.7, "device": "cpu"}) == "cpu"
    assert vram_class_gb(15.93) == 16 and vram_class_gb(7.6) == 8 and vram_class_gb(None) is None
    assert vram_class_gb(79.2) == 80 and vram_class_gb(139.8) == 141


def test_env_fingerprint():
    env = {"gpu_name": "NVIDIA GeForce RTX 5060 Ti", "vram_class_gb": 16, "driver": "580.97",
           "torch": "2.14.1+cu130", "torch_cuda": "13.0", "transformers": "5.20.0", "os_family": "windows",
           "agbench_version": "0.1.0", "cpu_model": "x", "ram_gib": 32}
    fp = env_fingerprint(env)
    assert len(fp) == 12 and fp == env_fingerprint(dict(reversed(list(env.items()))))
    assert env_fingerprint({**env, "ram_gib": 64, "cpu_model": "y"}) == fp   # not part of the fingerprint
    assert env_fingerprint({**env, "driver": "581.01"}) != fp
    assert env_fingerprint({**env, "torch": "2.11.0+cu128"}) != fp


def test_arch_and_driver_rules():
    blackwell = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"]
    assert arch_supported((12, 0), blackwell)
    assert arch_supported((12, 0), ["sm_90", "compute_120"])
    assert not arch_supported((12, 0), ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"])
    assert not arch_supported((12, 0), ["sm_90", "compute_90"])        # old PTX does not count on 12.x
    assert arch_supported((8, 6), ["sm_80"]) and arch_supported((8, 6), ["sm_75", "compute_75"])
    assert not arch_supported((8, 6), ["sm_90", "sm_120"])
    assert driver_requirement("13.0", "linux") == (580, 0)
    assert driver_requirement("12.8", "windows") == (570, 65)
    assert driver_requirement("12.8", "linux") == (570, 26)
    assert driver_requirement(None, "linux") is None


# ---------------------------------------------------------------------------------------------
# Resume and budget
# ---------------------------------------------------------------------------------------------


def _rec(sid, status, detail="", fp="fp1", h="h1", t="2026-10-16T10:00:00Z"):
    return {"kind": "scenario", "scenario_id": sid, "scenario_hash": h, "env_fingerprint": fp,
            "status": status, "status_detail": detail, "finished_utc": t}


def test_resume_done_set_selection():
    recs = [
        {"kind": "run_start", "run_id": "x"},
        _rec("a", "ok"), _rec("b", "oom", "OOM at 16x1"),
        _rec("c", "skipped", "workers_exceed_cores: 16 workers > 8 logical cores"),
        _rec("d", "skipped", "budget: projected 16m > 15m"),
        _rec("e", "error", "RuntimeError: x"),
        _rec("f", "ok", fp="other"),                      # other environment
        _rec("g", "ok", h="old"),                         # scenario definition changed
        _rec("h", "error", t="2026-10-16T10:00:00Z"), _rec("h", "ok", t="2026-10-16T11:00:00Z"),
        _rec("i", "ok", t="2026-10-16T10:00:00Z"), _rec("i", "timeout", t="2026-10-16T11:00:00Z"),
        _rec("j", "skipped", "model_missing: xlsr"),
    ]
    idx = resume_index(recs, "fp1")
    d = {sid: resume_decision(idx, sid, "h1") for sid in "abcdefghij"}
    assert d == {"a": "cached", "b": "cached", "c": "cached", "d": None, "e": None, "f": None, "g": None,
                 "h": "cached", "i": "cached", "j": None}
    assert resume_decision(idx, "e", "h1", retry_failed=False) == "failed_earlier"
    assert resume_decision(idx, "d", "h1", retry_failed=False) is None


def test_budget_guard_math():
    assert calibration_factor([]) == 1.0
    assert calibration_factor([(60, 30), (30, 30), (15, 30)]) == 1.0
    assert calibration_factor([(10, 30), (20, 40)]) == pytest.approx((1 / 3 + 1 / 2) / 2)
    assert not over_budget(800, 30, 1.0, 900)
    assert over_budget(880, 30, 1.0, 900)
    assert not over_budget(880, 30, 0.5, 900)
    assert not over_budget(10_000, 300, 2.0, 0)          # 0 = unlimited


def test_child_env_rules():
    base = {"PATH": "x", "OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "2", "HF_HUB_OFFLINE": "0"}
    env = child_env(base, gpu=1, device="cuda", scenario_env={"OPENBLAS_NUM_THREADS": None,
                                                             "OMP_NUM_THREADS": None, "X": "1"})
    assert env["CUDA_VISIBLE_DEVICES"] == "1" and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["HF_HUB_OFFLINE"] == "1" and env["PYTHONUNBUFFERED"] == "1" and env["X"] == "1"
    assert "OPENBLAS_NUM_THREADS" not in env and "OMP_NUM_THREADS" not in env
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env
    assert child_env(base, gpu=0, device="cpu")["CUDA_VISIBLE_DEVICES"] == ""
    assert base["OPENBLAS_NUM_THREADS"] == "4"            # caller's dict untouched


# ---------------------------------------------------------------------------------------------
# Privacy sanitizer
# ---------------------------------------------------------------------------------------------


def _sanitize_with(rules, text):
    for pat, repl in rules:
        text = pat.sub(repl, text)
    return text


def test_sanitizer_rules():
    rules = build_sanitize_rules("/home/alice", "alice", "lab-box.example.org")
    s = _sanitize_with(rules, "File /home/alice/x.py by alice on lab-box.example.org (LAB-BOX); /home/alicex")
    assert s == "File ~/x.py by <user> on <host> (<host>); /home/alicex"
    win = build_sanitize_rules(r"C:\Users\Dev", "dev", "DESKTOP-AB12")
    t = _sanitize_with(win, r'File "C:\Users\Dev\bench\a.py" c:/users/dev/b --device cuda dev DESKTOP-AB12')
    assert t == 'File "~\\bench\\a.py" ~/b --device cuda <user> <host>'
    assert _sanitize_with(win, '"C:\\\\Users\\\\Dev\\\\x"') == '"~\\\\x"'
    # short names are not replaced (too many false hits)
    assert _sanitize_with(build_sanitize_rules("", "ab", "pc"), "ab pc") == "ab pc"


def test_sanitize_obj_types():
    import math

    out = sanitize_obj({"a": float("nan"), "b": [1, 2.5, float("inf")], "c": (True, None), "d": Path("x")})
    assert out == {"a": None, "b": [1, 2.5, None], "c": [True, None], "d": "x"}
    assert json.dumps(out, allow_nan=False)
    assert not math.isnan(sanitize_obj(1.0))


# ---------------------------------------------------------------------------------------------
# Files: JSONL append, truncated lines, lock
# ---------------------------------------------------------------------------------------------


def test_append_jsonl_and_truncated_last_line(tmp_path):
    p = tmp_path / "r.jsonl"
    append_jsonl(p, {"kind": "run_start", "n": 1})
    append_jsonl(p, {"kind": "scenario", "n": 2})
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"kind": "scenario", "n": 3, "trunc')        # writer died mid-line
    assert [r["n"] for r in load_records(p)] == [1, 2]
    append_jsonl(p, {"kind": "run_end", "n": 4})               # starts on a fresh line
    assert [r["n"] for r in load_records(p)] == [1, 2, 4]
    assert p.read_text(encoding="utf-8").endswith("\n")
    assert load_records(tmp_path / "missing.jsonl") == []


def test_run_lock(tmp_path):
    lock_path = tmp_path / ".lock"
    with RunLock(lock_path):
        assert lock_path.exists()
        with pytest.raises(LockHeld):
            RunLock(lock_path).acquire()
    assert not lock_path.exists()
    # stale lock: the process is gone
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    lock_path.write_text(json.dumps({"pid": proc.pid, "create_time": 1.0}), encoding="utf-8")
    lk = RunLock(lock_path).acquire()
    assert json.loads(lock_path.read_text(encoding="utf-8"))["pid"] == os.getpid()
    lk.release()
    assert not lock_path.exists()


# ---------------------------------------------------------------------------------------------
# Processes: timeout kills the whole tree (including a sleeping grandchild)
# ---------------------------------------------------------------------------------------------


def test_launch_timeout_kills_grandchild(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    script = textwrap.dedent(f"""
        import subprocess, sys, time
        g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        open({str(pid_file)!r}, "w").write(str(g.pid))
        print("spawned", flush=True)
        time.sleep(120)
    """)
    t0 = time.perf_counter()
    res = launch_process([sys.executable, "-c", script], env=dict(os.environ), log_path=tmp_path / "x.log",
                         timeout=3.0)
    assert res["timed_out"] is True and time.perf_counter() - t0 < 40
    gpid = int(pid_file.read_text())
    deadline = time.time() + 10
    while pid_alive(gpid) and time.time() < deadline:
        time.sleep(0.2)
    assert not pid_alive(gpid)
    assert "spawned" in (tmp_path / "x.log").read_text(encoding="utf-8")


def test_launch_exit_code_and_log_sanitized(tmp_path):
    home = str(Path.home())
    res = launch_process([sys.executable, "-c", f"import sys; print({home!r} + '/secret'); sys.exit(3)"],
                         env=dict(os.environ), log_path=tmp_path / "y.log", timeout=60)
    assert res["exit_code"] == 3 and res["timed_out"] is False
    text = (tmp_path / "y.log").read_text(encoding="utf-8")
    assert "~/secret" in text and home not in text


def test_kill_tree_missing_pid():
    assert kill_tree(2 ** 22 + 12345) == 0


# ---------------------------------------------------------------------------------------------
# Timing loops (CPU)
# ---------------------------------------------------------------------------------------------


def test_measure_loop_cpu_counts_and_stats():
    calls = []

    def step():
        calls.append(1)
        time.sleep(0.002)

    cfg = MeasureCfg(warmup_steps=2, warmup_min_s=0.0, measure_steps=7, measure_min_s=0.0)
    r = measure_loop(step, device="cpu", items_per_step=16, cfg=cfg)
    assert r["warmup_steps_done"] == 2 and r["n_steps"] == 7 and len(calls) == 9
    assert r["step_ms_p10"] <= r["step_ms_median"] <= r["step_ms_p90"]
    assert r["clips_per_s"] == pytest.approx(16 * 1000 / r["step_ms_median"])
    assert r["clips_per_s_wall"] == pytest.approx(16 * 7 / r["measure_s"])
    assert 1.5 < r["step_ms_median"] < 200 and r["step_ms_cv"] >= 0


def test_measure_loop_time_caps():
    cfg = MeasureCfg(warmup_steps=1000, warmup_min_s=0.0, warmup_max_s=0.05,
                     measure_steps=10_000, measure_min_s=0.0, measure_max_s=0.05)
    r = measure_loop(lambda: time.sleep(0.01), device="cpu", items_per_step=1, cfg=cfg)
    assert r["warmup_steps_done"] < 1000 and 5 <= r["n_steps"] < 100     # min 5 steps even when capped
    cfg2 = MeasureCfg(warmup_steps=1, warmup_min_s=0.0, measure_steps=2, measure_min_s=0.05)
    r2 = measure_loop(lambda: time.sleep(0.005), device="cpu", items_per_step=1, cfg=cfg2)
    assert r2["n_steps"] > 2 and r2["measure_s"] >= 0.05


def test_measure_for_duration_windows():
    summary, windows = measure_for_duration(lambda: time.sleep(0.005), device="cpu", items_per_step=4,
                                            duration_s=0.35, window_s=0.1, warmup=MeasureCfg.tiny())
    assert 3 <= len(windows) <= 5 and summary["n_windows"] == len(windows)
    assert all(w["clips_per_s"] > 0 and w["steps"] > 0 for w in windows)
    assert windows[-1]["t_s"] >= 0.35 and summary["sustain_ratio"] > 0
    assert summary["n_steps"] == sum(w["steps"] for w in windows) and "t0_perf" in summary


def test_cpu_loop_reports_cpu_time():
    def busy():  # ~20-40 ms of pure CPU per call (Windows process_time ticks are 15.6 ms)
        x = 0
        for i in range(300_000):
            x += i * i

    r = cpu_loop(busy, items_per_call=4, cfg=MeasureCfg(warmup_steps=1, warmup_min_s=0, measure_steps=5,
                                                         measure_min_s=0))
    assert r["calls"] == 5 and r["clips_per_s"] > 0
    assert r["ms_per_clip_median"] == pytest.approx(r["step_ms_median"] / 4)
    assert 0.3 < r["cpu_wall_ratio"] < 3.0 and r["cpu_ms_per_clip"] > 0


def test_percentile_matches_numpy():
    import numpy as np

    xs = [5.0, 1.0, 9.0, 3.0, 7.0, 2.0]
    for q in (0, 10, 50, 90, 100):
        assert percentile(xs, q) == pytest.approx(float(np.percentile(xs, q)))


# ---------------------------------------------------------------------------------------------
# Schema builders and validation
# ---------------------------------------------------------------------------------------------


def _records():
    env = {"os": "Linux", "os_family": "linux", "python": "3.12.3", "gpu_name": None}
    start = make_run_start(run_id="r1", suite="quick", gpu_slug="cpu", env_fingerprint="abc", env=env,
                           settings={"tiny": True})
    ok = make_scenario_record(
        run_id="r1", gpu_slug="cpu", env_fingerprint="abc", scenario_id="train.w2v2b.6s.bs16.bf16",
        scenario_hash="h", workload="train", workload_version=1, category="train", suite="quick",
        status="ok", status_detail="fits at 16x1", started_utc="2026-10-16T00:00:00Z", wall_s=3.0,
        params={"batch": 16}, measure=MeasureCfg().to_dict(),
        fallback={"ladder": [[16, 1]], "attempts": [att(16, 1, "ok")], "used": {"micro_batch": 16, "accum": 1},
                  "fits_requested": True},
        metrics={"clips_per_s": 50.0, "step_ms_median": 320.0, "loss_last": float("nan")},
        estimates=compute_estimates("train", {"clips_per_s": 50.0}, 6.0))
    oom = make_scenario_record(
        run_id="r1", gpu_slug="cpu", env_fingerprint="abc", scenario_id="train.w2v2b.6s.bs32.bf16",
        scenario_hash="h2", workload="train", workload_version=1, category="train", suite="full",
        status="oom", status_detail="OOM at 32x1", started_utc="2026-10-16T00:00:00Z", params={},
        measure={}, metrics={"clips_per_s": 1.0})
    err = make_scenario_record(
        run_id="r1", gpu_slug="cpu", env_fingerprint="abc", scenario_id="gpu.pcie", scenario_hash="h3",
        workload="basics.pcie", workload_version=1, category="basics", suite="quick", status="timeout",
        started_utc="2026-10-16T00:00:00Z", params={}, measure={},
        error={"type": "Timeout", "message": "x", "exit_code": None, "traceback_tail": ""})
    end = make_run_end(run_id="r1", wall_s=10.0, counts={"ok": 1, "oom": 1, "timeout": 1})
    return [start, ok, oom, err, end]


def test_schema_builders_valid_and_jsonschema_agrees():
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(json.loads(schema_path().read_text(encoding="utf-8")))
    recs = _records()
    for r in recs:
        assert validate_record(r) == [], (r.get("scenario_id"), validate_record(r))
        assert not list(validator.iter_errors(r))
        json.dumps(r, allow_nan=False)
    assert recs[1]["metrics"]["loss_last"] is None             # NaN -> null
    assert recs[2]["metrics"] == {}                            # non-ok rows carry no metrics
    assert recs[1]["estimates"]["asv5_train_epoch_min"] == pytest.approx(182357 / 50 / 60)
    assert recs[4]["counts"]["cached"] == 0


@pytest.mark.parametrize("mutate,needle", [
    (lambda r: r.pop("scenario_hash"), "missing field 'scenario_hash'"),
    (lambda r: r.update(status="great"), "status 'great'"),
    (lambda r: r.update(category="gpu"), "category 'gpu'"),
    (lambda r: r.update(metrics={}), "ok record without metrics.clips_per_s"),
    (lambda r: r.update(wall_s="3"), "field 'wall_s' should be number"),
    (lambda r: r.update(telemetry={"source": "magic"}), "telemetry.source"),
    (lambda r: r.update(status="error", metrics={}), "needs an error object"),
    (lambda r: r.update(status="oom"), "metrics must be empty unless status is ok"),
    (lambda r: r["fallback"]["attempts"][0].pop("accum"), "fallback.attempts[0].accum"),
])
def test_schema_validation_messages(mutate, needle):
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(json.loads(schema_path().read_text(encoding="utf-8")))
    rec = json.loads(json.dumps(_records()[1]))
    mutate(rec)
    problems = validate_record(rec)
    assert any(needle in p for p in problems), problems
    assert list(validator.iter_errors(rec)), "JSON Schema should also reject it"


def test_validate_unknown_kind():
    assert validate_record({"kind": "x"})[0].startswith("unknown kind")


# ---------------------------------------------------------------------------------------------
# Child helpers
# ---------------------------------------------------------------------------------------------


def test_child_helpers():
    import torch

    assert is_oom(torch.OutOfMemoryError("simulated"))
    assert is_oom(RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"))
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert not is_oom(RuntimeError("shape mismatch")) and not is_oom(ValueError("out of memory"))
    e = compute_estimates("infer", {"clips_per_s": 100.0}, 6.0)
    assert e["asv5_dev_pass_min"] == pytest.approx(140950 / 100 / 60)
    assert e["asv5_eval_pass_min"] == pytest.approx(680774 / 100 / 60)
    assert e["asv5_train_epoch_min"] is None and e["assumes_seconds"] == 6.0
    assert compute_estimates("basics", {"clips_per_s": 1.0}, None)["asv5_train_epoch_min"] is None
    assert all(v is None for v in compute_estimates("train", {}, 6.0).values())
    assert resolve_workers(4, tiny=False) == 4 and resolve_workers(16, tiny=True) == 2
    assert 1 <= resolve_workers("auto", tiny=False) <= 8
    assert len(downsample(list(range(1500)))) <= 600 and downsample([1, 2]) == [1, 2]
    windows = [{"t_s": 1.0, "t_start_s": 0.0, "clips_per_s": 10.0, "steps": 5},
               {"t_s": 2.0, "t_start_s": 1.0, "clips_per_s": 8.0, "steps": 4}]
    series = [{"t_s": 0.6, "temp_c": 50, "sm_clock_mhz": 2000, "power_w": 100},
              {"t_s": 1.4, "temp_c": 60, "sm_clock_mhz": 1900, "power_w": 150},
              {"t_s": 2.3, "temp_c": 70, "sm_clock_mhz": 1800, "power_w": 170}]
    m = merge_windows(windows, series, offset_s=0.5)
    assert [w["temp_c"] for w in m] == [55.0, 70.0] and m[1]["clips_per_s"] == 8.0
    assert m[0]["steps"] == 5 and merge_windows(windows, [], 0.0)[0]["temp_c"] is None


def test_run_context_spec_constructor(tmp_path):
    ctx = RunContext(scenario_id="x", workload="train", params={"batch": 16, "micro_batch": 8, "accum": 2},
                     device="cpu", tiny=True, measure=MeasureCfg.tiny(), seed=1, cache_dir=tmp_path,
                     keep_series=False, log=lambda m: None)
    assert (ctx.micro_batch, ctx.accum, ctx.gpu_index) == (8, 2, 0)
    assert MeasureCfg.from_dict(MeasureCfg.tiny().to_dict()) == MeasureCfg.tiny()


# ---------------------------------------------------------------------------------------------
# One stubbed run through the real runner and child processes (OOM ladder, error, resume)
# ---------------------------------------------------------------------------------------------

_STUB = '''
def run(ctx):
    ctx.log("stub workload " + ctx.scenario_id)
    p = ctx.params
    return {"metrics": {"clips_per_s": 10.0 * p.get("micro_batch", 1), "step_ms_median": 100.0,
                        "n_steps": 3, "warmup_s": 0.0},
            "extra": {"stub": True, "mb": p.get("micro_batch"), "accum": p.get("accum")},
            "timeseries": None}
'''


def test_stubbed_suite_ladder_error_and_resume(tmp_path, monkeypatch):
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    (stub_dir / "agbench_stub_wl.py").write_text(_STUB, encoding="utf-8")
    pp = os.pathsep.join([str(stub_dir), str(REPO)] + ([os.environ["PYTHONPATH"]]
                                                       if os.environ.get("PYTHONPATH") else []))
    monkeypatch.setenv("PYTHONPATH", pp)
    monkeypatch.setenv("AGBENCH_REGISTRY_OVERRIDE", "train=agbench_stub_wl:run,infer=agbench_stub_wl:run")
    monkeypatch.setenv("AGBENCH_SIMULATE_OOM_MB_GE", "2")
    monkeypatch.setenv("AGBENCH_SIMULATE_FAULT", "infer.w2v2b.6s.bs16.bf16:error")
    monkeypatch.setenv("AGBENCH_CACHE", str(tmp_path / "cache"))
    out = tmp_path / "res" / "cpu_test.jsonl"
    cmd = [sys.executable, "-m", "agbench", "run", "--tiny", "--device", "cpu", "--no-report", "--no-network",
           "--scenario", "train.w2v2b.6s.bs16.bf16,infer.w2v2b.6s.bs16.bf16", "--out", str(out)]
    p1 = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(tmp_path))
    assert p1.returncode == 1, p1.stdout + p1.stderr        # one scenario ended in error
    recs = load_records(out)
    assert [r["kind"] for r in recs] == ["run_start", "scenario", "scenario", "run_end"]
    assert all(validate_record(r) == [] for r in recs), [validate_record(r) for r in recs]
    train, infer = recs[1], recs[2]
    assert train["status"] == "ok" and train["status_detail"] == "1x2 after OOM at 2x1"
    assert [(a["micro_batch"], a["accum"], a["status"]) for a in train["fallback"]["attempts"]] == \
        [(2, 1, "oom"), (1, 2, "ok")]
    assert train["metrics"]["effective_batch"] == 2 and train["extra"]["stub"] is True
    assert train["estimates"]["asv5_train_epoch_min"] == pytest.approx(182357 / 10.0 / 60)
    assert infer["status"] == "error" and infer["error"]["message"] == "simulated"
    assert len(infer["fallback"]["attempts"]) == 1 and infer["metrics"] == {}
    assert recs[3]["counts"]["ok"] == 1 and recs[3]["counts"]["error"] == 1
    assert (out.parent / train["log_file"]).exists()
    assert "[ 1/2] train.w2v2b.6s.bs16.bf16 ... ok (1x2 after OOM at 2x1)" in p1.stdout
    # second run: the ok scenario is cached; the failed one is retried unless --no-retry-failed
    p2 = subprocess.run(cmd + ["--no-retry-failed"], capture_output=True, text=True, timeout=300,
                        cwd=str(tmp_path))
    assert p2.returncode == 0, p2.stdout + p2.stderr
    assert "train.w2v2b.6s.bs16.bf16 ... cached" in p2.stdout
    assert "failed earlier" in p2.stdout
    recs2 = load_records(out)
    assert [r["kind"] for r in recs2[4:]] == ["run_start", "run_end"]
    assert recs2[-1]["counts"]["cached"] == 2
    assert not (out.parent / ".lock").exists()


def test_parent_side_never_imports_torch():
    code = ("import sys\n"
            "import agbench.cli, agbench.runner, agbench.scenarios, agbench.schema, agbench.env, agbench.timing\n"
            "from agbench.runner import RunOptions, run_suite\n"
            "from agbench.scenarios import SUITES\n"
            "rc = run_suite(SUITES['full'], RunOptions(suite='full', dry_run=True))\n"
            "assert rc == 0 and 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)\n"
            "print('NO_TORCH_OK')\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                       cwd=str(REPO))
    assert p.returncode == 0 and "NO_TORCH_OK" in p.stdout, p.stdout + p.stderr
    assert "[49/49] train.w2v2b.6s.bs16.bf16.repeat" in p.stdout


# ---------------------------------------------------------------------------------------------
# doctor GPU branch with a faked RTX 5060 Ti (no GPU is touched)
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def fake_5060ti(monkeypatch):
    import torch

    import agbench.env as E

    gib = 1024 ** 3

    class Props:
        name = "NVIDIA GeForce RTX 5060 Ti"
        major, minor = 12, 0
        total_memory = int(15.93 * gib)
        uuid = "0b1c2d3e-0000-1111-2222-333344445555"

    state = {"arch": ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"],
             "driver": "580.97", "free": 15.2, "smoke": [], "width_max": 8}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i=0: Props)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: list(state["arch"]))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (int(state["free"] * gib), Props.total_memory))
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(E, "kernel_smoke", lambda: list(state["smoke"]))
    monkeypatch.setattr(E, "gpu_static_info", lambda i=0, u=None: {
        "source": "nvml", "driver": state["driver"], "power_limit_w": 180.0, "temp_c": 35, "power_w": 12.0,
        "sm_clock_mhz": 210, "mem_clock_mhz": 405, "sm_clock_max_mhz": 3090, "processes": [],
        "pcie": {"gen_cur": 1, "gen_max": 5, "width_cur": 8, "width_max": state["width_max"]}})
    return state


def _doctor(tmp_path):
    from agbench.env import doctor, format_doctor

    res = doctor(0, network=False, tiny=True, results_dir=tmp_path)
    return res, {c["name"]: c["status"] for c in res["checks"]}, format_doctor(res)


def test_doctor_fake_blackwell_ok(fake_5060ti, tmp_path):
    res, st, text = _doctor(tmp_path)
    assert res["status"] == "OK", text
    assert st["cuda_available"] == st["arch"] == st["kernel_smoke"] == st["driver"] == "ok"
    assert text.splitlines()[-1] == "DOCTOR: OK"
    env = res["env"]
    assert gpu_slug(env) == "rtx-5060-ti-16gb" and env["vram_class_gb"] == 16
    assert env["driver"] == "580.97" and env["gpu_capability"] == [12, 0]
    assert env["pcie"]["gen_max"] == 5 and env["doctor"]["status"] == "OK"
    json.dumps(res, allow_nan=False)


def test_doctor_fake_blackwell_failures(fake_5060ti, tmp_path):
    fake_5060ti["arch"] = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]   # cu126-like wheel
    fake_5060ti["smoke"] = ["bf16 matmul: RuntimeError: CUDA error: no kernel image is available"]
    res, st, text = _doctor(tmp_path)
    assert res["status"] == "FATAL" and st["arch"] == "fatal" and st["kernel_smoke"] == "fatal"
    assert "--extra cu130" in text and text.splitlines()[-1].startswith("DOCTOR: FATAL")
    fake_5060ti.update(arch=["sm_120"], smoke=[], driver="575.64.03")
    res, st, text = _doctor(tmp_path)
    assert res["status"] == "FATAL" and st["driver"] == "fatal" and "580" in text
    fake_5060ti.update(driver="581.15", free=9.0, width_max=4)
    res, st, _ = _doctor(tmp_path)
    assert res["status"] == "WARN" and st["vram"] == "warn" and st["pcie"] == "warn" and res["n_warn"] == 2


def test_doctor_saves_no_gpu_program_names(fake_5060ti, tmp_path, monkeypatch):
    import psutil

    import agbench.env as E

    fake_5060ti["free"] = 9.0
    base = E.gpu_static_info()
    monkeypatch.setattr(E, "gpu_static_info", lambda i=0, u=None: {**base, "processes": [{"pid": 4242}]})

    class FakeProc:
        def __init__(self, pid):
            pass

        def name(self):
            return "secretapp.exe"

    monkeypatch.setattr(psutil, "Process", FakeProc)
    res, st, text = _doctor(tmp_path)
    assert st["vram"] == "warn" and "1 other GPU process" in text
    notes = res.pop(E.CONSOLE_ONLY_KEY)
    assert notes == ["Other programs using the GPU: secretapp.exe"]
    assert "secretapp" not in json.dumps(res)          # neither the checks nor env["doctor"]["warnings"]


def test_doctor_names_cu128_when_driver_too_old_and_cuda_unavailable(fake_5060ti, tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fake_5060ti["driver"] = "575.64.03"
    res, st, text = _doctor(tmp_path)
    assert st["cuda_available"] == "fatal" and st["driver"] == "fatal", text
    assert "575.64.03 is older than 580.00" in text and "uv sync --frozen --extra cu128" in text
    fake_5060ti["driver"] = "590.10"
    res, st, text = _doctor(tmp_path)
    assert st["driver"] == "info" and "--extra cu130" in text


def test_uv_sync_hint_follows_driver():
    assert C.uv_sync_hint("575.64") == "uv sync --frozen --extra cu128"
    assert C.uv_sync_hint("580.97") == C.uv_sync_hint(None) == "uv sync --frozen --extra cu130"
    assert "--frozen" in C.TORCH_MISSING_MSG and "cu128" in C.TORCH_MISSING_MSG


def test_resume_retry_oom_reruns_only_oom():
    from agbench.runner import resume_decision, resume_index

    recs = [{"kind": "scenario", "env_fingerprint": "fp", "scenario_id": s, "scenario_hash": "h",
             "status": st, "finished_utc": "2026-10-16T00:00:0%dZ" % i}
            for i, (s, st) in enumerate([("a", "ok"), ("b", "oom")])]
    idx = resume_index(recs, "fp")
    assert resume_decision(idx, "a", "h") == resume_decision(idx, "b", "h") == "cached"
    assert resume_decision(idx, "a", "h", retry_oom=True) == "cached"
    assert resume_decision(idx, "b", "h", retry_oom=True) is None


def test_attempts_record_cap_and_free_vram():
    from agbench.runner import summarize_attempts

    att = [{"micro_batch": 16, "accum": 1, "status": "oom", "mem_cap_gib": 14.1, "vram_free_start_gib": 14.6},
           {"micro_batch": 8, "accum": 2, "status": "ok", "mem_cap_gib": 14.1, "vram_free_start_gib": 14.6}]
    _, _, fb = summarize_attempts(att, C.LADDER_16)
    assert fb["attempts"][0]["mem_cap_gib"] == 14.1 and fb["attempts"][0]["vram_free_start_gib"] == 14.6
    from agbench.report import fit_text

    rec = {"status": "ok", "fallback": fb}
    assert fit_text(rec) == "8x2 (OOM at 16x1 under a 14.1 GiB cap; 14.6 GiB were free)"


# ---------------------------------------------------------------------------------------------
# A killed runner leaves no child behind (agent tool timeouts kill the runner, not its children)
# ---------------------------------------------------------------------------------------------


def _wait(pred, timeout=30.0, step=0.2):
    t = time.time()
    while time.time() - t < timeout:
        v = pred()
        if v:
            return v
        time.sleep(step)
    return pred()


def test_child_exits_when_watched_process_is_gone(tmp_path):
    """The polling thread of watch_parent (the path every OS has; Linux also uses PDEATHSIG)."""
    import psutil

    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])
    tag = f"{sleeper.pid}:{psutil.Process(sleeper.pid).create_time()!r}"
    code = ("import time\nfrom agbench.child import watch_parent\n"
            f"assert watch_parent({tag!r}, interval_s=0.2)\ntime.sleep(60)\n")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), timeout=60, capture_output=True, text=True)
    sleeper.wait()
    assert p.returncode == 5, p.stdout + p.stderr
    assert time.time() - t0 < 20


@pytest.mark.skipif(os.name == "nt", reason="SIGKILL of the runner is simulated on POSIX")
def test_killed_runner_leaves_no_child_and_next_run_cleans_up(tmp_path):
    import signal

    import psutil

    from conftest import child_env as _env

    env = _env(tmp_path, {"AGBENCH_SIMULATE_FAULT": "cpu.decode:hang", "AGBENCH_NO_KEEP_AWAKE": "1"})
    cmd = [sys.executable, "-m", "agbench", "run", "--tiny", "--device", "cpu", "--no-report", "--no-network",
           "--scenario", "cpu.decode"]
    runner = subprocess.Popen(cmd, cwd=str(tmp_path), env=env, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        def child():
            try:
                kids = psutil.Process(runner.pid).children(recursive=True)
            except psutil.Error:
                return None
            for k in kids:
                try:
                    if "agbench.child" in " ".join(k.cmdline()):
                        return k
                except psutil.Error:  # e.g. the doctor process that just finished (zombie)
                    continue
            return None

        kid = _wait(child, timeout=240)
        assert kid is not None, "the scenario child never started"
        os.kill(runner.pid, signal.SIGKILL)             # what an agent tool's timeout does
        runner.wait(timeout=10)
        assert _wait(lambda: not kid.is_running() or kid.status() == psutil.STATUS_ZOMBIE, timeout=10)
    finally:
        if runner.poll() is None:
            runner.kill()
    assert (tmp_path / "results" / ".lock").exists()    # stale lock of the killed run
    # A leftover from an older kind of run (no watchdog) is stopped by the next run.
    logs = tmp_path / "results" / "logs" / "old"
    logs.mkdir(parents=True)
    fake = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "-m", "agbench.child",
                             "--out", str(logs / "x.a1.json")])
    try:
        from agbench.util import find_leftover_processes

        found = find_leftover_processes(tmp_path / "results")
        assert [p.pid for p in found] == [fake.pid]
        env.pop("AGBENCH_SIMULATE_FAULT")
        p2 = subprocess.run(cmd, cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300)
        assert p2.returncode == 0, p2.stdout + p2.stderr
        assert "Stopped 1 leftover benchmark process" in p2.stdout
        assert fake.wait(timeout=10) is not None
    finally:
        if fake.poll() is None:
            fake.kill()
