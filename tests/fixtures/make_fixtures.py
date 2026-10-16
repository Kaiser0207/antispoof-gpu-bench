"""Regenerate the hand-made sample result files in tests/fixtures/.

    python tests/fixtures/make_fixtures.py

The numbers are invented but plausible (they follow the pre-run estimates in SPEC section 6.3);
they are NOT measurements. The records follow the results schema of SPEC section 9 so that the
schema, report and privacy tests can run without a GPU. The output is deterministic.

Files written:
  rtx-5060-ti-16gb_20261016.jsonl  one GPU, a quick run then a full run (all 49 scenarios), with an
                                   OOM fallback, a budget skip that a later run completes, and sustain
  rtx-3090-24gb_20261016.jsonl     a second GPU, quick suite only (for multi-GPU reports)
  rtx-5060-ti-8gb_20261017.jsonl   a problem run: oom / timeout / error / crashed / skipped rows, no sustain
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from agbench.context import MeasureCfg  # noqa: E402
from agbench.registry import workload_version  # noqa: E402
from agbench.scenarios import SCENARIOS, SUITES, scenario_hash  # noqa: E402

ASV5_TRAIN, ASV5_DEV, ASV5_EVAL = 182_357, 140_950, 680_774

QUICK = list(SUITES["quick"])
ALL = list(SUITES["full"])
FULL_REST = [sid for sid in ALL if sid not in QUICK]
assert len(QUICK) == 18 and len(ALL) == 49 and len(set(ALL)) == 49


def h12(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12]


def category(sid: str) -> str:
    return SCENARIOS[sid].category


def workload(sid: str) -> str:
    return SCENARIOS[sid].workload


def parse_model_id(sid: str) -> dict:
    return dict(SCENARIOS[sid].params)


def params_for(sid: str) -> dict:
    return dict(SCENARIOS[sid].params)


def ladder_for(sid: str) -> list | None:
    lad = SCENARIOS[sid].ladder
    return [list(r) for r in lad] if lad else None


def measure_for(sid: str) -> dict:
    return (SCENARIOS[sid].measure or MeasureCfg()).to_dict()


def telemetry(temp: float, clock: float, power: float, *, n: int = 80, source: str = "nvml") -> dict:
    if source == "none":
        return {"source": "none"}

    def s(v: float, spread: float) -> dict:
        return {"min": round(v - spread, 1), "median": round(v, 1), "max": round(v + spread, 1)}

    return {"source": source, "interval_s": 0.5, "n": n, "temp_c": s(temp, 4), "sm_clock_mhz": s(clock, 60),
            "mem_clock_mhz": s(14001, 0), "power_w": s(power, 8), "util_pct": s(97, 3),
            "mem_used_gib": s(6.0, 0.5), "power_limit_w": 180.0, "throttle_reasons": []}


class Gpu:
    def __init__(self, slug: str, name: str, vram_gib: float, vram_class: int, os_name: str, os_family: str,
                 driver: str, power_limit: float, pcie: dict, cpu_logical: int, cpu_physical: int):
        self.slug, self.name, self.vram_gib, self.vram_class = slug, name, vram_gib, vram_class
        self.os_name, self.os_family, self.driver, self.power_limit = os_name, os_family, driver, power_limit
        self.pcie, self.cpu_logical, self.cpu_physical = pcie, cpu_logical, cpu_physical
        self.fingerprint = h12(slug, driver, "2.14.1+cu130", os_family)
        self.cap = round(vram_gib - 0.5 - (0.9 if os_family == "windows" else 0.0), 2)

    def env(self) -> dict:
        return {
            "os": self.os_name, "os_family": self.os_family, "python": "3.12.11",
            "torch": "2.14.1+cu130", "torch_cuda": "13.0", "cudnn": 92400, "transformers": "5.20.0",
            "numpy": "2.5.4", "scipy": "1.18.1", "soundfile": "0.14.0", "libsndfile": "1.2.2",
            "driver": self.driver, "gpu_name": self.name, "gpu_capability": [12, 0] if "50" in self.slug else [8, 6],
            "gpu_arch_list": ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"],
            "vram_total_gib": self.vram_gib, "vram_class_gb": self.vram_class, "pcie": self.pcie,
            "power_limit_w": self.power_limit, "cpu_model": "Example desktop CPU (fixture)",
            "cpu_logical": self.cpu_logical, "cpu_physical": self.cpu_physical, "ram_gib": 31.9,
            "blas": [{"internal_api": "openblas", "version": "0.3.30", "num_threads": self.cpu_logical}],
            "env_vars": {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "HF_ENDPOINT_set": False},
            "doctor": {"status": "OK", "warnings": []},
        }


class Writer:
    def __init__(self, gpu: Gpu, day: str):
        self.gpu, self.day, self.lines, self.clock = gpu, day, [], 0

    def ts(self, add_s: float = 0.0) -> str:
        self.clock += int(add_s)
        h, rem = divmod(self.clock, 3600)
        m, s = divmod(rem, 60)
        return f"{self.day}T{18 + h:02d}:{m:02d}:{s:02d}Z"

    def run_start(self, suite: str, budget: int) -> str:
        run_id = f"{self.day.replace('-', '')}T{18 + self.clock // 3600:02d}{(self.clock // 60) % 60:02d}00Z-{h12(self.gpu.slug, suite)[:6]}"
        self.run_id, self.suite, self.run_t0, self.counts = run_id, suite, self.clock, {}
        self.lines.append({
            "kind": "run_start", "schema_version": "1", "run_id": run_id, "suite": suite,
            "started_utc": self.ts(), "agbench_version": "0.1.0", "git_commit": None,
            "gpu_slug": self.gpu.slug, "env_fingerprint": self.gpu.fingerprint,
            "settings": {"budget_min": budget, "mem_cap_gib": None, "tiny": False, "device": "cuda",
                         "timeout_scale": 1.0},
            "env": self.gpu.env(),
        })
        return run_id

    def run_end(self, cached: int = 0, interrupted: bool = False) -> None:
        counts = {k: self.counts.get(k, 0) for k in ("ok", "oom", "error", "timeout", "crashed", "skipped")}
        counts["cached"] = cached
        self.lines.append({"kind": "run_end", "schema_version": "1", "run_id": self.run_id,
                           "finished_utc": self.ts(5), "wall_s": float(self.clock - self.run_t0),
                           "counts": counts, "interrupted": interrupted})

    def scenario(self, sid: str, status: str, *, wall: float, detail: str = "", metrics: dict | None = None,
                 attempts: list | None = None, extra: dict | None = None, error: dict | None = None,
                 tel: dict | None = None, timeseries: list | None = None, n_steps: int = 20) -> None:
        self.counts[status] = self.counts.get(status, 0) + 1
        started = self.ts()
        finished = self.ts(wall)
        ladder = ladder_for(sid)
        fallback = None
        if ladder is not None:
            attempts = attempts or [{"micro_batch": ladder[0][0], "accum": ladder[0][1], "status": status,
                                     "peak_reserved_gib": (metrics or {}).get("peak_reserved_gib"),
                                     "wall_s": wall}]
            for a in attempts:  # the runner records the cap and the free VRAM of every attempt
                a.setdefault("mem_cap_gib", self.gpu.cap)
                a.setdefault("vram_free_start_gib", round(self.gpu.vram_gib - 0.4, 2))
            used = None
            if status == "ok":
                used = {"micro_batch": attempts[-1]["micro_batch"], "accum": attempts[-1]["accum"]}
            fallback = {"ladder": ladder, "attempts": attempts, "used": used,
                        "fits_requested": status == "ok" and len(attempts) == 1}
        params = params_for(sid)
        cat = category(sid)
        est = {"asv5_train_epoch_min": None, "asv5_dev_pass_min": None, "asv5_eval_pass_min": None,
               "assumes_seconds": None}
        cps = (metrics or {}).get("clips_per_s")
        if status == "ok" and cps and cat in ("train", "e2e", "sustain"):
            est.update(asv5_train_epoch_min=round(ASV5_TRAIN / cps / 60, 2), assumes_seconds=params["seconds"])
        if status == "ok" and cps and cat == "infer":
            est.update(asv5_dev_pass_min=round(ASV5_DEV / cps / 60, 2),
                       asv5_eval_pass_min=round(ASV5_EVAL / cps / 60, 2), assumes_seconds=params["seconds"])
        measure = measure_for(sid)
        if status == "ok" and cat not in ("basics", "cpu"):
            measure.update(warmup_steps_done=5, warmup_s=6.1, n_steps=n_steps, measure_s=round(wall * 0.4, 1))
        self.lines.append({
            "kind": "scenario", "schema_version": "1", "run_id": self.run_id, "gpu_slug": self.gpu.slug,
            "env_fingerprint": self.gpu.fingerprint, "scenario_id": sid, "scenario_hash": scenario_hash(SCENARIOS[sid]),
            "workload": workload(sid), "workload_version": workload_version(workload(sid)), "category": cat,
            "suite": self.suite,
            "status": status, "status_detail": detail, "started_utc": started, "finished_utc": finished,
            "wall_s": float(wall), "params": params, "measure": measure, "fallback": fallback,
            "metrics": metrics if status == "ok" else {}, "estimates": est,
            "telemetry": tel if tel is not None else telemetry(68, 2790, 172),
            "timeseries": timeseries, "extra": extra or {}, "error": error,
            "log_file": f"logs/{self.run_id}/{sid}.a{len(attempts or [1])}.log",
        })

    def write(self, name: str) -> None:
        text = "".join(json.dumps(rec, sort_keys=False, separators=(", ", ": ")) + "\n" for rec in self.lines)
        (HERE / name).write_text(text, encoding="utf-8", newline="\n")


def model_metrics(gpu: Gpu, cps: float, batch: int, alloc: float, reserved: float, *, extra: dict | None = None) -> dict:
    step = batch * 1000.0 / cps
    m = {"clips_per_s": cps, "clips_per_s_wall": round(cps * 0.985, 2), "step_ms_median": round(step, 1),
         "step_ms_p10": round(step * 0.98, 1), "step_ms_p90": round(step * 1.03, 1),
         "step_ms_mean": round(step * 1.005, 1), "step_ms_cv": 0.018, "n_steps": 20, "effective_batch": batch,
         "peak_alloc_gib": alloc, "peak_reserved_gib": reserved, "vram_total_gib": gpu.vram_gib,
         "vram_free_start_gib": round(gpu.vram_gib - 0.4, 2), "mem_cap_gib": gpu.cap,
         "near_limit": reserved > 0.95 * gpu.cap, "sysmem_suspected": False,
         "loss_first": 0.693, "loss_last": 0.688, "loss_finite": True}
    m.update(extra or {})
    return m


def oom_attempt(mb: int, acc: int, peak: float, wall: float = 19.0) -> dict:
    return {"micro_batch": mb, "accum": acc, "status": "oom", "peak_reserved_gib": peak, "wall_s": wall}


def ok_attempt(mb: int, acc: int, peak: float, wall: float) -> dict:
    return {"micro_batch": mb, "accum": acc, "status": "ok", "peak_reserved_gib": peak, "wall_s": wall}


# name -> (clips/s, alloc GiB, reserved GiB, ladder attempts or None)
def model_table_5060ti16() -> dict:
    return {
        "train.w2v2b.6s.bs16.bf16": (56.3, 5.21, 6.42, None),
        "train.w2v2b.10s.bs16.bf16": (33.1, 7.74, 9.10, None),
        "train.w2v2b.2s.bs16.bf16": (158.0, 2.71, 3.30, None),
        "train.w2v2b.4s.bs16.bf16": (85.6, 3.98, 4.90, None),
        "train.w2v2b.8s.bs16.bf16": (42.2, 6.47, 7.70, None),
        "train.w2v2b.6s.bs16.bf16.gmean": (56.9, 5.19, 6.40, None),
        "train.w2v2b.6s.bs16.tf32": (38.4, 8.41, 9.60, None),
        "train.w2v2b.6s.bs16.fp16": (55.2, 5.24, 6.45, None),
        "train.w2v2b.6s.bs16.bf16.det": (52.1, 5.22, 6.43, None),
        "train.w2v2b.6s.bs16.bf16.ckpt": (43.4, 2.58, 3.20, None),
        "train.w2v2b.6s.bs8.bf16": (50.3, 3.31, 4.10, None),
        "train.w2v2b.6s.bs32.bf16": (58.2, 9.12, 10.40, None),
        "train.xlsr.6s.bs16.bf16": (19.8, 9.46, 10.21, [oom_attempt(16, 1, 14.62), ok_attempt(8, 2, 10.21, 52.0)]),
        "train.xlsr.6s.bs16.tf32": (12.1, 9.80, 10.90, [oom_attempt(16, 1, 14.70), oom_attempt(8, 2, 14.55),
                                                         ok_attempt(4, 4, 10.90, 70.0)]),
        "train.xlsr.6s.bs16.bf16.ckpt": (15.9, 6.95, 7.60, None),
        "train.w2v2l.6s.bs16.bf16": (18.9, 12.41, 13.20, None),
        "train.wavlmb.6s.bs16.bf16": (47.0, 6.48, 7.30, None),
        "train.fusion.4s.bs16.bf16": (21.4, 9.88, 10.60, [oom_attempt(16, 1, 14.66), ok_attempt(8, 2, 10.60, 50.0)]),
        "frozen.xlsr.6s.bs16.bf16": (61.2, 3.86, 4.40, None),
        "frozen.xlsr.6s.bs16.bf16.h40": (58.4, 4.17, 4.70, None),
        "frozen.w2v2b.6s.bs16.bf16": (181.0, 1.58, 2.00, None),
        "frozen.w2v2l.6s.bs16.bf16": (63.0, 3.90, 4.40, None),
        "infer.w2v2b.6s.bs16.bf16": (168.0, 0.98, 1.30, None),
        "infer.xlsr.6s.bs16.bf16": (52.4, 2.21, 2.60, None),
        "infer.w2v2b.6s.bs4.bf16": (139.0, 0.61, 0.80, None),
        "infer.xlsr.6s.bs4.bf16": (45.1, 1.49, 1.70, None),
        "infer.w2v2l.6s.bs16.bf16": (53.0, 2.20, 2.60, None),
        "infer.wavlmb.6s.bs16.bf16": (150.0, 1.10, 1.40, None),
        "infer.fusion.4s.bs16.bf16": (70.3, 2.62, 3.00, None),
        "e2e.w2v2b.6s.bs16.bf16": (54.9, 5.22, 6.44, None),
        "e2e.w2v2b.6s.bs16.bf16.rawboost": (51.2, 5.22, 6.44, None),
        "e2e.xlsr.6s.bs16.bf16": (19.5, 9.47, 10.22, [oom_attempt(16, 1, 14.60), ok_attempt(8, 2, 10.22, 61.0)]),
        "train.w2v2b.6s.bs16.bf16.repeat": (55.6, 5.21, 6.42, None),
    }


def model_table_3090() -> dict:
    return {
        "train.w2v2b.6s.bs16.bf16": (104.6, 5.20, 6.40, None),
        "infer.w2v2b.6s.bs16.bf16": (318.0, 0.98, 1.30, None),
        "train.xlsr.6s.bs16.bf16": (39.6, 13.90, 14.80, None),
        "infer.xlsr.6s.bs16.bf16": (99.0, 2.21, 2.60, None),
        "train.w2v2l.6s.bs16.bf16": (37.2, 12.40, 13.20, None),
        "train.fusion.4s.bs16.bf16": (41.5, 13.55, 14.70, None),
        "frozen.xlsr.6s.bs16.bf16": (112.0, 3.86, 4.40, None),
        "train.w2v2b.10s.bs16.bf16": (61.0, 7.70, 9.05, None),
        "e2e.w2v2b.6s.bs16.bf16": (99.2, 5.21, 6.43, None),
    }


def add_model(w: Writer, sid: str, row: tuple) -> None:
    cps, alloc, res, attempts = row
    params = parse_model_id(sid)
    batch = params["batch"]
    extra = {"attn_implementation": "eager" if params["model"] == "wavlmb" else "sdpa",
             "is_gradient_checkpointing": bool(params.get("grad_ckpt")), "precision_effective": params["precision"]}
    mextra = {}
    if params["mode"] == "e2e":
        workers = 5 if params.get("rawboost_p") else 2
        mextra = {"loader_wait_ms_median": 4.1, "loader_wait_frac": 0.014, "num_workers": workers,
                  "worker_startup_s": 3.2}
    ts = None
    tel = telemetry(66, 2805, 171)
    if params["mode"] == "sustain":
        windows = [56.6, 56.5, 56.2, 55.9, 55.6, 55.2, 54.9, 54.6, 54.4, 54.2]
        ts = [{"t_s": 30.0 * (i + 1), "clips_per_s": v, "temp_c": 61 + 1.8 * i, "sm_clock_mhz": 2850 - 15 * i,
               "power_w": 176 + (i % 2)} for i, v in enumerate(windows)]
        mextra = {"clips_per_s_first_window": windows[0], "clips_per_s_last_window": windows[-1],
                  "sustain_ratio": round(windows[-1] / windows[0], 3)}
        tel = telemetry(70, 2780, 177, n=600)
        tel["throttle_reasons"] = ["sw_power_cap"]
        cps = round(sum(windows) / len(windows), 1)
    detail = ""
    if attempts:
        u = attempts[-1]
        detail = f"{u['micro_batch']}x{u['accum']} after OOM at {attempts[0]['micro_batch']}x{attempts[0]['accum']}"
    elif ladder_for(sid):
        detail = f"fits at {ladder_for(sid)[0][0]}x{ladder_for(sid)[0][1]}"
    wall = 300.0 + 25 if params["mode"] == "sustain" else (30.0 if params["model"] in ("w2v2b", "wavlmb") else 50.0)
    w.scenario(sid, "ok", wall=wall + sum(a["wall_s"] for a in (attempts or [])[:-1]), detail=detail,
               metrics=model_metrics(w.gpu, cps, batch, alloc, res, extra=mextra), attempts=attempts,
               extra=extra, tel=tel, timeseries=ts)


def add_basics(w: Writer, fast: bool) -> None:
    k = 1.0 if fast else 1.55
    w.scenario("gpu.matmul", "ok", wall=25, metrics={
        "tflops": {"bf16": {"4096": round(44.1 * k, 1), "8192": round(46.8 * k, 1)},
                   "fp16": {"4096": round(44.9 * k, 1), "8192": round(47.4 * k, 1)},
                   "tf32": {"4096": round(22.4 * k, 1), "8192": round(23.0 * k, 1)},
                   "fp32": {"4096": round(21.1 * k, 1), "8192": round(22.3 * k, 1)}},
        "tflops_best": {"bf16": round(46.8 * k, 1), "fp16": round(47.4 * k, 1), "tf32": round(23.0 * k, 1),
                        "fp32": round(22.3 * k, 1)}}, tel=telemetry(64, 2820, 178))
    bw = 1.0 if fast else 2.05
    w.scenario("gpu.membw", "ok", wall=12, metrics={
        "gbps": {"copy": round(409.0 * bw), "read": round(426.0 * bw), "triad": round(404.0 * bw)}})
    link = w.gpu.pcie
    w.scenario("gpu.pcie", "ok", wall=12, metrics={
        "gbps": {"h2d_pinned": 26.1 if fast else 24.6, "d2h_pinned": 25.3 if fast else 24.9,
                 "h2d_pageable": 12.4 if fast else 11.0},
        "link": {"gen_cur": link["gen_cur"], "width_cur": link["width_cur"], "gen_max": link["gen_max"],
                 "width_max": link["width_max"]}})


def cpu_metrics(cps: float, workers: int, cpu_wall: float, **more) -> dict:
    m = {"clips_per_s": cps, "ms_per_clip_median": round(1000.0 / cps * max(workers, 1), 1),
         "cpu_ms_per_clip": round(1000.0 / cps * max(workers, 1) * cpu_wall, 1), "cpu_wall_ratio": cpu_wall,
         "num_workers": workers, "worker_startup_s": 2.9 if workers else 0.0}
    m.update(more)
    return m


def add_cpu(w: Writer, ids: list[str], scale: float = 1.0) -> None:
    none = telemetry(0, 0, 0, source="none")
    for sid in ids:
        if sid == "cpu.decode":
            w.scenario(sid, "ok", wall=15, tel=none, metrics=cpu_metrics(
                round(385.0 * scale, 1), 0, 1.0, decode_ms_per_audio_s=round(0.42 / scale, 3)))
        elif sid.startswith("cpu.rawboost"):
            b1 = sid.endswith("blas1")
            ms = (31.0, 3.1, 11.8) if b1 else (31.5, 3.2, 74.0)
            cps = round(1000.0 / sum(ms) * scale, 1)
            w.scenario(sid, "ok", wall=20, tel=none, metrics=cpu_metrics(
                cps, 0, 1.0 if b1 else 2.6, stage_ms={"lnl": ms[0], "isd": ms[1], "ssi": ms[2]}),
                extra={"threadpool": [{"internal_api": "openblas", "num_threads": 1 if b1 else w.gpu.cpu_logical}]})
        else:
            n = int(sid.split(".")[2][1:])
            if n > w.gpu.cpu_logical:
                w.scenario(sid, "skipped", wall=0.5, detail="workers_exceed_cores", tel=none)
                continue
            per_worker = 37.5 if sid.endswith("blas1") else 15.0
            eff = n if sid.endswith("blas1") else min(n, 2) + 0.4 * max(0, n - 2)
            eff = min(eff, w.gpu.cpu_physical * 1.25)
            w.scenario(sid, "ok", wall=25, tel=none, metrics=cpu_metrics(
                round(per_worker * eff * scale, 1), n, 1.0 if sid.endswith("blas1") else 1.9),
                extra={"threadpool": [{"internal_api": "openblas",
                                       "num_threads": 1 if sid.endswith("blas1") else w.gpu.cpu_logical}]})


def make_5060ti_16() -> None:
    gpu = Gpu("rtx-5060-ti-16gb", "NVIDIA GeForce RTX 5060 Ti", 15.93, 16, "Windows-11-10.0.26100", "windows",
              "580.97", 180.0, {"gen_cur": 5, "gen_max": 5, "width_cur": 8, "width_max": 8}, 12, 6)
    table = model_table_5060ti16()
    w = Writer(gpu, "2026-10-16")
    # Run 1: quick. The last two cpu.loader rows are budget-skipped and completed by run 2.
    w.run_start("quick", 15)
    add_basics(w, fast=True)
    for sid in QUICK[3:12]:
        add_model(w, sid, table[sid])
    add_cpu(w, ["cpu.decode", "cpu.rawboost.blas1", "cpu.rawboost.blasdefault", "cpu.loader.w4.blas1"])
    w.scenario("cpu.loader.w8.blas1", "skipped", wall=0.0, detail="budget", tel=telemetry(0, 0, 0, source="none"))
    w.scenario("cpu.loader.w4.blasdefault", "skipped", wall=0.0, detail="budget",
               tel=telemetry(0, 0, 0, source="none"))
    w.run_end()
    # Run 2: full, resumes (16 cached) and runs the rest.
    w.clock += 600
    w.run_start("full", 90)
    add_cpu(w, ["cpu.loader.w8.blas1", "cpu.loader.w4.blasdefault"])
    for sid in FULL_REST:
        if sid.startswith("cpu."):
            add_cpu(w, [sid])
        elif sid.startswith("sustain."):
            add_model(w, sid, (56.0, 5.21, 6.42, None))
        else:
            add_model(w, sid, table[sid])
    w.run_end(cached=16)
    w.write("rtx-5060-ti-16gb_20261016.jsonl")


def make_3090() -> None:
    gpu = Gpu("rtx-3090-24gb", "NVIDIA GeForce RTX 3090", 23.56, 24, "Linux-6.8.0-x86_64-with-glibc2.39", "linux",
              "580.82.09", 350.0, {"gen_cur": 4, "gen_max": 4, "width_cur": 16, "width_max": 16}, 16, 8)
    table = model_table_3090()
    w = Writer(gpu, "2026-10-16")
    w.run_start("quick", 15)
    add_basics(w, fast=False)
    for sid in QUICK[3:12]:
        add_model(w, sid, table[sid])
    add_cpu(w, QUICK[12:], scale=1.2)
    w.run_end()
    w.write("rtx-3090-24gb_20261016.jsonl")


def make_5060ti_8() -> None:
    gpu = Gpu("rtx-5060-ti-8gb", "NVIDIA GeForce RTX 5060 Ti", 7.96, 8, "Linux-6.8.0-x86_64-with-glibc2.39", "linux",
              "580.82.09", 180.0, {"gen_cur": 4, "gen_max": 5, "width_cur": 8, "width_max": 8}, 4, 4)
    w = Writer(gpu, "2026-10-17")
    w.run_start("quick", 15)
    add_basics(w, fast=True)
    add_model(w, "train.w2v2b.6s.bs16.bf16", (55.8, 5.21, 6.42, None))
    w.lines[-1]["metrics"]["near_limit"] = True
    oom = [oom_attempt(16, 1, 7.02), oom_attempt(8, 2, 7.01), oom_attempt(4, 4, 6.98), oom_attempt(2, 8, 6.97)]
    w.scenario("train.xlsr.6s.bs16.bf16", "oom", wall=80, detail="OOM at every rung (16x1 to 2x8)", attempts=oom)
    w.scenario("infer.xlsr.6s.bs16.bf16", "error", wall=14, detail="RuntimeError",
               error={"type": "RuntimeError", "message": "simulated: CUDA error: an illegal memory access was encountered",
                      "exit_code": 0, "traceback_tail": "Traceback (most recent call last):\n  File \"~/antispoof-gpu-bench/agbench/child.py\", line 1\nRuntimeError: simulated"})
    w.scenario("train.w2v2l.6s.bs16.bf16", "crashed", wall=9, detail="exit code -11, no result file",
               error={"type": "crashed", "message": "child exited with code -11 and wrote no result",
                      "exit_code": -11, "traceback_tail": ""})
    w.scenario("train.fusion.4s.bs16.bf16", "timeout", wall=240, detail="timeout after 240 s",
               error={"type": "timeout", "message": "attempt exceeded 240 s; process tree killed",
                      "exit_code": None, "traceback_tail": ""})
    w.scenario("frozen.xlsr.6s.bs16.bf16", "skipped", wall=0.2, detail="model_missing")
    w.scenario("train.w2v2b.10s.bs16.bf16", "skipped", wall=0.0, detail="budget")
    add_cpu(w, ["cpu.decode", "cpu.loader.w4.blas1", "cpu.loader.w8.blas1"], scale=0.8)
    w.run_end()
    w.write("rtx-5060-ti-8gb_20261017.jsonl")


if __name__ == "__main__":
    make_5060ti_16()
    make_3090()
    make_5060ti_8()
    print("fixtures written to", HERE.name)
