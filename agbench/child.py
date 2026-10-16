"""Child entry point (SPEC 8.2): ``python -m agbench.child --scenario ID --attempt k ...``.

Import-light: only the standard library at module level. torch, the workload module and the
telemetry sampler are imported inside main(), so DataLoader spawn workers that re-import
``__mp_main__`` stay cheap and side-effect free (Windows).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path

# Error texts that mean "ran out of GPU memory". Near the memory cap, cuDNN v9 often reports
# exhaustion as "no valid algorithm / no engine", and cuBLAS handle creation as NOT_INITIALIZED.
OOM_MARKERS = ("out of memory", "cublas_status_alloc_failed", "cudnn_status_alloc_failed",
               "unable to find a valid cudnn algorithm", "unable to find an engine",
               "cublas_status_not_initialized", "cudnn_status_not_supported")
# torch.use_deterministic_algorithms(True) raises this for an op without a deterministic kernel.
NONDETERMINISTIC_MARKER = "does not have a deterministic implementation"
MAX_TELEMETRY_POINTS = 600
PARENT_ENV = "AGBENCH_PARENT"  # "<pid>:<create_time>" of the runner, set by the runner
PARENT_EXIT_CODE = 5


def _parse(argv):
    p = argparse.ArgumentParser(prog="python -m agbench.child")
    p.add_argument("--scenario", required=True)
    p.add_argument("--attempt", type=int, default=1)
    p.add_argument("--micro-batch", type=int, default=None)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--smoke", action="store_true", help="real weights, tiny shapes and loops")
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--gpu", type=int, default=0, help="physical GPU index (NVML order)")
    p.add_argument("--mem-cap-gib", type=float, default=None)
    return p.parse_args(argv)


def watch_parent(env_value: str | None = None, interval_s: float = 1.0) -> bool:
    """Exit this process (``os._exit(5)``) as soon as the runner that started it is gone.

    A tool timeout or a killed console kills the runner but not its children (they run in their
    own process group so that a timeout can kill a whole tree). A leftover child would keep GPU
    memory and compute while the rerun measures. Linux: ``PR_SET_PDEATHSIG`` (immediate);
    every OS: a daemon thread that checks the runner's pid + start time every second.
    Returns True if watching started.
    """
    value = os.environ.get(PARENT_ENV) if env_value is None else env_value
    if not value:
        return False
    try:
        pid_s, _, ctime_s = value.partition(":")
        ppid, ctime = int(pid_s), (float(ctime_s) if ctime_s else None)
    except ValueError:
        return False
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            import signal

            libc = ctypes.CDLL(None, use_errno=True)
            libc.prctl(1, int(signal.SIGKILL), 0, 0, 0)  # PR_SET_PDEATHSIG
        except Exception:
            pass
    from .util import pid_alive

    if not pid_alive(ppid, ctime):
        os._exit(PARENT_EXIT_CODE)

    def _loop() -> None:
        while True:
            time.sleep(interval_s)
            if not pid_alive(ppid, ctime):
                try:
                    print("agbench child: the runner is gone; exiting", flush=True)
                finally:
                    os._exit(PARENT_EXIT_CODE)

    import threading

    threading.Thread(target=_loop, name="agbench-parent-watch", daemon=True).start()
    return True


def mem_cap_margin_gib(platform: str | None = None) -> float:
    """VRAM left unused below the free memory at start: Windows keeps a margin against the
    driver's silent spill into system memory; on Linux an allocator OOM is clean."""
    from . import constants as C

    plat = sys.platform if platform is None else platform
    return C.MEM_CAP_MARGIN_GIB_WINDOWS if plat.startswith("win") else C.MEM_CAP_MARGIN_GIB_OTHER


def is_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.OutOfMemoryError):
            return True
    except Exception:
        pass
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return any(m in msg for m in OOM_MARKERS)
    return False


def error_info(exc: BaseException, exit_code: int | None = None) -> dict:
    from .util import sanitize

    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    tail = "\n".join(tb.rstrip().splitlines()[-40:])
    return {"type": type(exc).__name__, "message": sanitize(str(exc))[:2000], "exit_code": exit_code,
            "traceback_tail": sanitize(tail)}


def resolve_workers(value, tiny: bool) -> int:
    """'auto' = min(8, max(1, physical_cores - 1)) (SPEC 6.1); tiny caps at 2."""
    from . import constants as C

    if value == "auto":
        phys = None
        try:
            import psutil

            phys = psutil.cpu_count(logical=False)
        except Exception:
            pass
        phys = phys or os.cpu_count() or 2
        value = min(8, max(1, phys - 1))
    value = int(value)
    return min(value, C.TINY_MAX_WORKERS) if tiny else value


def compute_estimates(category: str, metrics: dict, seconds) -> dict:
    """ASVspoof 5 time estimates from clips/s (SPEC 9.2); null where not applicable."""
    from . import constants as C
    from .schema import empty_estimates

    est = empty_estimates()
    cps = metrics.get("clips_per_s")
    if not isinstance(cps, (int, float)) or not cps or not math.isfinite(cps) or cps <= 0:
        return est
    est["assumes_seconds"] = float(seconds) if seconds is not None else None
    if category in ("train", "e2e", "sustain", "frozen"):
        est["asv5_train_epoch_min"] = C.ASV5_TRAIN_CLIPS / cps / 60.0
    if category == "infer":
        est["asv5_dev_pass_min"] = C.ASV5_DEV_CLIPS / cps / 60.0
        est["asv5_eval_pass_min"] = C.ASV5_EVAL_CLIPS / cps / 60.0
    return est


def downsample(series: list, max_points: int = MAX_TELEMETRY_POINTS) -> list:
    if not series or len(series) <= max_points:
        return list(series or [])
    stride = math.ceil(len(series) / max_points)
    return list(series[::stride])


def merge_windows(windows: list, series: list, offset_s: float) -> list:
    """Attach mean temperature / SM clock / power of the telemetry samples inside each window.

    ``offset_s`` = start of the measured part on the sampler's time axis.
    """
    out = []
    prev_end = 0.0
    for w in windows or []:
        if not isinstance(w, dict):
            continue
        t_end = float(w.get("t_s", 0.0))
        t_start = float(w.get("t_start_s", prev_end))
        prev_end = t_end
        a, b = offset_s + t_start, offset_s + t_end
        inside = [s for s in series or [] if isinstance(s.get("t_s"), (int, float)) and a <= s["t_s"] <= b]

        def mean(key):
            vals = [s.get(key) for s in inside if isinstance(s.get(key), (int, float))]
            return sum(vals) / len(vals) if vals else None

        rec = {"t_s": t_end, "clips_per_s": w.get("clips_per_s"), "temp_c": mean("temp_c"),
               "sm_clock_mhz": mean("sm_clock_mhz"), "power_w": mean("power_w")}
        for k in ("t_start_s", "steps", "step_ms_median"):
            if k in w:
                rec[k] = w[k]
        out.append(rec)
    return out


def _simulated_fault(scenario_id: str) -> None:
    spec = os.environ.get("AGBENCH_SIMULATE_FAULT", "")
    if not spec:
        return
    sid, _, kind = spec.rpartition(":")
    if sid != scenario_id:
        return
    if kind == "crash":
        sys.stdout.flush()
        os._exit(3)
    if kind == "hang":
        while True:
            time.sleep(3600)
    if kind == "error":
        raise RuntimeError("simulated")


def _simulated_oom(micro_batch: int | None) -> None:
    thr = os.environ.get("AGBENCH_SIMULATE_OOM_MB_GE")
    if not thr or micro_batch is None:
        return
    if int(micro_batch) >= int(thr):
        import torch

        raise torch.OutOfMemoryError("simulated")


def run_attempt(args, result: dict) -> None:
    """Fill ``result`` (the attempt JSON) for one attempt. Exceptions propagate to main()."""
    from . import constants as C
    from . import registry
    from .context import MeasureCfg, RunContext
    from .errors import SkipScenario
    from .scenarios import SCENARIOS, effective
    from .timing import peak_memory_gib, reset_peak_memory
    from .util import default_cache_dir, sanitize

    def log(msg: str) -> None:
        print(sanitize(str(msg)).encode("ascii", "replace").decode("ascii"), flush=True)

    if args.scenario not in SCENARIOS:
        raise SkipScenario("unknown_scenario", args.scenario)
    s = effective(SCENARIOS[args.scenario], tiny=args.tiny, device=args.device, smoke=args.smoke)
    device = args.device if s.needs_gpu or s.has_micro_batch() else "cpu"
    if s.category == "cpu":
        device = "cpu"
    result["device"] = device

    params = dict(s.params)
    if s.has_micro_batch():
        mb = args.micro_batch if args.micro_batch is not None else int(params["batch"])
        params["micro_batch"], params["accum"] = int(mb), int(args.accum)
    if "num_workers" in params:
        params["num_workers"] = resolve_workers(params["num_workers"], args.tiny or args.smoke)
        result["extra"]["num_workers_resolved"] = params["num_workers"]
    result["micro_batch"] = params.get("micro_batch")
    result["accum"] = params.get("accum")
    measure = s.measure or (MeasureCfg.tiny() if args.tiny else MeasureCfg())
    result["measure"] = measure.to_dict()

    log(f"agbench child: {s.id} attempt {args.attempt} device={device} "
        f"micro_batch={params.get('micro_batch')} accum={params.get('accum')} tiny={args.tiny}"
        + (" smoke=True" if args.smoke else ""))

    import torch

    seed = int(params.get("seed", C.DEFAULT_SEED))
    torch.manual_seed(seed)
    import random

    random.seed(seed)

    gpu_uuid = None
    cap_gib = None
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SkipScenario("no_gpu", "torch.cuda.is_available() is False")
        free, total = torch.cuda.mem_get_info()
        cap_gib = free / C.GIB - mem_cap_margin_gib()
        if args.mem_cap_gib:
            cap_gib = min(cap_gib, float(args.mem_cap_gib))
        frac = max(0.01, min(1.0, cap_gib * C.GIB / total))
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        result["memory"].update({"vram_total_gib": total / C.GIB, "vram_free_start_gib": free / C.GIB,
                                 "mem_cap_gib": cap_gib, "mem_cap_margin_gib": mem_cap_margin_gib()})
        from .env import gpu_uuid as _uuid

        gpu_uuid = _uuid()
        log(f"VRAM {total / C.GIB:.2f} GiB, free {free / C.GIB:.2f} GiB, cap {cap_gib:.2f} GiB")
    if params.get("deterministic"):
        # Strict, as in the reference training runs; a workload may fall back to warn-only below.
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
    if s.workload == "basics.pcie" and device != "cuda":
        raise SkipScenario("cpu_device", "PCIe transfer test needs a GPU")

    ctx = RunContext(scenario_id=s.id, workload=s.workload, params=params, device=device,
                     tiny=bool(args.tiny), measure=measure, seed=seed, cache_dir=default_cache_dir(),
                     keep_series=(s.category == "sustain"), log=log, gpu_index=int(args.gpu),
                     gpu_uuid=gpu_uuid, attempt=int(args.attempt))

    sampler = None
    t_sampler0 = time.perf_counter()
    if device == "cuda":
        try:
            from .telemetry import GpuSampler

            sampler = GpuSampler(device_index=int(args.gpu), interval_s=0.5,
                                 keep_series=True, uuid=gpu_uuid)
            t_sampler0 = time.perf_counter()
            sampler.__enter__()
            t_sampler0 = float(getattr(sampler, "t0_perf", t_sampler0))  # sample t_s are relative to it
        except Exception as e:
            sampler = None
            result["telemetry"] = {"source": "none", "error": f"{type(e).__name__}: {e}"[:300]}

    ret = None
    t_work_end = None
    try:
        _simulated_fault(s.id)
        _simulated_oom(params.get("micro_batch"))
        fn = registry.resolve(s.workload)
        reset_peak_memory(device)
        nondet = None
        try:
            ret = fn(ctx)
        except RuntimeError as e:
            if not (params.get("deterministic") and NONDETERMINISTIC_MARKER in str(e)) or is_oom(e):
                raise
            nondet = str(e).splitlines()[0][:300]
        if nondet is not None:
            # An op without a deterministic kernel in this torch build: rerun once in warn-only
            # mode (deterministic kernels wherever they exist) and say so in the record.
            log(f"strict deterministic mode failed ({nondet}); rerunning with warn_only=True")
            import gc

            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            reset_peak_memory(device)
            ctx.params["deterministic_warn_only"] = True
            ret = fn(ctx)
            result["extra"]["deterministic_strict_failed"] = sanitize(nondet)
        t_work_end = time.perf_counter()
    finally:
        series = []
        if sampler is not None:
            try:
                sampler.__exit__(None, None, None)
                result["telemetry"] = sampler.summary() or {"source": "none"}
                series = sampler.series() or []
            except Exception as e:
                result["telemetry"] = {"source": "none", "error": f"{type(e).__name__}: {e}"[:300]}
        peak_a, peak_r = peak_memory_gib(device)
        result["memory"].update({"peak_alloc_gib": peak_a, "peak_reserved_gib": peak_r})
        result["_series"] = series

    # ---- status ok: assemble metrics ----------------------------------------------------------
    ret = ret or {}
    metrics = dict(ret.get("metrics") or {})
    extra = result["extra"]
    extra.update(ret.get("extra") or {})
    t0_perf = metrics.pop("t0_perf", None)
    mem = result["memory"]
    if s.has_micro_batch():
        metrics.setdefault("effective_batch", params["micro_batch"] * params["accum"])
    metrics.update({k: mem.get(k) for k in ("peak_alloc_gib", "peak_reserved_gib", "vram_total_gib",
                                            "vram_free_start_gib", "mem_cap_gib")})
    near = bool(cap_gib and mem.get("peak_reserved_gib") is not None
                and mem["peak_reserved_gib"] > C.NEAR_LIMIT_FRAC * cap_gib)
    metrics["near_limit"] = near
    sysmem = False
    if sys.platform.startswith("win") and device == "cuda":
        used_max = ((result.get("telemetry") or {}).get("mem_used_gib") or {}).get("max")
        total = mem.get("vram_total_gib")
        if isinstance(used_max, (int, float)) and total and used_max > total * 1.001:
            sysmem = True
        cv = metrics.get("step_ms_cv")
        if near and isinstance(cv, (int, float)) and cv > 0.5:
            sysmem = True
    metrics["sysmem_suspected"] = sysmem
    for k in ("warmup_steps_done", "warmup_s", "n_steps", "measure_s"):
        if k in metrics:
            result["measure"][k] = metrics[k]
    result["metrics"] = metrics
    result["estimates"] = compute_estimates(s.category, metrics, params.get("seconds"))

    windows = ret.get("timeseries")
    series = result.get("_series") or []
    if windows:
        if t0_perf is not None:
            offset = t0_perf - t_sampler0
        else:  # assume the windows end when the workload returned
            end_rel = (t_work_end or time.perf_counter()) - t_sampler0
            offset = end_rel - float(windows[-1].get("t_s", 0.0))
        result["timeseries"] = merge_windows(windows, series, offset)
    if s.category == "sustain" and series:
        extra["telemetry_series"] = downsample(series)
    result["status"] = "ok"


def main(argv=None) -> int:
    args = _parse(argv)
    t0 = time.perf_counter()
    watch_parent()
    from .util import atomic_write_json, kill_tree, sanitize_obj, utc_now_iso

    result: dict = {
        "scenario_id": args.scenario, "attempt": args.attempt, "micro_batch": args.micro_batch,
        "accum": args.accum, "status": "error", "status_detail": "", "started_utc": utc_now_iso(),
        "metrics": {}, "measure": {}, "estimates": None, "telemetry": {"source": "none"},
        "timeseries": None, "extra": {}, "error": None,
        "memory": {"peak_alloc_gib": None, "peak_reserved_gib": None, "vram_total_gib": None,
                   "vram_free_start_gib": None, "mem_cap_gib": None},
    }
    try:
        run_attempt(args, result)
    except BaseException as e:  # noqa: BLE001 - classify everything, including KeyboardInterrupt
        from .errors import SkipScenario

        if isinstance(e, SkipScenario):
            result["status"], result["status_detail"] = "skipped", e.reason
            if e.detail:
                result["extra"]["skip_detail"] = e.detail
        elif is_oom(e):
            result["status"], result["status_detail"] = "oom", "out of memory"
            result["extra"]["oom_message"] = str(e).splitlines()[0][:300] if str(e) else ""
        else:
            result["status"] = "error"
            result["status_detail"] = f"{type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"
            result["error"] = error_info(e)
        if result["status"] != "ok":
            result["metrics"] = {}
        traceback.print_exc()
    result.pop("_series", None)
    result["finished_utc"] = utc_now_iso()
    result["wall_s"] = time.perf_counter() - t0
    try:
        atomic_write_json(Path(args.out), sanitize_obj(result))
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(4)
    print(f"agbench child: status {result['status']} {result.get('status_detail', '')}", flush=True)
    # Leave no stray DataLoader workers behind, and skip interpreter teardown (CUDA / NVML / worker
    # joins can stall at exit); the attempt JSON is already on disk.
    try:
        kill_tree(os.getpid(), timeout=3.0, include_parent=False)
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
