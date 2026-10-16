"""CPU data pipeline benchmarks (SPEC 6.1): FLAC decode, RawBoost, DataLoader workers.

None of these touch the GPU. BLAS thread settings come from the child's environment, which the
runner sets per scenario (``blas1``: ``OPENBLAS_NUM_THREADS=1``; ``blasdefault``: the four BLAS
thread variables removed). Each workload records what it actually saw (environment values and
``threadpoolctl`` info without file paths); if the environment does not match the scenario (for
example when called directly), it corrects it (``threadpoolctl`` in-process, ``os.environ`` for
spawned workers) and says so in ``extra.blas_fixup``.
"""

from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager

import numpy as np

from agbench.context import MeasureCfg
from agbench.errors import ScenarioError, SkipScenario
from agbench.util import median, percentile

BLAS_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "GOTO_NUM_THREADS")
SPIN_SETTLE_S = 0.15


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _log(ctx, msg: str) -> None:
    try:
        ctx.log(msg)
    except Exception:
        pass


def _seconds(ctx) -> float:
    s = float((ctx.params or {}).get("seconds", 6.0))
    return min(s, 2.0) if ctx.tiny else s


def logical_cores() -> int:
    """Logical CPUs this process may use (affinity-aware on Linux)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def physical_cores() -> int | None:
    try:
        import psutil

        return psutil.cpu_count(logical=False)
    except Exception:
        return None


def blas_mode(ctx) -> str | None:
    """'1', 'default' or None (not a BLAS-comparison scenario), from params or the id suffix."""
    v = (ctx.params or {}).get("blas")
    if v is None:
        sid = str(ctx.scenario_id)
        if sid.endswith(".blas1"):
            return "1"
        if sid.endswith(".blasdefault"):
            return "default"
        return None
    s = str(v).strip().lower()
    s = s[4:] if s.startswith("blas") else s
    if s in ("1", "one"):
        return "1"
    if s in ("default", "none", ""):
        return "default"
    return None


def blas_env() -> dict:
    return {v: os.environ.get(v) for v in BLAS_VARS}


def blas_env_matches(mode: str | None, env: dict) -> bool:
    if mode == "1":
        return env.get("OPENBLAS_NUM_THREADS") == "1"
    if mode == "default":
        return all(env.get(v) is None for v in BLAS_VARS)
    return True


def threadpool_summary() -> list[dict]:
    """threadpoolctl info without file paths (privacy)."""
    try:
        from threadpoolctl import threadpool_info

        keys = ("user_api", "internal_api", "prefix", "version", "num_threads", "threading_layer")
        return [{k: d.get(k) for k in keys if k in d} for d in threadpool_info()]
    except Exception:
        return []


@contextmanager
def _blas_in_process(mode: str | None, env_ok: bool):
    """Correct the in-process BLAS thread count when the environment does not match ``mode``."""
    if mode is None or env_ok:
        yield None
        return
    from threadpoolctl import threadpool_limits

    limit = 1 if mode == "1" else logical_cores()
    with threadpool_limits(limits=limit, user_api="blas"):
        yield f"threadpoolctl blas limit {limit} (environment did not match '{mode}')"


@contextmanager
def _blas_worker_env(mode: str | None):
    """Set the BLAS variables that spawned DataLoader workers inherit; restored afterwards."""
    saved = blas_env()
    fix = None
    try:
        if mode is not None and not blas_env_matches(mode, saved):
            if mode == "1":
                os.environ["OPENBLAS_NUM_THREADS"] = "1"
            else:
                for v in BLAS_VARS:
                    os.environ.pop(v, None)
            fix = f"worker environment set for '{mode}' (parent environment did not match)"
        yield fix
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def blas_spin_probe(n_samples: int = 96000, calls: int = 5,
                    settle_s: float = SPIN_SETTLE_S) -> dict:
    """CPU time burnt per ``np.linalg.norm`` call on a float64 vector, including the time BLAS
    worker threads keep spinning after the call returns (measured over ``settle_s`` of sleep).

    About 0 ms with one BLAS thread; up to ~100 ms per extra OpenBLAS thread when they spin.
    """
    x = np.random.default_rng(0).standard_normal(int(n_samples))
    np.linalg.norm(x, 2)
    time.sleep(settle_s)
    cpu_ms, call_ms = [], []
    for _ in range(int(calls)):
        c0 = time.process_time()
        w0 = time.perf_counter()
        np.linalg.norm(x, 2)
        call_ms.append((time.perf_counter() - w0) * 1e3)
        time.sleep(settle_s)
        cpu_ms.append((time.process_time() - c0) * 1e3)
    return {"calls": int(calls), "n_samples": int(n_samples), "settle_ms": settle_s * 1e3,
            "cpu_ms_per_call_median": round(median(cpu_ms), 3),
            "call_ms_median": round(median(call_ms), 4)}


def _tree_cpu_s() -> float:
    """CPU seconds of this process plus its live child processes (DataLoader workers)."""
    total = time.process_time()
    try:
        import psutil

        for c in psutil.Process().children(recursive=True):
            try:
                t = c.cpu_times()
                total += t.user + t.system
            except Exception:
                pass
    except Exception:
        pass
    return total


def _cpu_cfg(ctx, *, warmup_s: float, min_s: float, max_s: float, min_items: int,
             warmup_items: int = 3) -> MeasureCfg:
    """Shorter windows for the single-process CPU micro-benchmarks (per-item steps)."""
    if ctx.tiny:
        return MeasureCfg.tiny()
    base = ctx.measure if isinstance(ctx.measure, MeasureCfg) else MeasureCfg()
    return MeasureCfg(warmup_steps=warmup_items, warmup_min_s=min(base.warmup_min_s, warmup_s),
                      warmup_max_s=base.warmup_max_s,
                      measure_steps=max(min_items, base.measure_steps),
                      measure_min_s=min(base.measure_min_s, min_s),
                      measure_max_s=min(base.measure_max_s, max_s))


_LOOP_KEYS = ("clips_per_s", "clips_per_s_wall", "ms_per_clip_median", "cpu_ms_per_clip",
              "cpu_wall_ratio", "n_steps", "step_ms_p10", "step_ms_p90", "step_ms_cv",
              "warmup_steps_done", "warmup_s", "measure_s")


def _loop_metrics(m: dict) -> dict:
    return {k: m.get(k) for k in _LOOP_KEYS}


def _corpus(ctx):
    from agbench.data.synth import ensure_corpus

    return ensure_corpus(ctx.cache_dir, tiny=bool(ctx.tiny))


def _cores_extra() -> dict:
    return {"cpu_logical": os.cpu_count(), "cpu_available": logical_cores(),
            "cpu_physical": physical_cores()}


# ---------------------------------------------------------------------------------------------
# cpu.decode
# ---------------------------------------------------------------------------------------------


def run_decode(ctx) -> dict:
    """FLAC decode in one process: dataset-style items (whole-file decode + loop + crop), plus
    full-file decode ms per audio second and seek-based crop decode for comparison."""
    import soundfile as sf

    from agbench.data.dataset import SynthFlacDataset
    from agbench.data.synth import load_manifest
    from agbench.timing import cpu_loop

    seconds = _seconds(ctx)
    manifest = _corpus(ctx)
    ds = SynthFlacDataset(manifest, seconds=seconds, train=True, rawboost_p=0.0, seed=ctx.seed)
    order = np.random.default_rng(ctx.seed).permutation(len(ds))
    state = {"k": 0}

    def one_item():
        i = int(order[state["k"] % len(order)])
        state["k"] += 1
        ds[i]

    cfg = _cpu_cfg(ctx, warmup_s=1.0, min_s=4.0, max_s=8.0, min_items=40)
    m = cpu_loop(one_item, items_per_call=1, cfg=cfg)
    _log(ctx, f"decode items: {m['clips_per_s']:.1f} clips/s, "
              f"{m['ms_per_clip_median']:.2f} ms/clip median")

    # Full-file decode cost per second of audio.
    info = load_manifest(manifest)
    files = [(os.path.join(os.path.dirname(manifest), f["path"]), float(f["seconds"]))
             for f in info["files"]]
    budget_s = 0.2 if ctx.tiny else 1.5
    dec_s = aud_s = 0.0
    k = 0
    t_start = time.perf_counter()
    while k < len(files) and (k < 8 or time.perf_counter() - t_start < budget_s):
        path, secs = files[k]
        t0 = time.perf_counter()
        sf.read(path, dtype="float32")
        dec_s += time.perf_counter() - t0
        aud_s += secs
        k += 1
    decode_ms_per_audio_s = dec_s * 1e3 / aud_s if aud_s > 0 else None

    # Seek-based crop decode (reads only the crop's frames), for comparison.
    n = int(round(seconds * info["sr"]))
    seek_ms = []
    rng = np.random.default_rng(ctx.seed + 1)
    t_start = time.perf_counter()
    for path, secs in files:
        total = int(round(secs * info["sr"]))
        if total <= n:
            continue
        off = int(rng.integers(0, total - n + 1))
        t0 = time.perf_counter()
        sf.read(path, start=off, frames=n, dtype="float32")
        seek_ms.append((time.perf_counter() - t0) * 1e3)
        if len(seek_ms) >= 8 and time.perf_counter() - t_start > budget_s / 2:
            break

    metrics = {**_loop_metrics(m), "num_workers": 0, "worker_startup_s": 0.0,
               "decode_ms_per_audio_s": decode_ms_per_audio_s}
    extra = {
        "seconds": seconds, "processes": 1, "corpus_files": len(files),
        "corpus_mb": round(info.get("total_bytes", 0) / 1e6, 2),
        "flac_kbps": round(info["total_bytes"] * 8 / info["total_seconds"] / 1e3, 1)
        if info.get("total_seconds") else None,
        "full_decode_files": k, "full_decode_audio_s": round(aud_s, 2),
        "seek_crop_ms_median": round(median(seek_ms), 3) if seek_ms else None,
        "seek_crop_n": len(seek_ms),
        "libsndfile": getattr(sf, "__libsndfile_version__", None),
        "soundfile": getattr(sf, "__version__", None),
        "measure_effective": cfg.to_dict(),
        "note": "files come from the OS page cache (corpus written just before); item = "
                "whole-file decode + loop/crop + tensor, like the training loader",
        **_cores_extra(),
    }
    return {"metrics": metrics, "extra": extra, "timeseries": None}


# ---------------------------------------------------------------------------------------------
# cpu.rawboost
# ---------------------------------------------------------------------------------------------


def run_rawboost(ctx) -> dict:
    """RawBoost on pre-decoded crops in one process: clips/s, per-stage ms, CPU/wall ratio."""
    from agbench.data.dataset import SynthFlacDataset
    from agbench.data.rawboost import isd, lnl, rawboost, ssi
    from agbench.timing import cpu_loop

    p = ctx.params or {}
    seconds = _seconds(ctx)
    algo = int(p.get("algo", 4))
    prob = float(p.get("p", p.get("rawboost_p", 1.0)))
    mode = blas_mode(ctx)
    env = blas_env()
    env_ok = blas_env_matches(mode, env)

    manifest = _corpus(ctx)
    ds = SynthFlacDataset(manifest, seconds=seconds, train=True, rawboost_p=0.0, seed=ctx.seed)
    crops = [ds.load_crop(i)[0] for i in range(min(4 if ctx.tiny else 16, len(ds)))]

    stage_ms: dict[str, list[float]] = {"lnl": [], "isd": [], "ssi": []}
    stage_cpu: dict[str, list[float]] = {"lnl": [], "isd": [], "ssi": []}
    state = {"k": 0, "aug": 0}

    def one_item():
        k = state["k"]
        state["k"] += 1
        x = crops[k % len(crops)]
        rng = np.random.default_rng([int(ctx.seed), k])
        if prob < 1.0 and rng.random() >= prob:
            return
        state["aug"] += 1
        if algo != 4:
            rawboost(x, rng, algo=algo)
            return
        t0, c0 = time.perf_counter(), time.process_time()
        y = lnl(x, rng)
        t1, c1 = time.perf_counter(), time.process_time()
        y = isd(y, rng)
        t2, c2 = time.perf_counter(), time.process_time()
        y = ssi(y, rng)
        y = y.astype(np.float32)
        t3, c3 = time.perf_counter(), time.process_time()
        for name, a, b, ca, cb in (("lnl", t0, t1, c0, c1), ("isd", t1, t2, c1, c2),
                                   ("ssi", t2, t3, c2, c3)):
            stage_ms[name].append((b - a) * 1e3)
            stage_cpu[name].append((cb - ca) * 1e3)

    cfg = _cpu_cfg(ctx, warmup_s=1.0, min_s=8.0, max_s=12.0, min_items=30)
    with _blas_in_process(mode, env_ok) as fix:
        threads = threadpool_summary()
        m = cpu_loop(one_item, items_per_call=1, cfg=cfg)
        probe = blas_spin_probe(n_samples=crops[0].shape[0], calls=2 if ctx.tiny else 5)
    n_meas = int(m.get("n_steps") or 0)

    def _stage_median(d: dict) -> dict | None:
        if algo != 4 or not d["lnl"]:
            return None
        return {s: round(median(v[-n_meas:] if n_meas else v), 3) for s, v in d.items()}

    metrics = {**_loop_metrics(m), "num_workers": 0, "worker_startup_s": 0.0,
               "stage_ms": _stage_median(stage_ms)}
    _log(ctx, f"rawboost algo {algo}: {m['clips_per_s']:.1f} clips/s, cpu/wall "
              f"{m['cpu_wall_ratio']:.2f}, stages {metrics['stage_ms']}, "
              f"spin {probe['cpu_ms_per_call_median']:.1f} ms CPU per norm call")
    extra = {
        "algo": algo, "p": prob, "seconds": seconds, "n_crops": len(crops),
        "stage_cpu_ms": _stage_median(stage_cpu),
        "blas_mode": mode, "blas_env": env, "blas_env_matches": env_ok, "blas_fixup": fix,
        "threadpool": threads, "blas_spin_probe": probe,
        "measure_effective": cfg.to_dict(),
        "note": "decode excluded (crops decoded once); CPU time includes BLAS helper threads, "
                "so cpu_wall_ratio > 1 means threads spin besides the main thread",
        **_cores_extra(),
    }
    return {"metrics": metrics, "extra": extra, "timeseries": None}


# ---------------------------------------------------------------------------------------------
# cpu.loader
# ---------------------------------------------------------------------------------------------


def _workers(ctx) -> int:
    p = ctx.params or {}
    v = p.get("workers", p.get("num_workers"))
    if v is None:
        m = re.search(r"\.w(\d+)(\.|$)", str(ctx.scenario_id))
        v = m.group(1) if m else 4
    return int(v)


def loader_plan(min_batches: int, workers: int, prefetch: int) -> tuple[int, int]:
    """(batches to time, backlog batches to leave out of the rate) for ``cpu.loader``.

    When timing starts, workers whose start-up finished early may each hold up to ``prefetch``
    finished batches: up to ``prefetch * workers`` batches arrive "for free". They are timed but
    left out of the steady-state rate. The steady part has at least ``max(min_batches, 6 *
    workers)`` batches, and both parts are whole multiples of ``workers``, because workers that
    start together deliver in bursts of ``workers`` batches.
    """
    w = max(1, int(workers))
    backlog = max(0, int(prefetch)) * w if workers > 0 else 0
    steady = w * max(-(-int(min_batches) // w), 6)
    return backlog + steady, backlog


def steady_rate(arrivals: list[float], batch: int, workers: int, backlog: int) -> tuple[float | None, int]:
    """(clips/s, batches used) between the end of the backlog and the last whole burst.

    ``arrivals[0]`` is the start of timing, ``arrivals[k]`` the arrival of the k-th timed batch.
    Falls back to all timed batches when too few arrived after the backlog (time cap reached).
    """
    w = max(1, int(workers))
    n = len(arrivals) - 1
    start = backlog
    end = start + ((n - start) // w) * w if n > start else start
    if end - start < w:
        start, end = 0, n
    dt = arrivals[end] - arrivals[start] if end > start else 0.0
    if dt <= 0:
        return None, 0
    return (end - start) * batch / dt, end - start


def run_loader(ctx) -> dict:
    """DataLoader throughput with N spawn workers (bs16, 6 s, RawBoost p=0.5), no GPU consumer."""
    from agbench.data.dataset import SynthFlacDataset, make_loader

    p = ctx.params or {}
    workers = _workers(ctx)
    if ctx.tiny:
        workers = min(workers, 2)
    avail = logical_cores()
    if workers > avail:
        raise SkipScenario("workers_exceed_cores", f"{workers} workers > {avail} logical cores")
    batch = int(p.get("batch", 16))
    if ctx.tiny:
        batch = min(batch, 2)
    seconds = _seconds(ctx)
    rb_p = float(p.get("rawboost_p", 0.5))
    algo = int(p.get("algo", 4))
    pin = bool(p.get("pin_memory", False))
    persistent = bool(p.get("persistent_workers", False))
    prefetch = int(p.get("prefetch_factor", 2))
    if ctx.tiny:
        warm, meas, backlog, cap_s = max(1, workers), 3, 0, 60.0
    else:
        warm = max(int(p.get("warmup_batches", 2)), workers)  # every worker delivered once
        meas, backlog = loader_plan(int(p.get("measure_batches", 30)), workers, prefetch)
        cap_s = float(p.get("cap_s", 60.0))
    mode = blas_mode(ctx)
    env = blas_env()

    manifest = _corpus(ctx)
    ds = SynthFlacDataset(manifest, seconds=seconds, train=True, rawboost_p=rb_p, seed=ctx.seed,
                          rawboost_algo=algo)
    loader = make_loader(ds, batch_size=batch, num_workers=workers, pin_memory=pin,
                         persistent_workers=persistent and workers > 0, prefetch_factor=prefetch,
                         shuffle=True, num_samples=batch * (warm + meas), seed=ctx.seed)
    waits: list[float] = []
    capped = False
    with _blas_worker_env(mode) as fix:
        threads = threadpool_summary()
        t0 = time.perf_counter()
        it = iter(loader)
        try:
            xb, yb = next(it)
            startup = time.perf_counter() - t0
            for _ in range(warm - 1):
                next(it)
            _log(ctx, f"loader: {workers} workers, first batch after {startup:.1f} s")
            c0 = _tree_cpu_s()
            w0 = time.perf_counter()
            arrivals = [0.0]
            for k in range(meas):
                s = time.perf_counter()
                xb, yb = next(it)
                now = time.perf_counter()
                waits.append(now - s)
                arrivals.append(now - w0)
                if now - w0 >= cap_s and k + 1 >= min(5, meas):
                    capped = k + 1 < meas
                    break
            wall = time.perf_counter() - w0
            cpu_s = max(0.0, _tree_cpu_s() - c0)
        finally:
            del it
    n = int(round(seconds * ds.sr))
    if tuple(xb.shape) != (batch, n) or str(xb.dtype) != "torch.float32" \
            or tuple(yb.shape) != (batch,) or str(yb.dtype) != "torch.int64":
        raise ScenarioError(f"unexpected batch: {tuple(xb.shape)} {xb.dtype}, "
                            f"{tuple(yb.shape)} {yb.dtype}")
    clips = batch * len(waits)
    cps_wall = clips / wall if wall > 0 else None
    cps, steady_batches = steady_rate(arrivals, batch, workers, backlog)
    if cps is None:
        cps = cps_wall
    # Batches arrive in bursts (one per worker); per-clip time over sliding windows of
    # ``workers`` batches is a stable per-clip figure whose median is meaningful.
    win = max(1, workers)
    per_clip_ms = [(arrivals[k + win] - arrivals[k]) * 1e3 / (win * batch)
                   for k in range(len(arrivals) - win)]
    metrics = {
        "clips_per_s": cps, "clips_per_s_wall": cps_wall,
        "ms_per_clip_median": (median(per_clip_ms) if per_clip_ms
                               else (wall * 1e3 / clips if clips else None)),
        "cpu_ms_per_clip": cpu_s * 1e3 / clips if clips else None,
        "cpu_wall_ratio": cpu_s / wall if wall > 0 else None,
        "num_workers": workers, "worker_startup_s": startup,
        "n_steps": len(waits), "measure_s": wall, "warmup_steps_done": warm,
        "step_ms_p10": percentile([w * 1e3 for w in waits], 10),
        "step_ms_p90": percentile([w * 1e3 for w in waits], 90),
    }
    _log(ctx, f"loader: {cps:.1f} clips/s over {len(waits)} batches, cpu/wall "
              f"{metrics['cpu_wall_ratio']:.2f}")
    extra = {
        "batch": batch, "seconds": seconds, "rawboost_p": rb_p, "algo": algo,
        "warmup_batches": warm, "measure_batches": meas, "measured_batches": len(waits),
        "backlog_batches": backlog, "steady_batches": steady_batches,
        "cap_s": cap_s, "capped": capped, "pin_memory": pin, "persistent_workers": persistent,
        "prefetch_factor": prefetch, "multiprocessing_context": "spawn" if workers else None,
        "clips_per_s_per_worker": (cps / workers) if (cps and workers) else None,
        "blas_mode": mode, "blas_env": env, "blas_env_matches": blas_env_matches(mode, env),
        "blas_fixup": fix, "threadpool_parent": threads,
        "step_ms_meaning": "time blocked in next(iterator) per batch",
        "note": "no GPU consumer: this is the loader's maximum rate; CPU time = this process + "
                "worker processes; warm-up = max(2, workers) batches so every worker has "
                "finished its start-up; clips_per_s = steady-state rate after the first "
                "prefetch_factor x workers timed batches (which may have been ready before timing "
                "started), over whole bursts of num_workers batches; clips_per_s_wall = all timed "
                "batches; ms_per_clip_median = median over sliding windows of num_workers batches",
        **_cores_extra(),
    }
    return {"metrics": metrics, "extra": extra, "timeseries": None}
