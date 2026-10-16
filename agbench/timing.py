"""Step timing (SPEC 5.5): warm-up, CUDA-event timed steps, medians; CPU loops; memory stats.

torch is imported lazily (only when a CUDA device is used), so this module stays importable in
the runner process and in CPU-only code paths.
"""

from __future__ import annotations

import math
import os
import statistics
import time
from typing import Callable

from .context import MeasureCfg
from .util import median, percentile

GIB = 1024 ** 3


def _is_cuda(device: str) -> bool:
    return str(device).startswith("cuda")


def _sync(device: str) -> None:
    if _is_cuda(device):
        import torch

        torch.cuda.synchronize()


def _warmup(step_fn: Callable[[], None], device: str, cfg: MeasureCfg,
            log: Callable[[str], None] | None) -> tuple[int, float]:
    t0 = time.perf_counter()
    n = 0
    while True:
        step_fn()
        n += 1
        el = time.perf_counter() - t0
        if n >= cfg.warmup_steps and el >= cfg.warmup_min_s:
            break
        if el >= cfg.warmup_max_s:
            break
    _sync(device)
    warmup_s = time.perf_counter() - t0
    if log:
        log(f"warm-up done: {n} steps in {warmup_s:.1f} s")
    return n, warmup_s


def summarize_steps(step_ms: list[float], items_per_step: float, wall_s: float) -> dict:
    """The per-step statistics of SPEC 5.5 from a list of step times in ms."""
    n = len(step_ms)
    med = median(step_ms) if n else None
    mean = statistics.fmean(step_ms) if n else None
    cv = (statistics.pstdev(step_ms) / mean) if (n > 1 and mean) else 0.0
    return {
        "step_ms_median": med,
        "step_ms_p10": percentile(step_ms, 10) if n else None,
        "step_ms_p90": percentile(step_ms, 90) if n else None,
        "step_ms_mean": mean,
        "step_ms_cv": cv,
        "n_steps": n,
        "items_per_step": items_per_step,
        "clips_per_s": (items_per_step * 1000.0 / med) if med else None,
        "clips_per_s_wall": (items_per_step * n / wall_s) if wall_s > 0 else None,
    }


def measure_loop(step_fn: Callable[[], None], *, device: str, items_per_step: int, cfg: MeasureCfg,
                 log: Callable[[str], None] | None = None) -> dict:
    """Warm up, then time ``step_fn`` per step (CUDA events on GPU, perf_counter on CPU).

    Returns the keys of SPEC 5.5: step_ms_median/p10/p90/mean/cv, n_steps, clips_per_s,
    clips_per_s_wall, warmup_steps_done, warmup_s, measure_s (plus items_per_step).
    """
    cuda = _is_cuda(device)
    if cuda:
        import torch
    n_warm, warmup_s = _warmup(step_fn, device, cfg, log)

    floor = max(1, min(5, cfg.measure_steps))
    events: list = []
    host_ms: list[float] = []
    _sync(device)
    w0 = time.perf_counter()
    k = 0
    while True:
        if cuda:
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            step_fn()
            e1.record()
            events.append((e0, e1))
        else:
            s = time.perf_counter()
            step_fn()
            host_ms.append((time.perf_counter() - s) * 1000.0)
        k += 1
        el = time.perf_counter() - w0
        if k >= cfg.measure_steps and el >= cfg.measure_min_s:
            break
        if el >= cfg.measure_max_s and k >= floor:
            break
    _sync(device)
    wall = time.perf_counter() - w0
    step_ms = [a.elapsed_time(b) for a, b in events] if cuda else host_ms
    out = summarize_steps(step_ms, items_per_step, wall)
    out.update({"warmup_steps_done": n_warm, "warmup_s": warmup_s, "measure_s": wall})
    if log:
        cps = out["clips_per_s"]
        log(f"measured {k} steps in {wall:.1f} s: median {out['step_ms_median']:.2f} ms/step, "
            f"{cps:.2f} items/s" if cps else f"measured {k} steps")
    return out


def measure_for_duration(step_fn: Callable[[], None], *, device: str, items_per_step: int,
                         duration_s: float, window_s: float, warmup: MeasureCfg,
                         log: Callable[[str], None] | None = None) -> tuple[dict, list[dict]]:
    """Sustained run: warm up, then repeat ``step_fn`` for ``duration_s``; throughput per window.

    Window throughput is wall-clock based (the device is synchronised at each window boundary), so
    thermal or power throttling shows up. Returns (summary, windows); summary has the keys of
    ``measure_loop`` plus clips_per_s_first_window, clips_per_s_last_window, sustain_ratio and
    ``t0_perf`` (time.perf_counter() at the start of the measured part, for aligning telemetry).
    Each window: {"t_s" (end, from start), "t_start_s", "clips_per_s", "steps", "step_ms_median"}.
    """
    cuda = _is_cuda(device)
    if cuda:
        import torch
    n_warm, warmup_s = _warmup(step_fn, device, warmup, log)
    _sync(device)
    t0 = time.perf_counter()
    win_start = t0
    win_steps = 0
    win_ms: list[float] = []
    all_ms: list[float] = []
    pending: list = []
    windows: list[dict] = []

    def _close_window(now: float) -> None:
        nonlocal win_start, win_steps, win_ms, pending
        if cuda:
            torch.cuda.synchronize()
            now = time.perf_counter()
            win_ms.extend(a.elapsed_time(b) for a, b in pending)
            pending = []
        dt = now - win_start
        if win_steps:
            windows.append({
                "t_s": now - t0,
                "t_start_s": win_start - t0,
                "clips_per_s": items_per_step * win_steps / dt if dt > 0 else None,
                "steps": win_steps,
                "step_ms_median": median(win_ms),
            })
            all_ms.extend(win_ms)
        win_start, win_steps, win_ms = now, 0, []

    while True:
        if cuda:
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            step_fn()
            e1.record()
            pending.append((e0, e1))
        else:
            s = time.perf_counter()
            step_fn()
            win_ms.append((time.perf_counter() - s) * 1000.0)
        win_steps += 1
        now = time.perf_counter()
        if now - t0 >= duration_s:
            _close_window(now)
            break
        if now - win_start >= window_s:
            _close_window(now)
            if log:
                w = windows[-1]
                log(f"window t={w['t_s']:.0f}s: {w['clips_per_s']:.2f} items/s")
    wall = time.perf_counter() - t0
    out = summarize_steps(all_ms, items_per_step, wall)
    full = [w for w in windows if w["clips_per_s"]]
    first = full[0]["clips_per_s"] if full else None
    last = full[-1]["clips_per_s"] if full else None
    out.update({
        "warmup_steps_done": n_warm, "warmup_s": warmup_s, "measure_s": wall,
        "clips_per_s_first_window": first, "clips_per_s_last_window": last,
        "sustain_ratio": (last / first) if (first and last) else None,
        "n_windows": len(windows), "t0_perf": t0,
    })
    return out, windows


def _cpu_time_tree() -> float:
    """CPU seconds of this process + its live children + reaped children."""
    total = time.process_time()
    try:
        t = os.times()
        total += t.children_user + t.children_system
    except Exception:
        pass
    try:
        import psutil

        for c in psutil.Process().children(recursive=True):
            try:
                ct = c.cpu_times()
                total += ct.user + ct.system
            except psutil.Error:
                pass
    except Exception:
        pass
    return total


def cpu_loop(fn: Callable[[], None], *, items_per_call: int, cfg: MeasureCfg,
             log: Callable[[str], None] | None = None) -> dict:
    """Time a CPU function with perf_counter and process(+children) CPU time.

    Returns step statistics (as ``measure_loop``, per call) plus clips_per_s, ms_per_clip_median,
    cpu_ms_per_clip, cpu_wall_ratio, calls, warmup_steps_done, warmup_s, measure_s.
    """
    n_warm, warmup_s = _warmup(fn, "cpu", cfg, log)
    floor = max(1, min(5, cfg.measure_steps))
    ms: list[float] = []
    c0 = _cpu_time_tree()
    w0 = time.perf_counter()
    k = 0
    while True:
        s = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - s) * 1000.0)
        k += 1
        el = time.perf_counter() - w0
        if k >= cfg.measure_steps and el >= cfg.measure_min_s:
            break
        if el >= cfg.measure_max_s and k >= floor:
            break
    wall = time.perf_counter() - w0
    cpu_s = max(0.0, _cpu_time_tree() - c0)
    out = summarize_steps(ms, items_per_call, wall)
    items = items_per_call * k
    med = out["step_ms_median"]
    out.update({
        "calls": k,
        "ms_per_clip_median": (med / items_per_call) if med and items_per_call else None,
        "cpu_ms_per_clip": (cpu_s * 1000.0 / items) if items else None,
        "cpu_wall_ratio": (cpu_s / wall) if wall > 0 else None,
        "warmup_steps_done": n_warm, "warmup_s": warmup_s, "measure_s": wall,
    })
    return out


# ---------------------------------------------------------------------------------------------
# CUDA memory statistics (used by the child, never by workloads)
# ---------------------------------------------------------------------------------------------


def reset_peak_memory(device: str) -> None:
    if _is_cuda(device):
        import torch

        torch.cuda.reset_peak_memory_stats()


def peak_memory_gib(device: str) -> tuple[float | None, float | None]:
    """(max_memory_allocated, max_memory_reserved) in GiB; (None, None) on CPU or failure."""
    if not _is_cuda(device):
        return None, None
    try:
        import torch

        return (torch.cuda.max_memory_allocated() / GIB, torch.cuda.max_memory_reserved() / GIB)
    except Exception:
        return None, None


def finite_or_none(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None
