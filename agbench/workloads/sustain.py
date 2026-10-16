"""``sustain.*``: the training step repeated for ``duration_s`` with per-window clips/s (SPEC 5.4).

The child merges GPU telemetry (temperature, clocks, power) into the windows.
"""

from __future__ import annotations

from agbench.workloads import common as C
from agbench.workloads import train as T

DURATION_S = 300.0
WINDOW_S = 30.0
TINY_DURATION_S = 3.0
TINY_WINDOW_S = 1.0


def run(ctx) -> dict:
    from agbench.timing import measure_for_duration

    raw = dict(ctx.params or {})
    duration = float(raw.get("duration_s", DURATION_S))
    window = float(raw.get("window_s", WINDOW_S))
    if ctx.tiny:
        duration, window = min(duration, TINY_DURATION_S), min(window, TINY_WINDOW_S)

    s = T.setup(ctx)
    pool = C.make_micro_pool(s.p, device=ctx.device, seed=s.p.seed)
    step = C.make_train_step(s.model, s.opt, s.prec, C.Cycler(pool), s.p.accum, s.p.grad_clip,
                             s.tracker, s.scaler)
    C.log(ctx, f"sustain {s.p.model} {s.p.seconds:g}s {s.p.micro_batch}x{s.p.accum} "
               f"for {duration:g} s in {window:g} s windows")
    measured, windows = measure_for_duration(step, device=ctx.device, items_per_step=s.p.effective_batch,
                                             duration_s=duration, window_s=window, warmup=ctx.measure,
                                             log=ctx.log)
    rates = [w.get("clips_per_s") for w in windows if w.get("clips_per_s") is not None]
    first = rates[0] if rates else None
    last = rates[-1] if rates else None
    more = {
        "clips_per_s_first_window": first,
        "clips_per_s_last_window": last,
        "sustain_ratio": (last / first) if first and last is not None else None,
    }
    s.extra.update({"duration_s": duration, "window_s": window, "n_windows": len(windows)})
    return C.finish(measured, s.p, s.extra, tracker=s.tracker, more_metrics=more,
                    timeseries=list(windows))
