"""``e2e.*``: the training step fed by a real DataLoader over the synthetic FLAC corpus (SPEC 5.4, 5.8).

Time the host spends blocked in ``next(iterator)`` is recorded per step as loader wait.
"""

from __future__ import annotations

import math
import os
import statistics
import time

from agbench.workloads import common as C
from agbench.workloads import train as T

# The corpus has only 256 files; draw this many samples per loader pass so no epoch boundary
# (which would stall the pipeline and respawn non-persistent workers) falls inside the
# measured window (warm-up <= 60 s + measurement <= 60 s at up to ~500 clips/s).
TARGET_ITEMS = 65536
TARGET_ITEMS_TINY = 512
TINY_MAX_WORKERS = 2


def physical_cores() -> int:
    try:
        import psutil

        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:  # pragma: no cover
        pass
    return os.cpu_count() or 1


def resolve_workers(value, *, tiny: bool) -> int:
    """``int`` or ``"auto"`` (= min(8, max(1, physical_cores - 1))); tiny mode caps at 2."""
    if value is None or str(value).lower() == "auto":
        n = min(8, max(1, physical_cores() - 1))
    else:
        n = int(value)
    if tiny:
        n = min(n, TINY_MAX_WORKERS)
    return max(0, n)


class LoaderSource:
    """Pulls micro-batches from a DataLoader, moves them to the device, times the waits."""

    def __init__(self, loader, *, micro_batch: int, device: str, non_blocking: bool):
        import torch

        self.loader = loader
        self.micro_batch = micro_batch
        self.device = torch.device(device)
        self.non_blocking = non_blocking
        self.it = None
        self.step_wait = 0.0
        self.waits: list[float] = []
        self.epoch_restarts = 0
        self.short_batches = 0
        self.pending = None
        self.worker_startup_s = None

    def start(self) -> None:
        """Create the iterator and fetch the first batch (= worker start-up time)."""
        t0 = time.perf_counter()
        self.it = iter(self.loader)
        self.pending = next(self.it)
        self.worker_startup_s = time.perf_counter() - t0

    def _next_batch(self):
        if self.pending is not None:
            batch, self.pending = self.pending, None
            return batch
        try:
            return next(self.it)
        except StopIteration:
            self.epoch_restarts += 1
            self.it = iter(self.loader)
            return next(self.it)

    def __call__(self):
        t0 = time.perf_counter()
        while True:
            x, y = self._next_batch()
            if x.shape[0] == self.micro_batch:
                break
            self.short_batches += 1  # partial last batch of an epoch: skip it
        self.step_wait += time.perf_counter() - t0
        return (x.to(self.device, non_blocking=self.non_blocking),
                y.to(self.device, non_blocking=self.non_blocking))

    def close(self) -> None:
        it, self.it = self.it, None
        shutdown = getattr(it, "_shutdown_workers", None)
        if shutdown is not None:
            try:
                shutdown()
            except Exception:  # pragma: no cover
                pass


def with_wall_rate(measured: dict) -> dict:
    """Use the wall-clock rate as ``clips_per_s`` (SPEC 5.4).

    With W loader workers, batches arrive in bursts: one step waits for data and the next W-1 find
    batches already queued. The median step time is then the GPU-only time, and 16 / median hides
    exactly the loader stalls this scenario exists to show. The wall rate (clips / total measured
    time) counts them. The median-based rate is kept as ``clips_per_s_median_step``.
    """
    out = dict(measured)
    out["clips_per_s_median_step"] = measured.get("clips_per_s")
    out["clips_per_s"] = measured.get("clips_per_s_wall")
    return out


def run(ctx) -> dict:
    import torch

    from agbench.data.dataset import SynthFlacDataset, make_loader
    from agbench.data.synth import ensure_corpus
    from agbench.timing import measure_loop

    raw = dict(ctx.params or {})
    s = T.setup(ctx)
    p = s.p
    workers = resolve_workers(raw.get("num_workers", raw.get("workers", 2)), tiny=ctx.tiny)
    rawboost_p = float(raw.get("rawboost_p", 0.0))
    pin = bool(raw.get("pin_memory", False)) and C.is_cuda(ctx.device)
    persistent = bool(raw.get("persistent_workers", False)) and workers > 0
    prefetch = int(raw.get("prefetch_factor", 2))

    manifest = ensure_corpus(ctx.cache_dir, tiny=ctx.tiny)
    ds = SynthFlacDataset(manifest, seconds=p.seconds, train=True, rawboost_p=rawboost_p, seed=p.seed)
    target = TARGET_ITEMS_TINY if ctx.tiny else TARGET_ITEMS
    num_samples = max(1, math.ceil(target / p.micro_batch)) * p.micro_batch
    loader = make_loader(ds, batch_size=p.micro_batch, num_workers=workers, pin_memory=pin,
                         persistent_workers=persistent, prefetch_factor=prefetch, shuffle=True,
                         drop_last=True, num_samples=num_samples, seed=p.seed)
    src = LoaderSource(loader, micro_batch=p.micro_batch, device=ctx.device, non_blocking=pin)
    inner = C.make_train_step(s.model, s.opt, s.prec, src, p.accum, p.grad_clip, s.tracker, s.scaler)

    def step() -> None:
        src.step_wait = 0.0
        inner()
        src.waits.append(src.step_wait)

    C.log(ctx, f"e2e {p.model} {p.seconds:g}s {p.micro_batch}x{p.accum} workers={workers} "
               f"rawboost_p={rawboost_p:g} pin={pin} persistent={persistent}")
    try:
        src.start()
        C.log(ctx, f"loader first batch after {src.worker_startup_s:.1f} s")
        measured = measure_loop(step, device=ctx.device, items_per_step=p.effective_batch,
                                cfg=ctx.measure, log=ctx.log)
    finally:
        src.close()

    measured = with_wall_rate(measured)
    n = int(measured.get("n_steps") or 0)
    waits_ms = [w * 1000.0 for w in (src.waits[-n:] if n > 0 else [])]
    wait_median = statistics.median(waits_ms) if waits_ms else None
    step_total_ms = float(measured.get("step_ms_mean") or 0.0) * n
    wait_frac = (sum(waits_ms) / step_total_ms) if step_total_ms > 0 else None
    more = {
        "loader_wait_ms_median": wait_median,
        "loader_wait_frac": min(1.0, wait_frac) if wait_frac is not None else None,
        "num_workers": workers,
        "worker_startup_s": round(src.worker_startup_s or 0.0, 3),
    }
    s.extra.update({
        "rate_basis": "clips_per_s = clips / wall-clock time of the measured steps (loader stalls included); "
                      "clips_per_s_median_step = batch / median step time",
        "loader": {"num_workers": workers, "rawboost_p": rawboost_p, "rawboost_algo": 4 if rawboost_p > 0 else None,
                   "pin_memory": pin, "persistent_workers": persistent, "prefetch_factor": prefetch,
                   "multiprocessing_context": "spawn" if workers > 0 else None,
                   "corpus_files": len(ds), "samples_per_pass": num_samples,
                   "epoch_restarts": src.epoch_restarts, "short_batches_skipped": src.short_batches},
        "physical_cores": physical_cores(), "logical_cores": os.cpu_count(),
        "blas_env": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                                     "MKL_NUM_THREADS", "GOTO_NUM_THREADS")},
        "torch_threads": torch.get_num_threads(),
    })
    return C.finish(measured, p, s.extra, tracker=s.tracker, more_metrics=more)
