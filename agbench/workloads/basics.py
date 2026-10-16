"""GPU basics (SPEC 6.1): matmul TFLOPS, device memory bandwidth, host<->device bandwidth.

Each operation is timed in batches: a short warm-up, a calibration that picks how many repeats
make one batch last at least ``batch_min_s``, then ``n_batches`` batches bracketed by CUDA events
(perf_counter on CPU). The reported value uses the median batch. torch is imported inside the
functions only.
"""

from __future__ import annotations

import math
import statistics
import time
from typing import Callable

from agbench.errors import SkipScenario

MIB = 1024 ** 2
DTYPES = ("bf16", "fp16", "tf32", "fp32")
# PCIe payload bandwidth per lane and direction in GB/s, after line encoding (gen 6: FLIT).
PCIE_LANE_GBPS = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}


def _is_cuda(device: str) -> bool:
    return str(device).startswith("cuda")


def _is_oom(exc: BaseException) -> bool:
    import torch

    if isinstance(exc, torch.OutOfMemoryError):
        return True
    msg = str(exc)
    return ("out of memory" in msg or "CUBLAS_STATUS_ALLOC_FAILED" in msg
            or "CUDNN_STATUS_ALLOC_FAILED" in msg)


def time_op(op: Callable[[], object], *, device: str, n_batches: int, batch_min_s: float,
            warmup: int = 3, calib_s: float = 0.05, max_reps: int = 100_000,
            during: Callable[[], None] | None = None) -> dict:
    """Median time of ``op`` over ``n_batches`` batches of ``reps`` back-to-back calls.

    ``during`` (CUDA only) is called once after all batches are queued, while the GPU is still
    working on them (used to read the PCIe link state under load).
    """
    import torch

    cuda = _is_cuda(device)

    def sync():
        if cuda:
            torch.cuda.synchronize()

    for _ in range(max(0, warmup)):
        op()
    sync()
    reps, dt = 1, 0.0
    while True:
        t0 = time.perf_counter()
        for _ in range(reps):
            op()
        sync()
        dt = time.perf_counter() - t0
        if dt >= calib_s or reps >= max_reps:
            break
        reps = min(max_reps, max(reps * 2, math.ceil(reps * calib_s / max(dt, 1e-9) * 1.1)))
    per_op = dt / reps
    reps = int(min(max_reps, max(1, math.ceil(batch_min_s / max(per_op, 1e-12)))))

    batch_s: list[float] = []
    if cuda:
        pairs = []
        for _ in range(n_batches):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(reps):
                op()
            e1.record()
            pairs.append((e0, e1))
        if during is not None:
            during()
        torch.cuda.synchronize()
        batch_s = [a.elapsed_time(b) / 1000.0 for a, b in pairs]
    else:
        for _ in range(n_batches):
            t0 = time.perf_counter()
            for _ in range(reps):
                op()
            batch_s.append(time.perf_counter() - t0)
    op_s = [b / reps for b in batch_s]
    mean = statistics.fmean(op_s)
    return {"reps": reps, "n_batches": len(op_s), "op_ms_median": statistics.median(op_s) * 1e3,
            "op_ms_min": min(op_s) * 1e3, "op_ms_max": max(op_s) * 1e3,
            "op_ms_cv": (statistics.pstdev(op_s) / mean) if len(op_s) > 1 and mean > 0 else 0.0}


def _batch_cfg(ctx, gpu_default_min_s: float) -> tuple[int, float]:
    p = ctx.params or {}
    if ctx.tiny:
        return 3, 0.01 if not _is_cuda(ctx.device) else 0.02
    if not _is_cuda(ctx.device):
        return 5, 0.05
    return int(p.get("n_batches", 10)), float(p.get("batch_min_s", gpu_default_min_s))


def _log(ctx, msg: str) -> None:
    try:
        ctx.log(msg)
    except Exception:
        pass


# ---------------------------------------------------------------------------------------------
# gpu.matmul
# ---------------------------------------------------------------------------------------------


def run_matmul(ctx) -> dict:
    """Square matmul TFLOPS (2 n^3 / t) per dtype (bf16, fp16, tf32, fp32) and size."""
    import torch

    p = ctx.params or {}
    cuda = _is_cuda(ctx.device)
    dev = torch.device(ctx.device)
    if cuda:
        sizes = [1024] if ctx.tiny else [int(n) for n in (p.get("sizes") or (4096, 8192))]
    else:
        sizes = [int(p.get("cpu_size", 256))]
    dtypes = [str(d) for d in (p.get("dtypes") or DTYPES)]
    n_batches, batch_min_s = _batch_cfg(ctx, 0.2)
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "tf32": torch.float32,
                   "fp32": torch.float32}
    tflops: dict = {}
    best: dict = {}
    details: dict = {}
    errors: dict = {}
    prev = torch.get_float32_matmul_precision()
    gen = torch.Generator(device=dev)
    ramp_s = 0.0
    if cuda and not ctx.tiny:
        # Bring the GPU out of idle clocks before the first timed size (~0.5 s of bf16 work).
        n0 = min(4096, max(sizes))
        a0 = torch.randn(n0, n0, device=dev, dtype=torch.bfloat16)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.5:
            for _ in range(4):
                torch.mm(a0, a0)
            torch.cuda.synchronize()
        ramp_s = time.perf_counter() - t0
        del a0
    try:
        for dt in dtypes:
            if dt not in torch_dtype:
                raise ValueError(f"unknown dtype {dt!r}; expected one of {DTYPES}")
            torch.set_float32_matmul_precision("high" if dt == "tf32" else "highest")
            tflops[dt] = {}
            for n in sizes:
                try:
                    gen.manual_seed(int(ctx.seed))
                    a = torch.randn(n, n, device=dev, dtype=torch_dtype[dt], generator=gen)
                    b = torch.randn(n, n, device=dev, dtype=torch_dtype[dt], generator=gen)
                    c = torch.empty(n, n, device=dev, dtype=torch_dtype[dt])

                    def op(a=a, b=b, c=c):
                        torch.mm(a, b, out=c)

                    r = time_op(op, device=ctx.device, n_batches=n_batches,
                                batch_min_s=batch_min_s)
                    finite = bool(torch.isfinite(c).all().item())
                    del a, b, c
                except Exception as exc:  # unsupported dtype on this device: record, go on
                    if _is_oom(exc):
                        raise
                    errors[f"{dt}/{n}"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    continue
                value = 2.0 * n ** 3 / (r["op_ms_median"] / 1e3) / 1e12
                tflops[dt][str(n)] = round(value, 3)
                details[f"{dt}/{n}"] = {**{k: round(v, 4) if isinstance(v, float) else v
                                           for k, v in r.items()}, "finite": finite}
                _log(ctx, f"matmul {dt} n={n}: {value:.2f} TFLOPS "
                          f"({r['reps']} reps x {r['n_batches']} batches)")
            if tflops[dt]:
                best[dt] = max(tflops[dt].values())
    finally:
        torch.set_float32_matmul_precision(prev)
    if not best:
        raise RuntimeError(f"no matmul measurement succeeded: {errors}")
    extra = {
        "sizes": sizes, "dtypes": dtypes, "n_batches": n_batches, "batch_min_s": batch_min_s,
        "clock_ramp_s": round(ramp_s, 3), "details": details, "errors": errors,
        "fp16_reduced_precision_reduction":
            bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction),
        "bf16_reduced_precision_reduction":
            bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        "note": ("bf16/fp16 tensors with torch defaults (fp32 accumulate); tf32 = fp32 tensors "
                 "with set_float32_matmul_precision('high'); fp32 = 'highest'"
                 + ("" if cuda else "; CPU run: tf32 is plain fp32")),
    }
    return {"metrics": {"tflops": tflops, "tflops_best": best}, "extra": extra,
            "timeseries": None}


# ---------------------------------------------------------------------------------------------
# gpu.membw
# ---------------------------------------------------------------------------------------------


def _fit_buffer_mib(mib: int, n_buffers: int) -> tuple[int, str | None]:
    """Shrink the buffer size (powers of two) when the per-process cap or free VRAM is smaller."""
    import torch

    try:
        free, total = torch.cuda.mem_get_info()
        frac = float(torch.cuda.get_per_process_memory_fraction())
        budget = min(free, frac * total) - torch.cuda.memory_reserved() - 256 * MIB
    except Exception:
        return mib, None
    per = budget * 0.9 / n_buffers
    if per >= mib * MIB:
        return mib, None
    new = max(16, 2 ** int(math.floor(math.log2(max(per / MIB, 1.0)))))
    return min(mib, new), f"reduced from {mib} MiB to fit the memory cap/free VRAM"


def run_membw(ctx) -> dict:
    """Device memory bandwidth: copy (2x bytes), read-reduce sum (1x), triad add (3x)."""
    import torch

    p = ctx.params or {}
    cuda = _is_cuda(ctx.device)
    dev = torch.device(ctx.device)
    mib = 64 if (ctx.tiny or not cuda) else int(p.get("buffer_mib", 1024))
    note = None
    if cuda:
        mib, note = _fit_buffer_mib(mib, 3)
    n = mib * MIB // 4
    n_batches, batch_min_s = _batch_cfg(ctx, 0.1)
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(ctx.seed))
    a = torch.rand(n, device=dev, dtype=torch.float32, generator=gen)
    b = torch.rand(n, device=dev, dtype=torch.float32, generator=gen)
    c = torch.empty_like(a)
    nbytes = n * 4

    def copy():
        c.copy_(a)

    def read():
        torch.sum(a)

    def triad():
        torch.add(a, b, out=c)

    wanted = [str(o) for o in (p.get("ops") or ("copy", "read", "triad"))]
    table = {"copy": (copy, 2), "read": (read, 1), "triad": (triad, 3)}
    unknown = [o for o in wanted if o not in table]
    if unknown:
        raise ValueError(f"unknown membw op(s) {unknown}; expected {sorted(table)}")
    gbps: dict = {}
    details: dict = {}
    for name in wanted:
        op, factor = table[name]
        r = time_op(op, device=ctx.device, n_batches=n_batches, batch_min_s=batch_min_s)
        gbps[name] = round(factor * nbytes / (r["op_ms_median"] / 1e3) / 1e9, 2)
        details[name] = {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
        _log(ctx, f"membw {name}: {gbps[name]:.1f} GB/s")
    extra = {"buffer_mib": mib, "buffer_note": note, "n_batches": n_batches,
             "batch_min_s": batch_min_s, "details": details,
             "units": "GB/s = 1e9 bytes/s; copy counts read+write, triad 2 reads + 1 write"}
    return {"metrics": {"gbps": gbps}, "extra": extra, "timeseries": None}


# ---------------------------------------------------------------------------------------------
# gpu.pcie
# ---------------------------------------------------------------------------------------------


def _gpu_ids(ctx) -> tuple[int, str | None]:
    import torch

    from agbench.telemetry import device_index_from_env

    index = getattr(ctx, "gpu_index", None)
    if index is None:
        index = device_index_from_env(0)
    uuid = getattr(ctx, "gpu_uuid", None)
    if not uuid:
        try:
            uuid = str(torch.cuda.get_device_properties(torch.device(ctx.device)).uuid)
        except Exception:
            uuid = None
    return int(index), uuid


def run_pcie(ctx) -> dict:
    """Host<->device copies of 256 MiB: pinned H2D, pinned D2H, pageable H2D; link under load."""
    import torch

    from agbench.telemetry import pcie_link

    if not _is_cuda(ctx.device):
        raise SkipScenario("cpu_device", "gpu.pcie needs a CUDA device")
    p = ctx.params or {}
    dev = torch.device(ctx.device)
    mib = 32 if ctx.tiny else int(p.get("buffer_mib", 256))
    n_batches, batch_min_s = _batch_cfg(ctx, 0.1)
    nbytes = mib * MIB
    index, uuid = _gpu_ids(ctx)

    src = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    src.fill_(7)
    dst = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    pageable = torch.empty(nbytes, dtype=torch.uint8)
    pageable.fill_(7)  # touch every page before timing
    d = torch.empty(nbytes, dtype=torch.uint8, device=dev)

    link_load: dict = {}

    def read_link():
        link_load.update(pcie_link(index, uuid))

    def h2d():
        d.copy_(src, non_blocking=True)

    def d2h():
        dst.copy_(d, non_blocking=True)

    def h2d_pageable():
        d.copy_(pageable, non_blocking=False)

    wanted = [str(m) for m in (p.get("modes") or ("h2d_pinned", "d2h_pinned", "h2d_pageable"))]
    table = {"h2d_pinned": h2d, "d2h_pinned": d2h, "h2d_pageable": h2d_pageable}
    unknown = [m for m in wanted if m not in table]
    if unknown:
        raise ValueError(f"unknown pcie mode(s) {unknown}; expected {sorted(table)}")
    gbps: dict = {}
    details: dict = {}
    for name in wanted:
        op = table[name]
        # read the link state under load during the first measured direction
        during = read_link if not link_load else None
        r = time_op(op, device=ctx.device, n_batches=n_batches, batch_min_s=batch_min_s,
                    during=during)
        gbps[name] = round(nbytes / (r["op_ms_median"] / 1e3) / 1e9, 2)
        details[name] = {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
        _log(ctx, f"pcie {name}: {gbps[name]:.2f} GB/s")
    torch.cuda.synchronize()
    # Round-trip check only when D2H ran after an H2D copy (both sources hold the value 7).
    ok = None
    if "d2h_pinned" in wanted and any(m.startswith("h2d") for m in
                                      wanted[:wanted.index("d2h_pinned")]):
        ok = bool(torch.equal(dst[:4096], src[:4096]))
    link_idle = pcie_link(index, uuid)
    link = {k: link_load.get(k) for k in ("gen_cur", "width_cur", "gen_max", "width_max")}
    theo = None
    if link["gen_cur"] in PCIE_LANE_GBPS and link["width_cur"]:
        theo = round(PCIE_LANE_GBPS[link["gen_cur"]] * link["width_cur"], 2)
    extra = {"buffer_mib": mib, "n_batches": n_batches, "batch_min_s": batch_min_s,
             "details": details, "copy_check_ok": ok, "link_source": link_load.get("source"),
             "link_idle_after": {k: link_idle.get(k) for k in link},
             "link_theoretical_gbps": theo,
             "units": "GB/s = 1e9 bytes/s per direction; theoretical = link payload rate"}
    return {"metrics": {"gbps": gbps, "link": link}, "extra": extra, "timeseries": None}
