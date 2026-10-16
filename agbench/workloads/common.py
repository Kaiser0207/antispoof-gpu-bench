"""Shared pieces of the model workloads (SPEC sections 5.1-5.4).

Precision contexts, determinism, model + head construction, optimizer, synthetic
batches generated on the device, and the step builders for train / frozen / infer.
"""

from __future__ import annotations

import contextlib
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import torch
from torch import nn

from agbench.heads import FrozenHeadSet, make_head
from agbench.models import MODELS, encoder_keys, load_encoder

SR = 16000
TINY_MAX_SECONDS = 2.0
TINY_MAX_BATCH = 2
POOL_BATCHES = 4  # distinct synthetic effective batches cycled through
TINY_CPU_THREADS = 2


# --------------------------------------------------------------------------- params


@dataclass
class ModelParams:
    model: str
    encoders: tuple[str, ...]  # encoder keys (two for "fusion")
    mode: str
    seconds: float
    batch: int  # requested effective batch
    micro_batch: int
    accum: int
    precision: str
    head: str
    grad_ckpt: bool
    deterministic: bool
    lr_encoder: float
    lr_head: float
    weight_decay: float
    grad_clip: float
    seed: int
    head_sets: int
    lr_heads: float  # frozen-mode heads optimizer
    conv_tf32: bool = True  # cuDNN convolutions may use TF32 for fp32 inputs (PyTorch default)
    flash_sdp: bool = True  # flash attention allowed in scaled_dot_product_attention
    deterministic_warn_only: bool = False  # set by the child after a strict-mode failure

    @property
    def effective_batch(self) -> int:
        return self.micro_batch * self.accum

    @property
    def n_samples(self) -> int:
        return int(round(self.seconds * SR))


def model_params(ctx) -> ModelParams:
    """Read ``ctx.params`` with SPEC defaults; apply the tiny-mode caps (2 s, batch 2)."""
    p = dict(ctx.params or {})
    batch = int(p.get("batch", 16))
    mb = int(p.get("micro_batch") or batch)
    acc = int(p.get("accum") or max(1, batch // max(mb, 1)))
    seconds = float(p.get("seconds", 6.0))
    if ctx.tiny:
        seconds = min(seconds, TINY_MAX_SECONDS)
        if mb * acc > TINY_MAX_BATCH:
            acc = min(acc, TINY_MAX_BATCH)
            mb = max(1, TINY_MAX_BATCH // acc)
    if mb < 1 or acc < 1:
        raise ValueError(f"bad micro_batch/accum: {mb}x{acc}")
    model = str(p.get("model", "w2v2b"))
    encoders = tuple(str(k) for k in (p.get("encoders") or encoder_keys(model)))
    for k in encoders:
        if k not in MODELS:
            raise ValueError(f"unknown encoder key {k!r}")
    head = "fusion" if len(encoders) > 1 else str(p.get("head") or "pyramid")
    mode = str(p.get("mode", ctx.workload))
    # frozen scenarios carry the heads' learning rate as ``lr_head`` (1e-3, SPEC 5.3)
    lr_heads = p.get("lr_heads", p.get("lr_head", 1e-3) if mode == "frozen" else 1e-3)
    return ModelParams(
        model=model,
        encoders=encoders,
        mode=mode,
        seconds=seconds,
        batch=batch,
        micro_batch=mb,
        accum=acc,
        precision=str(p.get("precision", "bf16")),
        head=head,
        grad_ckpt=bool(p.get("grad_ckpt", False)),
        deterministic=bool(p.get("deterministic", False)),
        lr_encoder=float(p.get("lr_encoder", 1e-5)),
        lr_head=float(p.get("lr_head", 1e-4)),
        weight_decay=float(p.get("weight_decay", 0.01)),
        grad_clip=float(p.get("grad_clip", 1.0)),
        seed=int(p.get("seed", getattr(ctx, "seed", 1234))),
        head_sets=int(p.get("head_sets", 1)),
        lr_heads=float(lr_heads),
        conv_tf32=bool(p.get("conv_tf32", True)),
        flash_sdp=bool(p.get("flash_sdp", True)),
        deterministic_warn_only=bool(p.get("deterministic_warn_only", False)),
    )


def prepare(ctx) -> tuple[ModelParams, dict, "Precision"]:
    """Params + process-wide settings every model workload applies first: tiny-CPU thread cap,
    determinism flags, precision mode, seed. Returns (params, determinism info, precision)."""
    p = model_params(ctx)
    if ctx.tiny and not is_cuda(ctx.device):
        # Tiny CPU runs are for tests/CI only; many intra-op threads on a loaded host make
        # them 20x slower without telling us anything.
        torch.set_num_threads(max(1, min(TINY_CPU_THREADS, torch.get_num_threads())))
    det = apply_determinism(p.deterministic, warn_only=p.deterministic_warn_only)
    det.update(apply_sdp(p.flash_sdp))
    prec = Precision(p.precision, ctx.device, conv_tf32=p.conv_tf32)
    torch.manual_seed(p.seed)
    return p, det, prec


def is_cuda(device: str) -> bool:
    return str(device).startswith("cuda")


def log(ctx, msg: str) -> None:
    fn = getattr(ctx, "log", None)
    if fn is not None:
        fn(msg)


# --------------------------------------------------------------------------- precision


class Precision:
    """Precision mode of SPEC 5.2. On CPU, fp16 is mapped to bf16 (no GradScaler) and
    tf32/fp32 to plain fp32; ``effective`` records what actually ran.

    On CUDA the fp32 math mode is set through PyTorch's new per-backend API only
    (``torch.backends.cuda.matmul.fp32_precision`` and ``torch.backends.cudnn.conv.fp32_precision``,
    "tf32" or "ieee"); mixing it with the legacy ``set_float32_matmul_precision`` / ``allow_tf32``
    makes later legacy reads raise in recent torch.
    """

    def __init__(self, precision: str, device: str, *, conv_tf32: bool = True):
        if precision not in ("bf16", "fp16", "tf32", "fp32"):
            raise ValueError(f"unknown precision {precision!r}")
        self.requested = precision
        self.device_type = "cuda" if is_cuda(device) else "cpu"
        self.dtype: torch.dtype | None = None
        self.use_scaler = False
        if self.device_type == "cpu":
            if precision in ("bf16", "fp16"):
                self.dtype, self.effective = torch.bfloat16, "bf16"
            else:
                self.effective = "fp32"
        else:
            self.effective = precision
            # bf16/fp16: fp32 matmuls IEEE (PyTorch default), cuDNN conv TF32 (PyTorch default) unless
            # conv_tf32=False (reference settings); tf32: TF32 matmuls too; fp32: IEEE everywhere.
            set_fp32_precision(matmul="tf32" if precision == "tf32" else "ieee",
                               conv="tf32" if (conv_tf32 and precision != "fp32") else "ieee")
            if precision == "bf16":
                self.dtype = torch.bfloat16
            elif precision == "fp16":
                self.dtype, self.use_scaler = torch.float16, True

    def autocast(self):
        if self.dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=self.dtype)

    def make_scaler(self, training: bool = True):
        if training and self.use_scaler:
            return torch.amp.GradScaler("cuda")
        return None

    def describe(self) -> dict:
        mm, conv = fp32_precision()
        return {
            "precision": self.requested,
            "precision_effective": self.effective,
            "autocast_dtype": str(self.dtype).replace("torch.", "") if self.dtype else None,
            "grad_scaler": self.use_scaler,
            "matmul_fp32_precision": mm,
            "cudnn_conv_fp32_precision": conv,
        }


def set_fp32_precision(*, matmul: str, conv: str) -> None:
    """fp32 math mode, new API only: ``matmul`` / ``conv`` each "tf32" or "ieee"."""
    try:
        torch.backends.cuda.matmul.fp32_precision = matmul
        torch.backends.cudnn.conv.fp32_precision = conv
    except AttributeError:  # torch < 2.9 (not the pinned builds): legacy flags
        torch.backends.cuda.matmul.allow_tf32 = matmul == "tf32"
        torch.backends.cudnn.allow_tf32 = conv == "tf32"


def fp32_precision() -> tuple[str | None, str | None]:
    """(matmul, cudnn conv) fp32 precision as set through the new API ("none" = PyTorch default)."""
    try:
        return (str(torch.backends.cuda.matmul.fp32_precision), str(torch.backends.cudnn.conv.fp32_precision))
    except AttributeError:
        return (None, None)


def _set_cudnn_fp32_precision(value: str) -> None:
    """cuDNN convolution fp32 precision: "tf32" (PyTorch default) or "ieee" (new API only)."""
    try:
        torch.backends.cudnn.conv.fp32_precision = value
    except AttributeError:  # older torch without the fp32_precision API
        torch.backends.cudnn.allow_tf32 = value == "tf32"


def apply_determinism(enabled: bool, *, warn_only: bool = False) -> dict:
    """Deterministic algorithms as in the reference training runs: strict
    ``torch.use_deterministic_algorithms(True)`` (an op without a deterministic kernel raises; the
    child then reruns once with ``warn_only``). ``CUBLAS_WORKSPACE_CONFIG`` comes from the scenario
    env (set before torch is imported)."""
    torch.backends.cudnn.benchmark = False  # PyTorch default, mirrors the reference runs
    if enabled:
        torch.use_deterministic_algorithms(True, warn_only=bool(warn_only))
        torch.backends.cudnn.deterministic = True
    mode = ("warn_only" if torch.is_deterministic_algorithms_warn_only_enabled() else "strict") \
        if torch.are_deterministic_algorithms_enabled() else "off"
    out = {
        "deterministic": bool(enabled),
        "deterministic_mode": mode,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    try:
        import torch.utils.deterministic as tud

        out["fill_uninitialized_memory"] = bool(tud.fill_uninitialized_memory)
    except Exception:  # pragma: no cover
        pass
    return out


def apply_sdp(flash: bool) -> dict:
    """Allow or forbid flash attention in ``scaled_dot_product_attention`` (the reference runs
    switched it off for clips longer than 4 s in deterministic mode); record all SDPA backends."""
    b = torch.backends.cuda
    b.enable_flash_sdp(bool(flash))
    out = {"sdp_flash_requested": bool(flash)}
    for name, fn in (("flash", "flash_sdp_enabled"), ("mem_efficient", "mem_efficient_sdp_enabled"),
                     ("math", "math_sdp_enabled"), ("cudnn", "cudnn_sdp_enabled")):
        try:
            out[f"sdp_{name}_enabled"] = bool(getattr(b, fn)())
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------- model


def normalize_wave(x: torch.Tensor) -> torch.Tensor:
    """Per-utterance normalisation (inside the model forward, SPEC 5.1)."""
    return (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-7)


class WaveClassifier(nn.Module):
    """Waveform ``[B, T]`` -> normalise -> encoder(s) ``last_hidden_state`` -> head -> logits."""

    def __init__(self, encoders: Sequence[nn.Module], head: nn.Module):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.head = head

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        x = normalize_wave(wave)
        hs = [enc(x).last_hidden_state for enc in self.encoders]
        return self.head(hs[0]) if len(hs) == 1 else self.head(hs)


def load_encoders(ctx, p: ModelParams, *, grad_ckpt: bool | None = None) -> list[nn.Module]:
    out = []
    for key in p.encoders:
        log(ctx, f"loading encoder {key} ({'tiny random' if ctx.tiny else MODELS[key].hf_id})")
        out.append(load_encoder(key, tiny=ctx.tiny, device=ctx.device,
                                grad_ckpt=p.grad_ckpt if grad_ckpt is None else grad_ckpt))
    return out


def build_classifier(ctx, p: ModelParams, *, grad_ckpt: bool | None = None) -> WaveClassifier:
    encoders = load_encoders(ctx, p, grad_ckpt=grad_ckpt)
    head = make_head(p.head, [e.config.hidden_size for e in encoders])
    return WaveClassifier(encoders, head.to(torch.device(ctx.device)))


def encoder_extra(encoders: Sequence[nn.Module], n_samples: int) -> dict:
    infos = []
    for e in encoders:
        info = dict(getattr(e, "agbench_info", {}))
        info["grad_ckpt"] = bool(e.is_gradient_checkpointing)  # actual flag at measurement time
        infos.append(info)
    frames = int(encoders[0]._get_feat_extract_output_lengths(torch.tensor(n_samples)).item())
    return {"encoders": infos, "frames_per_clip": frames,
            "attn_implementation": [i.get("attn_implementation") for i in infos]}


def n_params(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def make_optimizer(model: WaveClassifier, p: ModelParams) -> torch.optim.Optimizer:
    """AdamW (default foreach implementation), encoder lr 1e-5, head lr 1e-4, wd 0.01 on both."""
    return torch.optim.AdamW(
        [
            {"params": [q for q in model.encoders.parameters() if q.requires_grad], "lr": p.lr_encoder},
            {"params": [q for q in model.head.parameters() if q.requires_grad], "lr": p.lr_head},
        ],
        lr=p.lr_head,
        weight_decay=p.weight_decay,
    )


# --------------------------------------------------------------------------- synthetic data


def synth_waves(n: int, n_samples: int, *, device: str, seed: int) -> torch.Tensor:
    """Speech-like float32 waveforms ``[n, n_samples]`` generated on ``device``.

    Harmonic series of a slowly moving f0 (90-250 Hz) with 1/k amplitudes up to 4 kHz, a
    3-6 Hz syllabic envelope and a little noise, peak-normalised to 0.3-0.9. Content does not
    change GPU compute; it only keeps activations in a realistic range.
    """
    dev = torch.device(device)
    g = torch.Generator(device=dev)
    g.manual_seed(int(seed))

    def u(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(*shape, generator=g, device=dev)

    t = torch.arange(n_samples, device=dev, dtype=torch.float32) / SR
    f0_base = u(90.0, 250.0, n, 1)
    vib = 0.1 * f0_base * torch.sin(2 * math.pi * u(0.5, 2.0, n, 1) * t + u(0, 6.28, n, 1))
    f0 = f0_base + vib
    phase = 2 * math.pi * torch.cumsum(f0, dim=1) / SR
    x = torch.zeros(n, n_samples, device=dev)
    for k in range(1, 21):
        amp = (k * f0_base < 4000.0).float() / k
        x += amp * torch.sin(k * phase)
    env = 0.5 * (1 + torch.sin(2 * math.pi * u(3.0, 6.0, n, 1) * t + u(0, 6.28, n, 1)))
    x = x * env.pow(2) + 0.02 * torch.randn(n, n_samples, generator=g, device=dev)
    peak = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
    return (x / peak * u(0.3, 0.9, n, 1)).contiguous()


def balanced_labels(n: int, *, device: str, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) + 7)
    y = (torch.arange(n) % 2)[torch.randperm(n, generator=g)]
    return y.to(torch.device(device), dtype=torch.int64)


def make_micro_pool(p: ModelParams, *, device: str, seed: int, n_batches: int = POOL_BATCHES,
                    host_pinned: bool = False) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """``n_batches`` distinct effective batches, pre-split into micro-batches (flat list).

    Generated on ``device``. With ``host_pinned`` the waves are moved to pinned host memory
    (inference input path: H2D copy per step).
    """
    micro: list[tuple[torch.Tensor, torch.Tensor]] = []
    for b in range(n_batches):
        x = synth_waves(p.effective_batch, p.n_samples, device=device, seed=seed + 1000 * b)
        y = balanced_labels(p.effective_batch, device=device, seed=seed + 1000 * b)
        if host_pinned:
            x = x.cpu()
            y = y.cpu()
            if is_cuda(device):
                x, y = x.pin_memory(), y.pin_memory()
        for k in range(p.accum):
            sl = slice(k * p.micro_batch, (k + 1) * p.micro_batch)
            micro.append((x[sl], y[sl]))
    return micro


class Cycler:
    """Endless iterator over a list of micro-batches."""

    def __init__(self, items: list):
        self.items, self.i = items, 0

    def __call__(self):
        item = self.items[self.i]
        self.i = (self.i + 1) % len(self.items)
        return item


# --------------------------------------------------------------------------- loss tracking


class LossTracker:
    """Keeps the first/last loss and an all-finite flag on the device (no per-step sync)."""

    def __init__(self):
        self.first: torch.Tensor | None = None
        self.last: torch.Tensor | None = None
        self.finite: torch.Tensor | None = None
        self.n = 0

    def add(self, value: torch.Tensor) -> None:
        v = value.detach().float()
        if self.first is None:
            self.first = v
        self.last = v
        ok = torch.isfinite(v).all()
        self.finite = ok if self.finite is None else (self.finite & ok)
        self.n += 1

    def summary(self, prefix: str = "loss") -> dict:
        if self.first is None:
            return {f"{prefix}_first": None, f"{prefix}_last": None, f"{prefix}_finite": False}
        first, last = float(self.first.mean().item()), float(self.last.mean().item())
        return {f"{prefix}_first": first, f"{prefix}_last": last,
                f"{prefix}_finite": bool(self.finite.item())}


# --------------------------------------------------------------------------- step builders


def make_train_step(model: nn.Module, opt: torch.optim.Optimizer, prec: Precision,
                    next_micro: Callable[[], tuple[torch.Tensor, torch.Tensor]], accum: int,
                    grad_clip: float, tracker: LossTracker, scaler=None) -> Callable[[], None]:
    """One optimizer step over ``accum`` micro-batches (SPEC 5.1)."""
    params = [q for q in model.parameters() if q.requires_grad]
    ce = nn.CrossEntropyLoss()

    def step() -> None:
        total = None
        for _ in range(accum):
            x, y = next_micro()
            with prec.autocast():
                logits = model(x)
            loss = ce(logits.float(), y) / accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total = loss.detach() if total is None else total + loss.detach()
        if scaler is not None:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        tracker.add(total)

    return step


def make_frozen_step(encoder: nn.Module, heads: nn.ModuleList, opt: torch.optim.Optimizer,
                     prec: Precision, next_micro: Callable[[], tuple[torch.Tensor, torch.Tensor]],
                     accum: int, tracker: LossTracker, scaler=None) -> Callable[[], None]:
    """No-grad encoder forward with all hidden states, then heads forward/backward/AdamW."""
    ce = nn.CrossEntropyLoss()

    def step() -> None:
        total = None
        for _ in range(accum):
            x, y = next_micro()
            with torch.no_grad(), prec.autocast():
                hs = encoder(normalize_wave(x), output_hidden_states=True).hidden_states
            with prec.autocast():
                outs = [o for hset in heads for o in hset(hs)]
            loss = sum(ce(o.float(), y) for o in outs) / accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total = loss.detach() if total is None else total + loss.detach()
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        tracker.add(total)

    return step


class ScoreSink:
    """Keeps the last scores and an all-finite flag on the device (no per-step sync)."""

    def __init__(self):
        self.last: torch.Tensor | None = None
        self.finite: torch.Tensor | None = None
        self.n = 0

    def update(self, scores: torch.Tensor) -> None:
        ok = torch.isfinite(scores).all()
        self.finite = ok if self.finite is None else (self.finite & ok)
        self.last = scores
        self.n += scores.shape[0]

    def summary(self) -> dict:
        finite = bool(self.finite.item()) if self.finite is not None else False
        mean = float(self.last.float().mean().item()) if self.last is not None else None
        return {"scores_finite": finite, "score_mean_last": mean, "clips_scored": self.n}


def make_infer_step(model: nn.Module, prec: Precision, next_micro: Callable[[], tuple[torch.Tensor, torch.Tensor]],
                    accum: int, device: str, sink: ScoreSink) -> Callable[[], None]:
    """Pinned host batch -> device (non_blocking), inference_mode + autocast, softmax score."""
    dev = torch.device(device)
    non_blocking = is_cuda(device)

    def step() -> None:
        with torch.inference_mode():
            for _ in range(accum):
                xh, _y = next_micro()
                x = xh.to(dev, non_blocking=non_blocking)
                with prec.autocast():
                    logits = model(x)
                sink.update(torch.softmax(logits.float(), dim=-1)[:, 1])

    return step


def frozen_heads(dim: int, n_states: int, head_sets: int) -> nn.ModuleList:
    return nn.ModuleList(FrozenHeadSet(dim, n_states) for _ in range(head_sets))


# --------------------------------------------------------------------------- results


def base_extra(p: ModelParams, prec: Precision, det: dict, tiny: bool, device: str) -> dict:
    extra: dict[str, Any] = {
        "model": p.model, "mode": p.mode, "seconds": p.seconds, "samples_per_clip": p.n_samples,
        "batch_requested": p.batch, "micro_batch": p.micro_batch, "accum": p.accum,
        "effective_batch": p.effective_batch, "head": p.head, "tiny": bool(tiny),
        "device": str(device), "torch": torch.__version__,
    }
    extra.update(prec.describe())
    extra.update(det)
    if not is_cuda(device):
        extra["cpu_threads"] = torch.get_num_threads()
    return extra


def finish(measured: dict, p: ModelParams, extra: dict, *, tracker: LossTracker | None = None,
           sink: ScoreSink | None = None, more_metrics: dict | None = None,
           timeseries: list | None = None) -> dict:
    """Assemble the workload return value ``{"metrics", "extra", "timeseries"}``."""
    metrics = dict(measured)
    metrics["effective_batch"] = p.effective_batch
    if tracker is not None:
        metrics.update(tracker.summary())
    if sink is not None:
        s = sink.summary()
        metrics.update({"loss_first": None, "loss_last": None, "loss_finite": s["scores_finite"]})
        extra.update(s)
    if more_metrics:
        metrics.update(more_metrics)
    extra["params_resolved"] = asdict(p)
    return {"metrics": metrics, "extra": extra, "timeseries": timeseries}
