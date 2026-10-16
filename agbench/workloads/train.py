"""``train.*``: full fine-tune of encoder + head on synthetic GPU batches (SPEC 5.1, 5.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agbench.workloads import common as C


@dataclass
class TrainSetup:
    p: C.ModelParams
    prec: C.Precision
    model: Any  # WaveClassifier in train mode
    opt: Any
    scaler: Any
    tracker: C.LossTracker
    extra: dict


def setup(ctx) -> TrainSetup:
    """Model, optimizer and precision for the training recipe; shared by train / e2e / sustain."""
    p, det, prec = C.prepare(ctx)
    model = C.build_classifier(ctx, p)
    model.train()
    opt = C.make_optimizer(model, p)
    scaler = prec.make_scaler(training=True)
    extra = C.base_extra(p, prec, det, ctx.tiny, ctx.device)
    extra.update(C.encoder_extra(model.encoders, p.n_samples))
    extra.update({
        "grad_ckpt_requested": p.grad_ckpt,
        "params_encoder_m": round(C.n_params(model.encoders) / 1e6, 3),
        "params_head_m": round(C.n_params(model.head) / 1e6, 4),
        "optimizer": "AdamW(foreach default)", "lr_encoder": p.lr_encoder, "lr_head": p.lr_head,
        "weight_decay": p.weight_decay, "grad_clip": p.grad_clip, "layerdrop": 0.0,
        "spec_augment": False,
    })
    return TrainSetup(p, prec, model, opt, scaler, C.LossTracker(), extra)


def run(ctx) -> dict:
    from agbench.timing import measure_loop

    s = setup(ctx)
    pool = C.make_micro_pool(s.p, device=ctx.device, seed=s.p.seed)
    step = C.make_train_step(s.model, s.opt, s.prec, C.Cycler(pool), s.p.accum, s.p.grad_clip,
                             s.tracker, s.scaler)
    C.log(ctx, f"train {s.p.model} {s.p.seconds:g}s {s.p.micro_batch}x{s.p.accum} {s.prec.effective}")
    measured = measure_loop(step, device=ctx.device, items_per_step=s.p.effective_batch,
                            cfg=ctx.measure, log=ctx.log)
    return C.finish(measured, s.p, s.extra, tracker=s.tracker)
