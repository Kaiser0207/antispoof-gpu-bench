"""``frozen.*``: frozen encoder (no grad, all hidden states) feeding probe head sets (SPEC 5.3, 5.4)."""

from __future__ import annotations

from agbench.workloads import common as C


def run(ctx) -> dict:
    import torch

    from agbench.errors import ScenarioError
    from agbench.timing import measure_loop

    p, det, prec = C.prepare(ctx)
    if len(p.encoders) != 1:
        raise ScenarioError(f"frozen mode needs a single encoder, got {p.model!r}")
    (encoder,) = C.load_encoders(ctx, p, grad_ckpt=False)
    encoder.eval()
    encoder.requires_grad_(False)
    n_states = encoder.config.num_hidden_layers + 1
    heads = C.frozen_heads(encoder.config.hidden_size, n_states, p.head_sets).to(torch.device(ctx.device))
    heads.train()
    opt = torch.optim.AdamW(heads.parameters(), lr=p.lr_heads, weight_decay=p.weight_decay)
    scaler = prec.make_scaler(training=True)
    tracker = C.LossTracker()
    pool = C.make_micro_pool(p, device=ctx.device, seed=p.seed)
    step = C.make_frozen_step(encoder, heads, opt, prec, C.Cycler(pool), p.accum, tracker, scaler)

    extra = C.base_extra(p, prec, det, ctx.tiny, ctx.device)
    extra.update(C.encoder_extra([encoder], p.n_samples))
    extra.update({
        "hidden_states": n_states, "head_sets": p.head_sets,
        "heads": p.head_sets * heads[0].n_heads,
        "head_outputs": p.head_sets * heads[0].n_outputs,
        "params_heads_m": round(C.n_params(heads) / 1e6, 4),
        "optimizer": "AdamW(heads only)", "lr_heads": p.lr_heads, "weight_decay": p.weight_decay,
    })
    C.log(ctx, f"frozen {p.model} {p.seconds:g}s {p.micro_batch}x{p.accum} heads={extra['heads']}")
    measured = measure_loop(step, device=ctx.device, items_per_step=p.effective_batch,
                            cfg=ctx.measure, log=ctx.log)
    return C.finish(measured, p, extra, tracker=tracker)
