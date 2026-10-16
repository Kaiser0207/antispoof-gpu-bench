"""``infer.*``: Dev/Eval-style scoring from pinned host batches (SPEC 5.4)."""

from __future__ import annotations

from agbench.workloads import common as C


def run(ctx) -> dict:
    from agbench.timing import measure_loop

    p, det, prec = C.prepare(ctx)
    model = C.build_classifier(ctx, p, grad_ckpt=False)
    model.eval()
    pool = C.make_micro_pool(p, device=ctx.device, seed=p.seed, host_pinned=True)
    sink = C.ScoreSink()
    step = C.make_infer_step(model, prec, C.Cycler(pool), p.accum, ctx.device, sink)

    extra = C.base_extra(p, prec, det, ctx.tiny, ctx.device)
    extra.update(C.encoder_extra(model.encoders, p.n_samples))
    extra.update({"input_path": "pinned host -> device, non_blocking" if C.is_cuda(ctx.device)
                  else "host tensors (cpu device)",
                  "pinned": bool(pool[0][0].is_pinned()) if C.is_cuda(ctx.device) else False})
    C.log(ctx, f"infer {p.model} {p.seconds:g}s {p.micro_batch}x{p.accum} {prec.effective}")
    measured = measure_loop(step, device=ctx.device, items_per_step=p.effective_batch,
                            cfg=ctx.measure, log=ctx.log)
    return C.finish(measured, p, extra, sink=sink)
