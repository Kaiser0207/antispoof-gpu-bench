"""Command line (SPEC 7): python -m agbench doctor|fetch|prepare|list|quick|full|run|report.

The run commands (quick/full/run) orchestrate child processes and never import torch here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from . import constants as C
from .util import ascii_print, atomic_write_json, default_results_dir, local_date, repo_root, sanitize

# ---------------------------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------------------------


def _add_run_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("run options")
    g.add_argument("--gpu", type=int, default=0,
                   help="GPU index as listed by 'nvidia-smi -L' (default 0); only needed with several GPUs")
    g.add_argument("--out", type=Path, default=None,
                   help="results JSONL (default results/<gpu_slug>_<YYYYMMDD>.jsonl)")
    g.add_argument("--no-resume", action="store_true", help="ignore earlier results; run everything again")
    g.add_argument("--no-retry-failed", action="store_true",
                   help="on resume, also skip scenarios whose last status was error/timeout/crashed")
    g.add_argument("--retry-oom", action="store_true",
                   help="on resume, run again the scenarios that ended oom (use after closing a program that "
                        "held GPU memory)")
    g.add_argument("--budget-min", type=float, default=None,
                   help="soft time budget in minutes (default quick 15, full 90, run 0); 0 = unlimited")
    g.add_argument("--mem-cap-gib", type=float, default=None,
                   help="cap PyTorch's allocator at X GiB in every child (shared GPUs / smoke runs)")
    g.add_argument("--timeout-scale", type=float, default=1.0, help="multiply every timeout (default 1.0)")
    g.add_argument("--tiny", action="store_true",
                   help="tiny random models, tiny corpus, short loops; no downloads (tests, CI)")
    g.add_argument("--smoke", action="store_true",
                   help="maintainers only: real model weights, but batch 2, <= 2 s clips, short loops and small "
                        "GPU-basics buffers; checks the GPU code paths on a shared GPU (combine with "
                        "--mem-cap-gib). The numbers are NOT benchmark results")
    g.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                   help="default cuda; cpu runs every workload on the CPU (meant for --tiny)")
    g.add_argument("--allow-cpu-only", action="store_true",
                   help="maintainers only: if the doctor finds no usable GPU, still run the cpu.* scenarios "
                        "(the results then have no GPU numbers)")
    g.add_argument("--dry-run", action="store_true", help="print the plan with estimated times and exit")
    g.add_argument("--no-fetch", action="store_true",
                   help="do not download missing models (their scenarios are skipped: model_missing)")
    g.add_argument("--no-report", action="store_true", help="do not build the report at the end")
    g.add_argument("--no-network", action="store_true", help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agbench",
                                description="Synthetic-audio GPU benchmark for speech anti-spoofing workloads. "
                                            "Typical use: doctor, fetch, quick, (full).")
    p.add_argument("--version", action="version", version=f"agbench {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    d = sub.add_parser("doctor", help="check Python, torch build, GPU, driver, disk, models")
    d.add_argument("--gpu", type=int, default=0, help="GPU index as listed by 'nvidia-smi -L' (default 0)")
    d.add_argument("--json", type=Path, default=None, help="also write the result to this file")
    d.add_argument("--no-network", action="store_true", help="skip the Hugging Face reachability check")
    d.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help=argparse.SUPPRESS)
    d.add_argument("--tiny", action="store_true", help=argparse.SUPPRESS)
    d.add_argument("--results-dir", type=Path, default=None, help=argparse.SUPPRESS)

    f = sub.add_parser("fetch", help="download the pinned Hugging Face models (~3.3 GB)")
    f.add_argument("--models", default=",".join(C.MODEL_KEYS),
                   help=f"comma-separated keys (default all: {','.join(C.MODEL_KEYS)})")
    f.add_argument("--check", action="store_true", help="only verify what is cached")
    f.add_argument("--json", type=Path, default=None, help=argparse.SUPPRESS)

    pr = sub.add_parser("prepare", help="generate the synthetic FLAC corpus")
    pr.add_argument("--tiny", action="store_true")

    ls = sub.add_parser("list", help="list scenarios with estimated times")
    ls.add_argument("--suite", choices=("quick", "full"), default="full")
    ls.add_argument("--format", choices=("table", "json"), default="table")

    for name, helptext in (("quick", "most informative subset (about 10-15 min on an RTX 5060 Ti)"),
                           ("full", "all 49 scenarios (about 35-50 min in total; resumes after quick)")):
        sp = sub.add_parser(name, help=helptext)
        _add_run_options(sp)
    r = sub.add_parser("run", help="run selected scenarios")
    r.add_argument("--scenario", action="append", required=True, metavar="ID",
                   help="scenario id (repeatable, or comma-separated)")
    _add_run_options(r)

    rp = sub.add_parser("report", help="build summary.md, comparison.csv and figures from results")
    rp.add_argument("jsonl", nargs="*", type=Path, help="results files (default: all results/*.jsonl)")
    rp.add_argument("--out", type=Path, default=None, help="output directory")
    rp.add_argument("--references", type=Path, default=None, help="reference_gpus.json")
    return p


# ---------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------


def _set_gpu_env(gpu: int, device: str) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if device == "cpu" else str(gpu)


def cmd_doctor(a) -> int:
    _set_gpu_env(a.gpu, a.device)
    from .env import CONSOLE_ONLY_KEY, doctor, format_doctor, gpu_slug

    res = doctor(a.gpu, network=not a.no_network, device=a.device, tiny=a.tiny,
                 results_dir=a.results_dir or default_results_dir())
    console_notes = res.pop(CONSOLE_ONLY_KEY, None) or []
    ascii_print(format_doctor(res))
    if a.json is None:  # interactive use only: the runner saves this command's output in results/logs
        for note in console_notes:
            ascii_print(f"    {note}")
    rdir = Path(a.results_dir) if a.results_dir else default_results_dir()
    slug = gpu_slug(res.get("env") or {})
    try:
        atomic_write_json(rdir / f"doctor_{slug}_{local_date()}.json", res)
        if a.json:
            atomic_write_json(Path(a.json), res)
    except OSError as e:
        ascii_print(f"WARNING: could not write the doctor result: {sanitize(str(e))}")
    return 2 if res["status"] == "FATAL" else 0


def cmd_fetch(a) -> int:
    keys = [k.strip() for k in str(a.models).split(",") if k.strip()]
    bad = [k for k in keys if k not in C.MODEL_KEYS]
    if bad:
        ascii_print(f"unknown model key(s): {', '.join(bad)} (known: {', '.join(C.MODEL_KEYS)})")
        return 2
    from . import hub

    rows = hub.check(keys) if a.check else hub.prefetch(keys)
    ascii_print(f"{'model':<8} {'status':<11} {'bytes':>14}  detail")
    for r in rows:
        ascii_print(f"{r.get('key', '?'):<8} {str(r.get('status')):<11} {str(r.get('bytes') or '-'):>14}  "
                    f"{sanitize(str(r.get('error') or r.get('path') or ''))[:160]}")
    if any("429" in str(r.get("error") or "") for r in rows):
        ascii_print("Hugging Face rate limit: wait and rerun `fetch`, or set HF_TOKEN to a read token")
    if a.json:
        atomic_write_json(Path(a.json), {r["key"]: {k: r.get(k) for k in ("status", "bytes", "error")}
                                         for r in rows})
    ok = all(r.get("status") in ("cached", "downloaded") for r in rows)
    if not ok and not a.check:
        ascii_print("Some downloads failed; rerun the same command (downloads resume). Mirrors: set "
                    "HF_ENDPOINT (and HF_HUB_DISABLE_XET=1); offline: copy a Hugging Face cache and set "
                    "HF_HUB_OFFLINE=1, or use AGBENCH_MODELS_DIR.")
    return 0 if ok else 1


def cmd_prepare(a) -> int:
    from .data.synth import ensure_corpus
    from .util import default_cache_dir

    manifest = ensure_corpus(default_cache_dir(), tiny=a.tiny)
    try:
        info = json.loads(Path(manifest).read_text(encoding="utf-8"))
        n = len(info.get("files", []))
        mb = (info.get("total_bytes") or 0) / 1e6
        ascii_print(f"corpus ready: {n} FLAC files, {mb:.1f} MB -> {sanitize(str(manifest))}")
    except Exception:
        ascii_print(f"corpus ready -> {sanitize(str(manifest))}")
    return 0


def cmd_list(a) -> int:
    from .scenarios import SCENARIOS, SUITES, scenario_hash, suite_est_s, timeout_s

    ids = SUITES[a.suite]
    if a.format == "json":
        rows = []
        for sid in ids:
            s = SCENARIOS[sid]
            rows.append({"id": s.id, "category": s.category, "workload": s.workload, "suites": list(s.suites),
                         "params": s.params, "needs_gpu": s.needs_gpu, "models": list(s.models),
                         "ladder": [list(r) for r in s.ladder] if s.ladder else None, "est_s": s.est_s,
                         "timeout_s": timeout_s(s), "mem_est_gib": s.mem_est_gib, "env": s.env,
                         "scenario_hash": scenario_hash(s)})
        print(json.dumps({"suite": a.suite, "n": len(rows), "est_s": suite_est_s(a.suite), "scenarios": rows},
                         indent=1))
        return 0
    ascii_print(f"{'#':>3}  {'scenario':<36} {'suites':<6} {'est s':>6} {'mem GiB':>8}  ladder")
    for i, sid in enumerate(ids, 1):
        s = SCENARIOS[sid]
        lad = " -> ".join(f"{m}x{k}" for m, k in s.ladder) if s.ladder else "-"
        mem = f"{s.mem_est_gib:.1f}" if s.mem_est_gib else "-"
        su = "Q F" if "quick" in s.suites else "F"
        ascii_print(f"{i:>3}  {sid:<36} {su:<6} {s.est_s:>6.0f} {mem:>8}  {lad}")
    tot = suite_est_s(a.suite)
    ascii_print(f"{len(ids)} scenarios in '{a.suite}', estimated {tot / 60:.1f} min + ~{C.SETUP_EST_S:.0f} s "
                f"setup = {(tot + C.SETUP_EST_S) / 60:.1f} min on an RTX 5060 Ti 16 GB (Linux; Windows ~x1.3)")
    return 0


def cmd_run(a) -> int:
    from .runner import RunOptions, run_suite
    from .scenarios import SUITES

    if a.tiny and a.smoke:
        ascii_print("ERROR: use either --tiny (random tiny models) or --smoke (real weights, small shapes)")
        return 2
    if a.cmd == "run":
        ids = [x.strip() for item in a.scenario for x in item.split(",") if x.strip()]
    else:
        ids = list(SUITES[a.cmd])
    opts = RunOptions(suite=a.cmd, gpu=a.gpu, out=a.out, resume=not a.no_resume,
                      retry_failed=not a.no_retry_failed, retry_oom=a.retry_oom, budget_min=a.budget_min,
                      mem_cap_gib=a.mem_cap_gib,
                      timeout_scale=a.timeout_scale, tiny=a.tiny, smoke=a.smoke, device=a.device,
                      allow_cpu_only=a.allow_cpu_only, dry_run=a.dry_run, fetch=not a.no_fetch,
                      report=not a.no_report, network=not a.no_network)
    try:
        return run_suite(ids, opts)
    except ValueError as e:
        ascii_print(f"ERROR: {e}")
        return 2


def cmd_report(a) -> int:
    from .report import build_report

    paths = list(a.jsonl)
    if not paths:
        rdir = default_results_dir()
        paths = sorted(p for p in rdir.glob("*.jsonl") if p.is_file())
        if not paths:
            ascii_print(f"no results/*.jsonl found in {sanitize(str(rdir))}")
            return 1
    refs = a.references
    if refs is None:
        root = repo_root()
        refs = (root / "references" / "reference_gpus.json") if root else Path("references/reference_gpus.json")
    summary = build_report(paths, a.out, references_path=refs)
    ascii_print(f"Report: {sanitize(str(summary))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd in ("doctor", "fetch", "prepare"):
        from .child import watch_parent

        watch_parent()  # started by a runner: exit if that runner is killed
    handlers = {"doctor": cmd_doctor, "fetch": cmd_fetch, "prepare": cmd_prepare, "list": cmd_list,
                "quick": cmd_run, "full": cmd_run, "run": cmd_run, "report": cmd_report}
    try:
        return int(handlers[args.cmd](args) or 0)
    except KeyboardInterrupt:
        ascii_print("Interrupted.")
        return 130
    except ModuleNotFoundError as e:
        if e.name == "torch":
            ascii_print(C.TORCH_MISSING_MSG)
            return 2
        raise
