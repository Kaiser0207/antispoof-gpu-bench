"""Readable report from results JSONL files: summary.md, comparison.csv and figures/*.png.

    build_report([Path("results/rtx-5060-ti-16gb_20261016.jsonl")])
    build_report([a_jsonl, b_jsonl])          # several GPUs -> one column per GPU

Uses the latest terminal record per (gpu_slug, scenario_id). Never imports torch; matplotlib is
used through ``agbench.plots`` and a missing or failing figure never stops the report.
Reference numbers (references/reference_gpus.json) come from our real training logs and are
labelled as such everywhere; they are not runs of this benchmark.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from . import constants as C

LOG_REF_LABEL = "reference: training log, data loading included, torch 2.2.2"
STATUSES = ("ok", "oom", "skipped", "error", "timeout", "crashed")

MODEL_NAMES = {
    "w2v2b": "W2V2 Base",
    "xlsr": "XLS-R 300M",
    "w2v2l": "W2V2 Large (LS-960)",
    "wavlmb": "WavLM Base",
    "fusion": "XLS-R 300M + WavLM Base",
}
MODEL_SHORT = {"w2v2b": "W2V2-B", "xlsr": "XLS-R", "w2v2l": "W2V2-L", "wavlmb": "WavLM-B",
               "fusion": "XLS-R+WavLM"}
VARIANT_NAMES = {"gmean": "global-mean head", "det": "reference numerics", "ckpt": "act. checkpointing",
                 "h40": "40 heads", "rawboost": "RawBoost", "repeat": "repeat after soak"}
PRECISION_NAMES = {"bf16": "bf16", "fp16": "fp16+GradScaler", "tf32": "TF32", "fp32": "fp32"}

# (label, GPU-only training id, with-loader id or None, matching inference id)
TRAIN_WORKLOADS = [
    ("W2V2 Base, 6 s", "train.w2v2b.6s.bs16.bf16", "e2e.w2v2b.6s.bs16.bf16", "infer.w2v2b.6s.bs16.bf16"),
    ("XLS-R 300M, 6 s", "train.xlsr.6s.bs16.bf16", "e2e.xlsr.6s.bs16.bf16", "infer.xlsr.6s.bs16.bf16"),
    ("W2V2 Large (LS-960), 6 s", "train.w2v2l.6s.bs16.bf16", None, "infer.w2v2l.6s.bs16.bf16"),
    ("WavLM Base, 6 s", "train.wavlmb.6s.bs16.bf16", None, "infer.wavlmb.6s.bs16.bf16"),
    ("XLS-R 300M + WavLM Base, 4 s", "train.fusion.4s.bs16.bf16", None, "infer.fusion.4s.bs16.bf16"),
]
ANCHOR = "train.w2v2b.6s.bs16.bf16"
XLSR_ANCHOR = "train.xlsr.6s.bs16.bf16"
VARIANTS = [
    ("W2V2 Base: bf16 autocast (anchor)", ANCHOR, ANCHOR),
    ("W2V2 Base: fp16 autocast + GradScaler", "train.w2v2b.6s.bs16.fp16", ANCHOR),
    ("W2V2 Base: TF32 (fp32 weights, TF32 matmul)", "train.w2v2b.6s.bs16.tf32", ANCHOR),
    ("W2V2 Base: bf16, numerics of our training logs (strict deterministic algorithms, TF32 off, "
     "flash attention off)", "train.w2v2b.6s.bs16.bf16.det", ANCHOR),
    ("W2V2 Base: bf16, activation checkpointing on", "train.w2v2b.6s.bs16.bf16.ckpt", ANCHOR),
    ("W2V2 Base: bf16, global-mean head", "train.w2v2b.6s.bs16.bf16.gmean", ANCHOR),
    ("W2V2 Base: bf16, batch 8", "train.w2v2b.6s.bs8.bf16", ANCHOR),
    ("W2V2 Base: bf16, batch 32 (no fallback)", "train.w2v2b.6s.bs32.bf16", ANCHOR),
    ("W2V2 Base: anchor repeated after the 300 s sustained run", "train.w2v2b.6s.bs16.bf16.repeat", ANCHOR),
    ("XLS-R 300M: bf16 autocast (anchor)", XLSR_ANCHOR, XLSR_ANCHOR),
    ("XLS-R 300M: TF32", "train.xlsr.6s.bs16.tf32", XLSR_ANCHOR),
    ("XLS-R 300M: bf16, activation checkpointing on", "train.xlsr.6s.bs16.bf16.ckpt", XLSR_ANCHOR),
]
INFER_ROWS = ["infer.w2v2b.6s.bs16.bf16", "infer.w2v2b.6s.bs4.bf16", "infer.xlsr.6s.bs16.bf16",
              "infer.xlsr.6s.bs4.bf16", "infer.w2v2l.6s.bs16.bf16", "infer.wavlmb.6s.bs16.bf16",
              "infer.fusion.4s.bs16.bf16"]
LENGTH_IDS = [(s, f"train.w2v2b.{s}s.bs16.bf16") for s in (2, 4, 6, 8, 10)]

FIGURE_NAMES = {
    "fig1": "fig1_training_throughput.png",
    "fig2": "fig2_memory_fit.png",
    "fig3": "fig3_length_scaling.png",
    "fig4": "fig4_cpu_pipeline.png",
    "fig5": "fig5_sustained.png",
}


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------


@dataclass
class GpuRuns:
    """Everything known about one GPU (one gpu_slug) across the given JSONL files."""

    slug: str
    run_starts: list = field(default_factory=list)
    run_ends: list = field(default_factory=list)
    candidates: dict = field(default_factory=dict)   # scenario_id -> [(seq, record)]
    records: dict = field(default_factory=dict)      # scenario_id -> chosen record
    sources: list = field(default_factory=list)
    reference_label: str | None = None               # set for references/agbench_runs entries
    ref_key: str | None = None

    @property
    def env(self) -> dict:
        return (self.run_starts[-1].get("env") or {}) if self.run_starts else {}

    @property
    def tiny(self) -> bool:
        """True if any run was a tiny/smoke/CPU test run (numbers not comparable with real GPUs)."""
        return any((r.get("settings") or {}).get("tiny") or (r.get("settings") or {}).get("smoke")
                   or (r.get("settings") or {}).get("device") == "cpu"
                   for r in self.run_starts)

    @property
    def smoke(self) -> bool:
        return any((r.get("settings") or {}).get("smoke") for r in self.run_starts)

    @property
    def label(self) -> str:
        if self.reference_label:
            return self.reference_label
        base = gpu_label(self.env.get("gpu_name"), self.env.get("vram_class_gb"), self.slug)
        if self.smoke:
            return f"{base} (smoke test)"
        if self.tiny and self.env.get("gpu_name") and self.slug != "cpu":
            return f"{base} (tiny test)"
        return base

    def rec(self, sid: str) -> dict | None:
        return self.records.get(sid)

    def ok(self, sid: str) -> dict | None:
        r = self.records.get(sid)
        return r if r and r.get("status") == "ok" else None

    def metric(self, sid: str, key: str):
        r = self.ok(sid)
        return _num((r.get("metrics") or {}).get(key)) if r else None


def gpu_label(name: str | None, vram_class, slug: str) -> str:
    if not name:
        return slug
    label = re.sub(r"^(NVIDIA\s+)?(GeForce\s+)?", "", str(name)).strip()
    if vram_class and not re.search(r"\d+\s*GB$", label, re.I):
        label = f"{label} {vram_class} GB"
    return label


def _num(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _read_records(path: Path) -> list[dict]:
    try:
        from .schema import load_records

        return list(load_records(path))
    except Exception:  # noqa: BLE001 - fall back to a tolerant local reader
        out = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
        return out


def _skip_reason(rec: dict) -> str:
    return re.split(r"[\s:]", str(rec.get("status_detail") or ""), maxsplit=1)[0]


def _choose(cands: list) -> dict:
    """Latest terminal record; a retryable skip (budget, ...) never hides a real result."""
    retryable = getattr(C, "RETRYABLE_SKIP_REASONS", frozenset({"budget"}))

    def key(item):
        seq, rec = item
        soft = rec.get("status") == "skipped" and _skip_reason(rec) in retryable
        return (0 if soft else 1, str(rec.get("finished_utc") or ""), seq)

    return max(cands, key=key)[1]


def load_runs(paths: list[Path]) -> list[GpuRuns]:
    gpus: dict[str, GpuRuns] = {}
    run_slug: dict[str, str] = {}
    seq = 0
    for path in paths:
        path = Path(path)
        for rec in _read_records(path):
            seq += 1
            kind = rec.get("kind")
            slug = rec.get("gpu_slug") or run_slug.get(str(rec.get("run_id")), "unknown")
            if kind == "run_start":
                run_slug[str(rec.get("run_id"))] = slug
            g = gpus.setdefault(slug, GpuRuns(slug))
            if path.name not in g.sources:
                g.sources.append(path.name)
            if kind == "run_start":
                g.run_starts.append(rec)
            elif kind == "run_end":
                g.run_ends.append(rec)
            elif kind == "scenario" and rec.get("scenario_id"):
                g.candidates.setdefault(rec["scenario_id"], []).append((seq, rec))
    out = []
    for g in gpus.values():
        g.records = {sid: _choose(c) for sid, c in g.candidates.items()}
        if g.records or g.run_starts:
            out.append(g)
    return out


# ---------------------------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------------------------


@dataclass
class RefValue:
    gpu_key: str
    label: str
    value: float
    is_estimate: bool = False
    cpu_bound: bool = False
    source: str = "log"          # "log" (training log) or "agbench" (a run of this benchmark)

    def cell(self) -> str:
        s = _fmt_ref(self.value)
        if self.is_estimate:
            s += " (est.)"
        if self.cpu_bound:
            s += " (CPU-bound)"
        return s


_REF_FIELDS = {
    "train": {"clips_per_s": "train_clips_per_s", "asv5_train_epoch_min": "train_min_per_epoch",
              "peak_alloc_gib": "peak_alloc_gib", "peak_reserved_gib": "peak_reserved_gib"},
    "infer": {"clips_per_s": "dev_clips_per_s", "asv5_dev_pass_min": "dev_min"},
}


def _mode_of(sid: str) -> str:
    return sid.split(".", 1)[0]


def _swap_mode(sid: str, mode: str) -> str:
    parts = sid.split(".", 1)
    return f"{mode}.{parts[1]}" if len(parts) == 2 else sid


class References:
    def __init__(self, data: dict | None, base_dir: Path | None, path: Path | None = None):
        self.data = data or {}
        self.base_dir = base_dir
        self.path = path
        self.gpus = [g for g in self.data.get("gpus", []) if isinstance(g, dict)]
        self.estimates = [e for e in self.data.get("estimates", []) if isinstance(e, dict)]
        self.agbench_runs: dict[str, GpuRuns] = {}
        self.warnings: list[str] = []
        for g in self.gpus:
            for entry in g.get("agbench_runs") or []:
                self._load_agbench_run(g, entry)

    @property
    def available(self) -> bool:
        return bool(self.gpus)

    def _load_agbench_run(self, g: dict, entry) -> None:
        rel = entry.get("file") if isinstance(entry, dict) else entry
        if not rel or self.base_dir is None:
            return
        p = (self.base_dir / str(rel)).resolve()
        if not p.exists():
            self.warnings.append(f"reference run file not found: {rel}")
            return
        runs = load_runs([p])
        if runs:
            run = runs[0]
            run.reference_label = f"{g.get('label', g.get('gpu_key'))} (this benchmark)"
            run.ref_key = g.get("gpu_key")
            self.agbench_runs[str(g.get("gpu_key"))] = run

    def label(self, gpu_key: str) -> str:
        for g in self.gpus:
            if g.get("gpu_key") == gpu_key:
                return str(g.get("label", gpu_key))
        return gpu_key

    def log_value(self, gpu: dict, sid: str, metric: str) -> RefValue | None:
        mode = _mode_of(sid)
        kind = "infer" if mode == "infer" else ("train" if mode in ("train", "e2e", "sustain") else None)
        if kind is None:
            return None
        fname = _REF_FIELDS[kind].get(metric)
        for e in gpu.get("log_references") or []:
            maps = [str(m) for m in e.get("maps_to") or []]
            direct = sid in maps
            # Dev throughput logged with a training run applies to the matching inference scenario.
            derived = kind == "infer" and any(_swap_mode(m, "infer") == sid for m in maps)
            if not (direct or derived):
                continue
            val = _num(e.get(fname)) if fname else None
            if val is None and kind == "infer" and metric == "asv5_dev_pass_min":
                dev = _num(e.get("dev_clips_per_s"))
                val = C.ASV5_DEV_CLIPS / dev / 60.0 if dev else None
            if val is None:
                continue
            return RefValue(str(gpu.get("gpu_key")), str(gpu.get("label")), val,
                            is_estimate=bool(e.get("is_estimate")), cpu_bound=bool(e.get("cpu_bound")))
        return None

    def value(self, gpu_key: str, sid: str, metric: str, *, prefer_agbench: bool = True) -> RefValue | None:
        run = self.agbench_runs.get(gpu_key)
        if prefer_agbench and run is not None:
            v = run.metric(sid, metric)
            if v is None and metric.startswith("asv5_"):
                v = _num(((run.ok(sid) or {}).get("estimates") or {}).get(metric))
            if v is not None:
                return RefValue(gpu_key, self.label(gpu_key), v, source="agbench")
        for g in self.gpus:
            if g.get("gpu_key") == gpu_key:
                return self.log_value(g, sid, metric)
        return None

    def keys(self) -> list[str]:
        return [str(g.get("gpu_key")) for g in self.gpus]

    def estimate_for(self, slug: str) -> dict | None:
        for e in self.estimates:
            if e.get("gpu_key") == slug:
                return e
        return None


def _resolve_refs_path(p: Path | str | None) -> Path | None:
    if p is None:
        return None
    p = Path(p)
    if p.exists():
        return p
    if not p.is_absolute():
        alt = Path(__file__).resolve().parent.parent / p
        if alt.exists():
            return alt
    return None


def load_references(path: Path | str | None) -> References:
    rp = _resolve_refs_path(path)
    if rp is None:
        refs = References(None, None)
        if path is not None:
            refs.warnings.append(f"reference file not found ({Path(path).name}); the report has no reference "
                                 "columns")
        return refs
    try:
        data = json.loads(rp.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        refs = References(None, None)
        refs.warnings.append(f"reference file unreadable: {type(exc).__name__}")
        return refs
    return References(data, rp.parent, rp)


# ---------------------------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------------------------


def _fmt(v, kind: str = "num") -> str:
    v = _num(v)
    if v is None:
        return "-"
    if kind == "gib":
        return f"{v:.1f}"
    if kind == "min":
        return f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}"
    if kind == "ratio":
        return f"{v:.2f}x"
    if kind == "pct":
        return f"{100 * v:.0f} %"
    a = abs(v)
    if a >= 100 or (a >= 10 and v == int(v)):
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _fmt_ref(v) -> str:
    """Reference numbers are shown exactly as recorded (101.8, not 102)."""
    v = _num(v)
    return "-" if v is None else f"{v:g}"


def _md(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _table(header: list[str], rows: list[list[str]], align: list[str] | None = None) -> list[str]:
    align = align or (["l"] + ["r"] * (len(header) - 1))
    sep = ["---:" if a == "r" else ":---" for a in align]
    out = ["| " + " | ".join(_md(h) for h in header) + " |", "| " + " | ".join(sep) + " |"]
    for r in rows:
        out.append("| " + " | ".join(_md(c) for c in r) + " |")
    return out


def describe(sid: str) -> str:
    """Short human label, e.g. 'XLS-R 6 s, batch 16, bf16 (train)'."""
    p = sid.split(".")
    if p[0] in ("train", "e2e", "frozen", "infer", "sustain") and len(p) >= 5:
        model = MODEL_SHORT.get(p[1], p[1])
        secs = p[2].rstrip("s")
        bs = p[3].replace("bs", "")
        prec = PRECISION_NAMES.get(p[4], p[4])
        var = f", {VARIANT_NAMES.get(p[5], p[5])}" if len(p) > 5 else ""
        mode = {"e2e": "train + loader"}.get(p[0], p[0])
        return f"{model} {secs} s, batch {bs}, {prec}{var} ({mode})"
    return sid


def fit_text(rec: dict | None) -> str:
    """Which micro-batch x accumulation ran, e.g. '16x1', '8x2 (OOM at 16x1)', 'OOM at every split'."""
    if not rec:
        return "-"
    status = rec.get("status")
    fb = rec.get("fallback") or {}
    attempts = [a for a in fb.get("attempts") or [] if isinstance(a, dict)]
    if status == "oom":
        if attempts:
            return f"OOM at every split ({attempts[0].get('micro_batch')}x{attempts[0].get('accum')} to " \
                   f"{attempts[-1].get('micro_batch')}x{attempts[-1].get('accum')}{_cap_text(attempts[0])})"
        return "OOM"
    if status != "ok":
        return status or "-"
    used = fb.get("used")
    if isinstance(used, dict) and used.get("micro_batch"):
        s = f"{used.get('micro_batch')}x{used.get('accum', 1)}"
        first_oom = next((a for a in attempts if a.get("status") == "oom"), None)
        if first_oom:
            s += f" (OOM at {first_oom.get('micro_batch')}x{first_oom.get('accum')}{_cap_text(first_oom)})"
        return s
    eb = (rec.get("metrics") or {}).get("effective_batch") or (rec.get("params") or {}).get("batch")
    return f"{eb}x1" if eb else "-"


def _cap_text(attempt: dict) -> str:
    """' under a 14.1 GiB cap; 14.6 GiB free' for an OOM attempt (empty if unknown): tells 'does not
    fit in this card' apart from 'did not fit next to the desktop / other programs'."""
    cap, free = _num(attempt.get("mem_cap_gib")), _num(attempt.get("vram_free_start_gib"))
    if cap is None:
        return ""
    return f" under a {cap:.1f} GiB cap" + (f"; {free:.1f} GiB were free" if free is not None else "")


def _fit_short(rec: dict | None) -> str:
    """Bar annotation: only when a fallback was needed."""
    if not rec or rec.get("status") != "ok":
        return ""
    fb = rec.get("fallback") or {}
    used = fb.get("used")
    if isinstance(used, dict) and fb.get("fits_requested") is False:
        return f"(ran as {used.get('micro_batch')}x{used.get('accum', 1)})"
    return ""


def _peak_gib(rec: dict | None):
    if not rec:
        return None
    if rec.get("status") == "ok":
        return _num((rec.get("metrics") or {}).get("peak_reserved_gib"))
    attempts = [a for a in (rec.get("fallback") or {}).get("attempts") or [] if isinstance(a, dict)]
    for a in reversed(attempts):
        if _num(a.get("peak_reserved_gib")) is not None:
            return _num(a.get("peak_reserved_gib"))
    return None


def _memory_sort_key(sid: str):
    """Group the memory figure by mode, then model, then length / batch / variant."""
    p = sid.split(".")
    modes = ["train", "e2e", "frozen"]
    models = list(MODEL_NAMES)
    try:
        secs = float(p[2].rstrip("s"))
        bs = int(p[3].replace("bs", ""))
    except (IndexError, ValueError):
        secs, bs = 0.0, 0
    variant = p[5] if len(p) > 5 else ""
    return (modes.index(p[0]) if p[0] in modes else 9, models.index(p[1]) if len(p) > 1 and p[1] in models else 9,
            p[4] if len(p) > 4 else "", variant != "", variant, secs, bs)


def _date_of(iso: str | None) -> str | None:
    if not iso or len(iso) < 10:
        return None
    return iso[:10]


# ---------------------------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------------------------


class _Builder:
    def __init__(self, gpus: list[GpuRuns], refs: References, out_dir: Path):
        from . import plots

        self.plots = plots
        self.gpus = gpus
        self.refs = refs
        self.out = out_dir
        self.fig_dir = out_dir / "figures"
        self.lines: list[str] = []
        self.colors = {}
        for i, g in enumerate(gpus):
            self.colors[g.slug] = plots.MEASURED_COLORS[i % len(plots.MEASURED_COLORS)]
        self.multi = len(gpus) > 1
        self.figures_written: list[str] = []
        self.section_no = 0

    # -- small helpers ----------------------------------------------------------------------
    def ref_color(self, key: str) -> str:
        return self.plots.REFERENCE_COLORS.get(key, self.plots.OTHER_REFERENCE_COLOR)

    def add(self, *lines: str) -> None:
        self.lines.extend(lines)

    def section(self, title: str) -> None:
        self.section_no += 1
        self.add(f"## {self.section_no}. {title}", "")

    def figure(self, key: str, title: str, fn, method: str) -> None:
        name = FIGURE_NAMES[key]
        try:
            ok = fn(self.fig_dir / name)
        except Exception as exc:  # noqa: BLE001 - a figure must never stop the report
            self.add(f"*{title}: the figure could not be drawn ({type(exc).__name__}: {_md(str(exc))[:160]}).*", "")
            return
        if ok is False:
            self.add(f"*{title}: no data in these results (the scenarios were not run or did not finish).*", "")
            return
        self.figures_written.append(name)
        self.add(f"![{title}](figures/{name})", "", f"*Method:* {method}", "")

    def gpu_col(self, g: GpuRuns) -> list[str]:
        return [g.label] if self.multi else []

    def gpu_head(self) -> list[str]:
        return ["GPU"] if self.multi else []

    # -- sections ---------------------------------------------------------------------------
    def header(self) -> None:
        g0 = self.gpus[0]
        if self.multi:
            self.add("# GPU benchmark comparison: " + " vs ".join(g.label for g in self.gpus), "")
        else:
            env = g0.env
            name = re.sub(r"^NVIDIA\s+", "", str(env.get("gpu_name") or g0.slug))
            vc = env.get("vram_class_gb")
            test = " - smoke test, not a benchmark" if g0.smoke else (
                " - tiny test, not a benchmark" if g0.tiny and g0.slug != "cpu" else "")
            self.add(f"# GPU benchmark: {name}" + (f" ({vc} GB)" if vc else "") + test, "")
        for g in self.gpus:
            env = g.env
            dates = sorted({d for d in (_date_of(r.get("started_utc")) for r in g.run_starts) if d})
            suites = []
            for r in g.run_starts:
                if r.get("suite") and r["suite"] not in suites:
                    suites.append(r["suite"])
            counts = {s: 0 for s in STATUSES}
            for r in g.records.values():
                counts[r.get("status", "error")] = counts.get(r.get("status", "error"), 0) + 1
            count_txt = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
            torch_txt = env.get("torch", "?")
            if env.get("torch_cuda"):
                torch_txt += f" (CUDA {env['torch_cuda']})"
            parts = [
                ("dates " if len(dates) > 1 else "date ") + (", ".join(dates) or "?"),
                "suite " + (" + ".join(suites) or "?"),
                str(env.get("os", env.get("os_family", "?"))),
                f"driver {env.get('driver') or '-'}",
                f"torch {torch_txt}",
                f"{len(g.records)} scenarios: {count_txt or 'none'}",
            ]
            prefix = f"**{g.label}:** " if self.multi else ""
            self.add(prefix + " · ".join(parts), "")

    def checks(self) -> None:
        notes = []
        for g in self.gpus:
            if g.smoke:
                notes.append(f"{g.label}: this is a **smoke test run** (real weights, but batch 2, clips of at "
                             "most 2 s, short loops, often on a shared GPU under a memory cap). Its numbers only "
                             "show that the GPU code paths work; they say nothing about GPU speed and are not "
                             "compared with the references.")
            elif g.tiny:
                notes.append(f"{g.label}: this is a **tiny / CPU test run** (random tiny models, short loops). "
                             "Its numbers only show that the harness works; they say nothing about GPU speed "
                             "and are not compared with the references.")
        for g in self.gpus:
            who = f"{g.label}: " if self.multi else ""
            for sid, r in g.records.items():
                m = r.get("metrics") or {}
                if m.get("sysmem_suspected"):
                    notes.append(f"{who}`{sid}` may have spilled into shared system memory (Windows); its speed is "
                                 "not trustworthy.")
                if m.get("loss_finite") is False:
                    notes.append(f"{who}`{sid}` produced a non-finite loss.")
                cv = _num(m.get("step_ms_cv"))
                if cv is not None and cv > 0.15:
                    notes.append(f"{who}`{sid}` had noisy step times (coefficient of variation {cv:.2f}); "
                                 "something else may have used the GPU or CPU.")
            for sid, r in g.records.items():
                failed = (r.get("extra") or {}).get("deterministic_strict_failed")
                if failed and r.get("status") == "ok":
                    notes.append(f"{who}`{sid}`: an operation has no deterministic kernel in this torch build "
                                 f"({_md(str(failed)[:160])}); it ran with deterministic kernels where available "
                                 "(warn-only mode) instead of strict mode.")
            if not g.tiny and g.records and not any(_mode_of(s) == "sustain" and r.get("status") == "ok"
                                                    for s, r in g.records.items()):
                notes.append(f"{who}no sustained run in these results: every rate comes from a measurement window "
                             "of at most 60 s, so a card that slows down when hot can look faster here than in a "
                             "long training run. Run `full` for the 5-minute thermal check.")
            doc = (g.env.get("doctor") or {})
            for w in doc.get("warnings") or []:
                notes.append(f"{who}doctor warning: {_md(w)}")
            if g.run_ends and any(r.get("interrupted") for r in g.run_ends):
                notes.append(f"{who}a run was interrupted; rerunning the same command fills the gaps.")
        for g in self.gpus:
            who = f"{g.label}: " if self.multi else ""
            failed = sorted(s for s, r in g.records.items() if r.get("status") in ("error", "timeout", "crashed"))
            oom = sorted(s for s, r in g.records.items() if r.get("status") == "oom")
            if failed:
                notes.append(f"{who}{len(failed)} scenario(s) failed (error, timeout or crash): "
                             + ", ".join(f"`{s}`" for s in failed[:6]) + (" ..." if len(failed) > 6 else "")
                             + ". See *Scenarios that did not finish ok*.")
            if oom:
                notes.append(f"{who}{len(oom)} scenario(s) did not fit in GPU memory even at the smallest batch "
                             "split (a valid result): " + ", ".join(f"`{s}`" for s in oom[:6]) + ".")
        torches = {g.env.get("torch") for g in self.gpus if g.env.get("torch")}
        if len(torches) > 1:
            notes.append("The GPUs ran different torch builds (" + ", ".join(sorted(torches)) +
                         "); part of any difference can come from the software, not the GPU.")
        notes.extend(self.refs.warnings)
        if notes:
            self.add("**Check first:**", "")
            self.add(*[f"- {n}" for n in notes[:20]])
            if len(notes) > 20:
                self.add(f"- ... and {len(notes) - 20} more (see comparison.csv and the logs).")
            self.add("")

    def how_to_read(self) -> None:
        self.add(
            "> **How to read this report.** Speed is in **clips per second** (higher is better). One clip is one "
            "audio excerpt of the stated length (6 s unless noted) at 16 kHz. **GPU only** numbers feed synthetic "
            "audio that is already on the GPU, so they measure the GPU alone. **With loader** numbers also read "
            "and decode FLAC files in 2 DataLoader worker processes and use the numerics of our training logs "
            "(strict deterministic algorithms, TF32 off, flash attention off for 6 s clips), like a real training "
            "run; they are clips divided by wall-clock time, so loader stalls count. Reference numbers "
            "for our GPUs (RTX 3090 and RTX A4500 workstations, A100 40 GB on Google Colab) come from real training "
            "logs (data loading included, torch 2.2.2, strict deterministic algorithms, TF32 off), **not from this "
            "benchmark**, "
            "so ratios against them are indicative, not exact. *Batch 16x1* means 16 clips per step; *8x2* means "
            "the GPU ran out of memory at 16 and the same 16 clips were processed as 2 micro-batches of 8.",
            "")

    def fig_training(self) -> None:
        groups = [w[0] for w in TRAIN_WORKLOADS]
        series = []
        for g in self.gpus:
            vals, loader = {}, {}
            for label, tid, eid, _ in TRAIN_WORKLOADS:
                r = g.rec(tid)
                if r and r.get("status") == "ok":
                    vals[label] = (g.metric(tid, "clips_per_s"), _fit_short(r))
                elif r:
                    what = fit_text(r) if r.get("status") == "oom" else f"not measured ({r.get('status')})"
                    vals[label] = (None, (f"{g.label}: " if self.multi else "") + what)
                if eid and g.ok(eid):
                    loader[label] = (g.metric(eid, "clips_per_s"), _fit_short(g.rec(eid)))
            series.append({"label": f"{g.label}: GPU only (this benchmark)", "color": self.colors[g.slug],
                           "style": "solid", "values": vals})
            if loader:
                series.append({"label": f"{g.label}: with data loader, reference numerics (this benchmark)",
                               "color": self.colors[g.slug], "style": "hollow", "values": loader})
        for key in ([] if all(g.tiny for g in self.gpus) else self.refs.keys()):
            run = self.refs.agbench_runs.get(key)
            if run is not None:
                vals = {lab: (run.metric(tid, "clips_per_s"), "") for lab, tid, _, _ in TRAIN_WORKLOADS
                        if run.metric(tid, "clips_per_s") is not None}
                if vals:
                    series.append({"label": run.label + ": GPU only", "color": self.ref_color(key),
                                   "style": "solid", "values": vals})
            vals = {}
            for lab, tid, _, _ in TRAIN_WORKLOADS:
                gref = next(x for x in self.refs.gpus if x.get("gpu_key") == key)
                rv = self.refs.log_value(gref, tid, "clips_per_s")
                if rv and not rv.cpu_bound:
                    vals[lab] = (rv.value, "est." if rv.is_estimate else "")
            if vals:
                series.append({"label": f"{self.refs.label(key)}: {LOG_REF_LABEL}", "color": self.ref_color(key),
                               "style": "hatched", "values": vals, "fmt": _fmt_ref})
        has_data = any(v[0] is not None for s in series if not s["style"] == "hatched" for v in s["values"].values())

        def draw(path):
            if not has_data:
                return False
            return self.plots.grouped_hbars(path, groups=groups, series=series,
                                            title="Training throughput (full fine-tune, batch 16, bf16)",
                                            xlabel="clips per second (higher is better)")

        self.section("Training throughput vs our GPUs")
        self.figure("fig1", "Figure 1: Training throughput vs our GPUs", draw,
                    "one optimizer step = forward + backward + AdamW update over an effective batch of 16 clips "
                    "(bf16 autocast, encoder LR 1e-5, head LR 1e-4, grad clip 1.0, LayerDrop 0). Solid bars: median "
                    "step time of this benchmark (CUDA events, warm-up excluded) on synthetic audio already on the "
                    "GPU. Hollow bars: wall-clock rate (clips / total time, loader waits included) with FLAC decoding "
                    "in 2 DataLoader workers and the numerics of our training logs (strict deterministic "
                    "algorithms, TF32 off, flash attention off for 6 s). Hatched bars: our training logs (data "
                    "loading included, torch 2.2.2, same numerics); 'est.' marks "
                    "an estimate, not a completed epoch. '(ran as 8x2)' means batch 16 did not fit and was split.")

    def asv5_table(self) -> None:
        self.section("What this means for ASVspoof 5")
        ref_keys = [k for k in ("rtx-3090-24gb", "rtx-a4500-20gb") if k in self.refs.keys()]
        header = self.gpu_head() + ["Workload", "clips/s (GPU only)", "clips/s (with loader, ref. numerics)",
                                    "batch that fitted",
                                    "peak GiB", "train epoch (min)", "Dev pass (min)", "epoch + Dev (min)",
                                    "Eval pass (min)"]
        header += [f"vs {self.refs.label(k)}" for k in ref_keys]
        rows = []
        used_gpu_only_for_ratio = False
        for label, tid, eid, iid in TRAIN_WORKLOADS:
            for g in self.gpus:
                tr = g.rec(tid)
                if tr is None and (eid is None or g.rec(eid) is None):
                    continue
                cps = g.metric(tid, "clips_per_s")
                lcps = g.metric(eid, "clips_per_s") if eid else None
                ep = _num(((g.ok(tid) or {}).get("estimates") or {}).get("asv5_train_epoch_min"))
                if ep is None and cps:
                    ep = C.ASV5_TRAIN_CLIPS / cps / 60.0
                icps = g.metric(iid, "clips_per_s")
                dev = C.ASV5_DEV_CLIPS / icps / 60.0 if icps else None
                ev = C.ASV5_EVAL_CLIPS / icps / 60.0 if icps else None
                row = self.gpu_col(g) + [label, _fmt(cps), _fmt(lcps), fit_text(tr), _fmt(_peak_gib(tr), "gib"),
                                         _fmt(ep, "min"), _fmt(dev, "min"),
                                         _fmt(ep + dev if ep is not None and dev is not None else None, "min"),
                                         _fmt(ev, "min")]
                for k in ref_keys:
                    if g.tiny:
                        row.append("smoke run, n/a" if g.smoke else "tiny run, n/a")
                        continue
                    rv = self.refs.value(k, tid, "clips_per_s")
                    if rv is None and eid:
                        rv = self.refs.value(k, eid, "clips_per_s")
                    if rv is None:
                        row.append("-")
                        continue
                    if rv.cpu_bound:
                        row.append("CPU-bound, not comparable")
                        continue
                    mine = cps if rv.source == "agbench" else (lcps if lcps is not None else cps)
                    if mine is None:
                        row.append("-")
                        continue
                    cell = _fmt(mine / rv.value, "ratio")
                    if rv.source == "log" and lcps is None:
                        cell += "*"
                        used_gpu_only_for_ratio = True
                    if rv.is_estimate:
                        cell += " (ref est.)"
                    row.append(cell)
                rows.append(row)
        if not rows:
            self.add("*No training scenarios in these results.*", "")
            return
        align = ["l"] * len(self.gpu_head()) + ["l", "r", "r", "l", "r", "r", "r", "r", "r"] + ["r"] * len(ref_keys)
        self.add(*_table(header, rows, align), "")
        notes = [
            f"Train epoch = {C.ASV5_TRAIN_CLIPS:,} clips, Dev pass = {C.ASV5_DEV_CLIPS:,} clips, Eval pass = "
            f"{C.ASV5_EVAL_CLIPS:,} clips (ASVspoof 5). Real clips are 4-10 s and are centre-cropped or looped to the "
            "training length, so the cost per clip is constant. Minutes use the GPU-only rate; Dev and Eval use the "
            "matching batch-16 inference scenario (see *Inference*). A real run is slower if the CPU cannot keep up "
            "(see *Can the CPU keep up?*).",
            "Ratios = this GPU / the reference GPU's training-log rate (above 1.00x = faster). They use the "
            "with-loader rate when it was measured, because the logs include data loading; that scenario also uses "
            "the logs' numerics (strict deterministic algorithms, TF32 off, flash attention off for 6 s clips) and "
            "counts loader stalls (clips / wall-clock time).",
        ]
        if used_gpu_only_for_ratio:
            notes.append("* compared using the GPU-only rate (no with-loader scenario for this workload); the "
                         "reference includes data loading and ran with strict deterministic algorithms, so this "
                         "ratio can flatter this GPU (by about the cost shown for the reference-numerics variant "
                         "in *Precision and recipe variants*).")
        for k in ref_keys:
            if k in self.refs.agbench_runs:
                lab = self.refs.label(k)
                notes.append(f"vs {lab}: uses {lab}'s run of this benchmark (GPU only vs GPU only) where it "
                             "has the scenario, otherwise its training log.")
        if any("(ref est.)" in c for r in rows for c in r):
            notes.append("(ref est.) = the reference value itself is an estimate (see *Environment and provenance*).")
        self.add(*[f"- {n}" for n in notes], "")
        self.reference_rows()
        for g in self.gpus:
            est = self.refs.estimate_for(g.slug)
            if not est:
                continue
            parts = []
            for key, tid, name in (("train_w2v2b_6s_clips_per_s", ANCHOR, "W2V2 Base train"),
                                   ("train_xlsr_6s_clips_per_s", XLSR_ANCHOR, "XLS-R train"),
                                   ("train_fusion_4s_clips_per_s", "train.fusion.4s.bs16.bf16", "fusion train"),
                                   ("dev_w2v2b_6s_clips_per_s", "infer.w2v2b.6s.bs16.bf16", "W2V2 Base Dev scoring")):
                e = est.get(key)
                if e is None:
                    continue
                txt = f"{e[0]}-{e[2]} (central {e[1]})" if isinstance(e, list) and len(e) == 3 else f"{e}"
                parts.append(f"{name} expected {txt}, measured {_fmt(g.metric(tid, 'clips_per_s'))}")
            if parts:
                who = f" for {g.label}" if self.multi else ""
                self.add(f"Pre-run estimate{who} (made before any run, basis: {_md(est.get('basis', '-'))}): "
                         + "; ".join(parts) + " clips/s.", "")

    def reference_rows(self) -> None:
        rows = []
        for g in self.refs.gpus:
            for e in g.get("log_references") or []:
                tr = _num(e.get("train_clips_per_s"))
                dev = _num(e.get("dev_clips_per_s"))
                ep = _num(e.get("train_min_per_epoch"))
                if ep is None and tr:
                    ep = C.ASV5_TRAIN_CLIPS / tr / 60.0
                devmin = _num(e.get("dev_min")) or (C.ASV5_DEV_CLIPS / dev / 60.0 if dev else None)
                flag = []
                if e.get("is_estimate"):
                    flag.append("estimate")
                if e.get("cpu_bound"):
                    flag.append("CPU-bound, not comparable")
                rows.append([str(g.get("label")), str(e.get("workload")),
                             _fmt_ref(tr) + (f" ({', '.join(flag)})" if flag and tr else ""),
                             _fmt(ep, "min") if tr else "-", _fmt_ref(dev), _fmt(devmin, "min")])
        if rows:
            self.add("Our GPUs on the same workloads (training logs, not this benchmark):", "")
            self.add(*_table(["Reference GPU", "Workload", "train clips/s", "train epoch (min)", "Dev clips/s",
                              "Dev pass (min)"], rows, ["l", "l", "r", "r", "r", "r"]), "")

    def fig_memory(self) -> None:
        order = sorted(self._ordered_ids(lambda sid: _mode_of(sid) in ("train", "e2e", "frozen")),
                       key=_memory_sort_key)
        series, caps = [], []
        for g in self.gpus:
            vals = {}
            for sid in order:
                r = g.rec(sid)
                peak = _peak_gib(r)
                if peak is None:
                    continue
                if r.get("status") == "oom":
                    vals[describe(sid)] = (peak, "OOM", True)
                else:
                    fb = r.get("fallback") or {}
                    note = fit_text(r) if fb.get("fits_requested") is False else ""
                    vals[describe(sid)] = (peak, note, False)
            if vals:
                series.append({"label": f"{g.label}: peak reserved memory", "color": self.colors[g.slug],
                               "values": vals})
            capv = [_num((r.get("metrics") or {}).get("mem_cap_gib")) for r in g.records.values()]
            capv = [c for c in capv if c]
            if capv:
                caps.append({"label": f"{g.label}: usable cap {statistics.median(capv):.1f} GiB "
                                      "(VRAM free at start minus a margin)",
                             "value": statistics.median(capv), "color": self.colors[g.slug]})
        guides = [{"value": 7.0, "label": "about 7 GiB usable on an 8 GB card"},
                  {"value": 15.0, "label": "about 15 GiB usable on a 16 GB card (about 14 GiB on a Windows desktop)"}]
        rows = [describe(s) for s in order]

        def draw(path):
            if not series:
                return False
            return self.plots.memory_fit(path, rows=rows, series=series, caps=caps, guides=guides,
                                         title="Memory fit: peak GPU memory per scenario")

        self.section("Memory fit")
        self.figure("fig2", "Figure 2: Memory fit", draw,
                    "peak memory reserved by PyTorch's allocator during warm-up and measurement "
                    "(torch.cuda.max_memory_reserved), at the batch split that ran. Each scenario caps PyTorch at "
                    "the VRAM free at start minus a margin (0.5 GiB on Windows, 0.1 GiB elsewhere), so a scenario that "
                    "needs more stops with an out-of-memory "
                    "error instead of spilling into system memory; it is then retried in a fresh process with half "
                    "the micro-batch and twice the accumulation (16x1, 8x2, 4x4, 2x8). The annotation shows the "
                    "split that ran when 16x1 did not fit. Dashed lines: typical usable memory on 8 GB and 16 GB cards.")
        near = [f"`{sid}`" + (f" ({g.label})" if self.multi else "") for g in self.gpus
                for sid, r in g.records.items() if (r.get("metrics") or {}).get("near_limit")]
        if near:
            self.add("Within 5 % of the cap (may not fit with other programs open): " + ", ".join(near) + ".", "")

    def fig_length_and_variants(self) -> None:
        series = []
        for g in self.gpus:
            pts = [(float(s), g.metric(sid, "clips_per_s"), g.metric(sid, "step_ms_median")) for s, sid in LENGTH_IDS]
            pts = [p for p in pts if p[1] is not None and p[2] is not None]
            if pts:
                series.append({"label": g.label, "color": self.colors[g.slug], "points": pts})

        def draw(path):
            if not series or max(len(s["points"]) for s in series) < 2:
                return False
            return self.plots.length_scaling(path, series=series,
                                             title="W2V2 Base training vs clip length" +
                                                   ("" if self.multi else f" ({self.gpus[0].label})"))

        self.section("Clip length, precision and other variants")
        self.figure("fig3", "Figure 3: Clip length scaling", draw,
                    "W2V2 Base full fine-tune, batch 16, bf16, GPU only, at 2, 4, 6, 8 and 10 s crops (the quick "
                    "suite runs 6 and 10 s; the full suite adds the rest). Left: clips per second; right: median "
                    "milliseconds per optimizer step.")
        header = self.gpu_head() + ["Variant", "scenario", "clips/s", "vs anchor", "batch that fitted", "peak GiB"]
        rows = []
        for label, sid, anchor in VARIANTS:
            for g in self.gpus:
                r = g.rec(sid)
                if r is None:
                    continue
                cps, base = g.metric(sid, "clips_per_s"), g.metric(anchor, "clips_per_s")
                ratio = cps / base if cps is not None and base else None
                rows.append(self.gpu_col(g) + [label, f"`{sid}`", _fmt(cps), _fmt(ratio, "ratio"), fit_text(r),
                                               _fmt(_peak_gib(r), "gib")])
        if rows:
            self.add("Precision and recipe variants, relative to the bf16 anchor of the same model "
                     "(1.00x = same speed; GPU only):", "")
            self.add(*_table(header, rows, (["l"] if self.multi else []) + ["l", "l", "r", "r", "l", "r"]), "")

    def infer_table(self) -> None:
        self.section("Inference (Dev / Eval scoring)")
        header = self.gpu_head() + ["Scenario", "clips/s", "ms per batch", "batch that fitted", "peak GiB",
                                    "Dev pass (min)", "Eval pass (min)", "reference Dev clips/s (training logs)"]
        rows = []
        for sid in INFER_ROWS + sorted(s for g in self.gpus for s in g.records
                                       if _mode_of(s) == "infer" and s not in INFER_ROWS):
            refs = []
            for k in self.refs.keys():
                rv = self.refs.value(k, sid, "clips_per_s", prefer_agbench=False)
                if rv:
                    refs.append(f"{rv.cell()} ({self.refs.label(k)})")
            for g in self.gpus:
                r = g.rec(sid)
                if r is None:
                    continue
                cps = g.metric(sid, "clips_per_s")
                rows.append(self.gpu_col(g) + [describe(sid), _fmt(cps), _fmt(g.metric(sid, "step_ms_median")),
                                               fit_text(r), _fmt(_peak_gib(r), "gib"),
                                               _fmt(C.ASV5_DEV_CLIPS / cps / 60 if cps else None, "min"),
                                               _fmt(C.ASV5_EVAL_CLIPS / cps / 60 if cps else None, "min"),
                                               "; ".join(refs) or "-"])
        if not rows:
            self.add("*No inference scenarios in these results.*", "")
            return
        self.add(*_table(header, rows, (["l"] if self.multi else []) + ["l", "r", "r", "l", "r", "r", "r", "l"]), "")
        self.add("*Method:* encoder in eval mode under `torch.inference_mode()` and bf16 autocast, batch copied "
                 "from pinned host memory each step, pyramid head and softmax score; median time per batch. "
                 "Reference Dev rates were measured inside training runs (FLAC loading included), so they are "
                 "indicative only.", "")

    def fig_cpu(self) -> None:
        series, demand = [], []
        for g in self.gpus:
            for blas, dashed, txt in (("1", False, "OPENBLAS_NUM_THREADS=1"),
                                      ("default", True, "default BLAS threads")):
                pts = []
                for sid, r in g.records.items():
                    m = re.fullmatch(r"cpu\.loader\.w(\d+)\.blas(\w+)", sid)
                    if m and m.group(2) == blas and g.metric(sid, "clips_per_s") is not None:
                        pts.append((int(m.group(1)), g.metric(sid, "clips_per_s")))
                if pts:
                    series.append({"label": f"{g.label} PC: loader, RawBoost p=0.5, {txt}",
                                   "color": self.colors[g.slug], "dashed": dashed, "points": pts})
            need = g.metric(ANCHOR, "clips_per_s")
            if need:
                demand.append({"label": f"{g.label}: W2V2 Base training needs {_fmt(need)} clips/s (GPU only)",
                               "color": self.colors[g.slug], "value": need})

        def draw(path):
            if not series:
                return False
            return self.plots.cpu_pipeline(path, series=series, demand=demand,
                                           title="Can the CPU keep up? Data loader throughput vs workers")

        self.section("Can the CPU keep up?")
        self.figure("fig4", "Figure 4: Can the CPU keep up?", draw,
                    "a PyTorch DataLoader (spawn workers, batch 16) reads synthetic 4-10 s FLAC files, crops or loops "
                    "them to 6 s and applies RawBoost algorithm 4 (LnL + ISD + SSI) to half of the clips; no GPU work. "
                    "Warm-up: one batch per worker (at least 2); then at least max(30, 6 x workers) batches are timed "
                    "(60 s cap), and clips/s is the steady rate after the first 2 x workers timed batches (which "
                    "may have been prefetched before timing started), over whole bursts of one batch per worker. "
                    "Solid lines: OPENBLAS_NUM_THREADS=1 in the "
                    "workers; dashed lines: BLAS thread variables unset (BLAS threads may spin and slow the workers). "
                    "Dotted horizontal line: the rate the GPU consumes clips in W2V2 Base training; a loader below it "
                    "would make training CPU-bound.")
        for g in self.gpus:
            self._cpu_sentence(g)
        header = self.gpu_head() + ["Scenario", "what it measures", "workers", "BLAS threads", "clips/s",
                                    "ms per clip", "CPU time / wall", "details"]
        rows = []
        for sid in self._ordered_ids(lambda s: s.startswith("cpu.")):
            for g in self.gpus:
                r = g.rec(sid)
                if r is None:
                    continue
                m = r.get("metrics") or {}
                p = r.get("params") or {}
                kind = sid.split(".")[1] if "." in sid else sid
                what = {"decode": "FLAC decode + crop, 1 process",
                        "rawboost": "RawBoost algo 4 on every clip, 1 process",
                        "loader": "DataLoader, RawBoost on half the clips"}.get(kind, kind)
                tp = (r.get("extra") or {}).get("threadpool") or []
                blas = str(tp[0].get("num_threads")) if tp and isinstance(tp[0], dict) and tp[0].get("num_threads") \
                    else ("1" if sid.endswith("blas1") else ("default" if sid.endswith("blasdefault") else "-"))
                details = []
                if isinstance(m.get("stage_ms"), dict):
                    details.append("ms per stage: " + ", ".join(f"{k} {_fmt(v)}" for k, v in m["stage_ms"].items()))
                if m.get("decode_ms_per_audio_s") is not None:
                    details.append(f"decode {_fmt(m['decode_ms_per_audio_s'])} ms per audio second")
                if r.get("status") != "ok":
                    details.append(f"{r.get('status')}: {r.get('status_detail') or ''}".strip())
                workers = m.get("num_workers", p.get("num_workers", "-"))
                if kind in ("decode", "rawboost") or workers in (0, "0"):
                    workers = "1 process"
                rows.append(self.gpu_col(g) + [f"`{sid}`", what, str(workers), blas, _fmt(m.get("clips_per_s")),
                                               _fmt(m.get("ms_per_clip_median")), _fmt(m.get("cpu_wall_ratio")),
                                               "; ".join(details) or "-"])
        if rows:
            self.add(*_table(header, rows, (["l"] if self.multi else []) + ["l", "l", "r", "r", "r", "r", "r", "l"]),
                     "", "*CPU time / wall* above 1 in a single process means extra threads were busy (for example "
                         "BLAS threads spinning).", "")

    def _cpu_sentence(self, g: GpuRuns) -> None:
        need = g.metric(ANCHOR, "clips_per_s")
        pts1 = sorted((int(m.group(1)), g.metric(sid, "clips_per_s")) for sid in g.records
                      for m in [re.fullmatch(r"cpu\.loader\.w(\d+)\.blas1", sid)] if m and g.metric(sid, "clips_per_s"))
        who = f"{g.label} PC: " if self.multi else ""
        parts = []
        if pts1 and need:
            enough = [w for w, v in pts1 if v >= need]
            if enough:
                w = enough[0]
                v = dict(pts1)[w]
                parts.append(f"with OPENBLAS_NUM_THREADS=1, {w} worker{'s' if w > 1 else ''} deliver "
                             f"{_fmt(v)} clips/s with RawBoost on half the clips, enough for this GPU's W2V2 Base "
                             f"training rate of {_fmt(need)} clips/s")
            else:
                w, v = pts1[-1]
                parts.append(f"no tested worker count reached this GPU's W2V2 Base training rate of {_fmt(need)} "
                             f"clips/s with RawBoost (best: {_fmt(v)} clips/s with {w} workers), so training with "
                             "RawBoost would be CPU-bound on this PC")
        for w in (4, 8, 1):
            a = g.metric(f"cpu.loader.w{w}.blas1", "clips_per_s")
            b = g.metric(f"cpu.loader.w{w}.blasdefault", "clips_per_s")
            if a and b:
                parts.append(f"with {w} workers, default BLAS threads give {_fmt(b)} clips/s = {_fmt(b / a, 'ratio')} "
                             "the OPENBLAS_NUM_THREADS=1 rate")
                break
        rb1 = g.ok("cpu.rawboost.blas1")
        if rb1:
            ms = (rb1.get("metrics") or {}).get("ms_per_clip_median")
            if ms is not None:
                parts.append(f"RawBoost algo 4 costs {_fmt(ms)} ms per 6 s clip in one process")
        if parts:
            self.add(f"{who}" + "; ".join(parts)[0].upper() + "; ".join(parts)[1:] + ".", "")

    def basics_table(self) -> None:
        self.section("GPU basics")
        rows = []

        def add_row(label, getter, kind="num"):
            vals = [getter(g) for g in self.gpus]
            if any(v is not None for v in vals):
                rows.append([label] + [_fmt(v, kind) if not isinstance(v, str) else v for v in vals])

        for dt, name in (("bf16", "bf16"), ("fp16", "fp16"), ("tf32", "TF32"), ("fp32", "fp32 (strict)")):
            sizes = sorted({sz for g in self.gpus for sz in
                            (((g.ok("gpu.matmul") or {}).get("metrics") or {}).get("tflops") or {}).get(dt, {})},
                           key=lambda s: int(s) if str(s).isdigit() else 0)
            for sz in sizes:
                add_row(f"matmul {name}, n={sz} (TFLOPS)",
                        lambda g, dt=dt, sz=sz: _num(((((g.ok("gpu.matmul") or {}).get("metrics") or {})
                                                       .get("tflops") or {}).get(dt) or {}).get(sz)))
        for k, name in (("copy", "memory copy"), ("read", "memory read (sum)"), ("triad", "memory triad")):
            add_row(f"{name} (GB/s)", lambda g, k=k: _num((((g.ok("gpu.membw") or {}).get("metrics") or {})
                                                            .get("gbps") or {}).get(k)))
        for k, name in (("h2d_pinned", "PCIe host to GPU, pinned"), ("d2h_pinned", "PCIe GPU to host, pinned"),
                        ("h2d_pageable", "PCIe host to GPU, pageable")):
            add_row(f"{name} (GB/s)", lambda g, k=k: _num((((g.ok("gpu.pcie") or {}).get("metrics") or {})
                                                            .get("gbps") or {}).get(k)))

        def link(g):
            lk = (((g.ok("gpu.pcie") or {}).get("metrics") or {}).get("link")) or (g.env.get("pcie") or {})
            if not lk or lk.get("gen_cur") is None:
                return None
            return f"gen {lk.get('gen_cur')} x{lk.get('width_cur')} (max gen {lk.get('gen_max')} x{lk.get('width_max')})"

        add_row("PCIe link during transfer", link)
        if rows:
            self.add(*_table(["Measured"] + [g.label for g in self.gpus], rows), "")
            self.add("*Method:* matmul = 2n^3 floating-point operations per square matrix product, median of 10 "
                     "timed batches of at least 0.2 s each; memory = 1 GiB fp32 buffers (copy counts read + write); "
                     "PCIe = 256 MiB transfers. Approximate spec figures for context (not measured here):", "")
        else:
            self.add("*No GPU basics scenarios in these results.*", "")
        spec_rows = []
        for g in self.refs.gpus:
            spec_rows.append(self._spec_row(str(g.get("label")), g.get("spec_approx") or {}))
        for e in self.refs.estimates:
            spec_rows.append(self._spec_row(str(e.get("label")), e.get("spec_approx") or {}))
        if spec_rows:
            self.add(*_table(["GPU", "VRAM (GB)", "memory bandwidth (GB/s)", "FP32 (TFLOPS)",
                              "bf16 dense, FP32 accumulate (TFLOPS)", "PCIe"], spec_rows), "")

    @staticmethod
    def _spec_row(label: str, s: dict) -> list[str]:
        bf = s.get("bf16_dense_tflops_fp32acc")
        bf_txt = _fmt(bf)
        if bf is None and s.get("bf16_dense_tflops_fp32acc_derived") is not None:
            bf_txt = _fmt(s["bf16_dense_tflops_fp32acc_derived"]) + " (derived, not official)"
        return [label, _fmt(s.get("vram_gb")), _fmt(s.get("mem_bw_gbps")), _fmt(s.get("fp32_tflops")), bf_txt,
                str(s.get("pcie", "-"))]

    def fig_sustain(self) -> None:
        series, lines = [], []
        for g in self.gpus:
            r = g.ok("sustain.w2v2b.6s.bs16.bf16") or next(
                (x for s, x in g.records.items() if _mode_of(s) == "sustain" and x.get("status") == "ok"), None)
            if not r:
                continue
            ts = [t for t in r.get("timeseries") or [] if isinstance(t, dict) and _num(t.get("t_s")) is not None]
            windows = [(_num(t["t_s"]), _num(t.get("clips_per_s"))) for t in ts if _num(t.get("clips_per_s"))]
            tel = [(_num(t["t_s"]), _num(t.get("temp_c")), _num(t.get("sm_clock_mhz")), _num(t.get("power_w")))
                   for t in ts if any(_num(t.get(k)) is not None for k in ("temp_c", "sm_clock_mhz", "power_w"))]
            if windows or tel:
                series.append({"label": g.label, "color": self.colors[g.slug], "windows": windows, "telemetry": tel})
            m = r.get("metrics") or {}
            first, last = _num(m.get("clips_per_s_first_window")), _num(m.get("clips_per_s_last_window"))
            if first is None and windows:
                first, last = windows[0][1], windows[-1][1]
            ratio = _num(m.get("sustain_ratio")) or (last / first if first and last else None)
            thr = (r.get("telemetry") or {}).get("throttle_reasons") or []
            tmax = (((r.get("telemetry") or {}).get("temp_c")) or {}).get("max")
            who = f"{g.label}: " if self.multi else ""
            lines.append(f"{who}first 30 s window {_fmt(first)} clips/s, last window {_fmt(last)} clips/s "
                         f"(last / first = {_fmt(ratio, 'ratio')}); max temperature {_fmt(tmax)} deg C; throttle "
                         f"reasons reported: {', '.join(map(str, thr)) if thr else 'none'}.")
        if not series:
            return
        self.section("Sustained run (thermal and power behaviour)")
        self.figure("fig5", "Figure 5: Sustained run", lambda p: self.plots.sustained(
            p, series=series, title="W2V2 Base training for 300 s: speed, temperature, clock and power"),
            "the W2V2 Base training step (batch 16, bf16) repeated for 300 s; clips per second per 30 s window, "
            "GPU telemetry sampled every 0.5 s (NVML or nvidia-smi). A last/first ratio well below 1.00x means the "
            "card slows down when hot or power-limited.")
        self.add(*lines, "")

    def problems_table(self) -> None:
        rows = []
        for g in self.gpus:
            for sid in self._ordered_ids(lambda s: True):
                r = g.rec(sid)
                if r is None or r.get("status") == "ok":
                    continue
                err = r.get("error") or {}
                reason = r.get("status_detail") or ""
                if err.get("message"):
                    msg = str(err.get("message")).splitlines()[0][:160]
                    etype = str(err.get("type") or "")
                    full = f"{etype}: {msg}" if etype and etype not in msg else msg
                    reason = full if not reason or reason == etype else f"{reason}; {full}"
                attempts = [a for a in (r.get("fallback") or {}).get("attempts") or [] if isinstance(a, dict)]
                att = ", ".join(f"{a.get('micro_batch')}x{a.get('accum')} {a.get('status')}" for a in attempts)
                rows.append(self.gpu_col(g) + [f"`{sid}`", str(r.get("status")), reason or "-", att or "-",
                                               str(r.get("log_file") or "-")])
        self.section("Scenarios that did not finish ok")
        if not rows:
            self.add("All scenarios in these results finished ok.", "")
            return
        self.add(*_table(self.gpu_head() + ["Scenario", "status", "reason", "attempts", "log file (under results/)"],
                         rows, ["l"] * (len(self.gpu_head()) + 5)), "")
        self.add("*oom* is a valid result (the scenario does not fit on this card even at the smallest split); "
                 "*skipped* with reason *budget* is run by the next `full`; *error*, *timeout* and *crashed* are "
                 "retried once on the next run. The log files explain errors.", "")

    def environment(self) -> None:
        self.section("Environment and provenance")

        def e(g, *keys):
            v = g.env
            for k in keys:
                v = v.get(k) if isinstance(v, dict) else None
            return v

        def fmt_list(v):
            return ", ".join(str(x) for x in v) if isinstance(v, list) else (str(v) if v is not None else "-")

        fields = [
            ("GPU", lambda g: e(g, "gpu_name")),
            ("VRAM total (GiB) / class (GB)", lambda g: f"{_fmt(e(g, 'vram_total_gib'))} / {e(g, 'vram_class_gb')}"),
            ("compute capability", lambda g: ".".join(map(str, e(g, "gpu_capability") or [])) or None),
            ("NVIDIA driver", lambda g: e(g, "driver")),
            ("torch (CUDA)", lambda g: f"{e(g, 'torch')} ({e(g, 'torch_cuda')})" if e(g, "torch") else None),
            ("cuDNN", lambda g: e(g, "cudnn")),
            ("transformers", lambda g: e(g, "transformers")),
            ("Python", lambda g: e(g, "python")),
            ("OS", lambda g: e(g, "os")),
            ("CPU", lambda g: e(g, "cpu_model")),
            ("CPU cores (physical / logical)", lambda g: f"{e(g, 'cpu_physical')} / {e(g, 'cpu_logical')}"
                if e(g, "cpu_logical") else None),
            ("RAM (GiB)", lambda g: _fmt(e(g, "ram_gib")) if e(g, "ram_gib") else None),
            ("PCIe link at start", lambda g: (f"gen {e(g, 'pcie', 'gen_cur')} x{e(g, 'pcie', 'width_cur')} "
                                              f"(max gen {e(g, 'pcie', 'gen_max')} x{e(g, 'pcie', 'width_max')})")
                if e(g, "pcie", "gen_cur") is not None else None),
            ("power limit (W)", lambda g: _fmt(e(g, "power_limit_w")) if e(g, "power_limit_w") else None),
            ("BLAS", lambda g: ", ".join(f"{b.get('internal_api')} {b.get('version')} ({b.get('num_threads')} threads)"
                                        for b in e(g, "blas") or [] if isinstance(b, dict)) or None),
            ("doctor", lambda g: e(g, "doctor", "status")),
            ("agbench version", lambda g: fmt_list(sorted({str(r.get("agbench_version")) for r in g.run_starts}))),
            ("git commit", lambda g: fmt_list(sorted({str(r.get("git_commit")) for r in g.run_starts
                                                       if r.get("git_commit")})) or "-"),
            ("runs", lambda g: "; ".join(f"{r.get('suite')} {r.get('run_id')}" for r in g.run_starts) or None),
            ("results files", lambda g: ", ".join(g.sources)),
        ]
        rows = []
        for name, fn in fields:
            vals = []
            for g in self.gpus:
                try:
                    v = fn(g)
                except Exception:  # noqa: BLE001
                    v = None
                vals.append("-" if v in (None, "", "None / None") else str(v))
            if any(v != "-" for v in vals):
                rows.append([name] + vals)
        self.add(*_table(["Item"] + [g.label for g in self.gpus], rows, ["l"] * (len(self.gpus) + 1)), "")
        if self.refs.available:
            self.add("**Reference numbers (verbatim from `references/reference_gpus.json`).** "
                     + _md(self.refs.data.get("notes", "")), "")
            for g in self.refs.gpus:
                for ent in g.get("log_references") or []:
                    vals = []
                    for k in ("train_clips_per_s", "train_min_per_epoch", "dev_clips_per_s", "dev_min",
                              "peak_alloc_gib", "peak_reserved_gib"):
                        if ent.get(k) is not None:
                            vals.append(f"{k} {ent[k]}")
                    flags = [f for f in ("is_estimate", "cpu_bound") if ent.get(f)]
                    self.add(f"- {_md(g.get('label'))}, {_md(ent.get('workload'))}: {', '.join(vals)}"
                             + (f" [{', '.join(flags)}]" if flags else "")
                             + f". Provenance: {_md(ent.get('provenance', '-'))}.")
                runs = g.get("agbench_runs") or []
                if runs:
                    self.add(f"- {_md(g.get('label'))}: {len(runs)} run(s) of this benchmark listed in agbench_runs.")
            for est in self.refs.estimates:
                self.add(f"- {_md(est.get('label'))}: pre-run estimate, basis: {_md(est.get('basis', '-'))}.")
            self.add("")
        self.add("Generated by `python -m agbench report`. Raw numbers for every scenario and metric: "
                 "`comparison.csv` in this folder.", "")

    def _ordered_ids(self, pred) -> list[str]:
        order: list[str] = []
        try:
            from .scenarios import SUITES

            order = [s for s in SUITES.get("full", []) if pred(s)]
        except Exception:  # noqa: BLE001 - ordering is cosmetic
            order = []
        seen = set(order)
        for g in self.gpus:
            for sid in g.records:
                if sid not in seen and pred(sid):
                    order.append(sid)
                    seen.add(sid)
        present = {sid for g in self.gpus for sid in g.records}
        return [s for s in order if s in present]

    # -- CSV --------------------------------------------------------------------------------
    def comparison_rows(self) -> tuple[list[str], list[list]]:
        ref_keys = self.refs.keys()
        agb_keys = [k for k in ref_keys if k in self.refs.agbench_runs]
        header = (["scenario_id", "metric", "unit"] + [g.slug for g in self.gpus]
                  + [f"agbench_{k}" for k in agb_keys] + [f"ref_{k}" for k in ref_keys] + ["ref_source"])
        rows = []
        all_runs = self.gpus + [self.refs.agbench_runs[k] for k in agb_keys]
        for sid in self._ordered_ids(lambda s: True):
            metrics = self._csv_metrics(sid)
            for metric, unit, getter in metrics:
                vals = [getter(g) for g in all_runs]
                refvals, sources = [], []
                for k in ref_keys:
                    gref = next(x for x in self.refs.gpus if x.get("gpu_key") == k)
                    rv = self.refs.log_value(gref, sid, metric)
                    refvals.append("" if rv is None else rv.value)
                    if rv is not None:
                        flag = "estimate" if rv.is_estimate else ("CPU-bound, not comparable" if rv.cpu_bound else "")
                        sources.append(f"{k}: training log, data loading included, torch 2.2.2"
                                       + (f" ({flag})" if flag else ""))
                if all(v in (None, "") for v in vals) and all(v == "" for v in refvals):
                    continue
                rows.append([sid, metric, unit] + ["" if v is None else v for v in vals] + refvals
                            + ["; ".join(sources)])
        return header, rows

    @staticmethod
    def _csv_metrics(sid: str):
        def m(key):
            return lambda g: (g.ok(sid) or {}).get("metrics", {}).get(key)

        def status(g):
            r = g.rec(sid)
            return r.get("status") if r else None

        def fit(g):
            r = g.rec(sid)
            return fit_text(r) if r and r.get("fallback") else None

        def est(key):
            return lambda g: ((g.ok(sid) or {}).get("estimates") or {}).get(key)

        out = [("status", "", status)]
        mode = _mode_of(sid)
        if mode in ("train", "e2e", "frozen", "infer", "sustain"):
            out += [("clips_per_s", "clips/s", m("clips_per_s")), ("clips_per_s_wall", "clips/s", m("clips_per_s_wall")),
                    ("step_ms_median", "ms", m("step_ms_median")), ("step_ms_cv", "", m("step_ms_cv")),
                    ("batch_used", "micro x accum", fit), ("peak_alloc_gib", "GiB", m("peak_alloc_gib")),
                    ("peak_reserved_gib", "GiB", m("peak_reserved_gib")), ("mem_cap_gib", "GiB", m("mem_cap_gib"))]
            if mode in ("train", "e2e", "sustain"):
                out.append(("asv5_train_epoch_min", "min", est("asv5_train_epoch_min")))
            if mode == "infer":
                out += [("asv5_dev_pass_min", "min", est("asv5_dev_pass_min")),
                        ("asv5_eval_pass_min", "min", est("asv5_eval_pass_min"))]
            if mode == "e2e":
                out += [("clips_per_s_median_step", "clips/s", m("clips_per_s_median_step")),
                        ("loader_wait_frac", "fraction", m("loader_wait_frac")),
                        ("num_workers", "workers", m("num_workers"))]
            if mode == "sustain":
                out += [("sustain_ratio", "last/first", m("sustain_ratio"))]
        elif sid == "gpu.matmul":
            for dt in ("bf16", "fp16", "tf32", "fp32"):
                for sz in ("4096", "8192", "256"):
                    out.append((f"tflops_{dt}_{sz}", "TFLOPS",
                                lambda g, dt=dt, sz=sz: (((m("tflops")(g)) or {}).get(dt) or {}).get(sz)))
        elif sid == "gpu.membw":
            for k in ("copy", "read", "triad"):
                out.append((f"gbps_{k}", "GB/s", lambda g, k=k: (m("gbps")(g) or {}).get(k)))
        elif sid == "gpu.pcie":
            for k in ("h2d_pinned", "d2h_pinned", "h2d_pageable"):
                out.append((f"gbps_{k}", "GB/s", lambda g, k=k: (m("gbps")(g) or {}).get(k)))
            out.append(("link", "gen x width", lambda g: (lambda lk: f"gen{lk.get('gen_cur')} x{lk.get('width_cur')}"
                                                          if lk else None)(m("link")(g))))
        elif mode == "cpu":
            out += [("clips_per_s", "clips/s", m("clips_per_s")), ("ms_per_clip_median", "ms", m("ms_per_clip_median")),
                    ("cpu_ms_per_clip", "ms", m("cpu_ms_per_clip")), ("cpu_wall_ratio", "", m("cpu_wall_ratio")),
                    ("num_workers", "workers", m("num_workers"))]
            for k in ("lnl", "isd", "ssi"):
                out.append((f"stage_ms_{k}", "ms", lambda g, k=k: (m("stage_ms")(g) or {}).get(k)))
            out.append(("decode_ms_per_audio_s", "ms per audio s", m("decode_ms_per_audio_s")))
        else:
            out.append(("clips_per_s", "clips/s", m("clips_per_s")))
        return out

    # -- main -------------------------------------------------------------------------------
    def build(self) -> Path:
        self.out.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        for old in self.fig_dir.glob("fig*_*.png"):
            try:
                old.unlink()
            except OSError:
                pass
        self.header()
        self.checks()
        self.how_to_read()
        self.fig_training()
        self.asv5_table()
        self.fig_memory()
        self.fig_length_and_variants()
        self.infer_table()
        self.fig_cpu()
        self.basics_table()
        self.fig_sustain()
        self.problems_table()
        self.environment()
        header, rows = self.comparison_rows()
        with open(self.out / "comparison.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(header)
            w.writerows(rows)
        summary = self.out / "summary.md"
        text = "\n".join(self.lines).rstrip() + "\n"
        text = re.sub(r"\n{3,}", "\n\n", text)
        summary.write_text(text, encoding="utf-8", newline="\n")
        return summary


def _default_out_dir(paths: list[Path], gpus: list[GpuRuns]) -> Path:
    base = Path(paths[0]).resolve().parent
    if len(gpus) == 1:
        g = gpus[0]
        stems = {Path(p).stem for p in paths}
        if len(stems) == 1:
            stem = stems.pop()
            if re.fullmatch(re.escape(g.slug) + r"_\d{8}", stem):
                return base / f"{stem}_report"
        dates = sorted(d for d in (_date_of(r.get("started_utc")) for r in g.run_starts) if d)
        day = dates[-1].replace("-", "") if dates else _dt.date.today().strftime("%Y%m%d")
        return base / f"{g.slug}_{day}_report"
    return base / f"compare_{_dt.date.today().strftime('%Y%m%d')}_report"


def build_report(jsonl_paths: list[Path], out_dir: Path | None = None,
                 references_path: Path = Path("references/reference_gpus.json")) -> Path:
    """Write summary.md, comparison.csv and figures/*.png; return the path of summary.md.

    ``jsonl_paths``: one or more results files (one or several GPUs). ``out_dir`` defaults to
    ``<dir of the first file>/<gpu_slug>_<YYYYMMDD>_report`` (one GPU) or
    ``<dir>/compare_<YYYYMMDD>_report`` (several GPUs).
    """
    paths = [Path(p) for p in jsonl_paths]
    missing = [p for p in paths if not p.exists()]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise FileNotFoundError("no results files found: " + ", ".join(p.name for p in missing))
    gpus = load_runs(paths)
    if not gpus:
        raise ValueError("no benchmark records found in: " + ", ".join(p.name for p in paths))
    refs = load_references(references_path)
    for p in missing:
        refs.warnings.append(f"results file not found and ignored: {p.name}")
    out = Path(out_dir) if out_dir is not None else _default_out_dir(paths, gpus)
    return _Builder(gpus, refs, out).build()
