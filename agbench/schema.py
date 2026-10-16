"""Results JSONL records (SPEC 9): builders, validation, loading. No torch import."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import __version__
from .util import read_jsonl, sanitize_obj, utc_now_iso

SCHEMA_VERSION = "1"

STATUSES = ("ok", "oom", "error", "timeout", "crashed", "skipped")
TERMINAL_OK = ("ok", "oom", "skipped")
CATEGORIES = ("basics", "train", "frozen", "infer", "e2e", "cpu", "sustain")
MODEL_CATEGORIES = ("train", "frozen", "infer", "e2e", "sustain")
COUNT_KEYS = ("ok", "oom", "error", "timeout", "crashed", "skipped", "cached")
TELEMETRY_SOURCES = ("nvml", "nvidia-smi", "none")
ESTIMATE_KEYS = ("asv5_train_epoch_min", "asv5_dev_pass_min", "asv5_eval_pass_min", "assumes_seconds")


def empty_estimates() -> dict:
    return {k: None for k in ESTIMATE_KEYS}


def make_run_start(*, run_id: str, suite: str, gpu_slug: str, env_fingerprint: str, env: dict,
                   settings: dict, git_commit: str | None = None, started_utc: str | None = None) -> dict:
    return sanitize_obj({
        "kind": "run_start",
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "suite": suite,
        "started_utc": started_utc or utc_now_iso(),
        "agbench_version": __version__,
        "git_commit": git_commit,
        "gpu_slug": gpu_slug,
        "env_fingerprint": env_fingerprint,
        "settings": settings,
        "env": env,
    })


def make_scenario_record(*, run_id: str, gpu_slug: str, env_fingerprint: str, scenario_id: str,
                         scenario_hash: str, workload: str, workload_version: int, category: str,
                         suite: str, status: str, status_detail: str = "", started_utc: str,
                         finished_utc: str | None = None, wall_s: float = 0.0, params: dict,
                         measure: dict, fallback: dict | None = None, metrics: dict | None = None,
                         estimates: dict | None = None, telemetry: dict | None = None,
                         timeseries: list | None = None, extra: dict | None = None,
                         error: dict | None = None, log_file: str | None = None) -> dict:
    return sanitize_obj({
        "kind": "scenario",
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "gpu_slug": gpu_slug,
        "env_fingerprint": env_fingerprint,
        "scenario_id": scenario_id,
        "scenario_hash": scenario_hash,
        "workload": workload,
        "workload_version": int(workload_version),
        "category": category,
        "suite": suite,
        "status": status,
        "status_detail": status_detail or "",
        "started_utc": started_utc,
        "finished_utc": finished_utc or utc_now_iso(),
        "wall_s": float(wall_s),
        "params": params,
        "measure": measure,
        "fallback": fallback,
        "metrics": (metrics or {}) if status == "ok" else {},
        "estimates": estimates or empty_estimates(),
        "telemetry": telemetry or {"source": "none"},
        "timeseries": timeseries,
        "extra": extra or {},
        "error": error,
        "log_file": log_file,
    })


def make_run_end(*, run_id: str, wall_s: float, counts: dict, interrupted: bool = False,
                 finished_utc: str | None = None) -> dict:
    full_counts = {k: int(counts.get(k, 0)) for k in COUNT_KEYS}
    return sanitize_obj({
        "kind": "run_end",
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "finished_utc": finished_utc or utc_now_iso(),
        "wall_s": float(wall_s),
        "counts": full_counts,
        "interrupted": bool(interrupted),
    })


# ---------------------------------------------------------------------------------------------
# Validation (same rules as agbench/results_schema.json)
# ---------------------------------------------------------------------------------------------

_STR, _NUM, _INT, _BOOL, _OBJ, _ARR, _NULL = "string", "number", "integer", "boolean", "object", "array", "null"


def _type_ok(v: Any, t: str) -> bool:
    if t == _STR:
        return isinstance(v, str)
    if t == _NUM:
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == _INT:
        return isinstance(v, int) and not isinstance(v, bool)
    if t == _BOOL:
        return isinstance(v, bool)
    if t == _OBJ:
        return isinstance(v, dict)
    if t == _ARR:
        return isinstance(v, list)
    if t == _NULL:
        return v is None
    raise ValueError(t)


def _check(rec: dict, field: str, types: tuple[str, ...], problems: list[str], where: str = "") -> bool:
    label = f"{where}{field}"
    if field not in rec:
        problems.append(f"missing field '{label}'")
        return False
    if not any(_type_ok(rec[field], t) for t in types):
        problems.append(f"field '{label}' should be {' or '.join(types)}, got {type(rec[field]).__name__}")
        return False
    return True


_RUN_START_FIELDS = {
    "run_id": (_STR,), "suite": (_STR,), "started_utc": (_STR,), "agbench_version": (_STR,),
    "git_commit": (_STR, _NULL), "gpu_slug": (_STR,), "env_fingerprint": (_STR,),
    "settings": (_OBJ,), "env": (_OBJ,),
}
_SCENARIO_FIELDS = {
    "run_id": (_STR,), "gpu_slug": (_STR,), "env_fingerprint": (_STR,), "scenario_id": (_STR,),
    "scenario_hash": (_STR,), "workload": (_STR,), "workload_version": (_INT,), "category": (_STR,),
    "suite": (_STR,), "status": (_STR,), "status_detail": (_STR,), "started_utc": (_STR,),
    "finished_utc": (_STR,), "wall_s": (_NUM,), "params": (_OBJ,), "measure": (_OBJ,),
    "fallback": (_OBJ, _NULL), "metrics": (_OBJ,), "estimates": (_OBJ,), "telemetry": (_OBJ,),
    "timeseries": (_ARR, _NULL), "extra": (_OBJ,), "error": (_OBJ, _NULL), "log_file": (_STR, _NULL),
}
_RUN_END_FIELDS = {
    "run_id": (_STR,), "finished_utc": (_STR,), "wall_s": (_NUM,), "counts": (_OBJ,),
    "interrupted": (_BOOL,),
}


def _ok_metric_requirements(rec: dict) -> tuple[str, ...]:
    cat, wl = rec.get("category"), rec.get("workload")
    if cat in MODEL_CATEGORIES:
        return ("clips_per_s", "step_ms_median")
    if wl == "basics.matmul":
        return ("tflops",)
    if wl in ("basics.membw", "basics.pcie"):
        return ("gbps",)
    if cat == "cpu":
        return ("clips_per_s",)
    return ()


def validate_record(rec: dict) -> list[str]:
    """Human-readable problems with one record; an empty list means valid."""
    problems: list[str] = []
    if not isinstance(rec, dict):
        return ["record is not a JSON object"]
    kind = rec.get("kind")
    if kind not in ("run_start", "scenario", "run_end"):
        return [f"unknown kind {kind!r} (expected run_start, scenario or run_end)"]
    if rec.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version should be '{SCHEMA_VERSION}', got {rec.get('schema_version')!r}")
    fields = {"run_start": _RUN_START_FIELDS, "scenario": _SCENARIO_FIELDS, "run_end": _RUN_END_FIELDS}[kind]
    for name, types in fields.items():
        _check(rec, name, types, problems)

    if kind == "scenario":
        status = rec.get("status")
        if isinstance(status, str) and status not in STATUSES:
            problems.append(f"status {status!r} not in {list(STATUSES)}")
        cat = rec.get("category")
        if isinstance(cat, str) and cat not in CATEGORIES:
            problems.append(f"category {cat!r} not in {list(CATEGORIES)}")
        metrics = rec.get("metrics")
        if isinstance(metrics, dict):
            if status != "ok" and metrics:
                problems.append(f"metrics must be empty unless status is ok (status {status!r})")
            if status == "ok":
                for key in _ok_metric_requirements(rec):
                    want = _OBJ if key in ("tflops", "gbps") else _NUM
                    if metrics.get(key) is None:
                        problems.append(f"ok record without metrics.{key}")
                    elif not _type_ok(metrics[key], want):
                        problems.append(f"metrics.{key} should be {want}")
        tel = rec.get("telemetry")
        if isinstance(tel, dict):
            if tel.get("source") not in TELEMETRY_SOURCES:
                problems.append(f"telemetry.source {tel.get('source')!r} not in {list(TELEMETRY_SOURCES)}")
        est = rec.get("estimates")
        if isinstance(est, dict):
            for k, v in est.items():
                if v is not None and not _type_ok(v, _NUM):
                    problems.append(f"estimates.{k} should be number or null")
        fb = rec.get("fallback")
        if isinstance(fb, dict):
            if not isinstance(fb.get("attempts"), list):
                problems.append("fallback.attempts should be a list")
            else:
                for i, a in enumerate(fb["attempts"]):
                    if not isinstance(a, dict):
                        problems.append(f"fallback.attempts[{i}] should be an object")
                        continue
                    for f, t in (("micro_batch", (_INT,)), ("accum", (_INT,)), ("status", (_STR,))):
                        _check(a, f, t, problems, where=f"fallback.attempts[{i}].")
            if "ladder" in fb and fb["ladder"] is not None:
                lad = fb["ladder"]
                if not (isinstance(lad, list) and all(isinstance(r, list) and len(r) == 2 for r in lad)):
                    problems.append("fallback.ladder should be a list of [micro_batch, accum] pairs or null")
            if "used" in fb and fb["used"] is not None and not isinstance(fb["used"], dict):
                problems.append("fallback.used should be an object or null")
            if "fits_requested" in fb and not isinstance(fb["fits_requested"], bool):
                problems.append("fallback.fits_requested should be boolean")
        err = rec.get("error")
        if status in ("error", "timeout", "crashed") and err is None:
            problems.append(f"status {status!r} needs an error object")
        if isinstance(err, dict):
            for f in ("type", "message"):
                if not isinstance(err.get(f), str):
                    problems.append(f"error.{f} should be a string")
    elif kind == "run_end":
        counts = rec.get("counts")
        if isinstance(counts, dict):
            for k in COUNT_KEYS:
                if not _type_ok(counts.get(k), _INT):
                    problems.append(f"counts.{k} should be an integer")
    elif kind == "run_start":
        env = rec.get("env")
        if isinstance(env, dict):
            for f in ("os_family", "python"):
                if not isinstance(env.get(f), str):
                    problems.append(f"env.{f} should be a string")
    return problems


def load_records(path: Path) -> list[dict]:
    """All records of a results JSONL in file order; a truncated (unparseable) last line is ignored."""
    return read_jsonl(Path(path))


def schema_path() -> Path:
    return Path(__file__).with_name("results_schema.json")
