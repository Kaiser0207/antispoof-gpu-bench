"""Suite execution (SPEC 8): subprocess per attempt, timeouts, OOM ladder, resume, time budget.

The runner process never imports torch; children do all GPU work. The runner is the only
writer of the results JSONL.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from . import constants as C
from .registry import workload_version
from .scenarios import SCENARIOS, Scenario, effective, first_rung, scenario_hash, timeout_s
from .schema import (TERMINAL_OK, empty_estimates, load_records, make_run_end, make_run_start,
                     make_scenario_record, validate_record)
from .util import (LockHeld, RunLock, add_to_kill_on_exit_job, append_jsonl, ascii_print, default_results_dir,
                   find_leftover_processes, fmt_duration, kill_tree, local_date, median, repo_root, sanitize,
                   sanitize_file, tail_text, utc_now_iso)

PARENT_ENV = "AGBENCH_PARENT"  # same name as agbench.child.PARENT_ENV (kept import-free here)

FAILED_STATUSES = ("error", "timeout", "crashed")
ATTEMPT_MEMORY_KEYS = ("peak_alloc_gib", "peak_reserved_gib", "mem_cap_gib", "vram_free_start_gib")

# ---------------------------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------------------------


@dataclass
class RunOptions:
    suite: str = "run"                 # "quick" | "full" | "run"
    gpu: int = 0
    out: Path | None = None
    resume: bool = True
    retry_failed: bool = True
    retry_oom: bool = False            # re-measure scenarios whose earlier result was oom
    budget_min: float | None = None    # None = suite default
    mem_cap_gib: float | None = None
    timeout_scale: float = 1.0
    tiny: bool = False
    smoke: bool = False                # real weights, tiny shapes/loops (maintainers, shared GPUs)
    device: str = "cuda"
    allow_cpu_only: bool = False
    dry_run: bool = False
    fetch: bool = True
    report: bool = True
    network: bool = True
    python: str = field(default_factory=lambda: sys.executable)

    def budget_s(self) -> float:
        b = self.budget_min if self.budget_min is not None else C.DEFAULT_BUDGET_MIN.get(self.suite, 0.0)
        return max(0.0, float(b)) * 60.0


# ---------------------------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------------------------


def next_rung(ladder, attempts) -> tuple[int, int] | None:
    """Next (micro_batch, accum) to try, or None to stop (SPEC 8.4).

    ``attempts``: list of dicts with micro_batch, accum, status (in order). The first call (no
    attempts) returns the first rung; after an ``oom`` it returns the following rung; after any
    other status, or after the last rung, None.
    """
    rungs = [tuple(int(x) for x in r) for r in (ladder or [])]
    if not rungs:
        return None
    if not attempts:
        return rungs[0]
    last = attempts[-1]
    if last.get("status") != "oom":
        return None
    key = (int(last.get("micro_batch", -1)), int(last.get("accum", -1)))
    idx = rungs.index(key) if key in rungs else len(attempts) - 1
    return rungs[idx + 1] if idx + 1 < len(rungs) else None


def calibration_factor(finished: list[tuple[float, float]]) -> float:
    """Median of actual_wall / est_s over scenarios finished in this run (1.0 before the first)."""
    ratios = [w / e for w, e in finished if e and e > 0 and w is not None and w >= 0]
    return float(median(ratios)) if ratios else 1.0


def over_budget(elapsed_s: float, est_s: float, calibration: float, budget_s: float) -> bool:
    """True if starting a scenario now would be projected to end past the budget (0 = unlimited)."""
    if not budget_s or budget_s <= 0:
        return False
    return elapsed_s + est_s * calibration > budget_s


def resume_index(records: list[dict], env_fp: str) -> dict[tuple[str, str], dict]:
    """(scenario_id, scenario_hash) -> {"done": rec|None, "last": rec} for one env fingerprint."""
    idx: dict[tuple[str, str], dict] = {}
    scen = [r for r in records if r.get("kind") == "scenario" and r.get("env_fingerprint") == env_fp]
    scen.sort(key=lambda r: str(r.get("finished_utc") or ""))
    for r in scen:
        key = (r.get("scenario_id"), r.get("scenario_hash"))
        slot = idx.setdefault(key, {"done": None, "last": None})
        slot["last"] = r
        if is_done(r):
            slot["done"] = r
    return idx


def is_done(rec: dict) -> bool:
    st = rec.get("status")
    if st in ("ok", "oom"):
        return True
    if st == "skipped":
        return (rec.get("status_detail") or "").split(":")[0].strip() not in C.RETRYABLE_SKIP_REASONS
    return False


def resume_decision(index: dict, sid: str, shash: str, retry_failed: bool = True,
                    retry_oom: bool = False) -> str | None:
    """'cached' (finished earlier), 'failed_earlier' (skip, --no-retry-failed) or None (run it).

    ``retry_oom``: a scenario whose latest finished result is ``oom`` runs again (for example after
    another program held GPU memory during the first run)."""
    slot = index.get((sid, shash))
    if not slot:
        return None
    if slot["done"] is not None:
        if retry_oom and slot["done"].get("status") == "oom":
            return None
        return "cached"
    last = slot["last"]
    if last is not None and last.get("status") in FAILED_STATUSES and not retry_failed:
        return "failed_earlier"
    return None


def run_slug(slug: str, *, tiny: bool = False, smoke: bool = False) -> str:
    """Results slug of a run: test runs on a real GPU get their own slug (``rtx-3090-24gb-tiny``,
    ``...-smoke``), so their records never share a results file, resume entries or a report column
    with real benchmark runs. CPU runs stay ``cpu``."""
    if slug == "cpu":
        return slug
    if tiny:
        return f"{slug}-tiny"
    if smoke:
        return f"{slug}-smoke"
    return slug


def rung_str(mb, acc) -> str:
    return f"{mb}x{acc}"


def summarize_attempts(attempts: list[dict], ladder) -> tuple[str, str, dict | None]:
    """(status, status_detail, fallback) of a model scenario from its attempts."""
    if not attempts:
        return "error", "no attempt ran", None
    last = attempts[-1]
    ooms = [rung_str(a["micro_batch"], a["accum"]) for a in attempts if a["status"] == "oom"]
    st = last["status"]
    here = rung_str(last["micro_batch"], last["accum"])
    if st == "ok":
        detail = f"fits at {here}" if not ooms else f"{here} after OOM at {', '.join(ooms)}"
    elif st == "oom":
        detail = f"OOM at {', '.join(ooms)}"
    else:
        detail = last.get("status_detail") or st
        if ooms:
            detail = f"{st} at {here} after OOM at {', '.join(ooms)}: {detail}"
    used = {"micro_batch": last["micro_batch"], "accum": last["accum"]} if st == "ok" else None
    fallback = {
        "ladder": [list(r) for r in ladder] if ladder else None,
        "attempts": [{k: a.get(k) for k in ("micro_batch", "accum", "status", *ATTEMPT_MEMORY_KEYS,
                                            "wall_s", "log_file")}
                     for a in attempts],
        "used": used,
        "fits_requested": bool(st == "ok" and len(attempts) == 1),
    }
    return st, detail, fallback


def child_env(base: dict, *, gpu: int, device: str, scenario_env: dict | None = None,
              offline: bool = True) -> dict:
    """Environment for a child process (SPEC 8.7)."""
    env = dict(base)
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "HF_HUB_DISABLE_TELEMETRY": "1",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "" if device == "cpu" else str(gpu)
    for k, v in (scenario_env or {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = str(v)
    return env


def popen_kwargs() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


_PARENT_TAG: str | None = None


def parent_tag() -> str:
    """``<pid>:<create_time>`` of this (runner) process, passed to every child in AGBENCH_PARENT so
    that a child exits when the runner is killed (agbench.child.watch_parent)."""
    global _PARENT_TAG
    if _PARENT_TAG is None:
        try:
            import psutil

            ctime = psutil.Process(os.getpid()).create_time()
            _PARENT_TAG = f"{os.getpid()}:{ctime!r}"
        except Exception:
            _PARENT_TAG = str(os.getpid())
    return _PARENT_TAG


_ACTIVE: list[subprocess.Popen] = []


def launch_process(cmd: list[str], *, env: dict | None, log_path: Path | None, timeout: float | None,
                   cwd: str | None = None, poll_s: float = 0.25) -> dict:
    """Run ``cmd`` in its own process group; stdout+stderr -> ``log_path`` (None = inherit console).

    On timeout the whole process tree is killed. Ctrl+C kills the tree and re-raises.
    Returns {"exit_code", "timed_out", "wall_s"}.
    """
    t0 = time.perf_counter()
    logf = None
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        logf = open(log_path, "wb")
    env = dict(os.environ if env is None else env)
    env[PARENT_ENV] = parent_tag()
    try:
        proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                                stdout=logf if logf else None,
                                stderr=subprocess.STDOUT if logf else None, **popen_kwargs())
        add_to_kill_on_exit_job(proc)  # Windows: dies with the runner even if the runner is killed
        _ACTIVE.append(proc)
        timed_out = False
        try:
            while True:
                try:
                    rc = proc.wait(timeout=poll_s)
                    break
                except subprocess.TimeoutExpired:
                    if timeout and time.perf_counter() - t0 > timeout:
                        timed_out = True
                        kill_tree(proc.pid)
                        try:
                            rc = proc.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            rc = None
                        break
        except BaseException:
            kill_tree(proc.pid)
            try:
                proc.wait(timeout=15)
            except Exception:
                pass
            raise
        finally:
            if proc in _ACTIVE:
                _ACTIVE.remove(proc)
    finally:
        if logf:
            logf.close()
    if log_path is not None and not sanitize_file(Path(log_path)):
        ascii_print(f"WARNING: could not rewrite {Path(log_path).name} without private paths (file locked?)")
    return {"exit_code": rc, "timed_out": timed_out, "wall_s": time.perf_counter() - t0}


def git_commit() -> str | None:
    root = repo_root()
    if root is None or not (root / ".git").exists():
        return None
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        commit = out.stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10)
        if dirty.returncode == 0 and dirty.stdout.strip():
            commit += "-dirty"
        return commit or None
    except Exception:
        return None


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)


def metric_summary(rec: dict) -> str:
    m = rec.get("metrics") or {}
    wl = rec.get("workload")
    parts = []
    try:
        if wl == "basics.matmul":
            best = m.get("tflops_best") or {}
            if not best and isinstance(m.get("tflops"), dict):
                best = {k: max(v.values()) for k, v in m["tflops"].items() if isinstance(v, dict) and v}
            parts = [f"{k} {v:.3g}" for k, v in best.items() if isinstance(v, (int, float))]
            return ("TFLOPS " + ", ".join(parts)) if parts else ""
        if wl in ("basics.membw", "basics.pcie"):
            g = m.get("gbps") or {}
            return ", ".join(f"{k} {v:.3g} GB/s" for k, v in g.items() if isinstance(v, (int, float)))
        cps = m.get("clips_per_s")
        if isinstance(cps, (int, float)):
            parts.append(f"{cps:.1f} clips/s")
        pr = m.get("peak_reserved_gib")
        if isinstance(pr, (int, float)):
            parts.append(f"{pr:.1f} GiB")
    except Exception:
        pass
    return ", ".join(parts)


# ---------------------------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------------------------


class SetupError(RuntimeError):
    pass


class SuiteRunner:
    def __init__(self, ids: list[str], opts: RunOptions):
        unknown = [i for i in ids if i not in SCENARIOS]
        if unknown:
            raise ValueError(f"unknown scenario id(s): {', '.join(unknown)}")
        self.ids = list(dict.fromkeys(ids))
        self.opts = opts
        self.results_dir = Path(opts.out).parent if opts.out else default_results_dir()
        self.run_id = new_run_id()
        self.logs_dir = self.results_dir / "logs" / self.run_id
        self.out: Path | None = Path(opts.out) if opts.out else None
        self.counts = {k: 0 for k in ("ok", "oom", "error", "timeout", "crashed", "skipped", "cached")}
        self.finished: list[tuple[float, float]] = []
        self.t0 = time.perf_counter()
        self.excluded_s = 0.0          # time spent downloading models (not charged to the budget)
        self.env: dict = {}
        self.gpu_slug = "cpu"
        self.env_fp = ""
        self.no_gpu = False
        self.missing_models: set[str] = set()
        self.report_failed = False
        self.started_utc = utc_now_iso()

    # -- helpers ------------------------------------------------------------------------------
    def eff(self, sid: str) -> Scenario:
        return effective(SCENARIOS[sid], tiny=self.opts.tiny, device=self.opts.device, smoke=self.opts.smoke)

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def budget_elapsed(self) -> float:
        return self.elapsed() - self.excluded_s

    def rel(self, p: Path) -> str:
        """Path relative to the results directory (what records store)."""
        try:
            return Path(p).relative_to(self.results_dir).as_posix()
        except ValueError:
            return sanitize(str(p))

    @staticmethod
    def show(p: Path) -> str:
        """Path for console messages: relative to the working directory when possible."""
        try:
            return Path(p).resolve().relative_to(Path.cwd().resolve()).as_posix()
        except (ValueError, OSError):
            return sanitize(str(p))

    def setup_timeout(self, base_s: float) -> float:
        return base_s * max(1.0, self.opts.timeout_scale)   # --timeout-scale < 1 is for scenarios only

    def py(self, *args: str) -> list[str]:
        return [self.opts.python, "-m", *args]

    def base_env(self, offline: bool) -> dict:
        return child_env(os.environ, gpu=self.opts.gpu, device=self.opts.device, offline=offline)

    # -- plan ---------------------------------------------------------------------------------
    def print_plan(self) -> None:
        total = 0.0
        ascii_print(f"Plan ({self.opts.suite}, {len(self.ids)} scenarios"
                    + (", tiny" if self.opts.tiny else "") + (", smoke" if self.opts.smoke else "")
                    + f", device {self.opts.device}):")
        for i, sid in enumerate(self.ids, 1):
            s = self.eff(sid)
            lad = " -> ".join(rung_str(*r) for r in s.ladder) if s.ladder else "-"
            total += s.est_s
            ascii_print(f"[{i:>2}/{len(self.ids)}] {sid:<36} est {s.est_s:>5.0f} s  "
                        f"timeout {timeout_s(s, self.opts.timeout_scale):>5.0f} s  ladder {lad}")
        b = self.opts.budget_s()
        ascii_print(f"Estimated total: {fmt_duration(total + C.SETUP_EST_S)} on an RTX 5060 Ti 16 GB "
                    f"(scenarios {fmt_duration(total)} + setup ~{C.SETUP_EST_S:.0f} s); budget "
                    + (fmt_duration(b) if b else "unlimited")
                    + "; model downloads (~3.3 GB, first run only) not included")

    # -- setup steps --------------------------------------------------------------------------
    def run_doctor(self) -> dict:
        tmp = self.logs_dir / "doctor.json"
        cmd = self.py("agbench", "doctor", "--json", str(tmp), "--gpu", str(self.opts.gpu),
                      "--device", self.opts.device, "--results-dir", str(self.results_dir))
        if self.opts.tiny:
            cmd.append("--tiny")
        if not self.opts.network:
            cmd.append("--no-network")
        ascii_print("Running doctor ...")
        env = child_env(os.environ, gpu=self.opts.gpu, device=self.opts.device, offline=False)
        res = launch_process(cmd, env=env, log_path=self.logs_dir / "doctor.log",
                             timeout=self.setup_timeout(600))
        text = tail_text(self.logs_dir / "doctor.log", n_lines=60)
        if text:
            ascii_print(text)
        try:
            return json.loads(tmp.read_text(encoding="utf-8"))
        except Exception:
            raise SetupError(f"doctor did not produce a result (exit code {res['exit_code']}); "
                             f"see {self.rel(self.logs_dir / 'doctor.log')}")

    def check_models(self, keys: list[str]) -> None:
        if not keys:
            return
        offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes", "on")
        tmp = self.logs_dir / "fetch.json"
        download = self.opts.fetch and not offline
        cmd = self.py("agbench", "fetch", "--models", ",".join(keys), "--json", str(tmp))
        if not download:
            cmd.append("--check")
        ascii_print(("Fetching models: " if download else "Checking models: ") + ", ".join(keys))
        t = time.perf_counter()
        res = launch_process(cmd, env=child_env(os.environ, gpu=self.opts.gpu, device=self.opts.device,
                                                offline=False),
                             log_path=None if download else self.logs_dir / "fetch.log",
                             timeout=self.setup_timeout(4 * 3600))
        if download:
            self.excluded_s += time.perf_counter() - t
        try:
            status = json.loads(tmp.read_text(encoding="utf-8"))
        except Exception:
            status = {}
            ascii_print(f"WARNING: model check failed (exit code {res['exit_code']}); "
                        "scenarios whose models are missing will be skipped")
        for k in keys:
            st = (status.get(k) or {}).get("status")
            if st not in ("cached", "downloaded"):
                self.missing_models.add(k)
        if self.missing_models:
            ascii_print("WARNING: models not available: " + ", ".join(sorted(self.missing_models))
                        + " (their scenarios are recorded as skipped: model_missing; rerun "
                          "'uv run --no-sync python -m agbench fetch' and then the same command)")

    def prepare_corpus(self) -> None:
        cmd = self.py("agbench", "prepare")
        if self.opts.tiny:
            cmd.append("--tiny")
        ascii_print("Preparing synthetic FLAC corpus ...")
        res = launch_process(cmd, env=self.base_env(offline=True), log_path=self.logs_dir / "prepare.log",
                             timeout=self.setup_timeout(1800))
        if res["exit_code"] != 0:
            ascii_print(f"WARNING: corpus preparation failed (exit code {res['exit_code']}); see "
                        f"{self.rel(self.logs_dir / 'prepare.log')}")

    # -- records ------------------------------------------------------------------------------
    def write(self, rec: dict) -> None:
        problems = validate_record(rec)
        if problems:
            ascii_print("WARNING: record problems: " + "; ".join(problems[:5]))
        append_jsonl(self.out, rec)

    def skip_record(self, s: Scenario, reason: str, detail: str = "") -> dict:
        now = utc_now_iso()
        return make_scenario_record(
            run_id=self.run_id, gpu_slug=self.gpu_slug, env_fingerprint=self.env_fp, scenario_id=s.id,
            scenario_hash=scenario_hash(s), workload=s.workload, workload_version=workload_version(s.workload),
            category=s.category, suite=self.opts.suite, status="skipped",
            status_detail=reason + (f": {detail}" if detail else ""), started_utc=now, finished_utc=now,
            wall_s=0.0, params=s.params, measure=(s.measure.to_dict() if s.measure else {}), fallback=None,
            metrics={}, estimates=empty_estimates(), telemetry={"source": "none"}, timeseries=None,
            extra={}, error=None, log_file=None)

    # -- attempts -----------------------------------------------------------------------------
    def run_one_attempt(self, s: Scenario, k: int, rung: tuple[int, int] | None) -> dict:
        safe_id = s.id
        json_path = self.logs_dir / f"{safe_id}.a{k}.json"
        log_path = self.logs_dir / f"{safe_id}.a{k}.log"
        cmd = self.py("agbench.child", "--scenario", s.id, "--attempt", str(k), "--out", str(json_path),
                      "--device", self.opts.device, "--gpu", str(self.opts.gpu))
        if rung is not None:
            cmd += ["--micro-batch", str(rung[0]), "--accum", str(rung[1])]
        if self.opts.tiny:
            cmd.append("--tiny")
        if self.opts.smoke:
            cmd.append("--smoke")
        if self.opts.mem_cap_gib:
            cmd += ["--mem-cap-gib", str(self.opts.mem_cap_gib)]
        env = child_env(os.environ, gpu=self.opts.gpu, device=self.opts.device, scenario_env=s.env)
        tmo = timeout_s(s, self.opts.timeout_scale)
        res = launch_process(cmd, env=env, log_path=log_path, timeout=tmo)
        att = {"micro_batch": rung[0] if rung else None, "accum": rung[1] if rung else None,
               "wall_s": res["wall_s"], "log_file": self.rel(log_path), "exit_code": res["exit_code"],
               "json": None}
        data = None
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                data = None
        if res["timed_out"]:
            att.update(status="timeout", status_detail=f"timeout after {tmo:.0f} s",
                       error={"type": "Timeout", "message": f"no result within {tmo:.0f} s; process tree "
                              "killed", "exit_code": res["exit_code"],
                              "traceback_tail": tail_text(log_path, 40)})
        elif res["exit_code"] != 0 or data is None:
            att.update(status="crashed", status_detail=f"crashed (exit code {res['exit_code']})",
                       error={"type": "Crash", "message": f"child exited with code {res['exit_code']}"
                              + ("" if data else " without writing a result"),
                              "exit_code": res["exit_code"], "traceback_tail": tail_text(log_path, 40)})
        else:
            att.update(status=data.get("status", "error"), status_detail=data.get("status_detail", ""),
                       error=data.get("error"))
            mem = data.get("memory") or {}
            for k in ATTEMPT_MEMORY_KEYS:
                att[k] = mem.get(k)
            att["json"] = data
        for k in ATTEMPT_MEMORY_KEYS:
            att.setdefault(k, None)
        return att

    def run_scenario(self, s: Scenario) -> dict:
        started = utc_now_iso()
        t = time.perf_counter()
        attempts: list[dict] = []
        if s.has_micro_batch():
            ladder = s.ladder or (first_rung(s),)
            k = 0
            while True:
                rung = next_rung(ladder, attempts)
                if rung is None:
                    break
                k += 1
                attempts.append(self.run_one_attempt(s, k, rung))
            status, detail, fallback = summarize_attempts(attempts, s.ladder)
            if fallback is not None and not s.ladder:
                fallback["ladder"] = None
        else:
            attempts.append(self.run_one_attempt(s, 1, None))
            status, detail, fallback = attempts[0]["status"], attempts[0].get("status_detail", ""), None
        last = attempts[-1]
        data = last.get("json") or {}
        extra = dict(data.get("extra") or {})
        extra["env_vars"] = {k: v for k, v in child_env(os.environ, gpu=self.opts.gpu, device=self.opts.device,
                                                         scenario_env=s.env).items()
                             if k in C.ENV_ALLOWLIST}
        if data.get("device"):
            extra["device"] = data["device"]
        error = last.get("error") if status in FAILED_STATUSES else None
        rec = make_scenario_record(
            run_id=self.run_id, gpu_slug=self.gpu_slug, env_fingerprint=self.env_fp, scenario_id=s.id,
            scenario_hash=scenario_hash(s), workload=s.workload, workload_version=workload_version(s.workload),
            category=s.category, suite=self.opts.suite, status=status, status_detail=detail,
            started_utc=started, finished_utc=utc_now_iso(), wall_s=time.perf_counter() - t, params=s.params,
            measure=data.get("measure") or (s.measure.to_dict() if s.measure else {}), fallback=fallback,
            metrics=data.get("metrics") if status == "ok" else {},
            estimates=data.get("estimates") if status == "ok" else empty_estimates(),
            telemetry=data.get("telemetry") or {"source": "none"}, timeseries=data.get("timeseries"),
            extra=extra, error=error, log_file=last.get("log_file"))
        return rec

    # -- main loop ----------------------------------------------------------------------------
    def precheck(self, s: Scenario) -> tuple[str, str] | None:
        """Pre-launch skip (reason, detail) or None."""
        if self.no_gpu and (s.needs_gpu or s.has_micro_batch()):
            return "no_gpu", "doctor found no usable GPU (--allow-cpu-only)"
        miss = [m for m in s.models if m in self.missing_models]
        if miss and not self.opts.tiny:
            return "model_missing", ", ".join(miss)
        nw = SCENARIOS[s.id].params.get("num_workers")
        if s.workload == "cpu.loader" and isinstance(nw, int) and nw > (os.cpu_count() or 1):
            return "workers_exceed_cores", f"{nw} workers > {os.cpu_count()} logical cores"
        return None

    def run(self) -> int:
        if self.opts.dry_run:
            self.print_plan()
            return 0
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            ascii_print(C.TORCH_MISSING_MSG)
            return 2
        lock = RunLock(self.results_dir / ".lock")
        try:
            lock.acquire()
        except LockHeld as e:
            ascii_print(f"ERROR: {e}. Wait for it to finish (or delete the lock file if no run is active).")
            return 2
        old_term = None
        if os.name != "nt":
            def _term(signum, frame):
                raise KeyboardInterrupt
            old_term = signal.signal(signal.SIGTERM, _term)
        try:
            from .awake import keep_awake

            budget = self.opts.budget_s()
            with keep_awake(max_s=max(3600.0, 2 * budget) if budget else 4 * 3600.0):
                return self._run_locked()
        except KeyboardInterrupt:
            for p in list(_ACTIVE):
                kill_tree(p.pid)
            if self.out is not None and self.env_fp:
                try:
                    append_jsonl(self.out, make_run_end(run_id=self.run_id, wall_s=self.elapsed(),
                                                        counts=self.counts, interrupted=True))
                except Exception:
                    pass
            ascii_print("\nInterrupted. Rerun the same command to resume (finished scenarios are kept).")
            return 130
        finally:
            lock.release()
            if old_term is not None:
                signal.signal(signal.SIGTERM, old_term)

    def kill_leftovers(self) -> int:
        """Kill benchmark processes left by an earlier run on this results folder that was killed
        (we hold the lock, so no live run owns them). They would hold GPU memory and compute."""
        try:
            procs = find_leftover_processes(self.results_dir)
        except Exception:
            return 0
        for p in procs:
            try:
                kill_tree(p.pid)
            except Exception:
                pass
        if procs:
            ascii_print(f"Stopped {len(procs)} leftover benchmark process(es) from an earlier run that was "
                        "killed; they would have used the GPU during this run.")
        return len(procs)

    def _run_locked(self) -> int:
        o = self.opts
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.kill_leftovers()
        ascii_print(f"agbench {__version__} {o.suite}: {len(self.ids)} scenarios"
                    + (" (tiny)" if o.tiny else "") + (" (smoke: real weights, small shapes; not a benchmark)"
                                                       if o.smoke else "")
                    + f", device {o.device}, run {self.run_id}")
        try:
            doc = self.run_doctor()
        except SetupError as e:
            ascii_print(f"ERROR: {e}")
            return 2
        fatal_checks = [c for c in doc.get("checks", []) if c.get("status") == "fatal"]
        if fatal_checks:
            torch_bad = any(c.get("n") in (1, 2) for c in fatal_checks)
            if o.allow_cpu_only and not torch_bad:
                self.no_gpu = True
                ascii_print("Doctor found no usable GPU: running only the cpu.* scenarios (--allow-cpu-only).")
            else:
                ascii_print("ERROR: doctor reported a fatal problem (see above); nothing was run. "
                            "Fix it (see the troubleshooting table in AGENTS.md), then run the same command again.")
                return 2
        self.env = doc.get("env") or {}
        from .env import env_fingerprint, gpu_slug

        self.gpu_slug = run_slug(gpu_slug(self.env), tiny=o.tiny, smoke=o.smoke)
        self.env_fp = env_fingerprint(self.env)
        if self.out is None:
            self.out = self.results_dir / f"{self.gpu_slug}_{local_date()}.jsonl"
        ascii_print(f"GPU: {self.env.get('gpu_name') or 'none (cpu)'} -> results {self.show(self.out)}")

        # Resume index over earlier result files of this GPU.
        index: dict = {}
        if o.resume:
            files = set(self.results_dir.glob(f"{self.gpu_slug}_*.jsonl"))
            if self.out.exists():
                files.add(self.out)
            recs = [r for f in sorted(files) for r in load_records(f)]
            index = resume_index(recs, self.env_fp)

        plan: list[tuple[str, Scenario, str | None]] = []
        for sid in self.ids:
            s = self.eff(sid)
            dec = resume_decision(index, sid, scenario_hash(s), o.retry_failed, o.retry_oom) if o.resume else None
            plan.append((sid, s, dec))
        to_run = [s for _, s, d in plan if d is None]

        if not o.tiny and not self.no_gpu:
            keys = sorted({m for s in to_run for m in s.models})
            self.check_models(keys)
        if any(s.category in ("e2e", "cpu") for s in to_run):
            self.prepare_corpus()

        self.write(make_run_start(
            run_id=self.run_id, suite=o.suite, gpu_slug=self.gpu_slug, env_fingerprint=self.env_fp,
            env=self.env, git_commit=git_commit(), started_utc=self.started_utc,
            settings={"budget_min": (o.budget_s() / 60.0) or 0, "mem_cap_gib": o.mem_cap_gib, "tiny": o.tiny,
                      "smoke": o.smoke,
                      "device": o.device, "timeout_scale": o.timeout_scale, "gpu_index": o.gpu,
                      "resume": o.resume, "retry_failed": o.retry_failed, "retry_oom": o.retry_oom,
                      "fetch": o.fetch,
                      "n_scenarios": len(self.ids)}))

        budget = o.budget_s()
        n = len(plan)
        for i, (sid, s, dec) in enumerate(plan, 1):
            head = f"[{i:>2}/{n}] {sid} ..."
            if dec is not None:
                self.counts["cached"] += 1
                ascii_print(f"{head} cached" + (" (failed earlier, --no-retry-failed)"
                                                if dec == "failed_earlier" else ""))
                continue
            pre = self.precheck(s)
            cal = calibration_factor(self.finished)
            if pre is None and over_budget(self.budget_elapsed(), s.est_s, cal, budget):
                pre = ("budget", f"projected {fmt_duration(self.budget_elapsed() + s.est_s * cal)} > "
                                 f"{fmt_duration(budget)}")
            if pre is not None:
                rec = self.skip_record(s, *pre)
                self.write(rec)
                self.counts["skipped"] += 1
                ascii_print(f"{head} skipped ({pre[0]})")
                continue
            ascii_print(head, end=" ")
            rec = self.run_scenario(s)
            self.write(rec)
            st = rec["status"]
            self.counts[st] = self.counts.get(st, 0) + 1
            if st in ("ok", "oom", "error", "timeout", "crashed"):
                self.finished.append((rec["wall_s"], s.est_s))
            cal = calibration_factor(self.finished)
            left = sum(x.est_s for j, (_, x, d) in enumerate(plan, 1) if j > i and d is None) * cal
            ms = metric_summary(rec)
            detail = rec.get("status_detail") or ""
            if st not in ("ok",):
                detail = detail[:160]
            ascii_print(f"{st}" + (f" ({detail})" if detail else "") + (f" {ms}," if ms else "")
                        + f" {rec['wall_s']:.0f} s [elapsed {fmt_duration(self.elapsed())}, "
                          f"est left {fmt_duration(left)}]")
            if st in FAILED_STATUSES:
                ascii_print(f"        log: {rec.get('log_file')}")

        self.write(make_run_end(run_id=self.run_id, wall_s=self.elapsed(), counts=self.counts))
        c = self.counts
        ascii_print(f"DONE in {fmt_duration(self.elapsed())}: ok {c['ok']}, oom {c['oom']}, error {c['error']}, "
                    f"timeout {c['timeout']}, crashed {c['crashed']}, skipped {c['skipped']}, "
                    f"cached {c['cached']} -> {self.show(self.out)}")
        if o.report:
            self.build_report()
        failed = any(c[k] for k in FAILED_STATUSES)
        return 1 if (failed or self.report_failed) else 0

    def build_report(self) -> Path | None:
        paths = sorted(set(self.results_dir.glob(f"{self.gpu_slug}_*.jsonl")) | {self.out})
        out_dir = self.results_dir / f"{self.out.stem}_report"
        try:
            from .report import build_report

            refs = (repo_root() or Path(".")) / "references" / "reference_gpus.json"
            summary = build_report(paths, out_dir, references_path=refs)
            ascii_print(f"Report: {self.show(Path(summary))}")
            return Path(summary)
        except Exception as e:
            self.report_failed = True
            ascii_print(f"REPORT FAILED: {type(e).__name__}: {sanitize(str(e))[:300]} "
                        f"(retry with: uv run --no-sync python -m agbench report)")
            return None


def run_suite(ids: list[str], opts) -> int:
    """Run scenarios ``ids`` with ``opts`` (RunOptions); returns the exit code of SPEC 7."""
    return SuiteRunner(ids, opts).run()
