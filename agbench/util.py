"""Small helpers: atomic files, JSONL append, privacy sanitizer, process-tree kill, ASCII output.

Standard library + psutil only (the runner process imports this; it must never import torch).
"""

from __future__ import annotations

import getpass
import json
import math
import os
import re
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------------


def repo_root() -> Path | None:
    """The source checkout that contains this package (None for a non-editable install)."""
    root = Path(__file__).resolve().parent.parent
    return root if (root / "pyproject.toml").is_file() else None


def default_results_dir() -> Path:
    env = os.environ.get("AGBENCH_RESULTS")
    if env:
        return Path(env)
    root = repo_root()
    return (root / "results") if root else Path("results")


def default_cache_dir() -> Path:
    env = os.environ.get("AGBENCH_CACHE")
    if env:
        return Path(env)
    root = repo_root()
    return (root / "cache") if root else Path("cache")


# ---------------------------------------------------------------------------------------------
# Time and formatting
# ---------------------------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def local_date() -> str:
    return datetime.now().strftime("%Y%m%d")


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "?"
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def to_ascii(text: str) -> str:
    return str(text).encode("ascii", "replace").decode("ascii")


def ascii_print(*parts: Any, end: str = "\n", file=None) -> None:
    """Print ASCII only (Windows consoles may be cp1252) and flush."""
    stream = file or sys.stdout
    stream.write(to_ascii(" ".join(str(p) for p in parts)) + end)
    stream.flush()


# ---------------------------------------------------------------------------------------------
# Privacy sanitizer (SPEC 8.9)
# ---------------------------------------------------------------------------------------------

_SANITIZE_RULES: list[tuple[re.Pattern, str]] | None = None


def _identity() -> tuple[str, str, str]:
    try:
        home = str(Path.home())
    except Exception:
        home = ""
    try:
        user = getpass.getuser()
    except Exception:
        user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    try:
        host = socket.gethostname()
    except Exception:
        host = ""
    return home, user, host


def build_sanitize_rules(home: str, user: str, host: str) -> list[tuple[re.Pattern, str]]:
    rules: list[tuple[re.Pattern, str]] = []
    homes: set[str] = set()
    if home and len(home) > 1:
        h = home.rstrip("/\\")
        homes.update({h, h.replace("\\", "/"), h.replace("/", "\\")})
        if "\\" in h or "/" in h:  # escaped backslashes as they appear inside JSON/repr strings
            homes.add(h.replace("\\", "\\\\"))
    for h in sorted(homes, key=len, reverse=True):
        rules.append((re.compile(re.escape(h) + r"(?![A-Za-z0-9])", re.IGNORECASE), "~"))
    # Short host names (first label) and the full name; user name. Whole-token match only, so a
    # user called "dev" does not turn "--device" into "--<user>ice".
    names: list[tuple[str, str]] = []
    if host and len(host) > 2:
        names.append((host, "<host>"))
        short = host.split(".")[0]
        if short != host and len(short) > 2:
            names.append((short, "<host>"))
    if user and len(user) > 2:
        names.append((user, "<user>"))
    for name, repl in sorted(names, key=lambda x: len(x[0]), reverse=True):
        rules.append((re.compile(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])",
                                 re.IGNORECASE), repl))
    return rules


def sanitize(text: str) -> str:
    """Replace the home directory with ``~``, user name with ``<user>``, host name with ``<host>``."""
    global _SANITIZE_RULES
    if _SANITIZE_RULES is None:
        _SANITIZE_RULES = build_sanitize_rules(*_identity())
    out = str(text)
    for pat, repl in _SANITIZE_RULES:
        out = pat.sub(repl, out)
    return out


def sanitize_obj(obj: Any) -> Any:
    """Sanitize every string in a JSON-like structure; non-finite floats become None."""
    if isinstance(obj, str):
        return sanitize(obj)
    if isinstance(obj, bool) or obj is None or isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {sanitize(str(k)): sanitize_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_obj(v) for v in obj]
    if isinstance(obj, Path):
        return sanitize(str(obj))
    # numpy scalars and other number-likes
    for cast in (int, float):
        try:
            if hasattr(obj, "item"):
                return sanitize_obj(obj.item())
            return sanitize_obj(cast(obj))
        except Exception:
            continue
    return sanitize(str(obj))


# ---------------------------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------------------------


def dumps_line(rec: dict) -> str:
    return json.dumps(rec, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def replace_file(src: Path, dst: Path, *, retries: int | None = None, wait_s: float = 0.2) -> None:
    """``os.replace`` that retries on Windows, where a virus scanner or the search indexer can hold
    a newly written file for a moment (PermissionError, WinError 5 / 32)."""
    n = (10 if os.name == "nt" else 0) if retries is None else retries
    for i in range(n + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i >= n:
                raise
            time.sleep(wait_s)


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    replace_file(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def append_jsonl(path: Path, rec: dict) -> None:
    """Append one complete line, flush and fsync (a crash leaves at most a truncated last line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = dumps_line(rec) + "\n"
    # If an earlier writer died mid-line, start on a fresh line so the new record stays parseable.
    needs_nl = False
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_nl = f.read(1) != b"\n"
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        if needs_nl:
            f.write("\n")
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict]:
    """All parseable JSON object lines, in order; unparseable lines (a truncated tail) are skipped."""
    out: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
    except FileNotFoundError:
        pass
    return out


def tail_text(path: Path, n_lines: int = 40, max_bytes: int = 64_000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    return "\n".join(data.splitlines()[-n_lines:])


def sanitize_file(path: Path) -> bool:
    """Rewrite a (log) file through the sanitizer. Never raises: if the file cannot be rewritten
    (for example a locked file on Windows), it is left as it is and False is returned."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    clean = sanitize(text)
    if clean == text:
        return True
    try:
        atomic_write_text(Path(path), clean)
        return True
    except OSError:
        try:
            Path(path).with_name(Path(path).name + ".tmp").unlink()
        except OSError:
            pass
        return False


# ---------------------------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------------------------


def kill_tree(pid: int, timeout: float = 5.0, include_parent: bool = True) -> int:
    """Terminate a process and all its descendants; kill whatever survives ``timeout`` s.

    Returns the number of processes that were signalled.
    """
    import psutil

    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 0
    try:
        procs = parent.children(recursive=True)
    except psutil.Error:
        procs = []
    if include_parent:
        procs.append(parent)
    for p in procs:
        try:
            p.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass
    if alive:
        psutil.wait_procs(alive, timeout=timeout)
    return len(procs)


def pid_alive(pid: int, create_time: float | None = None) -> bool:
    import psutil

    try:
        p = psutil.Process(pid)
        if p.status() == psutil.STATUS_ZOMBIE:
            return False
        if create_time is not None and abs(p.create_time() - create_time) > 1.0:
            return False  # pid was reused by another process
        return True
    except psutil.Error:
        return False


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def find_leftover_processes(results_dir: Path) -> list:
    """Live processes from an earlier, killed run on the same results folder.

    Matches Python processes that run a scenario (``-m agbench.child``) or the runner's doctor
    (``-m agbench doctor``) and name a path inside ``results_dir`` (the attempt JSON of a child,
    ``--results-dir`` of the doctor). Call it only while holding the results lock: then no live run
    owns such a process. This process and its ancestors are never returned.
    """
    import psutil

    root = _norm(str(results_dir))
    me = os.getpid()
    skip = {me}
    try:
        skip.update(p.pid for p in psutil.Process(me).parents())
    except psutil.Error:
        pass
    found = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            if proc.info["pid"] in skip:
                continue
            cmd = proc.info.get("cmdline") or []
            if "-m" not in cmd:
                continue
            i = cmd.index("-m")
            mod = cmd[i + 1] if i + 1 < len(cmd) else ""
            sub = cmd[i + 2] if i + 2 < len(cmd) else ""
            if not (mod == "agbench.child" or (mod == "agbench" and sub == "doctor")):
                continue
            if any(_norm(a) == root or _norm(a).startswith(root + os.sep) for a in cmd if a and not a.startswith("-")):
                found.append(proc)
        except (psutil.Error, ValueError, OSError):
            continue
    return found


class _WinJob:
    """Windows Job Object with KILL_ON_JOB_CLOSE: when the runner dies (for example a tool timeout
    calls TerminateProcess on it), Windows closes the job handle and kills every process in it."""

    def __init__(self):
        self.handle = None
        if os.name != "nt":
            return
        try:
            import ctypes
            from ctypes import wintypes

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [(n, ctypes.c_ulonglong) for n in (
                    "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                    "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

            class BASIC(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                            ("SchedulingClass", wintypes.DWORD)]

            class EXTENDED(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                            ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                            ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.CreateJobObjectW.restype = wintypes.HANDLE
            k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            k32.SetInformationJobObject.restype = wintypes.BOOL
            k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
            k32.AssignProcessToJobObject.restype = wintypes.BOOL
            k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            job = k32.CreateJobObjectW(None, None)
            if not job:
                return
            info = EXTENDED()
            info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
                return  # 9 = JobObjectExtendedLimitInformation
            self._k32, self.handle = k32, job
        except Exception:
            self.handle = None

    def assign(self, proc) -> bool:
        if self.handle is None:
            return False
        try:
            return bool(self._k32.AssignProcessToJobObject(self.handle, int(proc._handle)))
        except Exception:
            return False


_JOB: _WinJob | None = None


def add_to_kill_on_exit_job(proc) -> bool:
    """Windows: put ``proc`` (a subprocess.Popen) in this process's kill-on-close job."""
    global _JOB
    if os.name != "nt":
        return False
    if _JOB is None:
        _JOB = _WinJob()
    return _JOB.assign(proc)


class LockHeld(RuntimeError):
    pass


class RunLock:
    """``results/.lock`` holding pid + start time; a lock whose process is gone is stale."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.acquired = False

    def acquire(self) -> "RunLock":
        import psutil

        self.path.parent.mkdir(parents=True, exist_ok=True)
        me = psutil.Process(os.getpid())
        payload = {"pid": os.getpid(), "create_time": me.create_time(), "started_utc": utc_now_iso()}
        for _ in range(3):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    info = json.loads(self.path.read_text(encoding="utf-8"))
                    holder, ctime = int(info.get("pid", -1)), info.get("create_time")
                except Exception:
                    info, holder, ctime = {}, -1, None
                if holder > 0 and pid_alive(holder, ctime):
                    raise LockHeld(f"another agbench run (pid {holder}, started "
                                   f"{info.get('started_utc', '?')}) holds {self.path.name}")
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            self.acquired = True
            return self
        raise LockHeld(f"could not acquire {self.path.name}")

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            info = json.loads(self.path.read_text(encoding="utf-8"))
            if int(info.get("pid", -1)) == os.getpid():
                self.path.unlink()
        except Exception:
            pass
        self.acquired = False

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


# ---------------------------------------------------------------------------------------------
# Stats (pure python; used by timing.py and the runner)
# ---------------------------------------------------------------------------------------------


def percentile(values: Iterable[float], q: float) -> float | None:
    """Linear-interpolation percentile (same as numpy's default), q in [0, 100]."""
    xs = sorted(float(v) for v in values)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def median(values: Iterable[float]) -> float | None:
    return percentile(values, 50.0)


def monotonic() -> float:
    return time.perf_counter()
