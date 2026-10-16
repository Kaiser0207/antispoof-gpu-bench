"""Model download and cache lookup (SPEC section 7.2).

Import-light (no torch, huggingface_hub imported lazily): the parent process uses it
for ``fetch``. ``HF_ENDPOINT`` (mirrors), ``HF_HOME``, ``HF_HUB_CACHE``, ``HF_HUB_OFFLINE``
and ``HF_TOKEN`` are honoured by huggingface_hub itself; the token is never read,
logged or written here.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

from agbench.models import MODELS, ModelSpec

ALLOW_PATTERNS = ["config.json", "preprocessor_config.json", "pytorch_model.bin"]
WEIGHTS_FILE = "pytorch_model.bin"
CONFIG_FILE = "config.json"
RETRIES = 3
BACKOFF_S = (5.0, 15.0, 45.0)
RATE_LIMIT_MSG = ("Hugging Face rate limit: wait and rerun `fetch`, "
                  "or set HF_TOKEN to a read token")
_TRUE = {"1", "true", "yes", "on"}


def _spec(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        raise KeyError(f"unknown model key {key!r}; known: {', '.join(MODELS)}") from None


def _tilde(text: str) -> str:
    """Replace the home directory prefix with ``~`` (both slash styles)."""
    home = str(Path.home())
    if len(home) > 1:
        for h in {home, home.replace("\\", "/"), home.replace("/", "\\")}:
            text = text.replace(h, "~")
    return text


def _check_folder(folder: Path, spec: ModelSpec) -> str | None:
    """Return a problem description, or None if the folder is usable."""
    cfg, weights = folder / CONFIG_FILE, folder / WEIGHTS_FILE
    if not cfg.is_file():
        return f"{CONFIG_FILE} missing"
    if not weights.is_file():
        return f"{WEIGHTS_FILE} missing"
    size = weights.stat().st_size
    if size != spec.weights_bytes:
        return f"{WEIGHTS_FILE} has {size} bytes, expected {spec.weights_bytes}"
    return None


def _verified_cache_file() -> Path:
    from agbench.util import default_cache_dir

    return default_cache_dir() / "verified_models.json"


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _file_id(path: Path) -> dict:
    st = Path(path).stat()
    return {"real": os.path.realpath(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def verify_hash(folder: Path, spec: ModelSpec) -> str | None:
    """Check pytorch_model.bin against the pinned sha256 (once per file: the result is remembered
    in cache/verified_models.json by real path, size and modification time). A mirror
    (HF_ENDPOINT) or a copied cache could otherwise serve a different file of the same size."""
    if not spec.sha256:
        return None
    import json

    weights = Path(folder) / WEIGHTS_FILE
    try:
        fid = _file_id(weights)
    except OSError as e:
        return f"{WEIGHTS_FILE} unreadable ({type(e).__name__})"
    cache_file = _verified_cache_file()
    try:
        known = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        known = {}
    if known.get(spec.key) == {**fid, "sha256": spec.sha256}:
        return None
    digest = sha256_file(weights)
    if digest != spec.sha256:
        return f"{WEIGHTS_FILE} sha256 {digest[:12]}... differs from the pinned {spec.sha256[:12]}..."
    known[spec.key] = {**fid, "sha256": spec.sha256}
    try:
        from agbench.util import atomic_write_json

        atomic_write_json(cache_file, known)
    except OSError:
        pass
    return None


def _models_dir_folder(key: str) -> Path | None:
    root = os.environ.get("AGBENCH_MODELS_DIR")
    return Path(root).expanduser() / key if root else None


def _hf_cache_folder(spec: ModelSpec) -> Path | None:
    """Snapshot folder of the pinned revision in the HF cache (pure filesystem lookup)."""
    from huggingface_hub import try_to_load_from_cache

    try:
        found = try_to_load_from_cache(spec.hf_id, WEIGHTS_FILE, revision=spec.revision)
    except Exception:
        return None
    return Path(found).parent if isinstance(found, str) else None


def locate(key: str) -> tuple[Path | None, list[str]]:
    """(usable folder or None, problems found on the way). AGBENCH_MODELS_DIR wins."""
    spec = _spec(key)
    problems: list[str] = []
    for where, folder in (("AGBENCH_MODELS_DIR", _models_dir_folder(key)),
                          ("HF cache", _hf_cache_folder(spec))):
        if folder is None or not folder.exists():
            continue
        problem = _check_folder(folder, spec)
        if problem is None:
            return folder, problems
        problems.append(f"{where}: {problem}")
    return None, problems


def resolve(key: str) -> Path:
    """Local folder with config.json + pytorch_model.bin of the pinned revision.

    Raises ``SkipScenario("model_missing")`` when the model is not available locally.
    """
    folder, _ = locate(key)
    if folder is None:
        from agbench.errors import SkipScenario

        raise SkipScenario("model_missing")
    return folder


def is_cached(key: str) -> bool:
    return locate(key)[0] is not None


def source_of(folder: Path) -> str:
    """'models_dir' or 'hf_cache' (recorded in results; never the path itself)."""
    root = os.environ.get("AGBENCH_MODELS_DIR")
    if root:
        try:
            Path(folder).resolve().relative_to(Path(root).expanduser().resolve())
            return "models_dir"
        except ValueError:
            pass
    return "hf_cache"


def _offline() -> bool:
    if os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in _TRUE:
        return True
    try:
        from huggingface_hub import constants

        return bool(constants.HF_HUB_OFFLINE)
    except Exception:  # pragma: no cover
        return False


def _status_code(exc: BaseException) -> int | None:
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    return int(code) if isinstance(code, int) else None


def _is_xet_error(exc: BaseException) -> bool:
    text = f"{type(exc).__module__}.{type(exc).__name__}: {exc}".lower()
    return "xet" in text


def _is_permanent(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in {"RepositoryNotFoundError", "RevisionNotFoundError", "GatedRepoError",
                "EntryNotFoundError", "RemoteEntryNotFoundError", "DisabledRepoError"}:
        return True
    return _status_code(exc) in {401, 403, 404}


def _disable_xet() -> None:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        from huggingface_hub import constants

        constants.HF_HUB_DISABLE_XET = True  # read at call time by file_download
    except Exception:  # pragma: no cover
        pass


def _result(key: str, folder: Path | None, status: str, error: str | None) -> dict:
    spec = _spec(key)
    nbytes = None
    if folder is not None and (folder / WEIGHTS_FILE).is_file():
        nbytes = (folder / WEIGHTS_FILE).stat().st_size
    return {"key": key, "hf_id": spec.hf_id, "revision": spec.revision,
            "path": _tilde(str(folder)) if folder is not None else None,
            "bytes": nbytes, "expected_bytes": spec.weights_bytes,
            "status": status, "error": _tilde(error)[:500] if error else None}


def check(keys: list[str]) -> list[dict]:
    """Verify only (``fetch --check``): status ``cached`` (size and sha256 match), ``bad`` or ``missing``."""
    out = []
    for key in keys:
        folder, problems = locate(key)
        if folder is not None:
            bad = verify_hash(folder, _spec(key))
            if bad:
                out.append(_result(key, folder, "bad", f"{bad}; rerun fetch to download it again"))
                continue
        out.append(_result(key, folder, "cached" if folder else "missing",
                           "; ".join(problems) or (None if folder else "not downloaded")))
    return out


def prefetch(keys: list[str], *, log: Callable[[str], None] | None = None,
             retries: int = RETRIES, backoff_s: tuple[float, ...] = BACKOFF_S) -> list[dict]:
    """Download the pinned revision of each model (config, preprocessor config, weights).

    Returns ``[{"key", "path", "bytes", "status": "cached"|"downloaded"|"failed", "error", ...}]``.
    Never raises for download problems; the caller decides what a failure means.
    """
    log = log or (lambda s: print(s, flush=True))
    out: list[dict] = []
    for key in keys:
        spec = _spec(key)
        folder, problems = locate(key)
        force = False
        if folder is not None:
            bad = verify_hash(folder, spec)
            if bad is None:
                out.append(_result(key, folder, "cached", None))
                continue
            if hub_folder_is_models_dir(folder):
                out.append(_result(key, folder, "failed", f"AGBENCH_MODELS_DIR: {bad}"))
                continue
            problems.append(f"HF cache: {bad}")
            force = True  # a foreign or corrupted file in the HF cache: download it again
        if _offline():
            msg = "not cached and HF_HUB_OFFLINE=1 (copy the HF cache or use AGBENCH_MODELS_DIR)"
            if problems:
                msg += "; " + "; ".join(problems)
            out.append(_result(key, None, "failed", msg))
            continue
        out.append(_download(spec, log, retries, backoff_s, force=force))
    return out


def hub_folder_is_models_dir(folder: Path) -> bool:
    return source_of(folder) == "models_dir"


def _download(spec: ModelSpec, log: Callable[[str], None], retries: int,
              backoff_s: tuple[float, ...], force: bool = False) -> dict:
    import huggingface_hub

    gb = spec.weights_bytes / 1e9
    log(f"fetch {spec.key}: {spec.hf_id}@{spec.revision[:10]} ({gb:.2f} GB)")
    xet_retried = False
    rate_limit_told = False
    last_error = "unknown error"
    attempt = 0
    while attempt <= retries:
        try:
            path = huggingface_hub.snapshot_download(
                spec.hf_id, revision=spec.revision, allow_patterns=ALLOW_PATTERNS,
                force_download=force)
            folder = Path(path)
            problem = _check_folder(folder, spec) or verify_hash(folder, spec)
            if problem is None:
                return _result(spec.key, folder, "downloaded", None)
            last_error = f"downloaded files failed verification: {problem}"
            force = True  # a truncated or foreign file: fetch it again from scratch
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - every hub error ends up in the result
            last_error = f"{type(exc).__name__}: {exc}"
            if _status_code(exc) == 429 and not rate_limit_told:
                log(RATE_LIMIT_MSG)
                rate_limit_told = True
            if (_is_xet_error(exc) and os.environ.get("HF_ENDPOINT") and not xet_retried):
                xet_retried = True
                log(f"fetch {spec.key}: Xet transfer failed via HF_ENDPOINT; retrying with HF_HUB_DISABLE_XET=1")
                _disable_xet()
                continue  # immediate extra attempt, not counted as a retry
            if _is_permanent(exc):
                break
        attempt += 1
        if attempt <= retries:
            wait = backoff_s[min(attempt - 1, len(backoff_s) - 1)] if backoff_s else 0.0
            log(f"fetch {spec.key}: attempt {attempt} failed ({_tilde(last_error)[:160]}); "
                f"retry in {wait:.0f} s")
            if wait > 0:
                time.sleep(wait)
    return _result(spec.key, None, "failed", last_error)
