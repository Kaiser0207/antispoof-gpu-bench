"""Environment capture, GPU slug / fingerprint, and the doctor checks (SPEC 7.1, 8.5, 8.9).

``gpu_slug`` and ``env_fingerprint`` are pure functions (runner-safe). ``capture_env`` and
``doctor`` import torch and therefore only run in child processes (or the ``doctor`` command).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import sys
from importlib import metadata
from pathlib import Path

from . import __version__
from . import constants as C
from .util import sanitize, sanitize_obj

# ---------------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------------


def os_family() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "macos"
    return sys.platform


def vram_class_gb(total_gib: float | None) -> int | None:
    if not total_gib:
        return None
    return min(C.VRAM_CLASSES_GB, key=lambda c: (abs(c - float(total_gib)), c))


def slug_from_name(name: str, total_gib: float | None) -> str:
    n = name.strip()
    for prefix in ("NVIDIA ", "GeForce "):
        if n.upper().startswith(prefix.upper()):
            n = n[len(prefix):]
    s = re.sub(r"[^a-z0-9]+", "-", n.lower()).strip("-")
    cls = vram_class_gb(total_gib)
    if cls and not s.endswith(f"{cls}gb"):
        s = f"{s}-{cls}gb"
    return s or "unknown-gpu"


def gpu_slug(env: dict) -> str:
    """``rtx-5060-ti-16gb`` style name; ``cpu`` when no GPU is used (SPEC 8.5)."""
    name = env.get("gpu_name")
    if not name or env.get("device") == "cpu":
        return "cpu"
    return slug_from_name(name, env.get("vram_total_gib"))


def env_fingerprint(env: dict) -> str:
    keys = ("gpu_name", "vram_class_gb", "driver", "torch", "torch_cuda", "transformers",
            "os_family", "agbench_version")
    payload = {k: env.get(k) for k in keys}
    if payload["agbench_version"] is None:
        payload["agbench_version"] = __version__
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def parse_version_tuple(v: str | None) -> tuple[int, ...]:
    if not v:
        return ()
    return tuple(int(x) for x in re.findall(r"\d+", str(v))[:3])


def driver_requirement(torch_cuda: str | None, family: str) -> tuple[int, int] | None:
    """Minimum driver for the CUDA version torch was built with (None = unknown / not needed)."""
    cv = parse_version_tuple(torch_cuda)
    if not cv:
        return None
    if cv[0] >= 13:
        return C.MIN_DRIVER_CUDA13
    if cv[:2] >= (12, 8):
        return C.MIN_DRIVER_CUDA128_WINDOWS if family == "windows" else C.MIN_DRIVER_CUDA128_LINUX
    return None


def arch_supported(capability, arch_list: list[str]) -> bool:
    """True if torch ships a kernel image (SASS of the same major) or usable PTX for this GPU.

    For compute capability >= 12.0 (Blackwell consumer, RTX 50xx) only sm_12x / compute_12x
    entries count (SPEC 7.1 check 4): older PTX would JIT slowly or not at all.
    """
    major, minor = int(capability[0]), int(capability[1])
    want = major * 10 + minor
    for a in arch_list:
        m = re.fullmatch(r"(sm|compute)_(\d+)a?", a)
        if not m:
            continue
        num = int(m.group(2))
        if m.group(1) == "sm" and num // 10 == major and num <= want:
            return True
        if m.group(1) == "compute" and num <= want and (major < 12 or num // 10 == major):
            return True
    return False


# ---------------------------------------------------------------------------------------------
# GPU static info (telemetry.nvml_static_info: NVML, else nvidia-smi); never raises
# ---------------------------------------------------------------------------------------------


def _gpu_processes(index: int, uuid: str | None) -> list[dict]:
    """Compute + graphics processes on the GPU (NVML only): [{"pid", "used_gib"}]."""
    procs: list[dict] = []
    try:
        import pynvml as nv  # provided by nvidia-ml-py

        nv.nvmlInit()
        try:
            try:
                h = nv.nvmlDeviceGetHandleByUUID(uuid) if uuid else None
            except Exception:
                h = None
            h = h or nv.nvmlDeviceGetHandleByIndex(int(index))
            for fn in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
                try:
                    for p in getattr(nv, fn)(h):
                        used = getattr(p, "usedGpuMemory", None)
                        procs.append({"pid": int(p.pid), "used_gib": (used / C.GIB) if used else None})
                except Exception:
                    pass
        finally:
            nv.nvmlShutdown()
    except Exception:
        pass
    return procs


def gpu_static_info(index: int = 0, uuid: str | None = None) -> dict:
    """driver, name, power limit, PCIe link (cur/max), clocks, temperature, GPU processes; {} if unknown."""
    try:
        from .telemetry import nvml_static_info

        info = dict(nvml_static_info(index, uuid) or {})
    except Exception:
        info = {}
    if info.get("source") in (None, "none"):
        return {}
    info["processes"] = _gpu_processes(index, uuid)
    return info


# ---------------------------------------------------------------------------------------------
# Host info
# ---------------------------------------------------------------------------------------------


def cpu_model() -> str | None:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if sys.platform.startswith("win"):
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except Exception:
        pass
    return platform.processor() or None


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def blas_info() -> list[dict]:
    try:
        from threadpoolctl import threadpool_info

        return [{"internal_api": i.get("internal_api"), "user_api": i.get("user_api"),
                 "version": i.get("version"), "num_threads": i.get("num_threads"),
                 "threading_layer": i.get("threading_layer")} for i in threadpool_info()]
    except Exception:
        return []


def env_vars_allowlisted() -> dict:
    out = {k: os.environ[k] for k in C.ENV_ALLOWLIST if k in os.environ}
    out["HF_ENDPOINT_set"] = bool(os.environ.get("HF_ENDPOINT"))
    return out


def host_info() -> dict:
    d = {"cpu_model": cpu_model(), "cpu_logical": os.cpu_count(), "cpu_physical": None, "ram_gib": None}
    try:
        import psutil

        d["cpu_physical"] = psutil.cpu_count(logical=False)
        d["ram_gib"] = psutil.virtual_memory().total / C.GIB
    except Exception:
        pass
    return d


# ---------------------------------------------------------------------------------------------
# capture_env (child only: imports torch)
# ---------------------------------------------------------------------------------------------


def capture_env(device_index: int = 0, *, device: str = "cuda") -> dict:
    """run_start.env (SPEC 9.1), sanitized and allow-listed. ``device_index`` = physical GPU index."""
    env: dict = {
        "os": platform.platform(),
        "os_family": os_family(),
        "python": platform.python_version(),
        "agbench_version": __version__,
        "device": device,
        "torch": None, "torch_cuda": None, "cudnn": None,
        "transformers": _pkg_version("transformers"),
        "huggingface_hub": _pkg_version("huggingface-hub"),
        "numpy": _pkg_version("numpy"),
        "scipy": _pkg_version("scipy"),
        "soundfile": _pkg_version("soundfile"),
        "libsndfile": None,
        "driver": None, "gpu_name": None, "gpu_capability": None, "gpu_arch_list": None,
        "vram_total_gib": None, "vram_class_gb": None, "gpu_index": device_index,
        "pcie": None, "power_limit_w": None,
    }
    try:
        import soundfile

        env["libsndfile"] = str(getattr(soundfile, "__libsndfile_version__", None))
    except Exception:
        pass
    env.update(host_info())
    try:
        import torch

        env["torch"] = torch.__version__
        env["torch_cuda"] = torch.version.cuda
        try:
            env["cudnn"] = torch.backends.cudnn.version()
        except Exception:
            pass
        env["torch_threads"] = torch.get_num_threads()
        if device != "cpu":
            env["gpu_arch_list"] = list(torch.cuda.get_arch_list())
            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                env["gpu_name"] = p.name
                env["gpu_capability"] = [int(p.major), int(p.minor)]
                env["vram_total_gib"] = round(p.total_memory / C.GIB, 3)
                env["vram_class_gb"] = vram_class_gb(env["vram_total_gib"])
                env["gpu_uuid_known"] = bool(gpu_uuid())
    except Exception as e:  # torch missing or broken: doctor reports it
        env["torch_error"] = f"{type(e).__name__}: {e}"[:300]
    if device != "cpu":
        st = gpu_static_info(device_index, gpu_uuid())
        env["driver"] = st.get("driver")
        env["pcie"] = st.get("pcie")
        env["power_limit_w"] = st.get("power_limit_w")
        env["sm_clock_max_mhz"] = st.get("sm_clock_max_mhz")
        if not env["gpu_name"] and st.get("name"):
            env["gpu_name_nvml"] = st.get("name")
    env["blas"] = blas_info()
    env["env_vars"] = env_vars_allowlisted()
    return sanitize_obj(env)


def gpu_uuid() -> str | None:
    """NVML-style uuid ("GPU-...") of torch device 0, or None."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        u = str(torch.cuda.get_device_properties(0).uuid)
        return u if u.startswith("GPU-") else f"GPU-{u}"
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------
# doctor (SPEC 7.1)
# ---------------------------------------------------------------------------------------------


def _hub_cache_dir() -> Path:
    try:
        from huggingface_hub import constants as hc

        return Path(hc.HF_HUB_CACHE)
    except Exception:
        home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        return Path(os.environ.get("HF_HUB_CACHE") or Path(home) / "hub")


def _free_gb(path: Path) -> float | None:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(p).free / 1e9
    except OSError:
        return None


def _rel_err(got, ref) -> float:
    import torch

    got = got.float().cpu()
    ref = ref.float().cpu()
    return float((got - ref).norm() / ref.norm().clamp_min(1e-12))


def kernel_smoke(rtol: float = 2e-2) -> list[str]:
    """bf16 matmul, conv1d and SDPA on the GPU vs CPU float32; returns problems (empty = fine)."""
    import torch
    import torch.nn.functional as F

    problems = []
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(256, 256, generator=g), torch.randn(256, 256, generator=g)
    x, w = torch.randn(2, 8, 4000, generator=g), torch.randn(16, 8, 10, generator=g)
    q, k, v = (torch.randn(2, 4, 128, 64, generator=g) for _ in range(3))
    tests = {
        "bf16 matmul": (lambda d, dt: a.to(d, dt) @ b.to(d, dt), a @ b),
        "conv1d": (lambda d, dt: F.conv1d(x.to(d, dt), w.to(d, dt), stride=5), F.conv1d(x, w, stride=5)),
        "sdpa": (lambda d, dt: F.scaled_dot_product_attention(q.to(d, dt), k.to(d, dt), v.to(d, dt)),
                 F.scaled_dot_product_attention(q, k, v)),
    }
    for name, (fn, ref) in tests.items():
        try:
            got = fn("cuda", torch.bfloat16)
            torch.cuda.synchronize()
            err = _rel_err(got, ref)
            if not err < rtol:
                problems.append(f"{name}: relative error {err:.3g} > {rtol}")
        except Exception as e:
            problems.append(f"{name}: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
    try:
        got = F.conv1d(x.cuda(), w.cuda(), stride=5)
        torch.cuda.synchronize()
        err = _rel_err(got, F.conv1d(x, w, stride=5))
        if not err < rtol:
            problems.append(f"fp32 conv1d: relative error {err:.3g} > {rtol}")
    except Exception as e:
        problems.append(f"fp32 conv1d: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
    return problems


def _network_check(timeout: float = 6.0) -> tuple[bool, str]:
    import urllib.error
    import urllib.request

    endpoint = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    req = urllib.request.Request(endpoint + "/api/models/facebook/wav2vec2-base", method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return True, f"HTTP {e.code}"  # reachable; the server answered
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"


def doctor(device_index: int = 0, network: bool = True, *, device: str = "cuda", tiny: bool = False,
           results_dir: Path | None = None, model_keys: tuple[str, ...] | None = None) -> dict:
    """Run the SPEC 7.1 checks. Returns {"status": "OK"|"WARN"|"FATAL", "checks": [...], "env": {...}}.

    ``device_index`` is the physical GPU index; torch sees it as device 0 (the caller sets
    CUDA_VISIBLE_DEVICES before importing torch).
    """
    checks: list[dict] = []
    fam = os_family()
    gpu_mode = device != "cpu"
    # NVML / nvidia-smi work without CUDA: read the driver first, so the fix hints can name the
    # right PyTorch build (driver >= 580 -> cu130, 570-579 -> cu128).
    st_pre: dict = gpu_static_info(device_index, None) if gpu_mode else {}
    hint = C.uv_sync_hint(st_pre.get("driver"))

    console_notes: list[str] = []  # printed by the doctor command, never saved

    def add(n, name, status, message, **kw):
        checks.append({"n": n, "name": name, "status": status, "message": sanitize(message), **kw})

    # 1 Python
    is64 = sys.maxsize > 2 ** 32
    pyok = sys.version_info[:2] == (3, 12) and is64
    add(1, "python", "ok" if pyok else "fatal",
        f"Python {platform.python_version()} {'64' if is64 else '32'}-bit"
        + ("" if pyok else " (need 3.12 64-bit: uv sync installs it)"))

    # 2 torch import and build
    torch = None
    try:
        import torch  # noqa: F811

        tv, tc = torch.__version__, torch.version.cuda
        if not gpu_mode or tiny:
            add(2, "torch", "ok", f"torch {tv} (CUDA {tc or 'none'})")
        elif tc is None:
            add(2, "torch", "fatal", f"torch {tv} is a CPU-only build: run \"{hint}\"")
        elif parse_version_tuple(tc)[:2] < (12, 8):
            add(2, "torch", "warn", f"torch {tv} built for CUDA {tc}: has no Blackwell (sm_120) kernels;"
                f" for RTX 50xx run \"{hint}\"")
        else:
            suffix_ok = tv.endswith(("+cu130", "+cu128"))
            add(2, "torch", "ok" if suffix_ok else "warn",
                f"torch {tv} (CUDA {tc})" + ("" if suffix_ok else
                                              f"; not the locked build, expected \"{hint}\""))
    except Exception as e:
        add(2, "torch", "fatal", f"{C.TORCH_MISSING_MSG} ({type(e).__name__}: {str(e)[:150]})")

    cuda_ok = False
    if gpu_mode and torch is not None:
        # 3 CUDA available
        try:
            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False
        add(3, "cuda_available", "ok" if cuda_ok else "fatal",
            "torch.cuda.is_available() = True" if cuda_ok else
            "torch.cuda.is_available() = False: check the NVIDIA driver (nvidia-smi must work; Blackwell "
            "needs driver >= 580 for the cu130 build or >= 570 for cu128) and that the torch build is "
            f"CUDA-enabled (\"{hint}\")")
    elif gpu_mode:
        add(3, "cuda_available", "fatal", "torch missing")
    else:
        for n, name in ((3, "cuda_available"), (4, "arch"), (5, "kernel_smoke"), (6, "driver"),
                        (7, "vram"), (8, "pcie")):
            add(n, name, "skip", "device=cpu")

    uuid = gpu_uuid() if cuda_ok else None
    st: dict = {}
    if gpu_mode:
        st = gpu_static_info(device_index, uuid) if uuid else st_pre
    if cuda_ok:
        # 4 arch list
        p = torch.cuda.get_device_properties(0)
        cap = (int(p.major), int(p.minor))
        arch = list(torch.cuda.get_arch_list())
        ok = arch_supported(cap, arch)
        add(4, "arch", "ok" if ok else "fatal",
            f"{p.name}, compute capability {cap[0]}.{cap[1]}, torch arch list {' '.join(arch)}"
            + ("" if ok else f" -> wrong torch wheel: reinstall it with \"{hint}\""))
        # 5 kernel smoke
        probs = kernel_smoke()
        add(5, "kernel_smoke", "ok" if not probs else "fatal",
            "bf16 matmul, conv1d, SDPA match the CPU" if not probs else
            "; ".join(probs) + " -> if this says 'no kernel image is available', the torch wheel lacks "
            f"this GPU's architecture: \"{hint}\"")
        # 6 driver
        drv = st.get("driver")
        need = driver_requirement(torch.version.cuda, fam)
        if drv is None:
            add(6, "driver", "warn", "driver version unknown (no NVML / nvidia-smi)")
        elif need and parse_version_tuple(drv)[:2] < need:
            add(6, "driver", "fatal", _driver_too_old(drv, need, torch.version.cuda))
        else:
            add(6, "driver", "ok", f"driver {drv}" + (f" (>= {need[0]}.{need[1]:02d})" if need else ""))
        # 7 VRAM
        try:
            free, total = torch.cuda.mem_get_info()
            free_g, total_g = free / C.GIB, total / C.GIB
            procs = [pp for pp in st.get("processes", []) if pp.get("pid") != os.getpid()]
            warn = free_g < total_g - 1.5
            msg = f"VRAM {total_g:.2f} GiB total, {free_g:.2f} GiB free"
            if warn:
                msg += " -> other programs hold VRAM; close them for clean numbers"
                if procs:
                    msg += f" ({len(procs)} other GPU process{'es' if len(procs) != 1 else ''})"
                    # Program names are shown on the console only, never saved (privacy).
                    console_notes.append("Other programs using the GPU: " + ", ".join(_process_names(procs)))
            add(7, "vram", "warn" if warn else "ok", msg, total_gib=total_g, free_gib=free_g)
        except Exception as e:
            add(7, "vram", "warn", f"could not read VRAM: {type(e).__name__}")
        # 8 PCIe
        pc = st.get("pcie") or {}
        if pc.get("width_max"):
            w = pc.get("width_max")
            add(8, "pcie", "warn" if w < 8 else "ok",
                f"PCIe gen {pc.get('gen_cur')}/{pc.get('gen_max')} (cur/max), width x{pc.get('width_cur')}"
                f"/x{w}" + (" -> link narrower than x8" if w < 8 else "")
                + " (an idle GPU may show a lower current gen)")
        else:
            add(8, "pcie", "info", "PCIe link unknown")
    elif gpu_mode:
        for n, name in ((4, "arch"), (5, "kernel_smoke")):
            add(n, name, "skip", "CUDA not available")
        # 6 driver: NVML works without CUDA, and a driver too old for this torch build is the most
        # common reason why CUDA is not available.
        drv = st.get("driver")
        need = driver_requirement(torch.version.cuda, fam) if torch is not None else None
        if drv is None:
            add(6, "driver", "skip", "CUDA not available and no NVML / nvidia-smi: is the NVIDIA driver installed?")
        elif need and parse_version_tuple(drv)[:2] < need:
            add(6, "driver", "fatal", _driver_too_old(drv, need, torch.version.cuda))
        else:
            add(6, "driver", "info", f"driver {drv}" + (f" (>= {need[0]}.{need[1]:02d})" if need else "")
                + " is new enough; CUDA is still not available (see check 3)")
        for n, name in ((7, "vram"), (8, "pcie")):
            add(n, name, "skip", "CUDA not available")

    # 9 Windows sysmem fallback reminder
    if fam == "windows" and gpu_mode:
        add(9, "sysmem_fallback", "info", "NVIDIA Control Panel -> Manage 3D settings -> 'CUDA - Sysmem "
            "Fallback Policy' = 'Prefer No Sysmem Fallback' for python.exe is recommended; the harness "
            "also caps memory at the VRAM free at start")
    else:
        add(9, "sysmem_fallback", "skip", "not Windows" if fam != "windows" else "device=cpu")

    # 10 disk + 11 models / network
    rdir = Path(results_dir) if results_dir else None
    missing: list[str] = []
    model_status: dict = {}
    if not tiny:
        keys = model_keys or ("w2v2b", "xlsr", "w2v2l", "wavlmb")
        try:
            from . import hub  # B

            for k in keys:
                try:
                    model_status[k] = "cached" if hub.is_cached(k) else "missing"
                except Exception as e:
                    model_status[k] = f"unknown ({type(e).__name__})"
        except Exception as e:
            model_status = {k: f"unknown (hub module: {type(e).__name__})" for k in keys}
        missing = [k for k, v in model_status.items() if v != "cached"]
    hub_dir = _hub_cache_dir()
    hub_free = _free_gb(hub_dir)
    if hub_free is None:
        add(10, "disk_models", "warn", "free space at the model cache unknown")
    elif missing and hub_free < 5.0:
        add(10, "disk_models", "fatal", f"{hub_free:.1f} GB free at the model cache, need >= 5 GB "
            f"to download {', '.join(missing)}")
    else:
        add(10, "disk_models", "ok", f"{hub_free:.1f} GB free at the model cache")
    if rdir is not None:
        rfree = _free_gb(rdir)
        if rfree is not None and rfree < 1.0:
            add(10, "disk_results", "warn", f"only {rfree:.2f} GB free at results/ (want >= 1 GB)")
        elif rfree is not None:
            add(10, "disk_results", "ok", f"{rfree:.1f} GB free at results/")
    if tiny:
        add(11, "models", "skip", "tiny mode uses random weights (no downloads)")
    else:
        add(11, "models", "ok" if not missing else "warn",
            ", ".join(f"{k}: {v}" for k, v in model_status.items())
            + ("" if not missing else " -> run: uv run --no-sync python -m agbench fetch"),
            models=model_status)
        offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes", "on")
        if not network or offline:
            add(11, "network", "skip", "skipped (--no-network or HF_HUB_OFFLINE=1)")
        else:
            reach, detail = _network_check()
            add(11, "network", "ok" if reach else "warn",
                f"Hugging Face endpoint {'reachable' if reach else 'NOT reachable'} ({detail})"
                + (" (custom HF_ENDPOINT)" if os.environ.get("HF_ENDPOINT") else ""))

    # 12 host
    h = host_info()
    blas = blas_info()
    blas_txt = ", ".join(f"{b.get('internal_api')} {b.get('version')} x{b.get('num_threads')}" for b in blas)
    ram_txt = f"{h['ram_gib']:.1f}" if h.get("ram_gib") else "?"
    add(12, "host", "info", f"{h.get('cpu_model')}, {h.get('cpu_physical')} cores / {h.get('cpu_logical')} "
        f"threads, {ram_txt} GiB RAM; BLAS: {blas_txt or 'unknown'}")

    # 13 GPU idle state
    if gpu_mode and st:
        add(13, "gpu_idle", "info", f"power limit {st.get('power_limit_w')} W, SM clock "
            f"{st.get('sm_clock_mhz')} MHz (max {st.get('sm_clock_max_mhz')}), mem clock "
            f"{st.get('mem_clock_mhz')} MHz, {st.get('temp_c')} C, {st.get('power_w')} W at idle",
            idle={k: st.get(k) for k in ("power_limit_w", "sm_clock_mhz", "sm_clock_max_mhz",
                                          "mem_clock_mhz", "temp_c", "power_w")})
    else:
        add(13, "gpu_idle", "skip", "no GPU telemetry" if gpu_mode else "device=cpu")

    n_fatal = sum(c["status"] == "fatal" for c in checks)
    n_warn = sum(c["status"] == "warn" for c in checks)
    status = "FATAL" if n_fatal else ("WARN" if n_warn else "OK")
    env = capture_env(device_index, device=device) if torch is not None else {
        "os": platform.platform(), "os_family": fam, "python": platform.python_version(),
        "agbench_version": __version__, "device": device, "gpu_name": None, "torch": None}
    env["doctor"] = {"status": status,
                     "warnings": [f"{c['name']}: {c['message']}" for c in checks if c["status"] == "warn"]}
    out = sanitize_obj({"status": status, "n_fatal": n_fatal, "n_warn": n_warn, "checks": checks,
                        "env": env, "agbench_version": __version__})
    out[CONSOLE_ONLY_KEY] = console_notes  # popped by cmd_doctor before anything is written
    return out


CONSOLE_ONLY_KEY = "_console_only"


def _driver_too_old(drv: str, need: tuple[int, int], torch_cuda: str | None) -> str:
    msg = (f"driver {drv} is older than {need[0]}.{need[1]:02d} needed by torch built for CUDA {torch_cuda}: "
           "update the NVIDIA driver (step 1 of AGENTS.md)")
    major = (parse_version_tuple(drv) or (0,))[0]
    if 570 <= major < 580 and parse_version_tuple(torch_cuda)[:1] >= (13,):
        msg += f', or run "{C.UV_SYNC_HINT_CU128}" (PyTorch for CUDA 12.8 works with driver 570-579)'
    return msg


def _process_names(procs: list[dict]) -> list[str]:
    names = []
    try:
        import psutil
    except Exception:
        return [f"pid {pp.get('pid')}" for pp in procs[:8]]
    for pp in procs[:8]:
        try:
            names.append(psutil.Process(pp["pid"]).name())
        except Exception:
            names.append(f"pid {pp.get('pid')}")
    return names


def format_doctor(result: dict) -> str:
    lines = [f"{'#':>2}  {'check':<16} {'status':<6} message"]
    for c in result["checks"]:
        lines.append(f"{c['n']:>2}  {c['name']:<16} {c['status'].upper():<6} {c['message']}")
    st = result["status"]
    n = result.get("n_fatal") if st == "FATAL" else result.get("n_warn")
    lines.append(f"DOCTOR: {st}" + (f" ({n})" if st != "OK" else ""))
    return "\n".join(lines)
