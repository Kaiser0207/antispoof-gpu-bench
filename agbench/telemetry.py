"""GPU telemetry: temperature, clocks, power, utilisation, memory, throttle reasons (SPEC 9.2).

``GpuSampler`` samples in a background thread through NVML (module ``pynvml`` from the
``nvidia-ml-py`` package) and falls back to a long-running ``nvidia-smi --query-gpu ... -lms``
process. Without either it reports ``{"source": "none"}``. Nothing here ever raises into the
caller, and nothing here creates a CUDA context. GPU UUIDs are used only to find the device and
are never returned.
"""

from __future__ import annotations

import os
import shutil
import statistics
import subprocess
import sys
import threading
import time

# NVML "clocks event reasons" (formerly "throttle reasons") bit -> name.
THROTTLE_BITS: tuple[tuple[int, str], ...] = (
    (0x1, "gpu_idle"),
    (0x2, "applications_clocks_setting"),
    (0x4, "sw_power_cap"),
    (0x8, "hw_slowdown"),
    (0x10, "sync_boost"),
    (0x20, "sw_thermal_slowdown"),
    (0x40, "hw_thermal_slowdown"),
    (0x80, "hw_power_brake_slowdown"),
    (0x100, "display_clock_setting"),
)
SAMPLE_KEYS = ("temp_c", "sm_clock_mhz", "mem_clock_mhz", "power_w", "util_pct", "mem_used_gib")
_GIB = 1024 ** 3
_MAX_SAMPLES = 20_000
_SMI_BASE = ("temperature.gpu", "clocks.sm", "clocks.mem", "power.draw", "utilization.gpu",
             "memory.used")
_SMI_REASONS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")


def decode_throttle(mask) -> list[str]:
    """Names of the set bits of an NVML clocks-event-reason mask (unknown bits as hex)."""
    try:
        m = int(mask)
    except (TypeError, ValueError):
        return []
    names = [name for bit, name in THROTTLE_BITS if m & bit]
    rest = m & ~sum(bit for bit, _ in THROTTLE_BITS)
    if rest:
        names.append(hex(rest))
    return names


# Indirections so tests can simulate machines without NVML / nvidia-smi.
def _import_pynvml():
    try:
        import pynvml  # provided by nvidia-ml-py

        return pynvml
    except Exception:
        return None


def _nvidia_smi_path() -> str | None:
    return shutil.which("nvidia-smi")


def _norm_uuid(uuid) -> str | None:
    if uuid is None:
        return None
    s = str(uuid).strip()
    if not s:
        return None
    return s if s.upper().startswith(("GPU-", "MIG-")) else "GPU-" + s


def _num(text: str) -> float | None:
    t = text.strip().strip("[]")
    if not t or t.upper().startswith(("N/A", "NOT SUPPORTED", "UNKNOWN", "ERR")):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _popen_kwargs() -> dict:
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


# ---------------------------------------------------------------------------------------------
# NVML backend
# ---------------------------------------------------------------------------------------------


class _Nvml:
    def __init__(self, nv, device_index: int = 0, uuid: str | None = None):
        nv.nvmlInit()
        self.nv = nv
        self.h = None
        self._closed = False
        try:
            u = _norm_uuid(uuid)
            if u:
                try:
                    self.h = nv.nvmlDeviceGetHandleByUUID(u)
                except Exception:
                    self.h = None
            if self.h is None:
                self.h = nv.nvmlDeviceGetHandleByIndex(int(device_index))
        except Exception:
            self.close()
            raise

    def _try(self, fn_name: str, *args):
        fn = getattr(self.nv, fn_name, None)
        if fn is None:
            return None
        try:
            return fn(self.h, *args)
        except Exception:
            return None

    def throttle_mask(self):
        m = self._try("nvmlDeviceGetCurrentClocksEventReasons")
        return m if m is not None else self._try("nvmlDeviceGetCurrentClocksThrottleReasons")

    def sample(self) -> dict:
        nv = self.nv
        temp = self._try("nvmlDeviceGetTemperature", getattr(nv, "NVML_TEMPERATURE_GPU", 0))
        sm = self._try("nvmlDeviceGetClockInfo", getattr(nv, "NVML_CLOCK_SM", 1))
        mem = self._try("nvmlDeviceGetClockInfo", getattr(nv, "NVML_CLOCK_MEM", 2))
        power = self._try("nvmlDeviceGetPowerUsage")
        util = self._try("nvmlDeviceGetUtilizationRates")
        meminfo = self._try("nvmlDeviceGetMemoryInfo")
        mask = self.throttle_mask()
        return {
            "temp_c": float(temp) if temp is not None else None,
            "sm_clock_mhz": float(sm) if sm is not None else None,
            "mem_clock_mhz": float(mem) if mem is not None else None,
            "power_w": float(power) / 1000.0 if power is not None else None,
            "util_pct": float(getattr(util, "gpu", None)) if util is not None else None,
            "mem_used_gib": (float(getattr(meminfo, "used", 0)) / _GIB
                             if meminfo is not None else None),
            "throttle": decode_throttle(mask) if mask is not None else [],
        }

    def power_limit_w(self) -> float | None:
        v = self._try("nvmlDeviceGetEnforcedPowerLimit")
        if v is None:
            v = self._try("nvmlDeviceGetPowerManagementLimit")
        return float(v) / 1000.0 if v is not None else None

    def pcie(self) -> dict:
        def _int(v):
            return int(v) if v is not None else None
        return {"gen_cur": _int(self._try("nvmlDeviceGetCurrPcieLinkGeneration")),
                "gen_max": _int(self._try("nvmlDeviceGetMaxPcieLinkGeneration")),
                "width_cur": _int(self._try("nvmlDeviceGetCurrPcieLinkWidth")),
                "width_max": _int(self._try("nvmlDeviceGetMaxPcieLinkWidth"))}

    def static(self) -> dict:
        nv = self.nv

        def _s(v):
            return v.decode() if isinstance(v, bytes) else (str(v) if v is not None else None)

        def _call(name, *args):
            fn = getattr(nv, name, None)
            try:
                return fn(*args) if fn is not None else None
            except Exception:
                return None

        cuda_ver = _call("nvmlSystemGetCudaDriverVersion")
        meminfo = self._try("nvmlDeviceGetMemoryInfo")
        default_limit = self._try("nvmlDeviceGetPowerManagementDefaultLimit")
        gpu_gen_max = self._try("nvmlDeviceGetGpuMaxPcieLinkGeneration")
        sm_max = self._try("nvmlDeviceGetMaxClockInfo", getattr(nv, "NVML_CLOCK_SM", 1))
        mem_max = self._try("nvmlDeviceGetMaxClockInfo", getattr(nv, "NVML_CLOCK_MEM", 2))
        s = self.sample()
        return {
            "source": "nvml",
            "name": _s(self._try("nvmlDeviceGetName")),
            "driver": _s(_call("nvmlSystemGetDriverVersion")),
            "cuda_driver": (f"{int(cuda_ver) // 1000}.{(int(cuda_ver) % 1000) // 10}"
                            if cuda_ver else None),
            "memory_total_gib": (float(meminfo.total) / _GIB if meminfo is not None else None),
            "power_limit_w": self.power_limit_w(),
            "power_default_limit_w": (float(default_limit) / 1000.0
                                      if default_limit is not None else None),
            "pcie": self.pcie(),
            "pcie_gpu_gen_max": int(gpu_gen_max) if gpu_gen_max is not None else None,
            "sm_clock_max_mhz": float(sm_max) if sm_max is not None else None,
            "mem_clock_max_mhz": float(mem_max) if mem_max is not None else None,
            "temp_c": s["temp_c"], "sm_clock_mhz": s["sm_clock_mhz"],
            "mem_clock_mhz": s["mem_clock_mhz"], "power_w": s["power_w"],
            "throttle_reasons": s["throttle"],
        }

    def close(self) -> None:
        """Balance our nvmlInit exactly once (NVML init/shutdown are reference counted)."""
        if self._closed:
            return
        self._closed = True
        try:
            self.nv.nvmlShutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------------------------
# nvidia-smi backend
# ---------------------------------------------------------------------------------------------


def _smi_target(device_index: int, uuid) -> str:
    return _norm_uuid(uuid) or str(int(device_index))


def _smi_query(smi: str, fields: list[str] | tuple[str, ...], target: str,
               timeout: float = 15.0) -> list[str] | None:
    """One ``--query-gpu`` call; returns the values of the first line, or None on failure."""
    try:
        cp = subprocess.run([smi, f"--query-gpu={','.join(fields)}",
                             "--format=csv,noheader,nounits", "-i", target],
                            capture_output=True, text=True, timeout=timeout, **_popen_kwargs())
    except Exception:
        return None
    if cp.returncode != 0 or not cp.stdout.strip():
        return None
    vals = [v.strip() for v in cp.stdout.strip().splitlines()[0].split(",")]
    return vals if len(vals) == len(fields) else None


def parse_smi_sample(values: list[str], with_reasons: bool) -> dict | None:
    """Turn one CSV line of ``_SMI_BASE`` (+ reasons) values into a sample dict."""
    if len(values) != len(_SMI_BASE) + (1 if with_reasons else 0):
        return None
    temp, sm, mem, power, util, used = (_num(v) for v in values[:6])
    throttle: list[str] = []
    if with_reasons:
        raw = values[6].strip()
        try:
            throttle = decode_throttle(int(raw, 16) if raw.lower().startswith("0x") else int(raw))
        except ValueError:
            throttle = []
    return {"temp_c": temp, "sm_clock_mhz": sm, "mem_clock_mhz": mem, "power_w": power,
            "util_pct": util, "mem_used_gib": used / 1024.0 if used is not None else None,
            "throttle": throttle}


class _Smi:
    def __init__(self, smi: str, device_index: int = 0, uuid=None):
        self.smi = smi
        self.target = _smi_target(device_index, uuid)
        self.fields: list[str] | None = None
        self.with_reasons = False
        for reasons in _SMI_REASONS:
            if _smi_query(smi, [*_SMI_BASE, reasons], self.target) is not None:
                self.fields, self.with_reasons = [*_SMI_BASE, reasons], True
                break
        if self.fields is None:
            if _smi_query(smi, list(_SMI_BASE), self.target) is None:
                raise RuntimeError("nvidia-smi query failed")
            self.fields = list(_SMI_BASE)
        self.proc: subprocess.Popen | None = None

    def power_limit_w(self) -> float | None:
        v = _smi_query(self.smi, ["power.limit"], self.target)
        return _num(v[0]) if v else None

    def run(self, interval_s: float, emit, stop: threading.Event) -> None:
        cmd = [self.smi, f"--query-gpu={','.join(self.fields)}", "--format=csv,noheader,nounits",
               "-i", self.target, "-lms", str(max(100, int(round(interval_s * 1000))))]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         text=True, bufsize=1, **_popen_kwargs())
        except Exception:
            return
        try:
            for line in self.proc.stdout:  # type: ignore[union-attr]
                if stop.is_set():
                    break
                s = parse_smi_sample([v.strip() for v in line.strip().split(",")],
                                     self.with_reasons)
                if s is not None:
                    emit(s)
        except Exception:
            pass
        finally:
            self.terminate()

    def terminate(self) -> None:
        p = self.proc
        if p is None:
            return
        try:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except Exception:
                    p.kill()
        except Exception:
            pass

    def static(self) -> dict:
        fields = ["name", "driver_version", "memory.total", "power.limit", "power.default_limit",
                  "pcie.link.gen.current", "pcie.link.gen.max", "pcie.link.width.current",
                  "pcie.link.width.max", "clocks.max.sm", "clocks.max.mem", "temperature.gpu",
                  "clocks.sm", "clocks.mem", "power.draw"]
        v = _smi_query(self.smi, fields, self.target) or [""] * len(fields)

        def _i(x):
            n = _num(x)
            return int(n) if n is not None else None
        total = _num(v[2])
        return {
            "source": "nvidia-smi", "name": v[0] or None, "driver": v[1] or None,
            "cuda_driver": None,
            "memory_total_gib": total / 1024.0 if total is not None else None,
            "power_limit_w": _num(v[3]), "power_default_limit_w": _num(v[4]),
            "pcie": {"gen_cur": _i(v[5]), "gen_max": _i(v[6]), "width_cur": _i(v[7]),
                     "width_max": _i(v[8])},
            "pcie_gpu_gen_max": None,
            "sm_clock_max_mhz": _num(v[9]), "mem_clock_max_mhz": _num(v[10]),
            "temp_c": _num(v[11]), "sm_clock_mhz": _num(v[12]), "mem_clock_mhz": _num(v[13]),
            "power_w": _num(v[14]), "throttle_reasons": [],
        }


# ---------------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------------


def _stats(values: list[float]) -> dict | None:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    return {"min": round(min(xs), 3), "median": round(statistics.median(xs), 3),
            "max": round(max(xs), 3)}


class GpuSampler:
    """Background sampler of one GPU. Use as a context manager; never raises.

    ``uuid`` (an NVML ``GPU-...`` id or the bare UUID that torch reports) wins over
    ``device_index`` (the NVML / nvidia-smi index, i.e. PCI bus order).
    """

    def __init__(self, device_index: int = 0, interval_s: float = 0.5, keep_series: bool = False,
                 uuid: str | None = None):
        self.device_index = int(device_index)
        self.interval_s = float(interval_s)
        self.keep_series = bool(keep_series)
        self.uuid = uuid
        self.source = "none"
        self.power_limit_w: float | None = None
        self._samples: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._backend = None
        self._t0 = time.perf_counter()

    # -- lifecycle --------------------------------------------------------------------------
    def __enter__(self) -> "GpuSampler":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> "GpuSampler":
        try:
            self._t0 = time.perf_counter()
            nv = _import_pynvml()
            if nv is not None:
                try:
                    self._backend = _Nvml(nv, self.device_index, self.uuid)
                    self.source = "nvml"
                except Exception:
                    self._backend = None
            if self._backend is None:
                smi = _nvidia_smi_path()
                if smi:
                    try:
                        self._backend = _Smi(smi, self.device_index, self.uuid)
                        self.source = "nvidia-smi"
                    except Exception:
                        self._backend = None
            if self._backend is None:
                self.source = "none"
                return self
            try:
                self.power_limit_w = self._backend.power_limit_w()
            except Exception:
                self.power_limit_w = None
            if isinstance(self._backend, _Nvml):
                self._emit(self._backend.sample())  # one sample right away
                target = self._run_nvml
            else:
                target = self._run_smi
            self._thread = threading.Thread(target=target, name="agbench-gpu-sampler", daemon=True)
            self._thread.start()
        except Exception:
            self.source = "none"
            self._backend = None
        return self

    def stop(self) -> None:
        try:
            self._stop.set()
            if isinstance(self._backend, _Smi):
                self._backend.terminate()
            if self._thread is not None:
                self._thread.join(timeout=5.0)
            if isinstance(self._backend, _Nvml):
                self._backend.close()
        except Exception:
            pass

    # -- sampling -----------------------------------------------------------------------------
    def _emit(self, s: dict) -> None:
        s = dict(s)
        s["t_s"] = round(time.perf_counter() - self._t0, 3)
        with self._lock:
            self._samples.append(s)
            if len(self._samples) > _MAX_SAMPLES:
                self._samples = self._samples[::2]

    def _run_nvml(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self._emit(self._backend.sample())
            except Exception:
                pass

    def _run_smi(self) -> None:
        try:
            self._backend.run(self.interval_s, self._emit, self._stop)
        except Exception:
            pass

    # -- results ------------------------------------------------------------------------------
    def summary(self) -> dict:
        """Telemetry object of SPEC 9.2; ``{"source": "none"}`` without NVML / nvidia-smi."""
        try:
            if self.source == "none":
                return {"source": "none"}
            with self._lock:
                samples = list(self._samples)
            out: dict = {"source": self.source, "interval_s": self.interval_s, "n": len(samples)}
            for key in SAMPLE_KEYS:
                out[key] = _stats([s.get(key) for s in samples])
            out["power_limit_w"] = self.power_limit_w
            seen: list[str] = []
            for s in samples:
                for r in s.get("throttle") or []:
                    if r not in seen:
                        seen.append(r)
            out["throttle_reasons"] = seen
            return out
        except Exception:
            return {"source": "none"}

    def series(self, max_points: int | None = 600) -> list[dict]:
        """Samples ``[{"t_s", "temp_c", ..., "throttle": [...]}]`` when ``keep_series``, else []."""
        try:
            if not self.keep_series:
                return []
            with self._lock:
                samples = [dict(s) for s in self._samples]
            if max_points and len(samples) > max_points:
                step = -(-len(samples) // int(max_points))
                samples = samples[::step]
            return [{"t_s": s.get("t_s"), **{k: s.get(k) for k in SAMPLE_KEYS},
                     "throttle": list(s.get("throttle") or [])} for s in samples]
        except Exception:
            return []

    @property
    def t0_perf(self) -> float:
        """``time.perf_counter()`` at start; sample ``t_s`` values are relative to it."""
        return self._t0


def _open_backend(device_index: int, uuid):
    nv = _import_pynvml()
    if nv is not None:
        try:
            return _Nvml(nv, device_index, uuid)
        except Exception:
            pass
    smi = _nvidia_smi_path()
    if smi:
        try:
            return _Smi(smi, device_index, uuid)
        except Exception:
            pass
    return None


def nvml_static_info(device_index: int = 0, uuid: str | None = None) -> dict:
    """Driver, name, power limits, PCIe link (cur/max), clocks and temperature now; never raises.

    Keys: source, name, driver, cuda_driver, memory_total_gib, power_limit_w,
    power_default_limit_w, pcie {gen_cur, gen_max, width_cur, width_max}, pcie_gpu_gen_max,
    sm_clock_max_mhz, mem_clock_max_mhz, temp_c, sm_clock_mhz, mem_clock_mhz, power_w,
    throttle_reasons. ``{"source": "none"}`` when neither NVML nor nvidia-smi works.
    """
    try:
        b = _open_backend(device_index, uuid)
        if b is None:
            return {"source": "none"}
        try:
            return b.static()
        finally:
            if isinstance(b, _Nvml):
                b.close()
    except Exception:
        return {"source": "none"}


def pcie_link(device_index: int = 0, uuid: str | None = None) -> dict:
    """Current/max PCIe generation and width now (``source`` says where it came from)."""
    empty = {"gen_cur": None, "gen_max": None, "width_cur": None, "width_max": None}
    try:
        nv = _import_pynvml()
        if nv is not None:
            try:
                b = _Nvml(nv, device_index, uuid)
                try:
                    return {**b.pcie(), "source": "nvml"}
                finally:
                    b.close()
            except Exception:
                pass
        smi = _nvidia_smi_path()
        if smi:
            v = _smi_query(smi, ["pcie.link.gen.current", "pcie.link.gen.max",
                                 "pcie.link.width.current", "pcie.link.width.max"],
                           _smi_target(device_index, uuid))
            if v:
                n = [_num(x) for x in v]
                return {"gen_cur": int(n[0]) if n[0] is not None else None,
                        "gen_max": int(n[1]) if n[1] is not None else None,
                        "width_cur": int(n[2]) if n[2] is not None else None,
                        "width_max": int(n[3]) if n[3] is not None else None,
                        "source": "nvidia-smi"}
    except Exception:
        pass
    return {**empty, "source": "none"}


def device_index_from_env(default: int = 0) -> int:
    """The physical index behind ``CUDA_VISIBLE_DEVICES`` when it is one integer (else default)."""
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return int(v) if v.isdigit() else int(default)
