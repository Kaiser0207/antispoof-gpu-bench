"""Builder C unit tests: synthetic audio, corpus, RawBoost, dataset/loader, telemetry, GPU basics
and CPU pipeline workloads (tiny, CPU only, no network, no GPU)."""

from __future__ import annotations

import json
import os
import pickle
import time
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")

from agbench import telemetry as T  # noqa: E402
from agbench.context import MeasureCfg, RunContext  # noqa: E402
from agbench.data import rawboost as RB  # noqa: E402
from agbench.data.dataset import SynthFlacDataset, make_loader, worker_init_fn  # noqa: E402
from agbench.data.synth import corpus_dir, ensure_corpus, load_manifest, speechlike  # noqa: E402
from agbench.errors import SkipScenario  # noqa: E402
from agbench.workloads import basics, cpu_pipeline  # noqa: E402

SR = 16000


@pytest.fixture(autouse=True)
def _restore_torch_globals():
    threads = torch.get_num_threads()
    prec = torch.get_float32_matmul_precision()
    yield
    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision(prec)


@pytest.fixture(scope="module")
def tiny_manifest(tmp_path_factory) -> Path:
    return ensure_corpus(tmp_path_factory.mktemp("cache"), tiny=True)


def _ctx(sid: str, workload: str, params: dict, cache_dir: Path, *, tiny: bool = True,
         logs: list | None = None) -> RunContext:
    return RunContext(scenario_id=sid, workload=workload, params=dict(params), device="cpu",
                      tiny=tiny, measure=MeasureCfg.tiny() if tiny else MeasureCfg(), seed=1234,
                      cache_dir=Path(cache_dir), keep_series=False,
                      log=(logs.append if logs is not None else (lambda m: None)))


def _snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    clean = clean.astype(np.float64)
    return 20.0 * np.log10(np.linalg.norm(clean) / np.linalg.norm(noisy.astype(np.float64) - clean))


# ---------------------------------------------------------------------------------------------
# speechlike
# ---------------------------------------------------------------------------------------------


def test_speechlike_range_dtype_determinism():
    x = speechlike(3 * SR, np.random.default_rng(7))
    assert x.dtype == np.float32 and x.shape == (3 * SR,)
    assert np.all(np.isfinite(x))
    assert float(np.max(np.abs(x))) <= 1.0
    assert 0.25 <= float(np.max(np.abs(x))) <= 0.95  # peak normalised to 0.3-0.9 (+ noise floor)
    assert float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) > 1e-3  # not silence
    y = speechlike(3 * SR, np.random.default_rng(7))
    assert np.array_equal(x, y)
    z = speechlike(3 * SR, np.random.default_rng(8))
    assert not np.array_equal(x, z)


@pytest.mark.parametrize("n", [0, 1, 100, 799, 8000])
def test_speechlike_short_lengths(n):
    x = speechlike(n, np.random.default_rng(0))
    assert x.shape == (n,) and x.dtype == np.float32
    assert np.all(np.isfinite(x)) and (n == 0 or float(np.max(np.abs(x))) <= 1.0)


# ---------------------------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------------------------


def test_ensure_corpus_tiny_contents(tmp_path):
    m = ensure_corpus(tmp_path, tiny=True)
    assert m == tmp_path / "synth_tiny_v1" / "manifest.json"
    data = load_manifest(m)
    assert data["version"] == 1 and data["sr"] == SR
    files = data["files"]
    assert len(files) == 8
    assert sorted(f["label"] for f in files) == [0] * 4 + [1] * 4  # balanced
    assert data["total_bytes"] == sum(f["bytes"] for f in files)
    for f in files:
        assert not os.path.isabs(f["path"]) and "/" not in f["path"] and "\\" not in f["path"]
        assert 1.0 <= f["seconds"] <= 2.0
        info = sf.info(str(m.parent / f["path"]))
        assert info.samplerate == SR and info.channels == 1
        assert info.format == "FLAC" and info.subtype == "PCM_16"
        assert info.frames == f["frames"] and abs(f["seconds"] - f["frames"] / SR) < 1e-4
        assert (m.parent / f["path"]).stat().st_size == f["bytes"]
    assert str(tmp_path) not in m.read_text(encoding="utf-8")  # relative paths only


def test_ensure_corpus_idempotent_and_repairs(tmp_path):
    m = ensure_corpus(tmp_path, tiny=True)
    first = m.read_text(encoding="utf-8")
    mtimes = {p.name: p.stat().st_mtime_ns for p in m.parent.glob("*.flac")}
    time.sleep(0.05)
    assert ensure_corpus(tmp_path, tiny=True) == m
    assert {p.name: p.stat().st_mtime_ns for p in m.parent.glob("*.flac")} == mtimes
    # A damaged file (size mismatch) triggers regeneration with identical content.
    victim = m.parent / load_manifest(m)["files"][3]["path"]
    victim.write_bytes(b"broken")
    ensure_corpus(tmp_path, tiny=True)
    assert m.read_text(encoding="utf-8") == first
    assert sf.info(str(victim)).frames > 0
    assert not list(m.parent.glob("*.tmp"))


def test_ensure_corpus_full_parameters_and_cache_env(tmp_path, monkeypatch):
    m = ensure_corpus(tmp_path, n_files=3, seed=5)  # full-mode lengths, few files
    data = load_manifest(m)
    assert m.parent.name == "synth_v1" and len(data["files"]) == 3
    assert all(4.0 <= f["seconds"] <= 10.0 for f in data["files"])
    # Different parameters -> regenerated, not silently reused.
    m2 = ensure_corpus(tmp_path, n_files=2, seed=5)
    assert len(load_manifest(m2)["files"]) == 2
    monkeypatch.setenv("AGBENCH_CACHE", str(tmp_path / "elsewhere"))
    assert corpus_dir(None, tiny=True) == tmp_path / "elsewhere" / "synth_tiny_v1"


# ---------------------------------------------------------------------------------------------
# RawBoost
# ---------------------------------------------------------------------------------------------


def test_rawboost_defaults_match_official_values():
    assert RB.DEFAULTS == {
        "nBands": 5, "minF": 20, "maxF": 8000, "minBW": 100, "maxBW": 1000,
        "minCoeff": 10, "maxCoeff": 100, "minG": 0, "maxG": 0,
        "minBiasLinNonLin": 5, "maxBiasLinNonLin": 20, "N_f": 5,
        "P": 10, "g_sd": 2, "SNRmin": 10, "SNRmax": 40,
    }


@pytest.fixture(scope="module")
def clip() -> np.ndarray:
    return speechlike(2 * SR, np.random.default_rng(11))


@pytest.mark.parametrize("algo", [1, 2, 3, 4, 5, 6, 7, 8])
def test_rawboost_algos_shape_finite_float32_deterministic(clip, algo):
    before = clip.copy()
    y1 = RB.rawboost(clip, np.random.default_rng(3), algo=algo)
    y2 = RB.rawboost(clip, np.random.default_rng(3), algo=algo)
    y3 = RB.rawboost(clip, np.random.default_rng(4), algo=algo)
    assert y1.shape == clip.shape and y1.dtype == np.float32
    assert np.all(np.isfinite(y1))
    assert np.array_equal(y1, y2)
    assert not np.array_equal(y1, y3)
    assert not np.array_equal(y1, clip)
    assert np.array_equal(clip, before)  # input untouched


def test_rawboost_algo0_and_errors(clip):
    y = RB.rawboost(clip, np.random.default_rng(0), algo=0)
    assert y.dtype == np.float32 and np.array_equal(y, clip)
    with pytest.raises(ValueError):
        RB.rawboost(clip, np.random.default_rng(0), algo=9)
    with pytest.raises(TypeError):
        RB.rawboost(clip, np.random.default_rng(0), algo=4, snr_min=10)


def test_ssi_snr_within_range(clip):
    x = clip.astype(np.float64)
    snrs = [_snr_db(x, RB.ssi(x, np.random.default_rng(k))) for k in range(12)]
    assert all(10.0 - 1e-6 <= s <= 40.0 + 1e-6 for s in snrs), snrs
    fixed = _snr_db(x, RB.ssi(x, np.random.default_rng(1), SNRmin=20, SNRmax=20))
    assert abs(fixed - 20.0) < 1e-6


def test_isd_touches_at_most_p_percent(clip):
    x = clip.astype(np.float64) * 0.2  # peak <= 0.18: |x + 2*x*f| <= 0.54, no renormalisation
    for k in range(5):
        y = RB.isd(x, np.random.default_rng(k), P=10)
        assert float(np.max(np.abs(y))) <= 1.0
        assert 0 < np.count_nonzero(y != x) <= 0.10 * x.size


def test_lnl_and_notch_fir(clip):
    y = RB.lnl(clip, np.random.default_rng(2))
    assert y.dtype == np.float64 and float(np.max(np.abs(y))) <= 1.0 + 1e-12
    assert abs(float(np.mean(y))) < 1e-9  # mean removed (then scaled at most)
    from scipy import signal

    keys = ("nBands", "minF", "maxF", "minBW", "maxBW", "minCoeff", "maxCoeff", "minG", "maxG")
    args = {k: RB.DEFAULTS[k] for k in keys}
    b = RB.notch_fir(np.random.default_rng(5), **args, sr=SR)
    assert b.ndim == 1 and b.size % 2 == 1  # odd: convolution of odd-length band filters
    _, h = signal.freqz(b, 1, fs=SR)
    assert abs(float(np.max(np.abs(h))) - 1.0) < 1e-9  # gain 0 dB
    args.update(minG=-5, maxG=-20)  # the LnL bias shift (min > max) must still work
    b2 = RB.notch_fir(np.random.default_rng(5), **args, sr=SR)
    peak_db = 20 * np.log10(np.max(np.abs(signal.freqz(b2, 1, fs=SR)[1])))
    assert -20.0 - 1e-6 <= peak_db <= -5.0 + 1e-6


# ---------------------------------------------------------------------------------------------
# Dataset and loader
# ---------------------------------------------------------------------------------------------


def test_dataset_crop_and_loop_lengths(tiny_manifest):
    data = load_manifest(tiny_manifest)
    ds_long = SynthFlacDataset(tiny_manifest, seconds=0.5, train=False)
    ds_short = SynthFlacDataset(tiny_manifest, seconds=5.0, train=False)  # files are 1-2 s
    assert len(ds_long) == len(ds_short) == 8
    for i, f in enumerate(data["files"]):
        wave, _ = sf.read(str(tiny_manifest.parent / f["path"]), dtype="float32")
        x, y = ds_long[i]
        assert x.dtype == torch.float32 and tuple(x.shape) == (8000,)
        assert y.dtype == torch.int64 and int(y) == f["label"]
        off = (wave.size - 8000) // 2  # centre crop
        assert np.array_equal(x.numpy(), wave[off:off + 8000])
        xs, _ = ds_short[i]
        assert tuple(xs.shape) == (80000,)
        tiled = np.tile(wave, -(-80000 // wave.size))
        off = (tiled.size - 80000) // 2
        assert np.array_equal(xs.numpy(), tiled[off:off + 80000])


def test_dataset_train_crop_deterministic_and_rawboost(tiny_manifest):
    ds = SynthFlacDataset(tiny_manifest, seconds=0.5, train=True, seed=3)
    a, b = ds[2][0], ds[2][0]
    assert torch.equal(a, b)
    offsets = set()
    for epoch in range(6):
        ds.set_epoch(epoch)
        offsets.add(ds[2][0][:8].numpy().tobytes())
    assert len(offsets) > 1  # random offsets differ between epochs
    ds.set_epoch(0)
    aug = SynthFlacDataset(tiny_manifest, seconds=0.5, train=True, seed=3, rawboost_p=1.0)
    xa = aug[2][0]
    assert xa.dtype == torch.float32 and xa.shape == a.shape
    assert not torch.equal(xa, a)
    assert torch.equal(xa, aug[2][0])
    pickle.loads(pickle.dumps(aug))  # spawn workers receive a pickled copy


def test_dataloader_two_spawn_workers(tiny_manifest):
    ds = SynthFlacDataset(tiny_manifest, seconds=1.0, train=True, rawboost_p=0.5)
    loader = make_loader(ds, batch_size=2, num_workers=2, pin_memory=False,
                         persistent_workers=False, num_samples=10, seed=1)
    assert loader.multiprocessing_context.get_start_method() == "spawn"
    batches = list(loader)
    assert len(batches) == 5
    for x, y in batches:
        assert tuple(x.shape) == (2, SR) and x.dtype == torch.float32
        assert tuple(y.shape) == (2,) and y.dtype == torch.int64
        assert torch.isfinite(x).all()
    plain = make_loader(ds, batch_size=3, num_workers=0, pin_memory=False,
                        persistent_workers=True, shuffle=False)
    assert len(list(plain)) == 2  # 8 items, drop_last
    worker_init_fn(0)  # callable outside a worker too


# ---------------------------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------------------------


def test_gpu_sampler_without_nvml_or_smi(monkeypatch):
    monkeypatch.setattr(T, "_import_pynvml", lambda: None)
    monkeypatch.setattr(T, "_nvidia_smi_path", lambda: None)
    with T.GpuSampler(device_index=0, interval_s=0.05, keep_series=True) as s:
        time.sleep(0.05)
    assert s.summary() == {"source": "none"}
    assert s.series() == []
    assert T.nvml_static_info(0) == {"source": "none"}
    assert T.pcie_link(0)["source"] == "none"


class _FakeNvml(types.SimpleNamespace):
    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_SM = 1
    NVML_CLOCK_MEM = 2

    def __init__(self):
        super().__init__(calls=[], n=0, inits=0, shutdowns=0)

    def nvmlInit(self):
        self.inits += 1

    def nvmlShutdown(self):
        self.shutdowns += 1

    def nvmlDeviceGetHandleByUUID(self, uuid):
        self.calls.append(("uuid", uuid))
        return "h"

    def nvmlDeviceGetHandleByIndex(self, i):
        self.calls.append(("index", i))
        return "h"

    def nvmlDeviceGetTemperature(self, h, kind):
        self.n += 1
        return 60 + self.n % 3

    def nvmlDeviceGetClockInfo(self, h, kind):
        return 2500 if kind == self.NVML_CLOCK_SM else 14000

    def nvmlDeviceGetPowerUsage(self, h):
        return 150_000

    def nvmlDeviceGetUtilizationRates(self, h):
        return types.SimpleNamespace(gpu=97, memory=40)

    def nvmlDeviceGetMemoryInfo(self, h):
        return types.SimpleNamespace(used=4 * 1024 ** 3, total=16 * 1024 ** 3, free=12 * 1024 ** 3)

    def nvmlDeviceGetCurrentClocksEventReasons(self, h):
        return 0x4 | 0x20

    def nvmlDeviceGetEnforcedPowerLimit(self, h):
        return 180_000


def test_gpu_sampler_with_fake_nvml(monkeypatch):
    fake = _FakeNvml()
    monkeypatch.setattr(T, "_import_pynvml", lambda: fake)
    with T.GpuSampler(device_index=1, interval_s=0.01, keep_series=True,
                      uuid="1234abcd-0000-0000-0000-000000000000") as s:
        time.sleep(0.15)
    summ = s.summary()
    assert summ["source"] == "nvml" and summ["n"] >= 2 and summ["interval_s"] == 0.01
    assert summ["sm_clock_mhz"] == {"min": 2500.0, "median": 2500.0, "max": 2500.0}
    assert summ["power_w"]["median"] == 150.0 and summ["power_limit_w"] == 180.0
    assert summ["util_pct"]["max"] == 97.0 and summ["mem_used_gib"]["max"] == 4.0
    assert 60.0 <= summ["temp_c"]["min"] <= summ["temp_c"]["max"] <= 62.0
    assert summ["throttle_reasons"] == ["sw_power_cap", "sw_thermal_slowdown"]
    assert fake.calls[0] == ("uuid", "GPU-1234abcd-0000-0000-0000-000000000000")
    assert fake.inits == fake.shutdowns == 1
    ser = s.series(max_points=3)
    assert 1 <= len(ser) <= 3
    assert set(ser[0]) == {"t_s", "temp_c", "sm_clock_mhz", "mem_clock_mhz", "power_w", "util_pct",
                           "mem_used_gib", "throttle"}
    s2 = T.GpuSampler(keep_series=False)
    assert s2.series() == []


def test_telemetry_parsers():
    assert T.decode_throttle(0) == []
    assert T.decode_throttle(0x104) == ["sw_power_cap", "display_clock_setting"]
    assert T.decode_throttle(0x1000) == ["0x1000"]
    s = T.parse_smi_sample(["71", "2602", "13801", "165.20", "99", "4096", "0x0000000000000004"],
                           with_reasons=True)
    assert s == {"temp_c": 71.0, "sm_clock_mhz": 2602.0, "mem_clock_mhz": 13801.0,
                 "power_w": 165.2, "util_pct": 99.0, "mem_used_gib": 4.0,
                 "throttle": ["sw_power_cap"]}
    s = T.parse_smi_sample(["[N/A]", "1000", "N/A", "[Not Supported]", "0", "512"], False)
    assert s["temp_c"] is None and s["power_w"] is None and s["mem_used_gib"] == 0.5
    assert T.parse_smi_sample(["1", "2"], False) is None


# ---------------------------------------------------------------------------------------------
# Workloads (tiny, CPU)
# ---------------------------------------------------------------------------------------------


def test_basics_tiny_cpu(tmp_path):
    params = {"sizes": [4096, 8192], "dtypes": ["bf16", "fp16", "tf32", "fp32"]}
    out = basics.run_matmul(_ctx("gpu.matmul", "basics.matmul", params, tmp_path))
    m = out["metrics"]
    assert set(m["tflops"]) == {"bf16", "fp16", "tf32", "fp32"}
    for dt, by_size in m["tflops"].items():
        assert list(by_size) == ["256"] and by_size["256"] > 0
        assert m["tflops_best"][dt] == by_size["256"]
    assert torch.get_float32_matmul_precision() == "highest"
    out = basics.run_membw(_ctx("gpu.membw", "basics.membw", {"buffer_mib": 1024}, tmp_path))
    assert set(out["metrics"]["gbps"]) == {"copy", "read", "triad"}
    assert all(v > 0 for v in out["metrics"]["gbps"].values())
    assert out["extra"]["buffer_mib"] == 64
    with pytest.raises(SkipScenario) as e:
        basics.run_pcie(_ctx("gpu.pcie", "basics.pcie", {"buffer_mib": 256}, tmp_path))
    assert e.value.reason == "cpu_device"


def test_time_op_cpu_counts_reps():
    calls = []
    r = basics.time_op(lambda: calls.append(1), device="cpu", n_batches=3, batch_min_s=0.001,
                       warmup=2, calib_s=0.001)
    assert r["n_batches"] == 3 and r["reps"] >= 1 and r["op_ms_median"] > 0
    assert len(calls) >= 2 + 3 * r["reps"]


_CPU_KEYS = ("clips_per_s", "ms_per_clip_median", "cpu_ms_per_clip", "cpu_wall_ratio",
             "num_workers", "worker_startup_s")


def test_cpu_decode_and_rawboost_tiny(tmp_path):
    logs: list[str] = []
    out = cpu_pipeline.run_decode(_ctx("cpu.decode", "cpu.decode", {"seconds": 6.0}, tmp_path,
                                       logs=logs))
    m = out["metrics"]
    assert all(m[k] is not None for k in _CPU_KEYS) and m["clips_per_s"] > 0
    assert m["decode_ms_per_audio_s"] > 0 and out["extra"]["seconds"] == 2.0  # tiny cap
    out = cpu_pipeline.run_rawboost(_ctx("cpu.rawboost.blas1", "cpu.rawboost",
                                         {"algo": 4, "p": 1.0, "seconds": 6.0, "blas": "1"},
                                         tmp_path, logs=logs))
    m = out["metrics"]
    assert all(m[k] is not None for k in _CPU_KEYS) and m["clips_per_s"] > 0
    assert set(m["stage_ms"]) == {"lnl", "isd", "ssi"}
    assert all(v > 0 for v in m["stage_ms"].values())
    ex = out["extra"]
    assert ex["blas_mode"] == "1" and "blas_spin_probe" in ex
    assert all("filepath" not in d for d in ex["threadpool"])
    blob = json.dumps(out)
    assert str(tmp_path) not in blob and str(Path.home()) not in blob
    assert all(line.isascii() for line in logs)


def test_cpu_loader_tiny_spawn(tmp_path):
    out = cpu_pipeline.run_loader(_ctx("cpu.loader.w4.blas1", "cpu.loader",
                                       {"num_workers": 4, "batch": 16, "seconds": 6.0,
                                        "rawboost_p": 0.5, "blas": "1"}, tmp_path))
    m = out["metrics"]
    assert all(m[k] is not None for k in _CPU_KEYS)
    assert m["num_workers"] == 2 and m["clips_per_s"] > 0 and m["worker_startup_s"] > 0
    assert out["extra"]["batch"] == 2 and out["extra"]["measured_batches"] == 3


def test_cpu_loader_skips_when_workers_exceed_cores(tmp_path, monkeypatch):
    monkeypatch.setattr(cpu_pipeline, "logical_cores", lambda: 2)
    ctx = _ctx("cpu.loader.w4.blas1", "cpu.loader", {"num_workers": 4}, tmp_path, tiny=False)
    with pytest.raises(SkipScenario) as e:
        cpu_pipeline.run_loader(ctx)
    assert e.value.reason == "workers_exceed_cores"
    assert not (tmp_path / "synth_v1").exists()  # skipped before any work


def test_blas_mode_and_env_helpers(tmp_path, monkeypatch):
    mk = lambda sid, params: _ctx(sid, "cpu.loader", params, tmp_path)  # noqa: E731
    assert cpu_pipeline.blas_mode(mk("x", {"blas": "1"})) == "1"
    assert cpu_pipeline.blas_mode(mk("x", {"blas": "default"})) == "default"
    assert cpu_pipeline.blas_mode(mk("cpu.loader.w4.blasdefault", {})) == "default"
    assert cpu_pipeline.blas_mode(mk("cpu.rawboost.blas1", {})) == "1"
    assert cpu_pipeline.blas_mode(mk("cpu.decode", {})) is None
    assert cpu_pipeline.blas_env_matches("1", {"OPENBLAS_NUM_THREADS": "1"})
    assert not cpu_pipeline.blas_env_matches("default", {"OMP_NUM_THREADS": "2"})
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    with cpu_pipeline._blas_worker_env("default") as fix:
        assert "OMP_NUM_THREADS" not in os.environ and fix
    assert os.environ["OMP_NUM_THREADS"] == "2"
    with cpu_pipeline._blas_worker_env("1"):
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert "OPENBLAS_NUM_THREADS" not in os.environ
    probe = cpu_pipeline.blas_spin_probe(n_samples=20000, calls=1, settle_s=0.01)
    assert probe["calls"] == 1 and probe["cpu_ms_per_call_median"] >= 0
