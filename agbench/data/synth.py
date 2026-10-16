"""Synthetic speech-like audio and the on-disk FLAC corpus (SPEC 5.6).

The audio only has to make FLAC compression and decoding cost realistic; its content does not
change GPU compute. Everything is driven by an explicit ``numpy.random.Generator``, so the corpus
is identical on every machine for the same seed and library versions.

No import-time side effects (this module is imported by DataLoader spawn workers).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

SR = 16000
CORPUS_VERSION = 1
GENERATOR = "speechlike_v1"
FULL_DIR = "synth_v1"
TINY_DIR = "synth_tiny_v1"
NOISE_FLOOR_DBFS = -50.0


# ---------------------------------------------------------------------------------------------
# Speech-like waveform
# ---------------------------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _butter_sos(order: int, cutoff_hz: float, btype: str, sr: int):
    from scipy import signal

    return signal.butter(order, cutoff_hz, btype=btype, fs=sr, output="sos")


def _smooth(x: np.ndarray, sr: int, tau_s: float = 0.004) -> np.ndarray:
    """One-pole low-pass, used to soften segment on/off edges (avoids clicks)."""
    from scipy import signal

    a = float(np.exp(-1.0 / (tau_s * sr)))
    return signal.lfilter([1.0 - a], [1.0, -a], x)


def _resonate(x: np.ndarray, freq: float, bw: float, sr: int) -> np.ndarray:
    """Two-pole formant resonator (unity gain at DC), one ``lfilter`` biquad."""
    from scipy import signal

    r = np.exp(-np.pi * bw / sr)
    a1 = -2.0 * r * np.cos(2.0 * np.pi * freq / sr)
    a2 = r * r
    return signal.lfilter([1.0 + a1 + a2], [1.0, a1, a2], x)


def _sawtooth_polyblep(f0: np.ndarray, sr: int) -> np.ndarray:
    """Band-limited sawtooth with per-sample f0: harmonics at k*f0 with 1/k amplitudes."""
    dt = f0 / sr
    phase = np.cumsum(dt)
    t = phase - np.floor(phase)
    saw = 2.0 * t - 1.0
    lo = t < dt
    u = t[lo] / dt[lo]
    saw[lo] -= u + u - u * u - 1.0
    hi = t > 1.0 - dt
    u = (t[hi] - 1.0) / dt[hi]
    saw[hi] -= u * u + u + u + 1.0
    return saw


def speechlike(n_samples: int, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Return ``n_samples`` of speech-like audio as float32 in [-1, 1].

    Alternating voiced / unvoiced / pause segments of 50-400 ms (about 60-75 % / 10-20 % /
    10-25 %). Voiced: sawtooth (harmonic series, 1/k amplitudes) on a random-walk f0 in
    90-250 Hz, three formant resonators (F1 300-900, F2 900-2500, F3 2500-3500 Hz) and a 4 kHz
    low-pass. Unvoiced: high-passed noise. A 3-6 Hz syllabic envelope, a -50 dBFS noise floor and
    peak normalisation to a random 0.3-0.9.
    """
    n = int(n_samples)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    from scipy import signal

    # 1) Segment plan.
    min_len, max_len = int(0.05 * sr), int(0.40 * sr)
    p = np.array([rng.uniform(0.60, 0.75), rng.uniform(0.10, 0.20), rng.uniform(0.10, 0.25)])
    p /= p.sum()
    n_seg = n // min_len + 2
    lens = rng.integers(min_len, max_len + 1, size=n_seg)
    kinds = rng.choice(3, size=n_seg, p=p)  # 0 voiced, 1 unvoiced, 2 pause
    # Avoid two pauses in a row (keeps the segments alternating between sound and silence).
    for k in np.flatnonzero((kinds[1:] == 2) & (kinds[:-1] == 2)) + 1:
        kinds[k] = 0
    ends = np.cumsum(lens)
    k_used = int(np.searchsorted(ends, n)) + 1
    starts = ends[:k_used] - lens[:k_used]
    per_sample = np.repeat(kinds[:k_used], lens[:k_used])[:n]
    voiced_mask = _smooth((per_sample == 0).astype(np.float64), sr)
    unvoiced_mask = _smooth((per_sample == 1).astype(np.float64), sr)

    # 2) Voiced source: random-walk f0 (10 ms control rate, log domain) -> sawtooth.
    hop = sr // 100
    n_ctrl = n // hop + 2
    log_lo, log_hi = np.log(90.0), np.log(250.0)
    walk = np.log(rng.uniform(100.0, 220.0)) + np.cumsum(rng.normal(0.0, 0.015, n_ctrl))
    walk = np.clip(walk, log_lo, log_hi)
    f0 = np.exp(np.interp(np.arange(n) / hop, np.arange(n_ctrl), walk))
    source = _sawtooth_polyblep(f0, sr)

    # 3) Formants per voiced segment (three biquad resonators each).
    voiced = np.zeros(n, dtype=np.float64)
    for k in np.flatnonzero(kinds[:k_used] == 0):
        a, b = int(starts[k]), int(min(ends[k], n))
        if b - a < 8:
            continue
        seg = source[a:b]
        seg = _resonate(seg, rng.uniform(300, 900), rng.uniform(60, 120), sr)
        seg = _resonate(seg, rng.uniform(900, 2500), rng.uniform(80, 200), sr)
        seg = _resonate(seg, rng.uniform(2500, 3500), rng.uniform(150, 300), sr)
        voiced[a:b] = seg
    voiced = signal.sosfilt(_butter_sos(6, 4000.0, "lowpass", sr), voiced)
    voiced *= voiced_mask

    # 4) Unvoiced: high-passed noise, scaled relative to the voiced level.
    noise = signal.sosfilt(_butter_sos(4, 2500.0, "highpass", sr), rng.standard_normal(n))
    v_rms = float(np.sqrt(np.mean(voiced * voiced))) or 0.1
    n_rms = float(np.sqrt(np.mean(noise * noise))) or 1.0
    unvoiced = noise * (rng.uniform(0.15, 0.35) * v_rms / n_rms) * unvoiced_mask

    # 5) Syllabic envelope (3-6 Hz), peak normalisation, noise floor.
    t = np.arange(n) / sr
    env = 0.35 + 0.65 * 0.5 * (1.0 - np.cos(2.0 * np.pi * rng.uniform(3.0, 6.0) * t
                                            + rng.uniform(0.0, 2.0 * np.pi)))
    y = (voiced + unvoiced) * env
    peak = float(np.max(np.abs(y)))
    if peak > 0.0:
        y *= rng.uniform(0.3, 0.9) / peak
    y += rng.standard_normal(n) * (10.0 ** (NOISE_FLOOR_DBFS / 20.0))
    np.clip(y, -1.0, 1.0, out=y)
    return y.astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Corpus on disk
# ---------------------------------------------------------------------------------------------


def default_cache_dir() -> Path:
    """AGBENCH_CACHE, else ``<repo>/cache`` (same rule as ``agbench.util.default_cache_dir``)."""
    try:
        from agbench.util import default_cache_dir as _dcd

        return _dcd()
    except Exception:
        return Path(os.environ.get("AGBENCH_CACHE") or "cache")


def corpus_dir(cache_dir: Path | str | None = None, *, tiny: bool = False) -> Path:
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    return root / (TINY_DIR if tiny else FULL_DIR)


def _corpus_params(tiny: bool, n_files: int, seed: int, min_s: float | None,
                   max_s: float | None) -> dict:
    if tiny:
        n_files = min(int(n_files), 8)
        min_s = 1.0 if min_s is None else float(min_s)
        max_s = 2.0 if max_s is None else float(max_s)
    else:
        min_s = 4.0 if min_s is None else float(min_s)
        max_s = 10.0 if max_s is None else float(max_s)
    if n_files < 1 or not (0.0 < min_s <= max_s):
        raise ValueError(f"bad corpus parameters: n_files={n_files}, min_s={min_s}, max_s={max_s}")
    return {"generator": GENERATOR, "n_files": int(n_files), "seed": int(seed),
            "min_s": min_s, "max_s": max_s, "tiny": bool(tiny)}


def _manifest_ok(manifest: Path, params: dict) -> bool:
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("version") != CORPUS_VERSION or data.get("sr") != SR:
            return False
        if any(data.get(k) != v for k, v in params.items()):
            return False
        files = data["files"]
        if len(files) != params["n_files"]:
            return False
        root = manifest.parent
        return all((root / f["path"]).stat().st_size == int(f["bytes"]) for f in files)
    except Exception:
        return False


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def ensure_corpus(cache_dir: Path | str | None = None, *, tiny: bool = False, n_files: int = 256,
                  seed: int = 20261016, min_s: float | None = None,
                  max_s: float | None = None) -> Path:
    """Create (once) the synthetic FLAC corpus and return the path of its ``manifest.json``.

    Full: ``n_files`` (256) files of uniform random 4-10 s under ``<cache>/synth_v1/``.
    Tiny: 8 files of 1-2 s under ``<cache>/synth_tiny_v1/``. 16-bit PCM FLAC, 16 kHz mono,
    balanced random labels. Idempotent: when the manifest matches the parameters and every file
    has its recorded size, nothing is written.
    """
    import soundfile as sf

    params = _corpus_params(tiny, n_files, seed, min_s, max_s)
    d = corpus_dir(cache_dir, tiny=tiny)
    manifest = d / "manifest.json"
    if _manifest_ok(manifest, params):
        return manifest
    d.mkdir(parents=True, exist_ok=True)

    n = params["n_files"]
    master = np.random.default_rng(params["seed"])
    durations = master.uniform(params["min_s"], params["max_s"], size=n)
    labels = np.resize(np.array([0, 1], dtype=np.int64), n)
    master.shuffle(labels)

    files = []
    for i in range(n):
        n_samp = max(1, int(round(float(durations[i]) * SR)))
        wave = speechlike(n_samp, np.random.default_rng([params["seed"], i]), SR)
        name = f"{i:04d}.flac"
        path = d / name
        tmp = d / f"{name}.{os.getpid()}.tmp"
        sf.write(str(tmp), wave, SR, format="FLAC", subtype="PCM_16")
        os.replace(tmp, path)
        files.append({"path": name, "seconds": round(n_samp / SR, 4), "frames": n_samp,
                      "label": int(labels[i]), "bytes": path.stat().st_size})
    data = {"version": CORPUS_VERSION, "sr": SR, **params, "files": files,
            "total_bytes": int(sum(f["bytes"] for f in files)),
            "total_seconds": round(float(sum(f["seconds"] for f in files)), 3)}
    _write_atomic(manifest, json.dumps(data, indent=1, sort_keys=True))
    return manifest


def load_manifest(manifest: Path | str) -> dict:
    return json.loads(Path(manifest).read_text(encoding="utf-8"))
