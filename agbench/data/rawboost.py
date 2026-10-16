"""RawBoost data augmentation, algorithms 1-8 (SPEC 5.7).

Re-implementation of H. Tak, M. Kamble, J. Patino, M. Todisco, N. Evans, "RawBoost: A Raw Data
Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing",
ICASSP 2022. It follows the official MIT-licensed implementation,
github.com/TakHemlata/RawBoost-antispoofing (Copyright (c) 2021 Hemlata): the same operations,
default parameters and parameter names, rewritten for this benchmark with no lines copied. The MIT
notice is reproduced in NOTICE and LICENSES/RawBoost-antispoofing-MIT.txt.

* LnL (algo 1): linear and non-linear convolutive noise. Sum over orders j = 1..N_f of
  ``x**j`` filtered by a random multi-band notch FIR; orders j >= 2 get a lower gain
  (shifted by the lin/non-lin bias). Mean removed, normalised if the peak exceeds 1.
* ISD (algo 2): impulsive signal-dependent noise on up to P % of the samples.
* SSI (algo 3): stationary signal-independent noise. White Gaussian noise coloured by a notch
  FIR, added at a random SNR in [SNRmin, SNRmax] dB. The SNR scaling uses ``np.linalg.norm``
  on float64 vectors, which goes through BLAS (``ddot``); OpenBLAS splits long vectors across
  its threads. That is the call that makes ``OPENBLAS_NUM_THREADS`` matter in loader workers,
  so it is kept on purpose.
* Algo 4 = 1 -> 2 -> 3 in series (the default in our training); 5 = 1 -> 2; 6 = 1 -> 3;
  7 = 2 -> 3; 8 = 1 and 2 in parallel (sum, normalised).

All randomness comes from an explicit ``numpy.random.Generator``. Stage functions work in
float64 and return float64; ``rawboost()`` returns float32. No import-time side effects.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

DEFAULTS: dict = {
    # notch filter bank (LnL and SSI)
    "nBands": 5, "minF": 20, "maxF": 8000, "minBW": 100, "maxBW": 1000,
    "minCoeff": 10, "maxCoeff": 100, "minG": 0, "maxG": 0,
    # LnL
    "minBiasLinNonLin": 5, "maxBiasLinNonLin": 20, "N_f": 5,
    # ISD
    "P": 10, "g_sd": 2,
    # SSI
    "SNRmin": 10, "SNRmax": 40,
}
_ALLOWED = frozenset(DEFAULTS) | {"sr"}
ALGOS = (0, 1, 2, 3, 4, 5, 6, 7, 8)


def _params(p: dict) -> dict:
    unknown = set(p) - _ALLOWED
    if unknown:
        raise TypeError(f"unknown RawBoost parameter(s): {sorted(unknown)}")
    out = dict(DEFAULTS)
    out["sr"] = 16000
    out.update(p)
    return out


def _peak_normalise(x: np.ndarray, always: bool) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.0 and (always or peak > 1.0):
        return x / peak
    return x


def notch_fir(rng: np.random.Generator, nBands: int, minF: float, maxF: float, minBW: float,
              maxBW: float, minCoeff: int, maxCoeff: int, minG: float, maxG: float,
              sr: int = 16000) -> np.ndarray:
    """Random multi-band notch FIR: ``nBands`` band-stop filters convolved together.

    Per band: centre frequency in [minF, maxF], bandwidth in [minBW, maxBW], an odd number of
    taps in [minCoeff, maxCoeff] (+1 if even); ``firwin`` with a Hamming window. The cascade is
    normalised so that its peak magnitude response (``freqz``) equals a random gain in
    [minG, maxG] dB.
    """
    nyq = sr / 2.0
    b = np.ones(1)
    for _ in range(int(nBands)):
        fc = rng.uniform(minF, maxF)
        bw = rng.uniform(minBW, maxBW)
        taps = int(rng.integers(int(minCoeff), int(maxCoeff)))
        if taps % 2 == 0:
            taps += 1
        lo = max(fc - bw / 2.0, 1e-3)
        hi = min(fc + bw / 2.0, nyq - 1e-3)
        band = signal.firwin(taps, [lo, hi], window="hamming", fs=sr)  # band-stop (pass_zero)
        b = np.convolve(band, b)
    # The bias shift can make minG > maxG (e.g. 0-5 and 0-20): draw from the ordered interval.
    gain_db = rng.uniform(min(minG, maxG), max(minG, maxG))
    _, h = signal.freqz(b, 1, fs=sr)
    return (10.0 ** (gain_db / 20.0)) * b / np.max(np.abs(h))


def _fir(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """FIR filter via ``lfilter`` with half-length delay compensation; same length as ``x``."""
    half = (b.shape[0] + 1) // 2
    padded = np.concatenate([x, np.zeros(b.shape[0] + 1, dtype=x.dtype)])
    return signal.lfilter(b, 1.0, padded)[half:half + x.shape[0]]


def _notch_args(q: dict, min_g: float, max_g: float) -> dict:
    return {"nBands": q["nBands"], "minF": q["minF"], "maxF": q["maxF"], "minBW": q["minBW"],
            "maxBW": q["maxBW"], "minCoeff": q["minCoeff"], "maxCoeff": q["maxCoeff"],
            "minG": min_g, "maxG": max_g, "sr": q["sr"]}


def lnl(x: np.ndarray, rng: np.random.Generator, **p) -> np.ndarray:
    """Linear and non-linear convolutive noise (algorithm 1). float64 in, float64 out."""
    q = _params(p)
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x)
    min_g, max_g = q["minG"], q["maxG"]
    for order in range(1, int(q["N_f"]) + 1):
        if order == 2:  # non-linear orders get a lower gain range
            min_g, max_g = q["minG"] - q["minBiasLinNonLin"], q["maxG"] - q["maxBiasLinNonLin"]
        b = notch_fir(rng, **_notch_args(q, min_g, max_g))
        y += _fir(x ** order, b)
    y -= np.mean(y)
    return _peak_normalise(y, always=False)


def isd(x: np.ndarray, rng: np.random.Generator, **p) -> np.ndarray:
    """Impulsive signal-dependent additive noise (algorithm 2). float64 in, float64 out."""
    q = _params(p)
    x = np.asarray(x, dtype=np.float64)
    y = x.copy()
    beta = rng.uniform(0.0, q["P"])  # percentage of samples to hit
    n = int(x.shape[0] * beta / 100.0)
    idx = rng.permutation(x.shape[0])[:n]
    f = (2.0 * rng.random(n) - 1.0) * (2.0 * rng.random(n) - 1.0)
    y[idx] = x[idx] + q["g_sd"] * x[idx] * f
    return _peak_normalise(y, always=False)


def ssi(x: np.ndarray, rng: np.random.Generator, **p) -> np.ndarray:
    """Stationary signal-independent additive noise (algorithm 3). float64 in, float64 out."""
    q = _params(p)
    x = np.asarray(x, dtype=np.float64)
    noise = rng.standard_normal(x.shape[0])
    b = notch_fir(rng, **_notch_args(q, q["minG"], q["maxG"]))
    noise = _peak_normalise(_fir(noise, b), always=True)
    snr_db = rng.uniform(q["SNRmin"], q["SNRmax"])
    # BLAS-backed norms on purpose (see the module docstring).
    scale = np.linalg.norm(x, 2) / (np.linalg.norm(noise, 2) * 10.0 ** (0.05 * snr_db))
    return x + noise * scale


def rawboost(x: np.ndarray, rng: np.random.Generator, algo: int = 4, sr: int = 16000,
             **overrides) -> np.ndarray:
    """Apply RawBoost algorithm ``algo`` (0 = none, 1-8) to a 1-D waveform; returns float32."""
    q = dict(overrides)
    q["sr"] = sr
    _params(q)  # validate names early
    x = np.asarray(x, dtype=np.float64)
    if algo == 0:
        y = x
    elif algo == 1:
        y = lnl(x, rng, **q)
    elif algo == 2:
        y = isd(x, rng, **q)
    elif algo == 3:
        y = ssi(x, rng, **q)
    elif algo == 4:
        y = ssi(isd(lnl(x, rng, **q), rng, **q), rng, **q)
    elif algo == 5:
        y = isd(lnl(x, rng, **q), rng, **q)
    elif algo == 6:
        y = ssi(lnl(x, rng, **q), rng, **q)
    elif algo == 7:
        y = ssi(isd(x, rng, **q), rng, **q)
    elif algo == 8:
        y = _peak_normalise(lnl(x, rng, **q) + isd(x, rng, **q), always=False)
    else:
        raise ValueError(f"RawBoost algo must be one of {ALGOS}, got {algo!r}")
    return np.asarray(y, dtype=np.float32)
