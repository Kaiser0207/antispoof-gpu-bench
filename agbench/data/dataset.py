"""Dataset and DataLoader over the synthetic FLAC corpus (SPEC 5.8).

Windows ``spawn`` safety: the dataset class, sampler and ``worker_init_fn`` live at module level,
the module has no import-time side effects, the dataset holds only picklable state (strings and
numbers), workers never touch CUDA, and there are no lambdas. RawBoost (and with it scipy) is
imported only when an item is actually augmented.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SynthFlacDataset(Dataset):
    """Items are ``(waveform float32 [seconds*sr], label int64)``.

    ``__getitem__(i)`` decodes the whole FLAC file (soundfile), loops it when shorter than
    ``seconds``, crops ``seconds*sr`` samples (random offset when ``train``, centre otherwise) and
    applies RawBoost (``rawboost_algo``, default 4) with probability ``rawboost_p``. The per-item
    RNG is seeded from ``(seed, epoch, i)``.
    """

    def __init__(self, manifest: Path | str, seconds: float = 6.0, train: bool = True,
                 rawboost_p: float = 0.0, seed: int = 1234, rawboost_algo: int = 4):
        manifest = Path(manifest)
        data = json.loads(manifest.read_text(encoding="utf-8"))
        self.root = str(manifest.parent)
        self.sr = int(data["sr"])
        self.files = [(str(f["path"]), int(f["label"])) for f in data["files"]]
        if not self.files:
            raise ValueError("empty corpus manifest")
        self.seconds = float(seconds)
        self.n_samples = int(round(self.seconds * self.sr))
        if self.n_samples < 1:
            raise ValueError(f"seconds must be > 0, got {seconds!r}")
        self.train = bool(train)
        self.rawboost_p = float(rawboost_p)
        self.rawboost_algo = int(rawboost_algo)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Change the epoch used in the per-item seed (not seen by already-started workers)."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.files)

    def load_crop(self, i: int) -> tuple[np.ndarray, int, np.random.Generator]:
        """Decode, loop and crop item ``i``; returns (float32 crop, label, the item's RNG)."""
        import soundfile as sf

        rel, label = self.files[i]
        rng = np.random.default_rng([self.seed, self.epoch, int(i)])
        wave, sr = sf.read(os.path.join(self.root, rel), dtype="float32", always_2d=False)
        if sr != self.sr:
            raise ValueError(f"sample rate {sr} != manifest {self.sr}: {rel}")
        if wave.ndim > 1:
            wave = wave[:, 0]
        n = self.n_samples
        if wave.shape[0] == 0:
            wave = np.zeros(n, dtype=np.float32)
        if wave.shape[0] < n:
            wave = np.tile(wave, math.ceil(n / wave.shape[0]))
        extra = wave.shape[0] - n
        off = int(rng.integers(0, extra + 1)) if self.train else extra // 2
        return wave[off:off + n].copy(), label, rng

    def __getitem__(self, i: int):
        crop, label, rng = self.load_crop(i)
        if self.rawboost_p > 0.0 and rng.random() < self.rawboost_p:
            from agbench.data.rawboost import rawboost

            crop = rawboost(crop, rng, algo=self.rawboost_algo, sr=self.sr)
        return torch.from_numpy(np.ascontiguousarray(crop, dtype=np.float32)), \
            torch.tensor(label, dtype=torch.int64)


class CycleSampler(torch.utils.data.Sampler):
    """Sequential indices 0..n-1 repeated until ``num_samples`` indices were produced."""

    def __init__(self, n: int, num_samples: int):
        self.n = int(n)
        self.num_samples = int(num_samples)

    def __iter__(self):
        return (k % self.n for k in range(self.num_samples))

    def __len__(self) -> int:
        return self.num_samples


def worker_init_fn(worker_id: int) -> None:
    """Seed the legacy global RNGs per worker and pre-import RawBoost (scipy) when the worker's
    dataset augments, so that import cost counts as worker start-up rather than as the first
    batch. Never touches CUDA or BLAS thread settings."""
    info = torch.utils.data.get_worker_info()
    seed = int(info.seed if info is not None else worker_id) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)
    if info is not None and float(getattr(info.dataset, "rawboost_p", 0.0) or 0.0) > 0.0:
        import agbench.data.rawboost  # noqa: F401


def make_loader(ds, *, batch_size: int, num_workers: int, pin_memory: bool,
                persistent_workers: bool, prefetch_factor: int = 2, shuffle: bool = True,
                drop_last: bool = True, num_samples: int | None = None,
                seed: int = 0) -> torch.utils.data.DataLoader:
    """DataLoader with ``multiprocessing_context="spawn"`` on every OS when ``num_workers > 0``.

    Extensions beyond SPEC 12 (all optional): ``drop_last`` (default True, every batch full);
    ``num_samples`` draws that many indices (several passes over the corpus without an epoch
    boundary, which would stall the pipeline and, without persistent workers, re-spawn workers);
    ``seed`` for the shuffling generator.
    """
    from torch.utils.data import DataLoader, RandomSampler

    gen = torch.Generator()
    gen.manual_seed(int(seed))
    kwargs: dict = {"batch_size": int(batch_size), "num_workers": int(num_workers),
                    "pin_memory": bool(pin_memory), "drop_last": bool(drop_last)}
    if num_samples is not None:
        kwargs["sampler"] = (RandomSampler(ds, replacement=False, num_samples=int(num_samples),
                                           generator=gen)
                             if shuffle else CycleSampler(len(ds), int(num_samples)))
    else:
        kwargs["shuffle"] = bool(shuffle)
        kwargs["generator"] = gen
    if num_workers > 0:
        kwargs.update(multiprocessing_context="spawn", persistent_workers=bool(persistent_workers),
                      prefetch_factor=int(prefetch_factor), worker_init_fn=worker_init_fn)
    return DataLoader(ds, **kwargs)
