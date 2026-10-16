"""Classification heads (SPEC section 5.3).

All heads take encoder hidden states ``[B, T', D]`` and return logits ``[B, n_classes]``.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

DEFAULT_BINS = (1, 2, 4)


def bin_bounds(T: int, n: int) -> list[tuple[int, int]]:
    """Interval boundaries of ``adaptive_avg_pool1d``: start = floor(i*T/n), end = ceil((i+1)*T/n)."""
    return [((i * T) // n, -((-(i + 1) * T) // n)) for i in range(n)]


def pyramid_pool(h: torch.Tensor, bins: Sequence[int] = DEFAULT_BINS) -> torch.Tensor:
    """``[B, T, D]`` -> ``[B, sum(bins) * D]``: mean of each time interval, bins in order."""
    T = h.shape[1]
    feats = [h[:, s:e].mean(dim=1) for n in bins for s, e in bin_bounds(T, n)]
    return torch.cat(feats, dim=-1)


class PyramidMeanHead(nn.Module):
    """Means over 1 + 2 + 4 time bins -> Linear(7*D, 256) -> Linear(256, 2), no activation."""

    def __init__(self, dim: int, bins=DEFAULT_BINS, proj: int = 256, n_classes: int = 2):
        super().__init__()
        self.bins = tuple(int(b) for b in bins)
        self.proj = nn.Linear(sum(self.bins) * dim, proj)
        self.out = nn.Linear(proj, n_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.out(self.proj(pyramid_pool(h, self.bins)))


class GlobalMeanHead(nn.Module):
    """Mean over time -> Linear(D, 256) -> Linear(256, 2)."""

    def __init__(self, dim: int, proj: int = 256, n_classes: int = 2):
        super().__init__()
        self.proj = nn.Linear(dim, proj)
        self.out = nn.Linear(proj, n_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.out(self.proj(h.mean(dim=1)))


class FusionHead(nn.Module):
    """One pyramid projection per encoder (Linear(7*D_i, 256)), concatenated -> Linear(256*k, 2).

    ``forward`` takes a sequence of hidden states, one per encoder, in ``dims`` order.
    """

    def __init__(self, dims: tuple[int, ...], bins=DEFAULT_BINS, proj: int = 256, n_classes: int = 2):
        super().__init__()
        self.bins = tuple(int(b) for b in bins)
        self.projs = nn.ModuleList(nn.Linear(sum(self.bins) * d, proj) for d in dims)
        self.out = nn.Linear(proj * len(dims), n_classes)

    def forward(self, hs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(hs) != len(self.projs):
            raise ValueError(f"FusionHead expects {len(self.projs)} hidden states, got {len(hs)}")
        z = [p(pyramid_pool(h, self.bins)) for p, h in zip(self.projs, hs)]
        return self.out(torch.cat(z, dim=-1))


class FrozenHeadSet(nn.Module):
    """One probe set = 4 heads trained jointly on frozen hidden states (loss = sum):

    1. pyramid head on the last hidden state;
    2. softmax-weighted sum of all hidden states (learned weights) -> pyramid head;
    3. one linear probe ``Linear(D, 2)`` per hidden state on its time mean (``n_states`` probes);
    4. MLP on the concatenated time means: ``Linear(D*n_states, 256)`` -> GELU -> ``Linear(256, 2)``.

    ``forward(hidden_states)`` returns a list of ``3 + n_states`` logits tensors ``[B, 2]``.
    """

    n_heads = 4

    def __init__(self, dim: int, n_states: int):
        super().__init__()
        self.n_states = int(n_states)
        self.last = PyramidMeanHead(dim)
        self.layer_logits = nn.Parameter(torch.zeros(self.n_states))
        self.weighted = PyramidMeanHead(dim)
        self.probes = nn.ModuleList(nn.Linear(dim, 2) for _ in range(self.n_states))
        self.mlp = nn.Sequential(nn.Linear(dim * self.n_states, 256), nn.GELU(), nn.Linear(256, 2))

    @property
    def n_outputs(self) -> int:
        return 3 + self.n_states

    def forward(self, hidden_states: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(hidden_states) != self.n_states:
            raise ValueError(f"expected {self.n_states} hidden states, got {len(hidden_states)}")
        dt = hidden_states[-1].dtype
        stack = torch.stack([h if h.dtype == dt else h.to(dt) for h in hidden_states])  # [L, B, T, D]
        w = torch.softmax(self.layer_logits.float(), dim=0).to(dt)
        mixed = torch.tensordot(w, stack, dims=1)  # [B, T, D]
        means = stack.mean(dim=2)  # [L, B, D]
        outs = [self.last(hidden_states[-1]), self.weighted(mixed)]
        outs.extend(probe(means[i]) for i, probe in enumerate(self.probes))
        outs.append(self.mlp(means.permute(1, 0, 2).reshape(means.shape[1], -1)))
        return outs


def make_head(kind: str, dims: Sequence[int]) -> nn.Module:
    """``pyramid`` / ``gmean`` for one encoder; ``fusion`` (or several dims) -> FusionHead."""
    dims = tuple(int(d) for d in dims)
    if kind == "fusion" or len(dims) > 1:
        return FusionHead(dims)
    if kind == "pyramid":
        return PyramidMeanHead(dims[0])
    if kind == "gmean":
        return GlobalMeanHead(dims[0])
    raise ValueError(f"unknown head {kind!r} (pyramid, gmean, fusion)")
