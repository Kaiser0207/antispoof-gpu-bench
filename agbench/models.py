"""Model registry and encoder loading (SPEC sections 2.3, 5.1 and 12).

Import-light: this module does not import torch or transformers at import time,
so the parent (runner) process can read ``MODELS`` without touching torch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import torch


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    revision: str
    weights_bytes: int
    hidden: int
    layers: int
    family: str  # "wav2vec2" | "wavlm"
    sha256: str = ""  # of pytorch_model.bin at ``revision`` (the Hugging Face LFS hash)


MODELS: dict[str, ModelSpec] = {
    "w2v2b": ModelSpec(
        "w2v2b", "facebook/wav2vec2-base",
        "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8", 380_267_417, 768, 12, "wav2vec2",
        "3249fe98bfc62fcbc26067f724716a6ec49d12c4728a2af1df659013905dff21"),
    "xlsr": ModelSpec(
        "xlsr", "facebook/wav2vec2-xls-r-300m",
        "1a640f32ac3e39899438a2931f9924c02f080a54", 1_269_737_156, 1024, 24, "wav2vec2",
        "d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0"),
    "w2v2l": ModelSpec(
        "w2v2l", "facebook/wav2vec2-large",
        "312b2410566b698c7a649068d413b2067848bd75", 1_269_691_161, 1024, 24, "wav2vec2",
        "6502e658c56f241a4f2baf9e6ea44821fa01828a54677b4a1106702d5b084d1c"),
    "wavlmb": ModelSpec(
        "wavlmb", "microsoft/wavlm-base",
        "efa81aae7ff777e464159e0f877d54eac5b84f81", 377_617_425, 768, 12, "wavlm",
        "b41ab60fdcdc113ff164212c72b371dcd420a5773ca1f6c8f54d6809225ce072"),
}

# Scenario model keys that combine several encoders (same waveform into each).
COMPOSITES: dict[str, tuple[str, ...]] = {"fusion": ("xlsr", "wavlmb")}

# Architecture flags of the real checkpoints (from each pinned config.json), mirrored
# by the tiny configs so the tiny models exercise the same code paths.
_ARCH_FLAGS: dict[str, dict[str, Any]] = {
    "w2v2b": {"feat_extract_norm": "group", "do_stable_layer_norm": False, "conv_bias": False},
    "xlsr": {"feat_extract_norm": "layer", "do_stable_layer_norm": True, "conv_bias": True},
    "w2v2l": {"feat_extract_norm": "group", "do_stable_layer_norm": False, "conv_bias": False},
    "wavlmb": {"feat_extract_norm": "group", "do_stable_layer_norm": False, "conv_bias": False},
}

# Applied to every config (tiny and real): SPEC 5.1.
_COMMON_OVERRIDES: dict[str, Any] = {"layerdrop": 0.0, "apply_spec_augment": False}

TINY_HIDDEN = 32
TINY_LAYERS = 2


def encoder_keys(model_key: str) -> tuple[str, ...]:
    """Scenario model key -> encoder keys (``fusion`` -> ``("xlsr", "wavlmb")``)."""
    if model_key in COMPOSITES:
        return COMPOSITES[model_key]
    if model_key in MODELS:
        return (model_key,)
    raise KeyError(f"unknown model key: {model_key!r}")


def _config_class(family: str):
    from transformers import Wav2Vec2Config, WavLMConfig

    return {"wav2vec2": Wav2Vec2Config, "wavlm": WavLMConfig}[family]


def _model_class(family: str):
    from transformers import Wav2Vec2Model, WavLMModel

    return {"wav2vec2": Wav2Vec2Model, "wavlm": WavLMModel}[family]


def tiny_config(key: str):
    """Tiny random config: hidden 32, 2 layers, conv_dim 16 x 7 with the real conv
    strides/kernels (so frames per second match: 49 frames per second of audio)."""
    spec = MODELS[key]
    kwargs: dict[str, Any] = dict(
        hidden_size=TINY_HIDDEN,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=2,
        intermediate_size=64,
        conv_dim=(16,) * 7,
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=4,
        **_ARCH_FLAGS[key],
        **_COMMON_OVERRIDES,
    )
    if spec.family == "wavlm":
        kwargs.update(num_buckets=32, max_bucket_distance=80)
    return _config_class(spec.family)(**kwargs)


def set_grad_ckpt(model, enabled: bool) -> None:
    """Set activation checkpointing explicitly and assert the resulting flag.

    transformers 5.20 turns checkpointing ON when a config carries the legacy key
    ``"gradient_checkpointing": true`` (facebook/wav2vec2-base does), so every load
    must call this (SPEC 2.2 trap).
    """
    if enabled:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()
    actual = bool(model.is_gradient_checkpointing)
    if actual is not bool(enabled):
        raise RuntimeError(
            f"activation checkpointing is {actual} after setting it to {bool(enabled)}")


def load_encoder(key: str, *, tiny: bool, device: str, grad_ckpt: bool = False) -> "torch.nn.Module":
    """Load one encoder (``Wav2Vec2Model`` / ``WavLMModel``), float32, on ``device``.

    Real weights come from ``agbench.hub.resolve(key)`` (raises ``SkipScenario("model_missing")``).
    LayerDrop 0, SpecAugment off, checkpointing set explicitly and asserted.
    The returned module carries ``agbench_info`` (dict) with load details for ``extra``.
    """
    import torch

    spec = MODELS[key]
    cls = _model_class(spec.family)
    t0 = time.perf_counter()
    info: dict[str, Any] = {"key": key, "hf_id": spec.hf_id, "tiny": bool(tiny)}
    if tiny:
        model = cls(tiny_config(key))
        info.update(revision=None, source="tiny_random_config", missing_keys=[], unexpected_keys=0)
    else:
        from agbench import hub

        folder = hub.resolve(key)
        config = _config_class(spec.family).from_pretrained(str(folder), **_COMMON_OVERRIDES)
        legacy_ckpt = bool(getattr(config, "gradient_checkpointing", False))
        with _quiet_transformers():
            model, load_info = cls.from_pretrained(
                str(folder), config=config, dtype=torch.float32, output_loading_info=True)
        missing = sorted(load_info.get("missing_keys") or [])
        unexpected = load_info.get("unexpected_keys") or []
        mismatched = load_info.get("mismatched_keys") or []
        if mismatched:
            raise RuntimeError(f"{spec.hf_id}: shape-mismatched weights: {sorted(mismatched)[:5]}")
        # The pre-training checkpoints only add quantizer/projection weights that the bare
        # encoder does not use; a missing encoder weight means a wrong or broken file.
        bad = [k for k in missing if k != "masked_spec_embed"]
        if bad:
            raise RuntimeError(f"{spec.hf_id}: {len(bad)} encoder weights missing, e.g. {bad[:5]}")
        info.update(revision=spec.revision, source=hub.source_of(folder),
                    missing_keys=missing, unexpected_keys=len(unexpected),
                    legacy_config_grad_ckpt=legacy_ckpt)
    model = model.to(dtype=torch.float32)
    set_grad_ckpt(model, grad_ckpt)
    model = model.to(torch.device(device))
    cfg = model.config
    if float(getattr(cfg, "layerdrop", 0.0)) != 0.0 or bool(getattr(cfg, "apply_spec_augment", False)):
        raise RuntimeError("layerdrop/apply_spec_augment overrides did not apply")
    info.update(
        family=spec.family,
        attn_implementation=getattr(cfg, "_attn_implementation", None),
        grad_ckpt=bool(model.is_gradient_checkpointing),
        hidden=int(cfg.hidden_size),
        layers=int(cfg.num_hidden_layers),
        params_m=round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
        load_s=round(time.perf_counter() - t0, 3),
    )
    model.agbench_info = info
    return model


class _quiet_transformers:
    """Silence the transformers load report / progress bar (it prints absolute cache paths)."""

    def __enter__(self):
        try:
            from transformers.utils import logging as hf_logging

            self._lg = hf_logging
            self._level = hf_logging.get_verbosity()
            hf_logging.set_verbosity_error()
            self._bar = hf_logging.is_progress_bar_enabled()
            hf_logging.disable_progress_bar()
        except Exception:  # pragma: no cover - logging helpers moved
            self._lg = None
        return self

    def __exit__(self, *exc):
        if self._lg is not None:
            self._lg.set_verbosity(self._level)
            if self._bar:
                self._lg.enable_progress_bar()
        return False
