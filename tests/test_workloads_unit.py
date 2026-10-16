"""Builder B unit tests: heads, model registry/loading, hub logic, model workloads (tiny, CPU).

CPU only, no network, no downloads: every model here is a tiny random config.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from agbench import heads as H  # noqa: E402
from agbench import hub  # noqa: E402
from agbench import models as M  # noqa: E402
from agbench.context import MeasureCfg, RunContext  # noqa: E402
from agbench.errors import SkipScenario  # noqa: E402
from agbench.workloads import common as C  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODEL_WORKLOADS = ("train", "frozen", "infer", "e2e", "sustain")


@pytest.fixture(autouse=True)
def _reset_torch_globals():
    """Workloads set process-global torch flags (a child runs one scenario); undo them per test."""
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.fp32_precision = "none"  # PyTorch defaults, new API only
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.backends.cuda.enable_flash_sdp(True)


def make_ctx(workload: str, params: dict, tmp_path: Path, **kw) -> RunContext:
    return RunContext(
        scenario_id=kw.pop("scenario_id", f"{workload}.unit"), workload=workload, params=params,
        device="cpu", tiny=True, measure=MeasureCfg.tiny(), seed=1234,
        cache_dir=tmp_path / "cache", keep_series=False, log=lambda s: None, **kw)


def base_params(**over) -> dict:
    p = {"model": "w2v2b", "mode": "train", "seconds": 2.0, "batch": 2, "precision": "bf16",
         "head": "pyramid", "grad_ckpt": False, "deterministic": False, "lr_encoder": 1e-5,
         "lr_head": 1e-4, "weight_decay": 0.01, "grad_clip": 1.0, "layerdrop": 0.0, "seed": 1234,
         "micro_batch": 2, "accum": 1}
    p.update(over)
    return p


def assert_finite_metrics(out: dict, effective_batch: int) -> dict:
    assert set(out) >= {"metrics", "extra", "timeseries"}
    m = out["metrics"]
    for k in ("clips_per_s", "clips_per_s_wall", "step_ms_median", "step_ms_mean"):
        assert isinstance(m[k], float) and math.isfinite(m[k]) and m[k] > 0, (k, m[k])
    assert m["n_steps"] >= 1
    assert m["effective_batch"] == effective_batch
    assert m["loss_finite"] is True
    return m


# --------------------------------------------------------------------------- heads


@pytest.mark.parametrize("T", [1, 2, 3, 5, 7, 8, 13, 99, 299])
def test_pyramid_pool_matches_adaptive_avg_pool(T):
    torch.manual_seed(0)
    h = torch.randn(3, T, 5, dtype=torch.float64)
    got = H.pyramid_pool(h, (1, 2, 4))
    ref = torch.cat([F.adaptive_avg_pool1d(h.transpose(1, 2), n).transpose(1, 2).reshape(3, -1)
                     for n in (1, 2, 4)], dim=-1)
    assert got.shape == (3, 7 * 5)
    torch.testing.assert_close(got, ref)


def test_bin_bounds_cover_the_sequence():
    for T in (1, 4, 10, 299):
        for n in (1, 2, 4):
            b = H.bin_bounds(T, n)
            assert b[0][0] == 0 and b[-1][1] == T and all(e > s for s, e in b)


def test_head_shapes_and_layers():
    h = torch.randn(4, 50, 32)
    head = H.PyramidMeanHead(32)
    assert head(h).shape == (4, 2)
    assert head.proj.in_features == 7 * 32 and head.proj.out_features == 256
    assert head.out.in_features == 256 and head.out.out_features == 2
    g = H.GlobalMeanHead(32)
    assert g(h).shape == (4, 2) and g.proj.in_features == 32
    fh = H.FusionHead((32, 16))
    assert fh([h, torch.randn(4, 50, 16)]).shape == (4, 2)
    assert [p.in_features for p in fh.projs] == [7 * 32, 7 * 16] and fh.out.in_features == 512
    with pytest.raises(ValueError):
        fh([h])
    assert isinstance(H.make_head("pyramid", [32]), H.PyramidMeanHead)
    assert isinstance(H.make_head("gmean", [32]), H.GlobalMeanHead)
    assert isinstance(H.make_head("fusion", [32, 16]), H.FusionHead)
    with pytest.raises(ValueError):
        H.make_head("nope", [32])


def test_frozen_head_set_outputs_and_grads():
    n_states, dim = 3, 16
    hs = [torch.randn(2, 20, dim) for _ in range(n_states)]
    hset = H.FrozenHeadSet(dim, n_states)
    outs = hset(hs)
    assert len(outs) == hset.n_outputs == 3 + n_states
    assert all(o.shape == (2, 2) for o in outs)
    assert hset.mlp[0].in_features == dim * n_states
    sum(o.sum() for o in outs).backward()
    assert hset.layer_logits.grad is not None and torch.isfinite(hset.layer_logits.grad).all()
    assert all(p.grad is not None for p in hset.parameters())


# --------------------------------------------------------------------------- model registry


def test_model_registry_matches_spec():
    assert set(M.MODELS) == {"w2v2b", "xlsr", "w2v2l", "wavlmb"}
    assert M.MODELS["w2v2l"].hf_id == "facebook/wav2vec2-large"
    assert M.MODELS["wavlmb"].family == "wavlm"
    assert sum(s.weights_bytes for s in M.MODELS.values()) == 3_297_313_159
    assert all(len(s.revision) == 40 for s in M.MODELS.values())
    assert M.encoder_keys("fusion") == ("xlsr", "wavlmb")
    assert M.encoder_keys("xlsr") == ("xlsr",)
    with pytest.raises(KeyError):
        M.encoder_keys("nope")


def test_models_and_hub_do_not_import_torch():
    code = ("import sys, agbench.models, agbench.hub; "
            "print(int('torch' in sys.modules), int('transformers' in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(REPO)}, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["0", "0"]


@pytest.mark.parametrize("key", sorted(M.MODELS))
@pytest.mark.parametrize("ckpt", [False, True])
def test_load_encoder_tiny_sets_checkpointing(key, ckpt):
    enc = M.load_encoder(key, tiny=True, device="cpu", grad_ckpt=ckpt)
    assert enc.is_gradient_checkpointing is ckpt
    assert enc.config.hidden_size == M.TINY_HIDDEN and enc.config.num_hidden_layers == M.TINY_LAYERS
    assert enc.config.layerdrop == 0.0 and enc.config.apply_spec_augment is False
    assert next(enc.parameters()).dtype == torch.float32
    info = enc.agbench_info
    assert info["grad_ckpt"] is ckpt and info["source"] == "tiny_random_config"
    expected_attn = "eager" if M.MODELS[key].family == "wavlm" else "sdpa"
    assert info["attn_implementation"] == expected_attn
    # same conv stack as the real models: 6 s -> 299 frames, 2 s -> 99 frames
    assert int(enc._get_feat_extract_output_lengths(torch.tensor(96000))) == 299
    assert int(enc._get_feat_extract_output_lengths(torch.tensor(32000))) == 99


def test_load_encoder_overrides_legacy_checkpointing_config(monkeypatch):
    """facebook/wav2vec2-base carries "gradient_checkpointing": true; loading must turn it off."""
    real = M.tiny_config

    def legacy(key):
        cfg = real(key)
        cfg.gradient_checkpointing = True
        return cfg

    monkeypatch.setattr(M, "tiny_config", legacy)
    probe = M._model_class("wav2vec2")(legacy("w2v2b"))
    assert probe.is_gradient_checkpointing is True  # the trap is real in this transformers
    enc = M.load_encoder("w2v2b", tiny=True, device="cpu", grad_ckpt=False)
    assert enc.is_gradient_checkpointing is False


# --------------------------------------------------------------------------- hub


def _fake_model_folder(root: Path, key: str, size: int | None = None) -> Path:
    folder = root / key
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    with open(folder / "pytorch_model.bin", "wb") as f:
        f.truncate(M.MODELS[key].weights_bytes if size is None else size)  # sparse file
    return folder


@pytest.fixture
def no_hf_cache(monkeypatch):
    monkeypatch.setattr(hub, "_hf_cache_folder", lambda spec: None)
    # the fake weights are sparse zero files of the right size: skip the sha256 check here
    # (tested on its own in test_verify_hash_*)
    monkeypatch.setattr(hub, "verify_hash", lambda folder, spec: None)
    monkeypatch.delenv("AGBENCH_MODELS_DIR", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    import huggingface_hub.constants as hc

    monkeypatch.setattr(hc, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(hc, "HF_HUB_DISABLE_XET", False)


def test_resolve_uses_models_dir(tmp_path, monkeypatch, no_hf_cache):
    folder = _fake_model_folder(tmp_path, "w2v2b")
    monkeypatch.setenv("AGBENCH_MODELS_DIR", str(tmp_path))
    assert hub.resolve("w2v2b") == folder
    assert hub.is_cached("w2v2b")
    assert hub.source_of(folder) == "models_dir"
    assert hub.check(["w2v2b"])[0]["status"] == "cached"


def test_resolve_missing_or_wrong_size_raises_skip(tmp_path, monkeypatch, no_hf_cache):
    _fake_model_folder(tmp_path, "xlsr", size=123)
    monkeypatch.setenv("AGBENCH_MODELS_DIR", str(tmp_path))
    for key in ("xlsr", "w2v2l"):
        with pytest.raises(SkipScenario) as exc:
            hub.resolve(key)
        assert exc.value.reason == "model_missing"
        assert not hub.is_cached(key)
    rec = hub.check(["xlsr"])[0]
    assert rec["status"] == "missing" and "123 bytes" in rec["error"]


def test_prefetch_offline_never_downloads(monkeypatch, no_hf_cache):
    import huggingface_hub

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda *a, **k: pytest.fail("download attempted while offline"))
    rec = hub.prefetch(["wavlmb"], log=lambda s: None)[0]
    assert rec["status"] == "failed" and "HF_HUB_OFFLINE" in rec["error"]


def _install_fake_download(monkeypatch, tmp_path, failures: list):
    """snapshot_download that raises the queued exceptions first, then returns a valid folder."""
    import huggingface_hub

    calls = []

    def fake(repo_id, *, revision, allow_patterns, force_download=False, **kw):
        calls.append({"repo_id": repo_id, "revision": revision, "allow_patterns": allow_patterns,
                      "xet_disabled": os.environ.get("HF_HUB_DISABLE_XET")})
        if failures:
            raise failures.pop(0)
        key = next(k for k, s in M.MODELS.items() if s.hf_id == repo_id)
        folder = tmp_path / "snap" / key
        if not folder.exists():
            _fake_model_folder(tmp_path / "snap", key)
        return str(folder)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    return calls


class _HTTPErr(Exception):
    def __init__(self, code: int):
        super().__init__(f"HTTP {code}")
        self.response = types.SimpleNamespace(status_code=code)


def test_prefetch_retries_then_downloads(tmp_path, monkeypatch, no_hf_cache):
    calls = _install_fake_download(monkeypatch, tmp_path, [ConnectionError("reset"), TimeoutError("slow")])
    rec = hub.prefetch(["w2v2b"], log=lambda s: None, backoff_s=(0.0,))[0]
    assert rec["status"] == "downloaded" and rec["bytes"] == M.MODELS["w2v2b"].weights_bytes
    assert len(calls) == 3
    assert calls[0]["revision"] == M.MODELS["w2v2b"].revision
    assert calls[0]["allow_patterns"] == ["config.json", "preprocessor_config.json", "pytorch_model.bin"]


def test_prefetch_xet_error_with_mirror_retries_without_xet(tmp_path, monkeypatch, no_hf_cache):
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example")
    calls = _install_fake_download(monkeypatch, tmp_path, [RuntimeError("XetDownloadError: cas failed")])
    rec = hub.prefetch(["w2v2b"], log=lambda s: None, retries=0, backoff_s=(0.0,))[0]
    assert rec["status"] == "downloaded"
    assert [c["xet_disabled"] for c in calls] == [None, "1"]


def test_prefetch_rate_limit_and_permanent_errors(tmp_path, monkeypatch, no_hf_cache):
    lines: list[str] = []
    calls = _install_fake_download(monkeypatch, tmp_path, [_HTTPErr(429)] * 3)
    rec = hub.prefetch(["w2v2b"], log=lines.append, retries=2, backoff_s=(0.0,))[0]
    assert rec["status"] == "failed" and len(calls) == 3
    assert sum(hub.RATE_LIMIT_MSG in s for s in lines) == 1
    calls = _install_fake_download(monkeypatch, tmp_path, [_HTTPErr(404)])
    rec = hub.prefetch(["xlsr"], log=lambda s: None, retries=3, backoff_s=(0.0,))[0]
    assert rec["status"] == "failed" and len(calls) == 1  # no retry on 404


def test_verify_hash_checks_once_and_detects_a_foreign_file(tmp_path, monkeypatch):
    import dataclasses
    import hashlib

    monkeypatch.setenv("AGBENCH_CACHE", str(tmp_path / "cache"))
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "pytorch_model.bin").write_bytes(b"weights")
    good = dataclasses.replace(M.MODELS["w2v2b"], sha256=hashlib.sha256(b"weights").hexdigest())
    bad = dataclasses.replace(good, sha256="0" * 64)
    assert "differs from the pinned" in hub.verify_hash(folder, bad)
    assert hub.verify_hash(folder, good) is None
    monkeypatch.setattr(hub, "sha256_file", lambda p: pytest.fail("hashed again"))
    assert hub.verify_hash(folder, good) is None  # remembered by path, size and mtime


def test_pinned_hashes_are_sha256():
    for spec in M.MODELS.values():
        assert len(spec.sha256) == 64 and int(spec.sha256, 16) >= 0


def test_check_reports_bad_hash(tmp_path, monkeypatch, no_hf_cache):
    _fake_model_folder(tmp_path, "w2v2b")
    monkeypatch.setenv("AGBENCH_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(hub, "verify_hash", lambda folder, spec: "pytorch_model.bin sha256 differs")
    rec = hub.check(["w2v2b"])[0]
    assert rec["status"] == "bad" and "sha256" in rec["error"]
    rec = hub.prefetch(["w2v2b"], log=lambda s: None)[0]
    assert rec["status"] == "failed" and "AGBENCH_MODELS_DIR" in rec["error"]


def test_prefetch_reports_cached_without_network(tmp_path, monkeypatch, no_hf_cache):
    import huggingface_hub

    _fake_model_folder(tmp_path, "w2v2b")
    monkeypatch.setenv("AGBENCH_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("network"))
    assert hub.prefetch(["w2v2b"], log=lambda s: None)[0]["status"] == "cached"


# --------------------------------------------------------------------------- common helpers


def test_model_params_tiny_caps_and_frozen_lr(tmp_path):
    ctx = make_ctx("train", base_params(seconds=6.0, batch=16, micro_batch=8, accum=2), tmp_path)
    p = C.model_params(ctx)
    assert (p.seconds, p.micro_batch, p.accum, p.effective_batch) == (2.0, 1, 2, 2)
    ctx = make_ctx("frozen", {"model": "xlsr", "mode": "frozen", "batch": 2, "lr_head": 1e-3,
                              "head": "probe_set", "micro_batch": 2, "accum": 1}, tmp_path)
    assert C.model_params(ctx).lr_heads == 1e-3
    ctx = make_ctx("train", base_params(model="fusion", head="pyramid"), tmp_path)
    p = C.model_params(ctx)
    assert p.encoders == ("xlsr", "wavlmb") and p.head == "fusion"


def test_precision_cpu_mapping():
    p = C.Precision("fp16", "cpu")
    assert p.effective == "bf16" and p.dtype == torch.bfloat16 and p.make_scaler() is None
    assert C.Precision("tf32", "cpu").effective == "fp32"
    with pytest.raises(ValueError):
        C.Precision("int8", "cpu")


def test_precision_cuda_flags_without_gpu():
    """Only global flags are set at construction (no CUDA call), so this runs on a CPU build."""
    want = {"bf16": ("ieee", "tf32", "bfloat16"), "tf32": ("tf32", "tf32", None),
            "fp32": ("ieee", "ieee", None), "fp16": ("ieee", "tf32", "float16")}  # new fp32_precision API
    for mode in ("fp32", "bf16", "tf32", "fp16", "fp32", "bf16"):
        d = C.Precision(mode, "cuda").describe()
        assert (d["matmul_fp32_precision"], d["cudnn_conv_fp32_precision"], d["autocast_dtype"]) == want[mode]
        assert d["grad_scaler"] is (mode == "fp16")


def test_synthetic_waves_and_labels():
    a = C.synth_waves(4, 16000, device="cpu", seed=3)
    b = C.synth_waves(4, 16000, device="cpu", seed=3)
    assert a.shape == (4, 16000) and a.dtype == torch.float32
    assert torch.equal(a, b) and torch.isfinite(a).all()
    peak = a.abs().amax(dim=1)
    assert bool(((peak > 0.29) & (peak < 0.91)).all())
    y = C.balanced_labels(16, device="cpu", seed=1)
    assert y.dtype == torch.int64 and int(y.sum()) == 8


# --------------------------------------------------------------------------- workloads, tiny CPU


@pytest.mark.parametrize("over", [
    {},
    {"head": "gmean"},
    {"precision": "fp16"},
    {"precision": "tf32"},
    {"deterministic": True},
    {"grad_ckpt": True},
    {"micro_batch": 1, "accum": 2},
    {"model": "wavlmb"},
    {"model": "fusion", "seconds": 4.0, "head": "fusion", "encoders": ["xlsr", "wavlmb"]},
], ids=["anchor", "gmean", "fp16", "tf32", "det", "ckpt", "accum", "wavlm", "fusion"])
def test_train_tiny_cpu(tmp_path, over):
    from agbench.workloads import train

    out = train.run(make_ctx("train", base_params(**over), tmp_path))
    assert_finite_metrics(out, 2)
    ex = out["extra"]
    assert ex["encoders"][0]["grad_ckpt"] is bool(over.get("grad_ckpt", False))
    assert ex["accum"] == over.get("accum", 1)
    if over.get("precision") == "fp16":
        assert ex["precision_effective"] == "bf16" and ex["grad_scaler"] is False
    if over.get("model") == "fusion":
        assert [e["key"] for e in ex["encoders"]] == ["xlsr", "wavlmb"]


@pytest.mark.parametrize("head_sets", [1, 2])
def test_frozen_tiny_cpu(tmp_path, head_sets):
    from agbench.workloads import frozen

    params = {"model": "xlsr", "mode": "frozen", "seconds": 2.0, "batch": 2, "precision": "bf16",
              "head": "probe_set", "head_sets": head_sets, "lr_head": 1e-3, "weight_decay": 0.01,
              "seed": 1234, "micro_batch": 1, "accum": 2}
    out = frozen.run(make_ctx("frozen", params, tmp_path))
    assert_finite_metrics(out, 2)
    assert out["extra"]["heads"] == 4 * head_sets
    assert out["extra"]["hidden_states"] == M.TINY_LAYERS + 1


@pytest.mark.parametrize("over", [{}, {"micro_batch": 1, "accum": 2},
                                  {"model": "fusion", "head": "fusion", "seconds": 4.0}])
def test_infer_tiny_cpu(tmp_path, over):
    from agbench.workloads import infer

    out = infer.run(make_ctx("infer", base_params(mode="infer", **over), tmp_path))
    m = assert_finite_metrics(out, 2)
    assert m["loss_first"] is None and out["extra"]["scores_finite"] is True
    assert out["extra"]["clips_scored"] >= 2 * m["n_steps"]


def test_sustain_tiny_cpu(tmp_path):
    from agbench.workloads import sustain

    out = sustain.run(make_ctx("sustain", base_params(mode="sustain", duration_s=2.0, window_s=0.5),
                               tmp_path))
    m = assert_finite_metrics(out, 2)
    assert out["timeseries"] and all(w["clips_per_s"] > 0 for w in out["timeseries"] if w["clips_per_s"])
    assert m["clips_per_s_first_window"] > 0 and m["sustain_ratio"] > 0


def test_e2e_tiny_cpu(tmp_path):
    pytest.importorskip("agbench.data.dataset")
    from agbench.workloads import e2e

    params = base_params(mode="e2e", num_workers=2, rawboost_p=0.5, pin_memory=True,
                         persistent_workers=True)
    out = e2e.run(make_ctx("e2e", params, tmp_path))
    m = assert_finite_metrics(out, 2)
    assert m["num_workers"] == 2 and m["worker_startup_s"] > 0
    assert m["loader_wait_ms_median"] >= 0 and 0 <= m["loader_wait_frac"] <= 1
    assert out["extra"]["loader"]["pin_memory"] is False  # no pinning on the CPU device


def test_resolve_workers():
    from agbench.workloads.e2e import resolve_workers

    assert resolve_workers(4, tiny=False) == 4
    assert resolve_workers(4, tiny=True) == 2
    assert 1 <= resolve_workers("auto", tiny=False) <= 8


def test_every_model_scenario_runs_tiny_on_cpu(tmp_path):
    """All train/frozen/infer rows of the real matrix, as the child would call them (tiny, CPU).
    e2e and sustain are covered above (they spawn workers / run for seconds)."""
    scen = pytest.importorskip("agbench.scenarios")
    from agbench.registry import resolve

    ran = 0
    for sid, s in scen.SCENARIOS.items():
        if s.workload not in ("train", "frozen", "infer"):
            continue
        eff = scen.effective(s, tiny=True, device="cpu")
        mb, acc = scen.first_rung(eff)
        params = {**eff.params, "micro_batch": mb, "accum": acc}
        out = resolve(s.workload)(make_ctx(s.workload, params, tmp_path, scenario_id=sid))
        assert_finite_metrics(out, mb * acc)
        torch.use_deterministic_algorithms(False)
        ran += 1
    assert ran >= 25


@pytest.mark.skipif(os.environ.get("AGBENCH_TEST_REAL_MODELS") != "1",
                    reason="maintainer check: set AGBENCH_TEST_REAL_MODELS=1 with models cached")
@pytest.mark.parametrize("key", sorted(M.MODELS))
def test_real_weights_load_on_cpu(key):
    if not hub.is_cached(key):
        pytest.skip(f"{key} not cached")
    enc = M.load_encoder(key, tiny=False, device="cpu")
    assert enc.is_gradient_checkpointing is False
    assert enc.config.hidden_size == M.MODELS[key].hidden
    assert enc.config.num_hidden_layers == M.MODELS[key].layers
    with torch.no_grad():
        out = enc(torch.randn(1, 16000), output_hidden_states=True)
    assert len(out.hidden_states) == M.MODELS[key].layers + 1
    assert torch.isfinite(out.last_hidden_state).all()


@pytest.mark.skipif(os.environ.get("AGBENCH_TEST_REAL_MODELS") != "1",
                    reason="maintainer check: set AGBENCH_TEST_REAL_MODELS=1 with models cached")
@pytest.mark.parametrize("model", ["w2v2b", "fusion"])
def test_real_weights_train_step_on_cpu(model):
    """One forward + backward of the real classifier (fusion = XLS-R + WavLM Base + FusionHead)."""
    keys = M.encoder_keys(model)
    missing = [k for k in keys if not hub.is_cached(k)]
    if missing:
        pytest.skip(f"{', '.join(missing)} not cached")
    torch.manual_seed(0)
    encs = [M.load_encoder(k, tiny=False, device="cpu") for k in keys]
    head = H.make_head("fusion" if model == "fusion" else "pyramid", [e.config.hidden_size for e in encs])
    clf = C.WaveClassifier(encs, head).train()
    x = C.synth_waves(2, 16000, device="cpu", seed=1)
    y = torch.tensor([0, 1])
    loss = F.cross_entropy(clf(x).float(), y)
    loss.backward()
    assert torch.isfinite(loss)
    no_grad = [n for n, q in clf.named_parameters() if q.requires_grad and q.grad is None]
    # masked_spec_embed is used only by SpecAugment masking, which is off (SPEC 5.1)
    assert all(n.endswith("masked_spec_embed") for n in no_grad), no_grad
    grads = [q.grad for q in clf.parameters() if q.grad is not None]
    assert len(grads) > 100 and all(torch.isfinite(g).all() for g in grads)
