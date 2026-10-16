# antispoof-gpu-bench: build specification (SPEC v1)

Status: specification for four parallel builders (A, B, C, D). Written 2026-10-16.
Every version, index URL, model id and file size below was checked on 2026-10-16 against
the official source named next to it (section 2). Builders implement exactly what this file says.
If something here is wrong or impossible, do not invent a workaround silently: implement the
closest safe behaviour, and record the deviation in your hand-off notes so this file can be updated.

---

## 0. Purpose in one paragraph

A small public benchmark that an AI coding agent can set up and run **with one command** on a
friend's PC (Windows 11 or Linux, NVIDIA RTX 5060 Ti: Blackwell, compute capability 12.0 = `sm_120`,
16 GB or 8 GB, PCIe 5.0 x8). It measures the GPU on workloads that mirror our real speech
anti-spoofing work (ASVspoof 5 spoof detection, fine-tuning self-supervised speech encoders),
plus the CPU data pipeline that feeds them. It uses **synthetic audio only** and **public Hugging
Face weights**. It produces one results file (`results/<gpu>_<date>.jsonl`) and a readable report
(`summary.md` + figures) that we compare with our own GPUs ("RTX 3090 (workstation)",
"RTX A4500 (workstation)", "A100 40 GB (Google Colab)") to decide whether a new GPU is worth buying.

Success = the friend's agent runs `quick` (10-15 min) and optionally `full` (about 32-42 min in total,
hard limit 90 min), nothing crashes the suite, and we receive numbers that answer:
1. How many clips/s does this GPU train W2V2 Base, XLS-R 300M, W2V2 Large (LS-960), WavLM Base and the
   XLS-R + WavLM fusion at our settings, and what does that mean in minutes per ASVspoof 5 epoch?
2. Which of these fit in 16 GB (or 8 GB) at batch 16, and which needed a smaller micro-batch?
3. How fast is scoring (Dev / Eval passes)?
4. Would the friend's CPU keep up (FLAC decode + RawBoost), and does `OPENBLAS_NUM_THREADS=1` matter?
5. Raw GPU facts: bf16/TF32/fp16/fp32 matmul TFLOPS, memory bandwidth, PCIe bandwidth, clocks,
   temperature, power, throttling.

## 1. Ground rules (all builders)

1. **Synthetic data only.** No dataset download, no real audio. Speech-like waveforms are generated
   at 16 kHz (section 5.6).
2. **Public weights only**, downloaded from Hugging Face at runtime, pinned to the commit SHAs in
   section 2.3. We never redistribute weights.
3. **Fresh code.** Write minimal new code. Do not copy code from any private project. RawBoost is
   written fresh from the paper and the official algorithm description (section 5.7).
4. **Privacy (public repo).** No secrets, tokens, e-mail addresses, personal names, host names, IP
   addresses or absolute home-directory paths in any committed file, log template or test fixture.
   Our machines are referred to only as "RTX 3090 (workstation)", "RTX A4500 (workstation)",
   "A100 40 GB (Google Colab)". Results written by the tool must not contain the user name, host
   name, or absolute paths under the home directory (replace the home prefix with `~`; section 8.9).
   `tests/test_privacy.py` (D) enforces generic patterns (section 13.4).
5. **Licence:** MIT for this repo. Third-party credits in `NOTICE` (D).
6. **Python 3.12 only**, Linux x86_64 and Windows x86_64 (AMD64). No macOS/aarch64 support promised.
7. **No `torch.compile`**, no Triton-dependent paths (Triton is not shipped for Windows; our reference
   runs did not compile). No `torchaudio` (section 3.2).
8. **The parent (runner) process never imports torch.** Only child processes touch the GPU.
9. **Console output is ASCII** (Windows consoles may be cp1252). Files are written as UTF-8 explicitly.
10. Keep it small: target < 4,000 lines of Python in `agbench/` excluding tests.

## 2. Verified facts (2026-10-16) that this design depends on

### 2.1 PyTorch, CUDA, driver

| Fact | Value | Source (official) |
|---|---|---|
| Current stable torch | **2.14.1** (PyPI upload 2026-09-30) | pypi.org/pypi/torch/json |
| torch 2.14 stable CUDA builds | CUDA 12.6, **13.0**, 13.2 | github.com/pytorch/pytorch `RELEASE.md` (release compatibility matrix) |
| Blackwell 12.0 in torch 2.14 builds | CUDA 13.0.3 and 13.2.1 builds: "Blackwell(10.0, 12.0+PTX)", PTX on Linux only. **CUDA 12.6 build has no Blackwell.** | same `RELEASE.md`, "PyTorch CUDA Support Matrix" |
| Wheels on index `https://download.pytorch.org/whl/cu130` | `torch-2.14.1+cu130` cp312 `manylinux_2_28_x86_64` and `win_amd64` present | directory listing of the index |
| Wheels on index `https://download.pytorch.org/whl/cu128` | newest is `torch-2.11.0+cu128` (cp312 Linux + Windows). No 2.12+ cu128 builds. | directory listing |
| `cu129` index | Linux-only for 2.12/2.13, nothing for 2.14. **Not used.** | directory listing |
| `cu132` index | `torch-2.14.1+cu132` present (Linux + Windows). Not used by default (no advantage for us). | directory listing |
| torch 2.11 cu128 arch list | CUDA 12.8.1: "Blackwell(10.0, 12.0)" (table titled for 2.9/2.10 on the 2.11 branch; the doctor re-checks at runtime) | `RELEASE.md` on branch `release/2.11` |
| PyPI default torch 2.14.1 on Linux | depends on `cuda-toolkit==13.0.3` (i.e. a cu130 build). Windows PyPI wheel is CPU-only. | PyPI metadata |
| Driver needed for CUDA 13.x | **>= 580** (Linux and Windows) | NVIDIA CUDA Toolkit Release Notes, table "Minimum Required Driver Version for CUDA Minor Version Compatibility" |
| Driver needed for CUDA 12.8 | >= 570.26 (Linux) / >= 570.65 (Windows) | same release notes, Table 4 |
| Wheel download sizes | cu130 Linux 0.55 GB (+ CUDA libs pulled from PyPI, roughly 3 GB); cu130 Windows 2.0 GB; cu128 Windows 2.75 GB; cu128 Linux 0.82 GB | HTTP Content-Length of the wheel URLs |
| torchaudio | last release **2.11.0** (2026-03-23); "transitioned into a maintenance phase"; no build for torch 2.12-2.14 | pypi.org/pypi/torchaudio/json |
| TF32 control in torch 2.14 | `torch.set_float32_matmul_precision("high")` sets `torch.backends.cuda.matmul.fp32_precision="tf32"`; default is `"none"` (= full fp32). cuDNN conv default `torch.backends.cudnn.conv.fp32_precision="tf32"`. | checked in a torch 2.14.1+cpu install |
| OOM exception class | `torch.OutOfMemoryError` | same install |

### 2.2 Other packages (latest on PyPI, 2026-10-16)

| Package | Version | Note |
|---|---|---|
| transformers | 5.20.0 | requires `huggingface-hub>=1.31,<3`, `safetensors>=0.8`; torch extra needs `torch>=2.6` |
| huggingface-hub | 2.2.0 | pulls `hf-xet` on x86_64/AMD64; env vars verified in its `constants.py`: `HF_ENDPOINT`, `HF_HOME`, `HF_HUB_CACHE`, `HF_HUB_OFFLINE`, `HF_HUB_DISABLE_XET`, `HF_HUB_DOWNLOAD_TIMEOUT`, `HF_HUB_ETAG_TIMEOUT`, `HF_HUB_DISABLE_TELEMETRY`, `HF_TOKEN_PATH` |
| numpy | 2.5.4 | requires Python >= 3.12 |
| scipy | 1.18.1 | requires Python >= 3.12 |
| soundfile | 0.14.0 | wheels for `win_amd64` and `manylinux_2_28_x86_64` bundle libsndfile 1.2.2 (FLAC supported) |
| nvidia-ml-py | 13.615.71 | provides module `pynvml` (do **not** use the deprecated `pynvml` PyPI package) |
| psutil | 7.2.2 | process-tree kill, CPU/RAM info |
| threadpoolctl | 3.7.0 | report BLAS library and thread counts |
| matplotlib | 3.11.2 | report figures (Agg backend) |
| pytest | 9.1.1 | dev group |
| uv | 0.13.1 latest (0.10+ is fine) | installer; `uv run --no-sync` exists (`UV_NO_SYNC`) |
| GitHub Actions | `actions/checkout@v7` (v7.0.1), `astral-sh/setup-uv@v10` (v10.3.0) | GitHub releases API |

Verified in a scratch CPU install (torch 2.14.1+cpu, transformers 5.20.0): tiny random
`Wav2Vec2Model` and `WavLMModel` configs run forward+backward under bf16 autocast with
`output_hidden_states=True`; `facebook/wav2vec2-base` and `facebook/wav2vec2-xls-r-300m` load from
their `pytorch_model.bin` (no safetensors file exists in these repos) without `accelerate`.
`Wav2Vec2Model._supports_sdpa = True` (default attention = `sdpa`); `WavLMModel._supports_sdpa = False`
(attention = `eager`).

**Trap found and confirmed:** `facebook/wav2vec2-base/config.json` contains the legacy key
`"gradient_checkpointing": true`, and transformers 5.20 still **turns activation checkpointing ON** when
loading it (`model.is_gradient_checkpointing == True`). Our reference W2V2 Base run had checkpointing
explicitly **off**. Therefore every model load must call `model.gradient_checkpointing_disable()` and
assert `model.is_gradient_checkpointing is False` (unless the scenario asks for checkpointing). The
other three configs have it false or absent.

### 2.3 Models (Hugging Face, verified via the public HF API)

| key | HF id | pinned revision (commit SHA) | weights file, bytes | arch | pre-training data | role |
|---|---|---|---|---|---|---|
| `w2v2b` | `facebook/wav2vec2-base` | `0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8` | `pytorch_model.bin`, 380,267,417 | 12 layers, 768, group-norm conv encoder, post-LN; 94.4 M params in `Wav2Vec2Model` | LibriSpeech 960 h | main workload |
| `xlsr` | `facebook/wav2vec2-xls-r-300m` | `1a640f32ac3e39899438a2931f9924c02f080a54` | `pytorch_model.bin`, 1,269,737,156 | 24 layers, 1024, layer-norm conv encoder, pre-LN (`do_stable_layer_norm`); 315.4 M params | 436 k h multilingual (includes MLS, so not allowed in the ASVspoof 5 open condition; benchmarked because we use it) | large workload |
| `w2v2l` | `facebook/wav2vec2-large` | `312b2410566b698c7a649068d413b2067848bd75` | `pytorch_model.bin`, 1,269,691,161 | 24 layers, 1024, group-norm conv encoder, post-LN (`do_stable_layer_norm: false`, `feat_extract_norm: "group"`) | **LibriSpeech 960 h only, no fine-tuning** (model card: "pretrained on 16kHz sampled speech audio ... should be fine-tuned"; dataset tag `librispeech_asr`) | compliant large model |
| `wavlmb` | `microsoft/wavlm-base` | `efa81aae7ff777e464159e0f877d54eac5b84f81` | `pytorch_model.bin`, 377,617,425 | 12 layers, 768, gated relative position bias, eager attention | LibriSpeech 960 h (model card) | fusion partner, own scenario |

Rejected alternatives for the compliant large model: `facebook/wav2vec2-large-lv60` (LibriLight 60 k h,
stable layer norm), `facebook/wav2vec2-large-960h` (CTC fine-tuned for ASR). Total download for the
four models: **3,297,313,159 bytes (3.30 GB, 3.07 GiB)** plus small `config.json` /
`preprocessor_config.json` files. Fetch only `config.json`, `preprocessor_config.json`, `pytorch_model.bin`.

Our fusion reference run used XLSR-53 (`facebook/wav2vec2-large-xlsr-53`) + WavLM Base. XLSR-53 and
XLS-R 300M have identical architecture configs (hidden 1024, 24 layers, 16 heads, FFN 4096, same conv
stack, layer-norm conv encoder, stable layer norm; checked by comparing both `config.json` files),
so compute per clip is the same. The benchmark uses XLS-R 300M for the fusion.

### 2.4 RTX 5060 Ti (public spec figures, approximate)

128-bit GDDR7 at 28 Gb/s = **448 GB/s**; 4,608 CUDA cores; FP32 ~23.7 TFLOPS; PCIe 5.0 x8; 180 W;
16 GB or 8 GB. NVIDIA does not publish a per-precision tensor table for this SKU; ~95 TFLOPS dense
FP16 (FP16 accumulate) is derived from the "759 AI TOPS" (FP4, sparse) figure. GeForce cards run
FP16/BF16 with FP32 accumulate (what PyTorch uses) at half that rate, so about **47 TFLOPS dense bf16**
(derived, not official). RTX 3090 for comparison: 936 GB/s, 35.6 TFLOPS FP32, 71 TFLOPS dense
bf16 with FP32 accumulate. Treat all of these as approximate; the benchmark measures the real values.

### 2.5 RawBoost licence and defaults

Official repository `TakHemlata/RawBoost-antispoofing` is **MIT** ("Copyright (c) 2021 Hemlata"),
as is `TakHemlata/SSL_Anti-spoofing`. We still write fresh code (rule 1.3) and credit the paper and
repo in `NOTICE`. Algorithm 4 = LnL convolutive noise -> ISD additive noise -> SSI additive noise,
in series. Defaults from the official `main.py`: `nBands=5, minF=20, maxF=8000, minBW=100,
maxBW=1000, minCoeff=10, maxCoeff=100, minG=0, maxG=0, minBiasLinNonLin=5, maxBiasLinNonLin=20,
N_f=5, P=10, g_sd=2, SNRmin=10, SNRmax=40`, sample rate 16 kHz.

### 2.6 Windows: system-memory fallback

NVIDIA Windows drivers since 536.40 may spill CUDA allocations into shared system memory when VRAM is
full instead of failing (very slow, looks like a slow GPU instead of an OOM). Driver 546.01+ has the
NVIDIA Control Panel setting "CUDA - Sysmem Fallback Policy" -> "Prefer No Sysmem Fallback".
The harness guards against silent spill itself (section 8.6); AGENTS.md also tells the agent how to
set the policy for `python.exe`.

## 3. Pinned environment (builder A owns `pyproject.toml`, `uv.lock`, `.python-version`)

### 3.1 Variants

| uv extra | torch | index | needs driver | when to use |
|---|---|---|---|---|
| `cu130` (default) | `torch==2.14.1` | `https://download.pytorch.org/whl/cu130` | >= 580 | normal case for an RTX 5060 Ti |
| `cu128` (fallback) | `torch==2.11.0` | `https://download.pytorch.org/whl/cu128` | >= 570.26 Linux / >= 570.65 Windows | driver 570-579 that cannot be updated |
| `cpu` (CI/tests) | `torch==2.14.1` | `https://download.pytorch.org/whl/cpu` | none | GitHub Actions, CPU-only tests |

`cu126` must never be used on Blackwell (no `sm_120` kernels; gives "no kernel image is available").

### 3.2 pyproject.toml (normative skeleton; A fills in details)

```toml
[project]
name = "agbench"
version = "0.1.0"
description = "Synthetic-audio GPU benchmark for speech anti-spoofing workloads"
requires-python = ">=3.12,<3.13"
license = "MIT"
dependencies = [
  "transformers==5.20.0",
  "huggingface-hub>=1.31,<3",
  "safetensors>=0.8",
  "numpy==2.5.4",
  "scipy==1.18.1",
  "soundfile==0.14.0",
  "nvidia-ml-py>=12.570",
  "psutil>=7.0",
  "threadpoolctl>=3.5",
  "matplotlib==3.11.2",
]

[project.optional-dependencies]
cu130 = ["torch==2.14.1"]
cu128 = ["torch==2.11.0"]
cpu   = ["torch==2.14.1"]

[dependency-groups]
dev = ["pytest==9.1.1"]

[tool.uv]
conflicts = [[{ extra = "cu130" }, { extra = "cu128" }, { extra = "cpu" }]]
environments = [
  "sys_platform == 'linux' and platform_machine == 'x86_64'",
  "sys_platform == 'win32' and platform_machine == 'AMD64'",
]

[tool.uv.sources]
torch = [
  { index = "pytorch-cu130", extra = "cu130" },
  { index = "pytorch-cu128", extra = "cu128" },
  { index = "pytorch-cpu",   extra = "cpu" },
]

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- `.python-version` contains `3.12`. `uv.lock` is committed and must resolve for both environments and
  all three extras (`uv lock` then `uv sync --locked --extra cpu` must work on CI).
- torchaudio is **not** a dependency: its last release (2.11.0) pins torch 2.11 and it is in maintenance
  mode. FLAC decoding uses soundfile; filters use scipy.
- No `accelerate` (not needed to load these models; verified).

### 3.3 Setup and run commands (these exact commands go into AGENTS.md)

```
# Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

git clone <repo-url> antispoof-gpu-bench
cd antispoof-gpu-bench
uv sync --locked --extra cu130          # or --extra cu128 for driver 570-579
uv run --no-sync python -m agbench doctor
uv run --no-sync python -m agbench fetch
uv run --no-sync python -m agbench quick
uv run --no-sync python -m agbench full     # optional, continues where quick stopped
```

`--no-sync` matters: a plain `uv run` re-syncs to the project default (no extra) and would remove
torch. If torch is missing, `agbench` prints: `torch is not installed: run "uv sync --locked --extra cu130"`.

Disk: about 25 GB free recommended (venv 5-7 GB, uv cache of similar size, models 3.3 GB,
synthetic corpus ~0.1 GB, results < 50 MB).

## 4. Package layout and ownership

Exactly one builder owns each file. Never edit a file you do not own; if you need a change there,
put a note in your hand-off. `SPEC.md` is owned by the spec author.

```
antispoof-gpu-bench/
  SPEC.md                         spec author
  pyproject.toml  uv.lock  .python-version  .gitignore          A
  agbench/__init__.py             A   (__version__ = "0.1.0")
  agbench/__main__.py             A   (calls agbench.cli.main())
  agbench/cli.py                  A   argparse CLI (section 7)
  agbench/runner.py               A   suite execution, subprocess, timeout, ladder, resume, budget
  agbench/child.py                A   child entry point (python -m agbench.child)
  agbench/context.py              A   RunContext, MeasureCfg dataclasses
  agbench/errors.py               A   SkipScenario, ScenarioError
  agbench/registry.py             A   workload key -> "module:function" strings (lazy import)
  agbench/scenarios.py            A   the scenario matrix as data (section 6) + suite order
  agbench/schema.py               A   record builders, validation, load_records()
  agbench/results_schema.json     A   JSON Schema (draft 2020-12) for scenario records
  agbench/env.py                  A   capture_env(), gpu_slug(), env_fingerprint(), doctor()
  agbench/timing.py               A   measure_loop(), measure_for_duration(), cuda memory stats
  agbench/constants.py            A   ASV5 clip counts, sample rate, defaults
  agbench/util.py                 A   atomic write, path sanitizer, process-tree kill, ascii print
  agbench/models.py               B   model registry (section 2.3), load_encoder(), tiny configs
  agbench/hub.py                  B   prefetch(), resolve(), cache checks, mirror/offline logic
  agbench/heads.py                B   PyramidMeanHead, GlobalMeanHead, FusionHead, frozen probe heads
  agbench/workloads/__init__.py   B   (empty)
  agbench/workloads/common.py     B   precision contexts, optimizer, synthetic GPU batches, step builders
  agbench/workloads/train.py      B   run(ctx)  (train.*)
  agbench/workloads/frozen.py     B   run(ctx)  (frozen.*)
  agbench/workloads/infer.py      B   run(ctx)  (infer.*)
  agbench/workloads/e2e.py        B   run(ctx)  (e2e.*; uses C's dataset/loader)
  agbench/workloads/sustain.py    B   run(ctx)  (sustain.*)
  agbench/workloads/basics.py     C   run_matmul(ctx), run_membw(ctx), run_pcie(ctx)
  agbench/workloads/cpu_pipeline.py C run_decode(ctx), run_rawboost(ctx), run_loader(ctx)
  agbench/data/__init__.py        C   (empty)
  agbench/data/synth.py           C   speechlike(), ensure_corpus()
  agbench/data/rawboost.py        C   lnl(), isd(), ssi(), rawboost(algo=4)
  agbench/data/dataset.py         C   SynthFlacDataset, make_loader(), worker_init_fn
  agbench/telemetry.py            C   GpuSampler (NVML, nvidia-smi fallback)
  agbench/report.py               D   build_report()
  agbench/plots.py                D   figure functions (matplotlib Agg)
  references/reference_gpus.json  D
  references/README.md            D
  README.md  AGENTS.md  CLAUDE.md  LICENSE  NOTICE             D
  scripts/bench.sh  scripts/bench.ps1                          D   optional one-line wrappers
  .github/workflows/ci.yml        D
  tests/conftest.py               D
  tests/test_schema.py            D
  tests/test_report.py            D
  tests/test_cli_e2e.py           D   tiny CPU suite end to end through real subprocesses
  tests/test_fallback_sim.py      D   simulated OOM ladder through the real runner
  tests/test_privacy.py           D
  tests/fixtures/*.jsonl          D   hand-written sample result files
  tests/test_runner_unit.py       A   (own unit tests)
  tests/test_workloads_unit.py    B   (own unit tests, tiny models, CPU)
  tests/test_data_unit.py         C   (own unit tests)
  results/.gitkeep                D   (results/* is git-ignored except .gitkeep)
```

`.gitignore` (A): `.venv/`, `__pycache__/`, `*.egg-info/`, `.pytest_cache/`, `cache/`, `results/*`,
`!results/.gitkeep`, `.privacy_denylist`.

## 5. Workload definitions (shared semantics)

### 5.1 Common training recipe (mirrors our reference runs)

- Full fine-tune of the encoder + head. Model weights fp32; optimizer `torch.optim.AdamW` with default
  implementation (foreach), two parameter groups: encoder lr **1e-5**, head lr **1e-4**, weight decay
  **0.01** on both, betas default. No LR schedule (constant; schedules do not change step time).
- Loss: `CrossEntropyLoss` on `logits.float()`, labels random 0/1 (balanced, fixed seed).
- Gradient clipping `clip_grad_norm_(all params, 1.0)` every optimizer step (reference runs used 1.0).
  With fp16 + GradScaler: `scaler.unscale_(opt)` before clipping.
- **LayerDrop 0** (`layerdrop=0.0`) for every model, **SpecAugment off** (`apply_spec_augment=False`),
  dropout as in each config (model in train mode).
- **Activation checkpointing off** unless the scenario variant is `.ckpt` (then
  `gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`). Always assert
  the actual flag after load (section 2.2 trap).
- Attention: transformers default (`sdpa` for wav2vec2 models, `eager` for WavLM). Record the actual
  `config._attn_implementation` in `extra`.
- Input: raw waveform `[B, T]` float32 on the GPU, per-utterance normalisation inside the model forward:
  `x = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-7)`.
  No attention mask (fixed length, no padding).
- Encoder output for train/infer: `last_hidden_state` `[B, T', D]` (T' = 299 frames for 6 s).
- One **step** = one optimizer step over the effective batch (= `micro_batch * accum`, normally 16),
  including all micro-batch forwards/backwards, clipping, `optimizer.step()` and
  `zero_grad(set_to_none=True)`. ms/step and clips/s always refer to the effective batch.
- `torch.backends.cudnn.benchmark = False` (PyTorch default; mirrors references).
- Deterministic mode only in the reference-numerics rows (`.det`, `e2e.w2v2b.6s.bs16.bf16`,
  `e2e.xlsr.6s.bs16.bf16`): strict `torch.use_deterministic_algorithms(True)` (warn-only only as a recorded
  fallback), `torch.backends.cudnn.deterministic = True`, env `CUBLAS_WORKSPACE_CONFIG=:4096:8` (set by the
  child before torch import, from the scenario `env`), cuDNN conv TF32 off and flash SDPA off for clips longer
  than 4 s, as in the reference runs (section 18).

### 5.2 Precision modes

| `precision` | meaning |
|---|---|
| `bf16` | `torch.autocast("cuda", dtype=torch.bfloat16)`; no GradScaler; matmul fp32 precision left at default ("highest"); cuDNN conv TF32 default |
| `fp16` | `torch.autocast("cuda", dtype=torch.float16)` + `torch.amp.GradScaler("cuda")` |
| `tf32` | no autocast, fp32 tensors, `torch.set_float32_matmul_precision("high")` (TF32 matmul), cuDNN conv TF32 default |
| `fp32` | strict fp32: matmul "highest" and `torch.backends.cudnn.conv.fp32_precision = "ieee"`; used only in `gpu.matmul` |

On CPU (tiny/CI mode) autocast uses `device_type="cpu"` with bf16; fp16 and GradScaler are mapped to
bf16 without scaler and recorded in `extra.precision_effective`.

### 5.3 Heads (B, `agbench/heads.py`)

- `PyramidMeanHead(dim, bins=(1, 2, 4), proj=256, n_classes=2)`: input `[B, T', D]`. For each bin
  count `n`, split time into `n` intervals with the same boundaries as `adaptive_avg_pool1d`
  (`start = floor(i*T/n)`, `end = ceil((i+1)*T/n)`), mean each interval (slice + mean, deterministic);
  concatenate all 1+2+4 = 7 means -> `[B, 7*D]` -> `Linear(7*D, 256)` -> `Linear(256, 2)`. No
  activation between the two linears (mirrors our head). This is the default head (`head="pyramid"`).
- `GlobalMeanHead(dim, proj=256)`: mean over time -> `Linear(D, 256)` -> `Linear(256, 2)` (`head="gmean"`).
- `FusionHead(dims=(1024, 768))`: one pyramid projection per encoder (`Linear(7*D_i, 256)`), concatenate
  -> `Linear(512, 2)`. Both encoders get the same waveform.
- Frozen probe set (`FrozenHeadSet(dim, n_layers_plus_1)`, one "set" = 4 heads trained jointly, loss = sum):
  1. pyramid head on the last hidden state;
  2. softmax-weighted sum of all hidden states (learned weights) -> pyramid head;
  3. one linear probe `Linear(D, 2)` per hidden state on its time mean (L+1 probes);
  4. MLP on concatenated time means of all hidden states: `Linear(D*(L+1), 256)` -> GELU -> `Linear(256, 2)`.
  `head_sets=10` (variant `.h40`) instantiates 10 independent sets = 40 heads (mirrors our frozen screens).
  Heads optimizer: AdamW lr 1e-3, wd 0.01.

### 5.4 Mode semantics

| mode | encoder | grad | what one step does |
|---|---|---|---|
| `train` | train mode | full fine-tune | section 5.1 |
| `frozen` | `eval()`, `requires_grad_(False)` | heads only | `torch.no_grad()` encoder forward with `output_hidden_states=True` (13 states for base, 25 for large) under autocast, then heads forward/backward/AdamW step |
| `infer` | `eval()` | none | copy the batch from pinned host memory (`non_blocking=True`), `torch.inference_mode()` + autocast, pyramid head, softmax score; scores stay on GPU (no per-step sync) |
| `e2e` | as `train` | full | batch comes from a real `DataLoader` over the synthetic FLAC corpus (section 5.8); time blocked in `next(iterator)` is recorded as `loader_wait_ms` |
| `sustain` | as `train` | full | `train` step repeated for `duration_s` (300 s); clips/s per `window_s` (30 s) window |

### 5.5 Measurement (A: `agbench/timing.py`)

`MeasureCfg` defaults (normal): `warmup_steps=5, warmup_min_s=5.0, warmup_max_s=60.0, measure_steps=20,
measure_min_s=10.0, measure_max_s=60.0`. Warm-up runs until it has done at least `warmup_steps` and
`warmup_min_s` (capped by `warmup_max_s`). Measurement runs until at least `measure_steps` and
`measure_min_s` (capped by `measure_max_s`, minimum 5 steps). Loop control uses host
`time.perf_counter()`. Tiny mode: `warmup_steps=1, warmup_min_s=0, measure_steps=3, measure_min_s=0`.

On CUDA each measured step is bracketed by `torch.cuda.Event(enable_timing=True)` records; one
`torch.cuda.synchronize()` at the end; per-step ms from `start.elapsed_time(end)`. Wall clock: synchronize
before the first measured step and after the last. Reported:

- `step_ms_median`, `step_ms_p10`, `step_ms_p90`, `step_ms_mean`, `step_ms_cv` (std/mean), `n_steps`
- `clips_per_s = items_per_step * 1000 / step_ms_median` (primary number everywhere)
- `clips_per_s_wall = items_per_step * n_steps / wall_s`
- `warmup_steps_done`, `warmup_s`, `measure_s`

The child (not the workload) resets `torch.cuda.reset_peak_memory_stats()` right before calling the
workload and reads `max_memory_allocated` / `max_memory_reserved` after it returns, so the peak includes
warm-up. The workload never needs to touch memory statistics.

### 5.6 Synthetic speech-like audio (C: `agbench/data/synth.py`)

`speechlike(n_samples, rng, sr=16000) -> np.ndarray[float32]`, vectorised, about 10 ms of CPU per second
of audio. Recipe: alternating voiced (60-75 %), unvoiced (10-20 %) and pause (10-25 %) segments of
50-400 ms; voiced = harmonic series of a random-walk f0 (90-250 Hz) with 1/k amplitudes up to 4 kHz,
passed through three resonators (F1 300-900 Hz, F2 900-2500 Hz, F3 2500-3500 Hz; `scipy.signal.lfilter`
biquads); unvoiced = high-passed noise; syllabic amplitude envelope 3-6 Hz; noise floor -50 dBFS; peak
normalised to a random 0.3-0.9. Content does not change GPU compute; it only makes FLAC compression and
decode cost realistic.

`ensure_corpus(cache_dir, n_files=256, seed=20261016, min_s=4.0, max_s=10.0) -> Path(manifest.json)`:
writes 16-bit PCM FLAC files (`soundfile.write(..., format="FLAC", subtype="PCM_16")`) with uniform random
durations in [4, 10] s and balanced random labels, under `cache/synth_v1/` (override with env
`AGBENCH_CACHE`). Idempotent: if the manifest exists and every file has the recorded size, do nothing.
Tiny mode: 8 files of 1-2 s under `cache/synth_tiny_v1/`.

### 5.7 RawBoost (C: `agbench/data/rawboost.py`)

Fresh implementation following Tak et al., "RawBoost: A Raw Data Boosting and Augmentation Method
applied to Automatic Speaker Verification Anti-Spoofing", ICASSP 2022, and the official algorithm
description (repo MIT). Functions take an explicit `rng: np.random.Generator` (no global RNG):

- `notch_fir(rng, nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, sr)`: per band draw
  centre frequency, bandwidth and an odd tap count; design with `scipy.signal.firwin(..., window="hamming",
  fs=sr)` and convolve the band filters together; gain-normalise using the peak of
  `scipy.signal.freqz` and a random gain in dB.
- `lnl(x, rng, N_f=5, ...)`: sum over orders i = 1..N_f of FIR-filtered `x**i` (gain range shifted by the
  lin/non-lin bias for i >= 2), remove mean, normalise if peak > 1. FIR filtering via `scipy.signal.lfilter`
  with half-length delay compensation (as in the official description).
- `isd(x, rng, P=10, g_sd=2)`: choose up to P % of samples, add `g_sd * x[p] * (u1 * u2)` with u in [-1, 1].
- `ssi(x, rng, SNRmin=10, SNRmax=40, ...)`: white Gaussian noise, coloured by a notch FIR, peak-normalised,
  scaled to a random SNR **using `np.linalg.norm(noise, 2)` and `np.linalg.norm(x, 2)`**. Keep these
  BLAS-backed norms: they are the calls that wake extra BLAS threads in loader workers, which is the
  effect `OPENBLAS_NUM_THREADS=1` vs default is meant to expose.
- `rawboost(x, rng, algo=4)` = `ssi(isd(lnl(x)))`. Algorithms 1, 2, 3 exposed for tests. Output float32.

### 5.8 Dataset and loader (C: `agbench/data/dataset.py`)

`SynthFlacDataset(manifest_path, seconds=6.0, train=True, rawboost_p=0.0, seed=1234)`:
`__getitem__(i)` reads the whole FLAC with `soundfile.read(path, dtype="float32")`; if shorter than
`seconds`, tile (loop) to at least that length; crop `seconds*16000` samples (random offset when
`train`, centre when not); with probability `rawboost_p` apply `rawboost(algo=4)`; per-item RNG seeded
from `(seed, epoch, i)`; returns `(torch.from_numpy(wave_float32), torch.tensor(label, dtype=torch.int64))`.

Windows `spawn` safety: the dataset class and `worker_init_fn` live at module level in this file, the
module has no import-time side effects, holds only picklable state (paths, numbers), never touches CUDA
in workers, no lambdas. `make_loader(dataset, batch_size, num_workers, pin_memory, persistent_workers,
prefetch_factor=2) -> torch.utils.data.DataLoader` passes `multiprocessing_context="spawn"` on every OS
when `num_workers > 0` (same behaviour on Linux and Windows; mirrors our reference runs).

## 6. Scenario matrix

ID grammar: `<mode>.<model>.<seconds>s.bs<batch>.<precision>[.<variant>]`, `gpu.<name>`,
`cpu.<name>[.w<N>].<blas>`. Model keys: `w2v2b`, `xlsr`, `w2v2l`, `wavlmb`, `fusion` (= `xlsr` + `wavlmb`).
Q = in `quick`, F = in `full` (full is a superset of quick). Ladder L16 = micro-batch x accumulation
attempts `16x1 -> 8x2 -> 4x4 -> 2x8` (the brief asked for 8x2 then 4x4; 2x8 is added as a last rung for
8 GB cards); L8 = `8x1 -> 4x2 -> 2x4`; "-" = no ladder (an OOM is the result).

Memory column: estimated peak **allocated** GiB at the requested batch, bf16 unless stated. Basis: W2V2
Base 6 s bs16 observed 5.25 GiB allocated / 6.63 GiB reserved (RTX 3090, torch 2.2.2): ~1.41 GiB weights +
grads + AdamW states, plus ~0.64 GiB activations per second of audio at bs16. XLS-R 300M 6 s bs16: ~15 GB
in use observed on the RTX A4500 (incl. context). XLSR-53 + WavLM Base 4 s bs16: 13.6 GiB allocated /
14.9 GiB reserved observed (RTX 3090). XLSR-53 frozen probe 4 s bs16, all hidden states: 2.58 GiB observed.
Other rows are scaled estimates (est).
Usable VRAM under the harness cap (section 8.6): 16 GB card ~14.9 GiB headless Linux, ~14.0 GiB Windows
with a desktop; 8 GB card ~7.0 / ~6.3 GiB. Outcome words: fits / tight (within ~10 % of the cap) /
-> NxM (expected rung) / OOM (recorded, by design).

Time column `est s` = estimated wall seconds per scenario on an RTX 5060 Ti 16 GB, Linux, including
process start (~8 s), model load, warm-up, measurement and any expected OOM attempt (section 6.3).

### 6.1 GPU basics, CPU pipeline, end-to-end

| id | workload | key params | suites | mem GiB | 16 GB | 8 GB | est s |
|---|---|---|---|---|---|---|---|
| `gpu.matmul` | basics.matmul | square n in {4096, 8192}; dtypes bf16, fp16, tf32, fp32; TFLOPS = 2n^3/t, median of 10 timed batches each >= 0.2 s | Q F | 0.8 | fits | fits | 25 |
| `gpu.membw` | basics.membw | 1 GiB fp32 buffers: copy (2x bytes/t), read-reduce `sum` (bytes/t), triad `add(a,b,out=c)` (3x bytes/t) | Q F | 3.0 | fits | fits | 12 |
| `gpu.pcie` | basics.pcie | 256 MiB: pinned H2D, pinned D2H, pageable H2D; NVML link gen/width read during transfer | Q F | 0.6 | fits | fits | 12 |
| `e2e.w2v2b.6s.bs16.bf16` | e2e | 2 spawn workers, RawBoost off, `pin_memory=False`, `persistent_workers=False` (exactly our reference loader); ladder L16 | Q F | 5.3 | fits | fits | 34 |
| `e2e.w2v2b.6s.bs16.bf16.rawboost` | e2e | RawBoost algo 4 p=0.5, workers = min(8, max(1, physical_cores-1)), `OPENBLAS_NUM_THREADS=1`, pinned, persistent; ladder L16 | F | 5.3 | fits | fits | 40 |
| `e2e.xlsr.6s.bs16.bf16` | e2e | as the first e2e row, ladder L16 | F | ~14-15 | tight, likely -> 8x2 | -> 2x8 | 61 |
| `cpu.decode` | cpu.decode | decode 6 s crops from the corpus, 1 process; also full-file decode ms per audio second | Q F | - | - | - | 15 |
| `cpu.rawboost.blas1` | cpu.rawboost | algo 4 on 6 s crops, p=1, 1 process, env `OPENBLAS_NUM_THREADS=1`; per-stage ms (LnL/ISD/SSI) and CPU-time/wall ratio | Q F | - | - | - | 20 |
| `cpu.rawboost.blasdefault` | cpu.rawboost | same, BLAS thread env vars removed | Q F | - | - | - | 20 |
| `cpu.loader.w4.blas1` / `.w8.blas1` | cpu.loader | DataLoader, bs16, 6 s, RawBoost p=0.5, N spawn workers, 2 warm-up + 30 measured batches (cap 40 s), no GPU | Q F | - | - | - | 25 each |
| `cpu.loader.w4.blasdefault` | cpu.loader | same with default BLAS threads | Q F | - | - | - | 25 |
| `cpu.loader.w1/.w2/.w12/.w16.blas1` | cpu.loader | as above | F | - | - | - | 25 each |
| `cpu.loader.w1/.w8.blasdefault` | cpu.loader | as above | F | - | - | - | 25 each |

`cpu.loader.wN.*` with N > logical cores -> status `skipped` (reason `workers_exceed_cores`).
BLAS env semantics: `blas1` sets `OPENBLAS_NUM_THREADS=1` only; `blasdefault` removes
`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `GOTO_NUM_THREADS` from the child env.
Both record `threadpoolctl.threadpool_info()` (library, version, num_threads).

### 6.2 Model scenarios

| id | mode | model | s | bs | prec | head / variant | ladder | suites | mem GiB | 16 GB | 8 GB | est s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `train.w2v2b.6s.bs16.bf16` (anchor) | train | w2v2b | 6 | 16 | bf16 | pyramid | L16 | Q F | 5.3 | fits | fits (tight on Windows) | 30 |
| `train.w2v2b.10s.bs16.bf16` | train | w2v2b | 10 | 16 | bf16 | pyramid | L16 | Q F | 7.8 | fits | -> 8x2 | 30 |
| `train.w2v2b.2s.bs16.bf16` | train | w2v2b | 2 | 16 | bf16 | pyramid | L16 | F | 2.7 | fits | fits | 30 |
| `train.w2v2b.4s.bs16.bf16` | train | w2v2b | 4 | 16 | bf16 | pyramid | L16 | F | 4.0 | fits | fits | 30 |
| `train.w2v2b.8s.bs16.bf16` | train | w2v2b | 8 | 16 | bf16 | pyramid | L16 | F | 6.5 | fits | tight / -> 8x2 | 30 |
| `train.w2v2b.6s.bs16.bf16.gmean` | train | w2v2b | 6 | 16 | bf16 | global mean | L16 | F | 5.3 | fits | fits | 30 |
| `train.w2v2b.6s.bs16.tf32` | train | w2v2b | 6 | 16 | tf32 | pyramid | L16 | F | ~8.5 est | fits | -> 8x2 | 31 |
| `train.w2v2b.6s.bs16.fp16` | train | w2v2b | 6 | 16 | fp16+GradScaler | pyramid | L16 | F | 5.3 | fits | fits | 30 |
| `train.w2v2b.6s.bs16.bf16.det` | train | w2v2b | 6 | 16 | bf16 | numerics of our reference runs: strict deterministic algorithms, TF32 off, flash SDPA off (section 18) | L16 | F | 5.3 | fits | fits | 32 |
| `train.w2v2b.6s.bs16.bf16.ckpt` | train | w2v2b | 6 | 16 | bf16 | activation checkpointing on (what the legacy config gives by default) | L16 | F | ~2.5 est | fits | fits | 35 |
| `train.w2v2b.6s.bs8.bf16` | train | w2v2b | 6 | 8 | bf16 | pyramid | L8 | F | 3.3 | fits | fits | 30 |
| `train.w2v2b.6s.bs32.bf16` | train | w2v2b | 6 | 32 | bf16 | pyramid | - | F | 9.1 | fits | OOM | 31 |
| `train.xlsr.6s.bs16.bf16` | train | xlsr | 6 | 16 | bf16 | pyramid | L16 | Q F | ~14-15 | tight, likely -> 8x2 (~9.5) | -> 4x4 (7.1, tight) / 2x8 (5.9) | 57 |
| `train.xlsr.6s.bs16.tf32` | train | xlsr | 6 | 16 | tf32 | pyramid | L16 | F | ~22 est | -> 8x2 (13.5, tight) / 4x4 | -> 2x8 (6.9, tight) or OOM | 94 |
| `train.xlsr.6s.bs16.bf16.ckpt` | train | xlsr | 6 | 16 | bf16 | activation checkpointing on | L16 | F | ~7 est | fits | tight / -> 8x2 | 44 |
| `train.w2v2l.6s.bs16.bf16` | train | w2v2l | 6 | 16 | bf16 | pyramid | L16 | Q F | ~12-13 est | fits (tight on Windows) | -> 4x4 (6.6) / 2x8 | 56 |
| `train.wavlmb.6s.bs16.bf16` | train | wavlmb | 6 | 16 | bf16 | pyramid (eager attention) | L16 | F | ~6.5 est | fits | tight / -> 8x2 | 30 |
| `train.fusion.4s.bs16.bf16` | train | xlsr+wavlmb | 4 | 16 | bf16 | FusionHead | L16 | Q F | 13.6 | tight, likely -> 8x2 (~9.9) | -> 2x8 (7.0, tight) or OOM | 59 |
| `frozen.xlsr.6s.bs16.bf16` | frozen | xlsr | 6 | 16 | bf16 | 1 head set (4 heads) | L16 | Q F | ~3.9 | fits | fits | 33 |
| `frozen.xlsr.6s.bs16.bf16.h40` | frozen | xlsr | 6 | 16 | bf16 | 10 head sets (40 heads) | L16 | F | ~4.2 | fits | fits | 33 |
| `frozen.w2v2b.6s.bs16.bf16` | frozen | w2v2b | 6 | 16 | bf16 | 1 head set | L16 | F | ~1.6 | fits | fits | 30 |
| `frozen.w2v2l.6s.bs16.bf16` | frozen | w2v2l | 6 | 16 | bf16 | 1 head set | L16 | F | ~3.9 | fits | fits | 33 |
| `infer.w2v2b.6s.bs16.bf16` | infer | w2v2b | 6 | 16 | bf16 | pyramid | L16 | Q F | ~1.0 | fits | fits | 30 |
| `infer.xlsr.6s.bs16.bf16` | infer | xlsr | 6 | 16 | bf16 | pyramid | L16 | Q F | ~2.2 | fits | fits | 33 |
| `infer.w2v2b.6s.bs4.bf16` | infer | w2v2b | 6 | 4 | bf16 | pyramid | - | F | ~0.6 | fits | fits | 30 |
| `infer.xlsr.6s.bs4.bf16` | infer | xlsr | 6 | 4 | bf16 | pyramid | - | F | ~1.5 | fits | fits | 33 |
| `infer.w2v2l.6s.bs16.bf16` | infer | w2v2l | 6 | 16 | bf16 | pyramid | L16 | F | ~2.2 | fits | fits | 33 |
| `infer.wavlmb.6s.bs16.bf16` | infer | wavlmb | 6 | 16 | bf16 | pyramid | L16 | F | ~1.1 | fits | fits | 30 |
| `infer.fusion.4s.bs16.bf16` | infer | xlsr+wavlmb | 4 | 16 | bf16 | FusionHead | L16 | F | ~2.6 | fits | fits | 35 |
| `sustain.w2v2b.6s.bs16.bf16` | sustain | w2v2b | 6 | 16 | bf16 | 300 s, 30 s windows, telemetry series kept | L16 | F | 5.3 | fits | fits | 320 |
| `train.w2v2b.6s.bs16.bf16.repeat` | train | w2v2b | 6 | 16 | bf16 | anchor repeated last, after the thermal soak | L16 | F | 5.3 | fits | fits | 30 |

Training scenarios use the recipe of 5.1; inference ladders split the effective batch into micro-batches
the same way (`micro_batch x accum` = effective batch). All model scenarios set `seed=1234`.

### 6.3 Suites, order and time budget

`quick` order (most informative first; the budget guard trims from the end):

1. `gpu.matmul`, `gpu.membw`, `gpu.pcie`
2. `train.w2v2b.6s.bs16.bf16`, `infer.w2v2b.6s.bs16.bf16`
3. `train.xlsr.6s.bs16.bf16`, `infer.xlsr.6s.bs16.bf16`
4. `train.w2v2l.6s.bs16.bf16`
5. `train.fusion.4s.bs16.bf16`
6. `frozen.xlsr.6s.bs16.bf16`
7. `train.w2v2b.10s.bs16.bf16`
8. `e2e.w2v2b.6s.bs16.bf16`
9. `cpu.decode`, `cpu.rawboost.blas1`, `cpu.rawboost.blasdefault`, `cpu.loader.w4.blas1`,
   `cpu.loader.w8.blas1`, `cpu.loader.w4.blasdefault`

`full` order: the quick list, then all remaining F rows grouped by model (base variants, xlsr variants,
w2v2l, wavlmb, fusion, frozen, infer, e2e, cpu), then `sustain.w2v2b.6s.bs16.bf16`, and
`train.w2v2b.6s.bs16.bf16.repeat` last.

Estimated totals on an RTX 5060 Ti 16 GB (sum of `est s` + ~45 s for doctor, corpus generation and report):

| suite | scenarios | Linux | Windows (x1.3 for slower process start / imports / spawn) | limit |
|---|---|---|---|---|
| quick | 18 | ~10 min | ~13 min | 15 min |
| full (incl. quick) | 49 | ~32 min | ~42 min | 90 min |

8 GB cards add OOM attempts (~15-20 s each, about 20 extra across `full`) -> roughly +6 min.

**Basis of the per-step estimates** (needed only for time budgeting; the benchmark measures the truth):
an RTX 5060 Ti runs our workloads at about **0.55x an RTX 3090 (range 0.45-0.67)**. Reasons:
(a) memory bandwidth 448 vs 936 GB/s = 0.48x; (b) FP32 vector 23.7 vs 35.6 TFLOPS = 0.67x and dense bf16
tensor (FP32 accumulate) ~47 vs 71 TFLOPS = 0.66x (5060 Ti figure derived, section 2.4); (c) our RTX A4500
ran W2V2 Base at 72.7 / 101.8 = 0.71x the RTX 3090 although it has more tensor throughput than the 3090,
which matches its bandwidth ratio (640/936 = 0.68x) and FP32 vector ratio (~0.66x): these workloads follow
bandwidth and vector throughput more than peak tensor FLOPS (caveat: different host CPUs, loader included).
The 5060 Ti's larger L2 (32 MB vs 6 MB) argues for the upper half of the range.
Resulting 5060 Ti (16 GB) expectations: W2V2 Base 6 s train ~56 clips/s (46-68) = 0.29 s/step, ~54 min per
ASVspoof 5 train epoch; Dev scoring ~165 clips/s (~14 min); Eval pass ~69 min; XLS-R 300M 6 s train ~20 clips/s
(17-25, with the expected 8x2 fallback) = ~150 min per epoch; XLS-R + WavLM 4 s fusion ~21 clips/s.
At half the central speed `quick` grows by only about 2 min, because measurement windows are capped
(`measure_max_s`) and fixed costs (process start, model load) dominate each scenario; the budget guard below
keeps it within 15 min in any case.

**Budget guard** (A): `--budget-min` defaults to 15 for `quick`, 90 for `full`, 0 = unlimited. Before
launching a scenario the runner projects `elapsed + est_s * calibration` where `calibration` = median of
(actual wall / est_s) over scenarios finished in this run (1.0 before the first). If the projection exceeds
the budget, the scenario is recorded as `skipped` with reason `budget` and the runner continues to the next
(cheaper ones may still fit). Budget-skipped scenarios are not "done" for resume, so `full` picks them up.

## 7. CLI (A: `agbench/cli.py`; `python -m agbench ...`)

```
python -m agbench doctor  [--gpu N] [--json PATH] [--no-network]
python -m agbench fetch   [--models KEY[,KEY...]] [--check]
python -m agbench prepare [--tiny]
python -m agbench list    [--suite quick|full] [--format table|json]
python -m agbench quick   [run options]
python -m agbench full    [run options]
python -m agbench run     --scenario ID [--scenario ID ...] [run options]
python -m agbench report  [JSONL ...] [--out DIR] [--references PATH]

run options:
  --gpu N              CUDA device index (default 0); sets CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=N in children
  --out PATH           results JSONL (default results/<gpu_slug>_<YYYYMMDD>.jsonl, local date at run start)
  --no-resume          ignore earlier results; run everything again
  --no-retry-failed    on resume, also skip scenarios whose last status was error/timeout/crashed
  --budget-min M       soft time budget (default quick 15, full 90, run 0); 0 = unlimited
  --mem-cap-gib X      cap PyTorch's allocator at X GiB in every child (for shared GPUs / smoke runs)
  --timeout-scale F    multiply every timeout (default 1.0)
  --tiny               tiny random models, tiny corpus, short loops; no downloads (tests, CI)
  --smoke              maintainers: real weights, but the tiny-mode shapes and loops (batch 2, <= 2 s,
                       MeasureCfg.tiny()) and small GPU-basics buffers; for code-path checks on a shared
                       GPU together with --mem-cap-gib (added at integration, see notes/VALIDATION.md)
  --device cuda|cpu    default cuda; cpu runs every workload on the CPU and is meant for --tiny (tests/CI)
  --allow-cpu-only     if the doctor finds no usable GPU, still run the cpu.* scenarios
  --dry-run            print the plan with estimated times and exit
  --no-fetch           do not download missing models (fail those scenarios as skipped: model_missing)
  --no-report          do not build the report at the end
```

`quick` / `full` / `run` = doctor (in a subprocess; abort with exit 2 on a fatal finding unless
`--allow-cpu-only`) -> fetch missing models (unless `--no-fetch` or offline) -> `prepare` corpus ->
scenarios -> `report`. Exit codes: 0 = all scenarios reached a terminal status in {ok, oom, skipped};
1 = at least one error/timeout/crashed; 2 = setup failure (doctor fatal, torch missing, lock held);
130 = interrupted (Ctrl+C).

Console output per scenario, ASCII, one line at start and one at end, e.g.
`[ 7/19] train.xlsr.6s.bs16.bf16 ... ok (8x2 after OOM at 16x1) 19.6 clips/s, 9.4 GiB, 61 s [elapsed 6m12s, est left 4m]`.

### 7.1 doctor checks (A: `agbench/env.py`)

| # | check | severity if failed |
|---|---|---|
| 1 | Python 3.12.x, 64-bit | fatal |
| 2 | `import torch` works; version ends with `+cu130` or `+cu128` (or `+cpu` with `--tiny`) | fatal (prints the uv sync command) |
| 3 | `torch.cuda.is_available()` | fatal (prints driver guidance) |
| 4 | device name, capability; if capability >= (12, 0) then `sm_120` or `compute_120` must be in `torch.cuda.get_arch_list()` | fatal ("wrong torch wheel: reinstall with --extra cu130") |
| 5 | kernel smoke on the GPU: bf16 matmul, conv1d, `scaled_dot_product_attention`, compared with CPU results (rtol 2e-2) | fatal (catches "no kernel image is available for execution on the device") |
| 6 | driver version (NVML, else `nvidia-smi`): >= 580 for a cu130 build, >= 570.26 Linux / 570.65 Windows for cu128 | fatal if below |
| 7 | VRAM total/free; warn if free < total - 1.5 GiB (other programs hold VRAM) and list GPU processes via NVML when available | warn |
| 8 | PCIe link: current/max generation and width; warn if max width < 8 | warn |
| 9 | Windows: remind about "CUDA - Sysmem Fallback Policy"; the harness cap still applies | info |
| 10 | free disk at `HF_HOME`/hub cache (>= 5 GB if models missing) and at `results/` (>= 1 GB) | fatal / warn |
| 11 | models cached? (per key, with expected bytes); network reachability of `HF_ENDPOINT` (HEAD, no token; skipped with `--no-network` or `HF_HUB_OFFLINE=1`) | warn |
| 12 | CPU model, logical/physical cores, RAM, numpy BLAS info (threadpoolctl) | info |
| 13 | power limit, current clocks, temperature at idle | info |

Writes `results/doctor_<gpu_slug>_<YYYYMMDD>.json` (same privacy rules). Prints a short ASCII table and
the line `DOCTOR: OK` / `DOCTOR: WARN (n)` / `DOCTOR: FATAL (n)`.

### 7.2 fetch (B: `agbench/hub.py`; A's CLI calls it)

`prefetch(keys)` calls `huggingface_hub.snapshot_download(repo_id, revision=<pinned sha>,
allow_patterns=["config.json", "preprocessor_config.json", "pytorch_model.bin"])`, verifies the weights file
size against section 2.3, retries 3 times with backoff. Honours `HF_ENDPOINT` (mirrors), `HF_HOME`,
`HF_HUB_CACHE`, `HF_TOKEN` (never logged or written anywhere). If `HF_ENDPOINT` is set and the download fails
with an Xet-related error, retry once with `HF_HUB_DISABLE_XET=1`. On HTTP 429 print: "Hugging Face rate
limit: wait and rerun `fetch`, or set HF_TOKEN to a read token". `AGBENCH_MODELS_DIR` (optional) points at
manually copied model folders `<dir>/<key>/{config.json,pytorch_model.bin}` used before the HF cache.
`--check` only verifies. Children always run with `HF_HUB_OFFLINE=1` and load from `resolve(key)`.

## 8. Execution design (A)

### 8.1 Processes

```
parent: python -m agbench quick           (no torch import; orchestrates; sole writer of the JSONL)
  |- child: python -m agbench doctor --json <tmp>
  |- child: python -m agbench prepare
  |- child: python -m agbench.child --scenario ID --attempt k --micro-batch m --accum a
  |         --out results/logs/<run_id>/<ID>.a<k>.json [--tiny] [--device D] [--mem-cap-gib X]
  |    `- DataLoader spawn workers (e2e.*, cpu.loader.*)
  `- child: python -m agbench report ...   (in-process is fine too; report does not import torch)
```

Children are started with `subprocess.Popen([sys.executable, "-m", ...], env=..., stdout/stderr -> log file)`,
`start_new_session=True` on POSIX, `creationflags=CREATE_NEW_PROCESS_GROUP` on Windows. Log file:
`results/logs/<run_id>/<ID>.a<k>.log`; the parent copies lines through the path sanitizer (8.9).

### 8.2 Child protocol (A: `agbench/child.py`)

1. Apply scenario `env` (already in the process env from the parent) and the memory cap: after CUDA init,
   `free, total = torch.cuda.mem_get_info()`; `cap = min(free - margin, mem_cap_gib or inf)` with margin
   0.5 GiB on Windows and 0.1 GiB elsewhere (section 18);
   `torch.cuda.set_per_process_memory_fraction(cap / total)`. Record `vram_total_gib`, `vram_free_start_gib`,
   `mem_cap_gib`.
2. Seed (`torch.manual_seed`, numpy Generator), build `RunContext`, start `GpuSampler` (C) on the same physical
   GPU (match NVML handle by `torch.cuda.get_device_properties(0).uuid`; fall back to the `--gpu` index, which
   equals the NVML index because `CUDA_DEVICE_ORDER=PCI_BUS_ID`), reset peak memory stats, call the workload function from the registry, stop sampler, read peak memory.
3. Classify: return value -> `ok`; `torch.OutOfMemoryError` or a `RuntimeError` whose message contains
   "out of memory", "CUBLAS_STATUS_ALLOC_FAILED" or "CUDNN_STATUS_ALLOC_FAILED" -> `oom`;
   `SkipScenario(reason)` -> `skipped`; any other exception -> `error` (type, message truncated to 2,000
   chars, traceback tail 40 lines, sanitized).
4. Write the attempt JSON atomically (`.tmp` + `os.replace`), exit 0. Any non-zero exit or a missing JSON
   means `crashed` to the parent.

Import-light rule: `agbench/child.py` imports torch and the workload module only inside `main()`, so DataLoader
spawn workers that re-import `__mp_main__` stay cheap and side-effect free.

### 8.3 Timeouts

Per attempt: `timeout_s = max(120, 4 * est_attempt_s) * timeout_scale` (sustain: `duration_s + 180`). On
timeout the parent kills the whole process tree (`psutil.Process(pid).children(recursive=True)` then the child;
terminate, wait 5 s, kill) and records `timeout`. Ctrl+C: kill the current tree, append `run_end` with
`interrupted: true`, exit 130.

### 8.4 OOM ladder (parent-driven, fresh process per rung)

For a scenario with a ladder, the parent runs rung 1; if the attempt status is `oom` it runs the next rung in a
**new** child (never retry inside a process that has just hit OOM: fragmentation and leaked references would
make later rungs fail falsely). Stop at the first `ok`, or at any non-OOM failure. All rungs OOM -> scenario
status `oom`. The record keeps every attempt (micro-batch, accum, status, peak memory, wall). A pure function
`next_rung(ladder, attempts) -> (mb, accum) | None` in `runner.py` makes this unit-testable.

### 8.5 Resume

Done-set = records in all `results/<gpu_slug>_*.jsonl` with the same `(scenario_id, scenario_hash,
env_fingerprint)` and status in {ok, oom, skipped (except reason `budget`)}. Those are skipped (console says
`cached`). error/timeout/crashed are retried once per run unless `--no-retry-failed`. A lock file
`results/.lock` (pid + start time) prevents two concurrent runs; a lock whose pid is dead is stale and removed.
JSONL appends: one complete line, `flush()` + `os.fsync()`; `load_records()` ignores a truncated last line.

- `scenario_hash` = first 12 hex chars of sha256 over canonical JSON
  `{id, workload, workload_version, params, ladder, measure}` (`sort_keys=True`, no whitespace).
- `env_fingerprint` = first 12 hex chars of sha256 over `{gpu_name, vram_class_gb, driver, torch, torch_cuda,
  transformers, os_family, agbench_version}`.
- `gpu_slug`: device name with leading "NVIDIA " and "GeForce " removed, lower case, non-alphanumerics -> "-",
  plus "-<vram class>gb" unless the name already ends with it. VRAM class = nearest of
  {4, 6, 8, 10, 11, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 141} to total GiB. Examples: `rtx-5060-ti-16gb`,
  `rtx-5060-ti-8gb`, `rtx-3090-24gb`, `rtx-a4500-20gb`, `a100-sxm4-40gb`. CPU tiny runs: `cpu`.
  Test runs on a GPU get their own slug, `<slug>-tiny` (`--tiny`) or `<slug>-smoke` (`--smoke`), so they never
  share a results file, resume entries or a report column with real runs (added at integration).

### 8.6 Memory cap and Windows sysmem fallback

The cap in 8.2 makes PyTorch raise OOM before the process can outgrow the VRAM that was free at start, so
Windows cannot silently spill into system memory. Each record also stores `near_limit = peak_reserved >
0.95 * cap` and, on Windows, `sysmem_suspected = true` if NVML reports used memory above total or if
`step_ms_cv > 0.5` while `near_limit`. `--mem-cap-gib` lowers the cap further for shared GPUs (maintainer
smoke runs use `--tiny` or a few small scenarios with `--mem-cap-gib 2.5`).

### 8.7 Environment for every child (set by the parent)

`PYTHONUNBUFFERED=1`, `PYTHONIOENCODING=utf-8`, `HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=<--gpu>`, plus the scenario's own `env` (BLAS settings,
`CUBLAS_WORKSPACE_CONFIG` for `.det`). The parent does not set `PYTORCH_CUDA_ALLOC_CONF`; if the user's
environment has it, its value is recorded in `env`.

### 8.8 Registry (A: `agbench/registry.py`) - exact names B and C must implement

| workload key | function | owner | `workload_version` |
|---|---|---|---|
| `train` | `agbench.workloads.train:run` | B | 1 |
| `frozen` | `agbench.workloads.frozen:run` | B | 1 |
| `infer` | `agbench.workloads.infer:run` | B | 1 |
| `e2e` | `agbench.workloads.e2e:run` | B | 1 |
| `sustain` | `agbench.workloads.sustain:run` | B | 1 |
| `basics.matmul` | `agbench.workloads.basics:run_matmul` | C | 1 |
| `basics.membw` | `agbench.workloads.basics:run_membw` | C | 1 |
| `basics.pcie` | `agbench.workloads.basics:run_pcie` | C | 1 |
| `cpu.decode` | `agbench.workloads.cpu_pipeline:run_decode` | C | 1 |
| `cpu.rawboost` | `agbench.workloads.cpu_pipeline:run_rawboost` | C | 1 |
| `cpu.loader` | `agbench.workloads.cpu_pipeline:run_loader` | C | 1 |

Bump a `workload_version` whenever what is measured changes; that invalidates resume for that workload.

### 8.9 Privacy of outputs

`util.sanitize(text)` replaces the user's home directory (from `Path.home()`, both slash styles) with `~`, and
the user name and host name (from `getpass.getuser()` / `socket.gethostname()`, if longer than 2 characters)
with `<user>` / `<host>`. Applied to error messages, tracebacks, log lines and every string in records. Records
never contain host name, user name, MAC/IP addresses, environment dumps or tokens. `env` lists only an
allow-list of variables: `CUDA_VISIBLE_DEVICES`, `CUDA_DEVICE_ORDER`, `PYTORCH_CUDA_ALLOC_CONF`,
`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `HF_HUB_OFFLINE`, and whether `HF_ENDPOINT`
is set (boolean only).

## 9. Results JSONL schema (A owns `schema.py` + `results_schema.json`; `SCHEMA_VERSION = "1"`)

One JSON object per line. Three kinds.

### 9.1 `run_start`

```json
{"kind": "run_start", "schema_version": "1", "run_id": "20261016T201500Z-3fa9c1",
 "suite": "quick", "started_utc": "2026-10-16T20:15:00Z", "agbench_version": "0.1.0",
 "git_commit": "abc1234 or null", "gpu_slug": "rtx-5060-ti-16gb", "env_fingerprint": "1f2e3d4c5b6a",
 "settings": {"budget_min": 15, "mem_cap_gib": null, "tiny": false, "device": "cuda", "timeout_scale": 1.0},
 "env": {
   "os": "Windows-11-10.0.26100", "os_family": "windows", "python": "3.12.11",
   "torch": "2.14.1+cu130", "torch_cuda": "13.0", "cudnn": 92400, "transformers": "5.20.0",
   "numpy": "2.5.4", "scipy": "1.18.1", "soundfile": "0.14.0", "libsndfile": "1.2.2",
   "driver": "580.97", "gpu_name": "NVIDIA GeForce RTX 5060 Ti", "gpu_capability": [12, 0],
   "gpu_arch_list": ["sm_75", "...", "sm_120"], "vram_total_gib": 15.93, "vram_class_gb": 16,
   "pcie": {"gen_cur": 5, "gen_max": 5, "width_cur": 8, "width_max": 8}, "power_limit_w": 180.0,
   "cpu_model": "...", "cpu_logical": 16, "cpu_physical": 8, "ram_gib": 31.9,
   "blas": [{"internal_api": "openblas", "version": "0.3.x", "num_threads": 16}],
   "env_vars": {"CUDA_VISIBLE_DEVICES": "0", "HF_ENDPOINT_set": false},
   "doctor": {"status": "OK", "warnings": []}}}
```

### 9.2 `scenario` (one per scenario per run, written after its last attempt)

| field | type | notes |
|---|---|---|
| `kind` | `"scenario"` | |
| `schema_version`, `run_id`, `gpu_slug`, `env_fingerprint` | str | |
| `scenario_id`, `scenario_hash`, `workload`, `workload_version`, `category`, `suite` | str/int | category in {basics, train, frozen, infer, e2e, cpu, sustain} |
| `status` | str | `ok` / `oom` / `error` / `timeout` / `crashed` / `skipped` |
| `status_detail` | str | short, e.g. `fits after OOM at 16x1`, `budget`, `workers_exceed_cores` |
| `started_utc`, `finished_utc`, `wall_s` | str/float | wall over all attempts |
| `params` | object | canonical scenario params (section 6), e.g. `{"model": "xlsr", "mode": "train", "seconds": 6.0, "batch": 16, "precision": "bf16", "head": "pyramid", "grad_ckpt": false, "deterministic": false, "lr_encoder": 1e-5, "lr_head": 1e-4, "weight_decay": 0.01, "grad_clip": 1.0, "layerdrop": 0.0, "seed": 1234}` |
| `measure` | object | MeasureCfg used + `warmup_steps_done`, `warmup_s`, `n_steps`, `measure_s` |
| `fallback` | object or null | `{"ladder": [[16,1],[8,2],[4,4],[2,8]], "attempts": [{"micro_batch": 16, "accum": 1, "status": "oom", "peak_reserved_gib": 14.6, "wall_s": 21.0}, ...], "used": {"micro_batch": 8, "accum": 2}, "fits_requested": false}` |
| `metrics` | object | see below; `{}` unless status is ok |
| `estimates` | object | see below |
| `telemetry` | object | `{"source": "nvml"\|"nvidia-smi"\|"none", "interval_s": 0.5, "n": 80, "temp_c": {"min":..,"median":..,"max":..}, "sm_clock_mhz": {..}, "mem_clock_mhz": {..}, "power_w": {..}, "util_pct": {..}, "mem_used_gib": {..}, "power_limit_w": 180.0, "throttle_reasons": ["sw_power_cap"]}` |
| `timeseries` | array or null | sustain only: per window `{"t_s", "clips_per_s", "temp_c", "sm_clock_mhz", "power_w"}`; telemetry samples downsampled to <= 600 points |
| `extra` | object | workload-specific details (attention implementation, actual checkpointing flags, threadpool info, per-stage RawBoost ms, worker count, ...) |
| `error` | object or null | `{"type", "message", "exit_code", "traceback_tail"}` (sanitized) |
| `log_file` | str | relative path under `results/logs/` |

`metrics` for model scenarios: `clips_per_s`, `clips_per_s_wall`, `step_ms_median`, `step_ms_p10`,
`step_ms_p90`, `step_ms_mean`, `step_ms_cv`, `n_steps`, `effective_batch`, `peak_alloc_gib`,
`peak_reserved_gib`, `vram_total_gib`, `vram_free_start_gib`, `mem_cap_gib`, `near_limit`,
`sysmem_suspected`, `loss_first`, `loss_last`, `loss_finite`; e2e adds `loader_wait_ms_median`,
`loader_wait_frac`, `num_workers`, `worker_startup_s`; sustain adds `clips_per_s_first_window`,
`clips_per_s_last_window`, `sustain_ratio`.
`metrics` for `gpu.matmul`: `tflops` = `{"bf16": {"4096": x, "8192": y}, "fp16": {...}, "tf32": {...},
"fp32": {...}}` plus `tflops_best` per dtype. `gpu.membw`: `gbps` = `{"copy": x, "read": y, "triad": z}`.
`gpu.pcie`: `gbps` = `{"h2d_pinned": x, "d2h_pinned": y, "h2d_pageable": z}`, `link` = `{gen_cur, width_cur,
gen_max, width_max}` sampled during the transfer.
`metrics` for `cpu.*`: `clips_per_s`, `ms_per_clip_median`, `cpu_ms_per_clip` (process + children CPU time),
`cpu_wall_ratio`, `num_workers`, `worker_startup_s`; `cpu.rawboost` adds `stage_ms` = `{"lnl", "isd", "ssi"}`;
`cpu.decode` adds `decode_ms_per_audio_s`.

`estimates` (A computes in the child from constants; `null` where not applicable):
`asv5_train_epoch_min = 182357 / clips_per_s / 60` (train, e2e, sustain);
`asv5_dev_pass_min = 140950 / clips_per_s / 60` and `asv5_eval_pass_min = 680774 / clips_per_s / 60` (infer).
Each estimate names the length it assumes (`"assumes_seconds": 6.0`). Real ASVspoof 5 clips are 4-10 s and are
centre-cropped or looped to the training length, so per-clip compute is constant.

### 9.3 `run_end`

`{"kind": "run_end", "schema_version": "1", "run_id": ..., "finished_utc": ..., "wall_s": ..., "counts":
{"ok": 17, "oom": 0, "error": 0, "timeout": 0, "crashed": 0, "skipped": 2, "cached": 0}, "interrupted": false}`

`schema.validate_record(rec) -> list[str]` returns human-readable problems (empty = valid);
`results_schema.json` is the same rules as JSON Schema for external tools.

## 10. Report (D: `agbench/report.py`, `agbench/plots.py`)

`build_report(jsonl_paths, out_dir, references_path="references/reference_gpus.json") -> Path`.
Default output: `results/<gpu_slug>_<YYYYMMDD>_report/` with `summary.md`, `comparison.csv`, `figures/*.png`.
With JSONL files from several GPUs: `results/compare_<YYYYMMDD>_report/` and one column per `gpu_slug`.
Uses the latest terminal record per `(gpu_slug, scenario_id)`. No torch import. matplotlib `Agg`, 150 dpi,
colour-blind-safe palette, every axis labelled with units, every figure has a legend where more than one
series is shown, and a one-line "Method:" caption under it in `summary.md`. Figures first, tables after.

`summary.md` layout:
1. Title `GPU benchmark: <GPU name> (<VRAM class> GB)`; one line with date, suite, OS, driver, torch/CUDA,
   and status counts.
2. **Figure 1 "Training throughput vs our GPUs"**: horizontal bars, clips/s for the five training workloads
   (W2V2 Base 6 s, XLS-R 300M 6 s, W2V2 Large LS-960 6 s, WavLM Base 6 s, XLS-R + WavLM 4 s); this GPU solid,
   references hatched and labelled "reference: training log, data loading included, torch 2.2.2".
3. **Table "What this means for ASVspoof 5"**: rows = those five workloads; columns = clips/s (GPU only),
   clips/s with loader (where an e2e row exists), batch that fitted (16x1 or fallback rung), peak GiB,
   train epoch min, Dev pass min, Eval pass min, ratio vs RTX 3090 (workstation) ref, ratio vs RTX A4500
   (workstation) ref. Epoch time = train epoch + Dev pass, Dev/Eval from the matching `infer.*` row.
4. **Figure 2 "Memory fit"**: peak reserved GiB per training scenario vs the VRAM cap, with dashed guides at
   ~7 GiB (8 GB card) and ~15 GiB (16 GB card); fallback rung annotated.
5. Figure 3 length scaling (clips/s and step ms vs seconds, W2V2 Base); table precision variants
   (bf16/fp16/tf32/det/ckpt) as ratios to the anchor.
6. Table inference (bs4/bs16, all models).
7. **Figure 4 "Can the CPU keep up?"**: loader clips/s vs workers, `blas1` vs `blasdefault`, with a horizontal
   line at this GPU's measured W2V2 Base training clips/s; table of `cpu.*` rows incl. CPU/wall ratio.
8. Table GPU basics (TFLOPS per dtype and size, GB/s, PCIe link) next to approximate spec values from
   `reference_gpus.json` for context.
9. **Figure 5 "Sustained run"** (if present): clips/s per 30 s window plus temperature, SM clock and power over
   time; text line with first/last window ratio and throttle reasons.
10. Table of every non-ok scenario with status and short reason.
11. Environment table; reference provenance notes (verbatim from `reference_gpus.json`).

`comparison.csv` columns: `scenario_id, metric, unit, <gpu_slug> ..., ref_<gpu_key> ..., ref_source`.

## 11. References (D: `references/reference_gpus.json`)

Numbers from our real training logs, data loading included, torch 2.2.2. They are **not** runs of this
benchmark and must be labelled that way everywhere. Exact agbench numbers on our GPUs go into `agbench_runs`
later (left empty now).

```json
{
  "schema_version": "1",
  "notes": "Reference numbers come from real training logs (torch 2.2.2, transformers 4.x, data loading included, bf16 autocast, AdamW, grad clip 1.0, deterministic algorithms on, LayerDrop 0, activation checkpointing off, 2 spawn DataLoader workers unless noted). They are not runs of this benchmark. agbench_runs will hold exact numbers from this benchmark on the same GPUs later.",
  "gpus": [
    {"gpu_key": "rtx-3090-24gb", "label": "RTX 3090 (workstation)",
     "spec_approx": {"vram_gb": 24, "mem_bw_gbps": 936, "fp32_tflops": 35.6, "bf16_dense_tflops_fp32acc": 71},
     "log_references": [
       {"workload": "W2V2 Base, 6 s, batch 16, train", "maps_to": ["e2e.w2v2b.6s.bs16.bf16", "train.w2v2b.6s.bs16.bf16"],
        "train_clips_per_s": 101.8, "train_min_per_epoch": 29.9, "dev_clips_per_s": 303,
        "peak_alloc_gib": 5.25, "peak_reserved_gib": 6.63, "rawboost": false,
        "provenance": "median of 14 epoch records of one run; FLAC decode + crop in 2 spawn workers, no augmentation"},
       {"workload": "XLS-R 300M, 6 s, batch 16, train", "maps_to": ["e2e.xlsr.6s.bs16.bf16", "train.xlsr.6s.bs16.bf16"],
        "train_clips_per_s": 38, "is_estimate": true,
        "provenance": "estimate from partial runs and a loader emulation (~36.9 clips/s without augmentation); not a completed epoch"},
       {"workload": "XLSR-53 + WavLM Base fusion, 4 s, batch 16, train", "maps_to": ["train.fusion.4s.bs16.bf16"],
        "train_clips_per_s": 40.6, "train_min_per_epoch": 74.9, "peak_alloc_gib": 13.6, "peak_reserved_gib": 14.9,
        "provenance": "XLSR-53 has the same architecture as XLS-R 300M; this benchmark uses XLS-R 300M"}
     ],
     "agbench_runs": []},
    {"gpu_key": "rtx-a4500-20gb", "label": "RTX A4500 (workstation)",
     "spec_approx": {"vram_gb": 20, "mem_bw_gbps": 640},
     "log_references": [
       {"workload": "W2V2 Base, 6 s, batch 16, train", "maps_to": ["e2e.w2v2b.6s.bs16.bf16", "train.w2v2b.6s.bs16.bf16"],
        "train_clips_per_s": 72.7, "train_min_per_epoch": 41.8, "dev_clips_per_s": 236, "dev_min": 10.0,
        "provenance": "pooling-baseline run, 2 workers"},
       {"workload": "XLS-R 300M, 6 s, batch 16, train", "maps_to": ["e2e.xlsr.6s.bs16.bf16", "train.xlsr.6s.bs16.bf16"],
        "train_clips_per_s": 28.7, "train_min_per_epoch": 106,
        "provenance": "steady-state rate of a running job, 2 workers; ~15 GB of 20 GB in use"}
     ],
     "agbench_runs": []},
    {"gpu_key": "a100-40gb-colab", "label": "A100 40 GB (Google Colab)",
     "spec_approx": {"vram_gb": 40, "mem_bw_gbps": 1555},
     "log_references": [
       {"workload": "W2V2 Base, 6 s, Dev scoring", "maps_to": ["infer.w2v2b.6s.bs16.bf16"], "dev_clips_per_s": 450,
        "provenance": "Dev pass inside a training run"},
       {"workload": "W2V2 Base, 6 s, batch 16, train with RawBoost", "maps_to": ["e2e.w2v2b.6s.bs16.bf16.rawboost"],
        "train_clips_per_s": 91, "cpu_bound": true,
        "provenance": "limited by RawBoost on the CPU (observed 90.8 clips/s); not a GPU number - do not compute GPU ratios from it"}
     ],
     "agbench_runs": []}
  ],
  "estimates": [
    {"gpu_key": "rtx-5060-ti-16gb", "label": "RTX 5060 Ti 16 GB (pre-run estimate)",
     "spec_approx": {"vram_gb": 16, "mem_bw_gbps": 448, "fp32_tflops": 23.7, "bf16_dense_tflops_fp32acc_derived": 47, "pcie": "5.0 x8"},
     "basis": "0.55x RTX 3090 (range 0.45-0.67), see SPEC section 6.3",
     "train_w2v2b_6s_clips_per_s": [46, 56, 68], "dev_w2v2b_6s_clips_per_s": 165,
     "train_xlsr_6s_clips_per_s": [17, 20, 25], "train_fusion_4s_clips_per_s": 21}
  ]
}
```

The report shows reference ratios as `this / reference` with the reference label and a footnote; for the A100
train number it shows "CPU-bound, not comparable". `references/README.md` explains provenance and how to add
`agbench_runs` entries (copy the run's JSONL into `references/runs/` and list it).

## 12. Interfaces (exact Python contracts)

```python
# agbench/context.py (A)
@dataclass(frozen=True)
class MeasureCfg:
    warmup_steps: int = 5; warmup_min_s: float = 5.0; warmup_max_s: float = 60.0
    measure_steps: int = 20; measure_min_s: float = 10.0; measure_max_s: float = 60.0
    @staticmethod
    def tiny() -> "MeasureCfg": ...

@dataclass
class RunContext:
    scenario_id: str
    workload: str
    params: dict            # scenario params + {"micro_batch": int, "accum": int} for this attempt
    device: str             # "cuda" or "cpu"
    tiny: bool
    measure: MeasureCfg
    seed: int
    cache_dir: Path         # synthetic corpus root (AGBENCH_CACHE or ./cache)
    keep_series: bool       # sustain only
    log: Callable[[str], None]   # ASCII progress line to the child log

# agbench/errors.py (A)
class SkipScenario(Exception): reason: str      # SkipScenario("workers_exceed_cores")
class ScenarioError(Exception): ...

# Workload functions (B, C): registry targets
def run(ctx: RunContext) -> dict
# returns {"metrics": {...}, "extra": {...}, "timeseries": [...] | None}
# Must NOT catch OOM. Raise SkipScenario when not applicable. Must not reset/read CUDA peak memory stats.

# agbench/timing.py (A)
def measure_loop(step_fn: Callable[[], None], *, device: str, items_per_step: int, cfg: MeasureCfg,
                 log: Callable[[str], None] | None = None) -> dict    # keys of section 5.5
def measure_for_duration(step_fn, *, device: str, items_per_step: int, duration_s: float,
                         window_s: float, warmup: MeasureCfg) -> tuple[dict, list[dict]]   # sustain
def cpu_loop(fn: Callable[[], None], *, items_per_call: int, cfg: MeasureCfg) -> dict  # perf_counter + process CPU time

# agbench/models.py (B)
@dataclass(frozen=True)
class ModelSpec:
    key: str; hf_id: str; revision: str; weights_bytes: int; hidden: int; layers: int; family: str  # "wav2vec2"|"wavlm"
MODELS: dict[str, ModelSpec]                       # keys: w2v2b, xlsr, w2v2l, wavlmb
def load_encoder(key: str, *, tiny: bool, device: str, grad_ckpt: bool = False) -> torch.nn.Module
    # from resolve(key) (or a tiny random config when tiny=True: hidden 32, 2 layers, conv_dim 16 x7,
    # same conv strides/kernels so frame rates match); layerdrop=0, apply_spec_augment=False,
    # checkpointing set explicitly and asserted, float32 weights, moved to device.
def tiny_config(key: str): ...

# agbench/hub.py (B)
def prefetch(keys: list[str]) -> list[dict]       # [{"key", "path", "bytes", "status": "cached"|"downloaded"|"failed", "error"}]
def resolve(key: str) -> Path                     # local folder with config.json + pytorch_model.bin; raises SkipScenario("model_missing")
def is_cached(key: str) -> bool

# agbench/heads.py (B)
class PyramidMeanHead(nn.Module): def __init__(self, dim: int, bins=(1, 2, 4), proj: int = 256, n_classes: int = 2)
class GlobalMeanHead(nn.Module):  def __init__(self, dim: int, proj: int = 256, n_classes: int = 2)
class FusionHead(nn.Module):      def __init__(self, dims: tuple[int, ...], bins=(1, 2, 4), proj: int = 256, n_classes: int = 2)
class FrozenHeadSet(nn.Module):   def __init__(self, dim: int, n_states: int)   # forward(list_of_hidden_states) -> list of logits

# agbench/data/synth.py (C)
def speechlike(n_samples: int, rng: np.random.Generator, sr: int = 16000) -> np.ndarray   # float32, [-1, 1]
def ensure_corpus(cache_dir: Path, *, tiny: bool = False, n_files: int = 256, seed: int = 20261016) -> Path  # manifest.json
# manifest: {"version": 1, "sr": 16000, "files": [{"path": "relative.flac", "seconds": 7.31, "label": 0, "bytes": 123}], "total_bytes": ...}

# agbench/data/rawboost.py (C)
def rawboost(x: np.ndarray, rng: np.random.Generator, algo: int = 4, sr: int = 16000, **overrides) -> np.ndarray
def lnl(x, rng, **p) -> np.ndarray; def isd(x, rng, **p) -> np.ndarray; def ssi(x, rng, **p) -> np.ndarray
DEFAULTS: dict   # section 2.5 values

# agbench/data/dataset.py (C)
class SynthFlacDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: Path, seconds: float = 6.0, train: bool = True, rawboost_p: float = 0.0, seed: int = 1234)
def make_loader(ds, *, batch_size: int, num_workers: int, pin_memory: bool, persistent_workers: bool,
                prefetch_factor: int = 2, shuffle: bool = True) -> torch.utils.data.DataLoader
def worker_init_fn(worker_id: int) -> None

# agbench/telemetry.py (C)
class GpuSampler:
    def __init__(self, device_index: int = 0, interval_s: float = 0.5, keep_series: bool = False,
                 uuid: str | None = None)   # uuid wins over index when given
    def __enter__(self) -> "GpuSampler"; def __exit__(self, *exc) -> None
    def summary(self) -> dict      # telemetry object of section 9.2; never raises; {"source": "none"} without NVML/nvidia-smi
    def series(self) -> list[dict] # [{"t_s", "temp_c", "sm_clock_mhz", "mem_clock_mhz", "power_w", "util_pct", "mem_used_gib", "throttle": [...]}]
def nvml_static_info(device_index: int = 0) -> dict   # driver, power limit, pcie gen/width (cur/max); used by env.py

# agbench/schema.py (A)
SCHEMA_VERSION = "1"
def make_run_start(...) -> dict; def make_scenario_record(...) -> dict; def make_run_end(...) -> dict
def validate_record(rec: dict) -> list[str]
def load_records(path: Path) -> list[dict]          # skips a truncated last line, keeps order

# agbench/env.py (A)
def capture_env(device_index: int = 0) -> dict      # run_start.env (sanitized, allow-listed)
def gpu_slug(env: dict) -> str
def env_fingerprint(env: dict) -> str
def doctor(device_index: int = 0, network: bool = True) -> dict   # {"status": "OK"|"WARN"|"FATAL", "checks": [...]}

# agbench/scenarios.py (A)
@dataclass(frozen=True)
class Scenario:
    id: str; category: str; workload: str; params: dict; suites: tuple[str, ...]
    needs_gpu: bool; models: tuple[str, ...]; ladder: tuple[tuple[int, int], ...] | None
    est_s: float; env: dict[str, str | None]   # None = remove the variable from the child env
SCENARIOS: dict[str, Scenario]; SUITES: dict[str, list[str]]   # "quick", "full" in run order
def scenario_hash(s: Scenario, measure: MeasureCfg) -> str

# agbench/runner.py (A)
def next_rung(ladder, attempts) -> tuple[int, int] | None
def run_suite(ids: list[str], opts) -> int          # exit code of section 7

# agbench/report.py (D)
def build_report(jsonl_paths: list[Path], out_dir: Path | None = None,
                 references_path: Path = Path("references/reference_gpus.json")) -> Path   # summary.md
```

Test hooks (A implements in `child.py`, D uses):
- `AGBENCH_SIMULATE_OOM_MB_GE=<int>`: the child raises `torch.OutOfMemoryError("simulated")` before the
  workload when the attempt's micro-batch >= value.
- `AGBENCH_SIMULATE_FAULT=<scenario_id>:<crash|hang|error>`: crash = `os._exit(3)`, hang = sleep forever,
  error = raise `RuntimeError("simulated")`.

Tiny mode contract (`--tiny`): models from `tiny_config` (no downloads), batch overrides to 2 (ladders scaled
`2x1 -> 1x2`), seconds capped at 2 s, `MeasureCfg.tiny()`, corpus
`synth_tiny_v1`, loader workers capped at 2, `gpu.pcie` -> skipped on CPU, `gpu.matmul` uses n=256 on CPU,
`gpu.membw` uses 64 MiB on CPU, sustain duration 3 s with 1 s windows. The whole tiny `full` suite on CPU must
finish in under 5 minutes on a 2-core CI runner.

## 13. Tests and CI

### 13.1 Rules
CPU only, no network, no downloads, tiny random configs. `pytest -q` from the repo root after
`uv sync --locked --extra cpu`. Each test file < 60 s on CI.

### 13.2 Builder-owned unit tests
- A `tests/test_runner_unit.py`: `next_rung` sequences, scenario hash stability, gpu_slug examples of 8.5,
  resume done-set selection, budget guard math, path sanitizer, atomic append + truncated-line tolerance,
  process-tree kill of a sleeping grandchild (timeout path), suite membership (quick subset of full; ids unique;
  every scenario's workload exists in the registry).
- B `tests/test_workloads_unit.py`: heads shapes (pyramid boundaries equal `adaptive_avg_pool1d` results),
  `load_encoder(tiny=True)` checkpointing flag both ways, each B workload `run(ctx)` in tiny CPU mode returns
  finite metrics.
- C `tests/test_data_unit.py`: speechlike range/dtype/determinism; corpus idempotency; RawBoost each algo:
  shape, finite, float32, deterministic for a fixed rng, SNR of SSI within [10, 40] dB; dataset crop/loop
  lengths; DataLoader with 2 spawn workers yields correct shapes (this runs on the Windows CI job too);
  `GpuSampler` without NVML returns `{"source": "none"}`.

### 13.3 D-owned cross-cutting tests
- `test_schema.py`: fixtures validate; broken records produce the expected messages; JSON Schema file and
  `validate_record` agree on the fixtures.
- `test_report.py`: `build_report` on fixtures (one GPU, two GPUs, a file with oom/timeout/skipped rows, a
  file without sustain) writes `summary.md`, `comparison.csv` and the expected figure files; reference
  labels present; no exception on missing scenarios.
- `test_cli_e2e.py`: real subprocesses: `python -m agbench run --tiny --device cpu --scenario <a few ids>
  --out <tmp>` then `report`; asserts exit code 0, valid JSONL, resume on a second invocation runs nothing
  (`cached`), `--no-resume` reruns.
- `test_fallback_sim.py`: with `AGBENCH_SIMULATE_OOM_MB_GE=2` a tiny laddered scenario ends `ok` at `1x2`
  with two attempts recorded; with the threshold at 1 it ends `oom`; `AGBENCH_SIMULATE_FAULT` crash/hang/error
  give `crashed` / `timeout` (with `--timeout-scale` small) / `error`, and the suite continues to the next
  scenario.
- `test_privacy.py` (13.4).

### 13.4 Privacy test
Scans every git-tracked text file (and generated fixtures) for: e-mail addresses (regex; the only allowed
form is a GitHub `...@users.noreply.github.com` address);
IPv4 addresses (except `127.0.0.1` and `0.0.0.0`); absolute home paths (`/home/<name>/`, `/Users/<name>/`,
`C:\Users\<name>\`, except `<user>`/`~` placeholders); token-like strings (`hf_[A-Za-z0-9]{30,}`,
`ghp_[A-Za-z0-9]{30,}`). Additionally, if an untracked `.privacy_denylist` file exists (one word per line,
git-ignored, maintainers only), none of its words may appear (case-insensitive, whole-word match). The test file
itself must not contain any private word.

### 13.5 CI (D: `.github/workflows/ci.yml`)
On push and pull request: matrix `ubuntu-latest`, `windows-latest`; steps `actions/checkout@v7`,
`astral-sh/setup-uv@v10`, `uv sync --locked --extra cpu`, `uv run --no-sync pytest -q`. Timeout 20 min.
No GPU jobs.

## 14. Documentation (D)

**README.md (humans):** what it measures and why (one paragraph); one-command quick start (Linux and Windows);
the scenario table in plain words (what each group answers); how long it takes; how to read `summary.md`
(clips/s, ms/step, peak memory, fallback rung, epoch estimates, the reference labels and why they are not
apples-to-apples); how to compare GPUs (`report a.jsonl b.jsonl`); privacy statement (synthetic audio, what
the results file contains and does not contain); credits (RawBoost paper/repo, model cards) and licence.

**AGENTS.md (AI agent, step by step):**
1. Pre-flight: `nvidia-smi` -> read driver version; >= 580 -> `cu130`; 570-579 -> `cu128`; lower -> tell the
   human to update the NVIDIA driver (Windows: GeForce driver from nvidia.com or the NVIDIA App; Linux:
   distribution package `nvidia-driver-580` or newer; Blackwell needs NVIDIA's open kernel modules, the
   default `-open` packages). Check free disk >= 25 GB.
2. Install uv; clone; `uv sync --locked --extra cu130` (exact commands of 3.3 for both shells).
3. `uv run --no-sync python -m agbench doctor` -> must print `DOCTOR: OK` or `DOCTOR: WARN`.
4. `uv run --no-sync python -m agbench fetch` (3.3 GB; mirrors via `HF_ENDPOINT`; offline via a copied HF cache
   and `HF_HUB_OFFLINE=1`, or `AGBENCH_MODELS_DIR`).
5. Close games/browsers/other GPU apps; on Windows set "CUDA - Sysmem Fallback Policy" = "Prefer No Sysmem
   Fallback" for `python.exe` (optional; the harness also guards); keep the PC on AC power, do not let it sleep.
6. `uv run --no-sync python -m agbench quick` (expected 10-15 min). If time allows,
   `uv run --no-sync python -m agbench full` (expected +20-35 min; resumes, never repeats finished scenarios).
7. Verify: the console ends with a status count line; `results/<gpu>_<date>.jsonl` and
   `results/<gpu>_<date>_report/summary.md` exist.
8. **Send back** the whole `results/` folder zipped (it contains `*.jsonl`, `doctor_*.json`,
   `*_report/summary.md` and figures, and `logs/`). Nothing else is needed.

Troubleshooting table: driver too old for cu130 (use `cu128` or update); "no kernel image is available for
execution on the device" / doctor check 4-5 fails (wrong wheel, e.g. cu126 or CPU torch: rerun
`uv sync --locked --extra cu130`); `torch.cuda.is_available() == False`; plain `uv run` removed torch (use
`--no-sync`); OOM rows (expected on some rows; that is data, do not change settings); Windows spawn errors
("An attempt has been made to start a new process before the current process has finished its bootstrapping
phase" -> report as a bug, do not patch); Windows Defender slowing the first import (wait, or exclude the repo
folder); disk full; Hugging Face 429 / timeouts (`HF_TOKEN`, `HF_HUB_DOWNLOAD_TIMEOUT=60`, rerun `fetch`,
mirror via `HF_ENDPOINT` + `HF_HUB_DISABLE_XET=1`); WSL2 works but native is preferred; laptop GPUs and
power-saving modes distort results; a run interrupted -> just rerun the same command (resume).
Rules for the agent: do not edit code or settings to make a scenario pass, do not raise batch sizes, do not
upload results anywhere; only report.

**CLAUDE.md:** two lines: "Follow AGENTS.md exactly. Do not modify the benchmark code to make scenarios pass;
OOM and skipped rows are valid results."

**NOTICE:** RawBoost (Tak et al., ICASSP 2022; github.com/TakHemlata/RawBoost-antispoofing, MIT, Copyright (c)
2021 Hemlata) - this repo contains a re-implementation that follows the official MIT-licensed code (same
operations, default parameters and parameter names, no lines copied), so NOTICE reproduces its MIT copyright and
permission notice (full text also in LICENSES/RawBoost-antispoofing-MIT.txt); models are downloaded from Hugging
Face under their own licences (list the four ids).

**LICENSE:** MIT, copyright holder = the repository owner's GitHub handle, year 2026.

## 15. Builder hand-off checklist

Each builder delivers: their files (section 4) only; their unit tests passing on CPU; a short hand-off note
(in the final message to the orchestrator, not a file) listing deviations from this SPEC and anything another
builder must know.

- **A:** `uv lock` resolves; `uv sync --locked --extra cpu` works; `python -m agbench list --suite full`
  prints 49 scenarios (18 in quick) with est times summing to section 6.3; `--dry-run` works without torch installed in the
  parent path; tiny end-to-end works once B and C land (A can stub workloads in its own tests).
- **B:** all model scenarios pass in tiny CPU mode; real-model loading verified on any CUDA GPU available to
  the builder only under a memory cap (`--mem-cap-gib`), never on a busy shared GPU without permission.
- **C:** `cpu.*` scenarios run on CPU (real, not tiny) on the builder machine at low priority; GPU basics run in
  tiny CPU mode; RawBoost numbers sanity-checked (algo 4 on a 6 s crop takes tens of ms per clip in one process).
- **D:** README/AGENTS/CLAUDE/NOTICE/LICENSE; references file exactly as section 11; report on fixtures; CI file;
  privacy test passes on the whole tree.

Integration order: A skeleton (context, errors, registry, timing, schema) first, so B/C/D code against real
signatures; B and C in parallel; D's fixtures follow section 9 and do not wait for real runs. Final integration:
tiny `full` on CPU, then (maintainers) a capped smoke on a real GPU, then publish.

## 16. Publishing notes (for the orchestrator)

- Public GitHub repository under the user's account. Before the first commit set a repo-local
  `user.email` to the account's GitHub **noreply** address (the global git identity has no e-mail set, so git
  would otherwise derive one from the local user and host name).
- Run `pytest tests/test_privacy.py` with a maintainer `.privacy_denylist` present before every push.
- Do not commit `results/` content, `cache/`, or any log from a private machine.

## 17. Known risks

1. Throughput scale factor (0.55x RTX 3090) is an estimate; quick could exceed 15 min on a slow Windows
   install. Mitigation: capped measurement windows + budget guard (6.3).
2. XLS-R and fusion at bs16 sit at the 16 GB boundary; whether `16x1` fits may differ between Linux (headless)
   and Windows (desktop VRAM). The record shows free VRAM at start, the cap, and the rung that fitted, so this
   is interpretable.
3. References are training-log numbers (torch 2.2.2, deterministic mode, loader included). Direct ratios are
   indicative only until `agbench_runs` exist for our GPUs.
4. `cu128` fallback runs torch 2.11.0 (older kernels); results record the torch version, and the report flags
   cross-version comparisons.
5. NVML fields can be missing on some Windows/WSL drivers; telemetry degrades to `nvidia-smi` or `none`
   without failing scenarios.

## 18. As built: differences from this spec (maintainers)

This section replaces the builders' hand-off notes. AI agents running the benchmark: nothing here changes
AGENTS.md; the switches below are for the maintainers' tests only.

**Interfaces.** `RunContext` adds `gpu_index`, `gpu_uuid`, `attempt`. `Scenario` adds `measure` (per-scenario
`MeasureCfg`) and `mem_est_gib`. `measure_loop` / `measure_for_duration` / `cpu_loop` take `log=`;
`measure_for_duration` also returns `n_windows` and `t0_perf` (the child pops it to align telemetry). `doctor()`
takes `device`, `tiny`, `results_dir`, `model_keys`. `ModelSpec` carries the pinned `sha256` of
`pytorch_model.bin`; `hub.check` returns `cached` / `bad` / `missing`, and `fetch` verifies the hash once per file
(remembered in `cache/verified_models.json`). `ensure_corpus(None)` uses the default cache;
`make_loader` adds `drop_last`, `num_samples` (no epoch boundary inside a measurement), `seed`.

**Test hooks and hidden flags (never in AGENTS.md).** `AGBENCH_REGISTRY_OVERRIDE`, `AGBENCH_SIMULATE_FAULT`,
`AGBENCH_SIMULATE_OOM_MB_GE`, `AGBENCH_PRIVACY_DENYLIST`, `AGBENCH_NO_KEEP_AWAKE`; `doctor --device/--tiny/
--results-dir`, `fetch --json`, run option `--no-network`; `--tiny`, `--smoke`, `--mem-cap-gib`,
`--allow-cpu-only`.

**Runner behaviour.** New run option `--retry-oom`: on resume, scenarios whose latest result was `oom` run
again (after a program that held GPU memory was closed); AGENTS.md allows it, with `--gpu N`, as the only
extra options. Resume treats `budget`, `model_missing` and `no_gpu` skips as not done. The budget clock
excludes model downloads. Exit code 1 also when the report fails. Records store the effective params, and
`scenario_hash` covers them, so tiny / smoke runs never share resume entries with real ones. Every launched
process gets `AGBENCH_PARENT=<pid>:<create time>`: children exit within a second when the runner is gone
(Linux also `PR_SET_PDEATHSIG`; Windows also a kill-on-close Job Object), and a new run first kills leftover
`agbench` processes that point into its results folder. The runner holds a sleep inhibitor while it runs
(Windows `SetThreadExecutionState`; Linux `systemd-inhibit` when permitted). Attempt entries record
`mem_cap_gib` and `vram_free_start_gib`, so the report can say under which cap 16x1 ran out of memory. The
memory cap margin is 0.5 GiB on Windows (sysmem fallback) and 0.1 GiB elsewhere. A log that cannot be rewritten
by the sanitizer (a locked file on Windows) is left as it is with a warning; `os.replace` retries on Windows.

**Workloads.** Tiny CPU runs cap torch threads at 2. Precision sets the fp32 math mode through the new
per-backend API only (`torch.backends.cuda.matmul.fp32_precision`, `torch.backends.cudnn.conv.fp32_precision`);
never read the legacy getters in a child (in torch 2.14 they raise after a new-API change). Reference-numerics
rows (`train.w2v2b.6s.bs16.bf16.det`, `e2e.w2v2b.6s.bs16.bf16`, `e2e.xlsr.6s.bs16.bf16`) use strict
`use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, cuDNN conv TF32 off and flash SDPA
off for clips longer than 4 s, as the training logs did; if an op has no deterministic kernel, the child reruns
once in warn-only mode and records `extra.deterministic_strict_failed`. `e2e` `clips_per_s` is the wall-clock
rate (the median-step rate is kept as `clips_per_s_median_step`). `cpu.loader` times at least
max(30, 6 x workers) batches after a backlog of `prefetch_factor x workers` and reports the steady rate over
whole bursts (`clips_per_s_wall` covers everything timed). `load_encoder` fails on missing or mis-shaped encoder
weights (other than `masked_spec_embed`) and switches off the gradient checkpointing that the wav2vec2-base
config turns on. Out-of-memory classification also covers cuDNN "no valid algorithm / no engine" and cuBLAS
`NOT_INITIALIZED` near the cap.

**Data and basics.** The corpus is 256 FLAC files (about 1,800 s, 37 MB, built in about 5 s). RawBoost follows
the official MIT code operation by operation (FIR delay compensation identical for the same taps; algorithms
5-8 included). `worker_init_fn` pre-imports scipy so it counts as worker start-up. `basics.matmul` warms the
clocks for about 0.5 s; `basics.membw` shrinks its buffers under a memory cap.

**Costs to keep in mind.** `import transformers` takes several seconds per child (it pulls in
`torch._dynamo`); this dominates tiny runs. Real-weight loading of all four checkpoints and a forward + backward
pass of the W2V2 Base and fusion classifiers are covered by the opt-in check
`AGBENCH_TEST_REAL_MODELS=1 HF_HUB_OFFLINE=1 pytest tests/test_workloads_unit.py -k real_weights`.
After changing the scenario matrix, regenerate the fixtures with `python tests/fixtures/make_fixtures.py`.
