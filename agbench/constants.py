"""Fixed numbers used across the harness (no imports beyond the standard library)."""

SAMPLE_RATE = 16000
MODEL_KEYS = ("w2v2b", "xlsr", "w2v2l", "wavlmb")  # SPEC 2.3 (details in agbench/models.py)
DEFAULT_SEED = 1234

# ASVspoof 5 clip counts (used only for derived time estimates).
ASV5_TRAIN_CLIPS = 182_357
ASV5_DEV_CLIPS = 140_950
ASV5_EVAL_CLIPS = 680_774

GIB = 1024 ** 3
MIB = 1024 ** 2

# VRAM classes for gpu_slug / env_fingerprint (nearest to the total GiB).
VRAM_CLASSES_GB = (4, 6, 8, 10, 11, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 141)

# OOM ladders: (micro_batch, accum) rungs, tried in order in fresh processes.
LADDER_16 = ((16, 1), (8, 2), (4, 4), (2, 8))
LADDER_8 = ((8, 1), (4, 2), (2, 4))
LADDER_TINY = ((2, 1), (1, 2))
TINY_BATCH = 2
TINY_MAX_SECONDS = 2.0
TINY_MAX_WORKERS = 2

# Memory cap: the child leaves this much of the VRAM that was free at start unused. Windows keeps
# a margin against the driver's silent spill into system memory (sysmem fallback); elsewhere an
# allocator OOM is clean, so only a small margin for memory allocated outside PyTorch.
MEM_CAP_MARGIN_GIB_WINDOWS = 0.5
MEM_CAP_MARGIN_GIB_OTHER = 0.1
NEAR_LIMIT_FRAC = 0.95

# Minimum NVIDIA driver versions (NVIDIA CUDA Toolkit release notes).
MIN_DRIVER_CUDA13 = (580, 0)
MIN_DRIVER_CUDA128_LINUX = (570, 26)
MIN_DRIVER_CUDA128_WINDOWS = (570, 65)

UV_SYNC_HINT = 'uv sync --frozen --extra cu130'
UV_SYNC_HINT_CU128 = 'uv sync --frozen --extra cu128'
TORCH_MISSING_MSG = ('torch is not installed: run "uv sync --frozen --extra cu130" (NVIDIA driver 580 or '
                     'newer) or "uv sync --frozen --extra cu128" (driver 570-579)')


def uv_sync_hint(driver: str | None) -> str:
    """The uv sync command for this NVIDIA driver: >= 580 -> cu130, 570-579 -> cu128."""
    try:
        major = int(str(driver).split(".")[0])
    except (TypeError, ValueError):
        return UV_SYNC_HINT
    return UV_SYNC_HINT_CU128 if 570 <= major < 580 else UV_SYNC_HINT

# Timeouts and budgets.
MIN_TIMEOUT_S = 120.0
TIMEOUT_FACTOR = 4.0
SUSTAIN_TIMEOUT_EXTRA_S = 180.0
DEFAULT_BUDGET_MIN = {"quick": 15.0, "full": 90.0, "run": 0.0}
SETUP_EST_S = 45.0  # doctor + corpus + report, used in printed totals only

# Skip reasons that do not count as "done" for resume (the next run tries again).
RETRYABLE_SKIP_REASONS = frozenset({"budget", "model_missing", "no_gpu"})

# Allow-listed environment variables copied into records (values), plus HF_ENDPOINT (set: bool).
ENV_ALLOWLIST = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "PYTORCH_CUDA_ALLOC_CONF",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "HF_HUB_OFFLINE",
)
BLAS_THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "GOTO_NUM_THREADS")
