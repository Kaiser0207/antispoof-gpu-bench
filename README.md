# antispoof-gpu-bench

A small, self-contained GPU benchmark for **speech anti-spoofing training**: fine-tuning
self-supervised speech encoders (wav2vec 2.0, XLS-R, WavLM) for spoof detection, as in ASVspoof 5.
It answers one practical question: **how fast would this GPU train and score our models, and does
it have enough memory?** It uses synthetic audio only (no dataset download) and public Hugging Face
weights, runs with one command on Windows 11 or Linux, and writes one results file plus a readable
report that can be compared with other GPUs.

It was written to check an NVIDIA RTX 5060 Ti (Blackwell, 16 GB or 8 GB) before buying one, but it
runs on any recent NVIDIA GPU.

> Using an AI coding agent to run this? Point it at [AGENTS.md](AGENTS.md): it has the exact
> step-by-step commands, checks and troubleshooting.

## Quick start

You need: an NVIDIA GPU with driver **580 or newer** (570-579 also works, see below; on Linux an RTX 50
card needs the `-open` driver package), Windows 11 or Linux on x86-64, about **25 GB free disk**, and an
internet connection for the first run (PyTorch with its CUDA libraries about 4-5 GB on Linux and 3 GB
on Windows, model weights 3.3 GB).

**Linux**

```bash
git clone https://github.com/Kaiser0207/antispoof-gpu-bench.git
cd antispoof-gpu-bench
bash scripts/bench.sh              # installs uv if needed, sets up, checks, downloads, runs everything (~35-50 min)
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/Kaiser0207/antispoof-gpu-bench.git
cd antispoof-gpu-bench
powershell -ExecutionPolicy ByPass -File scripts\bench.ps1
```

The scripts run the `full` suite; pass `quick` (`bash scripts/bench.sh quick`) for the 10-15 minute
subset only. They only wrap the manual steps, which are the same on both systems:

```bash
uv sync --frozen --extra cu130          # PyTorch 2.14.1 for CUDA 13.0 (driver >= 580)
uv run --no-sync python -m agbench doctor
uv run --no-sync python -m agbench fetch
uv run --no-sync python -m agbench quick    # the most informative subset first (~10-15 min)
uv run --no-sync python -m agbench full     # the rest (~20-35 min more); repeats nothing quick finished
```

Use `--extra cu128` instead of `cu130` if the driver is 570-579 and cannot be updated (PyTorch
2.11.0). `--frozen` installs exactly what `uv.lock` lists. Always run with `uv run --no-sync`: a plain
`uv run` re-syncs the environment without the extra and removes PyTorch. With several GPUs, add
`--gpu N` (the index from `nvidia-smi -L`). The benchmark asks the operating system not to sleep while
it runs.

When it finishes, the results are in `results/`:

- `results/<gpu>_<date>.jsonl`: every number, one JSON line per scenario (the file to send back);
- `results/<gpu>_<date>_report/summary.md`: the readable report, with `comparison.csv` and
  `figures/*.png` next to it;
- `results/doctor_<gpu>_<date>.json` and `results/logs/`: environment check and per-scenario logs.

An interrupted run (Ctrl+C, sleep, a killed terminal) resumes when the same command is run again; the
child processes of a killed run stop on their own.

## What it measures

Every scenario runs in its own process with a timeout, so an out-of-memory error or a crash is
recorded and the suite moves on. The **quick** suite (18 scenarios, about 10-15 minutes on an
RTX 5060 Ti) covers the most informative cases; **full** (49 scenarios, about 35-50 minutes in
total) adds the rest and never repeats what quick already finished.

| Group | What it answers | Scenarios |
|---|---|---|
| Training | How many clips per second does the GPU train at our settings? | full fine-tune of W2V2 Base, XLS-R 300M, W2V2 Large (LibriSpeech 960 h only), WavLM Base, and the XLS-R + WavLM fusion; 6 s clips (fusion 4 s), batch 16, bf16 autocast, AdamW, gradient clipping |
| Memory fit | Does batch 16 fit in 16 GB (or 8 GB)? | every training scenario; if batch 16 runs out of memory it is retried in a fresh process as 8x2, 4x4, 2x8 (micro-batch x gradient accumulation) |
| Clip length | How does cost grow with clip length? | W2V2 Base at 2, 4, 6, 8, 10 s |
| Precision and recipe | What do bf16 / fp16 / TF32, the numerics of our training logs (strict deterministic algorithms, TF32 off, no flash attention), activation checkpointing, other heads and batch sizes cost? | W2V2 Base and XLS-R variants |
| Frozen encoder | How fast is screening many small heads on a frozen encoder? | XLS-R, W2V2 Base and Large with 4 or 40 heads on all hidden states |
| Inference | How long do Dev and Eval scoring passes take? | all models at batch 16, two at batch 4 |
| With data loader | Does real data loading slow training down, and how does this GPU compare with our training logs? | W2V2 Base and XLS-R with FLAC decoding in 2 DataLoader workers and the logs' numerics (the rows compared with our GPUs), and W2V2 Base with RawBoost augmentation |
| CPU pipeline | Can this PC's CPU feed the GPU? Does `OPENBLAS_NUM_THREADS=1` matter? | FLAC decoding, RawBoost (algorithm 4: LnL + ISD + SSI) per stage, DataLoader with 1-16 workers, with and without BLAS thread limits |
| GPU basics | Raw numbers | bf16 / fp16 / TF32 / fp32 matrix multiply TFLOPS, memory bandwidth, PCIe bandwidth and link |
| Sustained | Does the card slow down when hot? | 5 minutes of W2V2 Base training with temperature, clock and power, then the anchor repeated |

Model weights come from Hugging Face, pinned to exact revisions: `facebook/wav2vec2-base`,
`facebook/wav2vec2-xls-r-300m`, `facebook/wav2vec2-large`, `microsoft/wavlm-base`.

## How to read the report

`summary.md` starts with figures, then tables. Every figure has a one-line *Method* note under it.

- **clips/s** (clips per second, higher is better): audio clips of the stated length (6 s unless
  noted) processed per second. For training, one *step* is forward + backward + optimizer update over
  16 clips; clips/s = 16 / median step time. Warm-up steps are excluded; GPU time is measured with
  CUDA events.
- **ms/step**: the median time of one step. The spread (p10/p90, coefficient of variation) is in
  `comparison.csv`; a large spread means something else was using the machine.
- **GPU only vs with loader**: GPU-only scenarios feed synthetic audio that is already on the GPU, so
  they measure the GPU alone. *With loader* scenarios read and decode FLAC files in DataLoader worker
  processes like a real training run, so they also depend on the CPU; their clips/s is clips divided
  by wall-clock time, so waits for the loader count. The two plain with-loader rows also use the
  numerics of our training logs (strict deterministic algorithms, TF32 off, flash attention off for
  6 s clips), because they are the rows compared with those logs.
- **peak GiB**: the most memory PyTorch reserved during the scenario. Each scenario is capped at the
  memory that was free when it started minus a margin (0.5 GiB on Windows, 0.1 GiB on Linux), so
  nothing silently spills into system memory.
- **batch that fitted**: `16x1` means 16 clips per micro-batch. `8x2 (OOM at 16x1 under a 14.1 GiB
  cap; 14.6 GiB were free)` means 16 did not fit in the memory that was free and the same 16 clips
  were processed as two micro-batches of 8 with gradient accumulation (same result, usually a little
  slower). If much less memory was free than the card has, a desktop or another program used the
  rest. `OOM` means even 2x8 did not fit: that is a valid result, not a failure.
- **ASVspoof 5 estimates**: minutes for one training epoch (182,357 clips), one Dev pass
  (140,950 clips) and one Eval pass (680,774 clips) at the measured rate. Real clips are 4-10 s and
  are cropped or looped to the training length, so the cost per clip is constant.
- **Reference numbers** for our own GPUs (RTX 3090 and RTX A4500 workstations, A100 40 GB on Google
  Colab) come from **real training logs**, not from this benchmark: they include data loading, used
  torch 2.2.2, strict deterministic algorithms and no TF32, and some are estimates. The report labels them
  "reference: training log" and draws them hatched. Ratios against them are a good indication, not an
  exact comparison; exact comparisons come from running this benchmark on both GPUs (next section).
  Details: [references/README.md](references/README.md).

## Comparing GPUs

Run the benchmark on each GPU, collect the `results/*.jsonl` files in one folder and build one report:

```bash
uv run --no-sync python -m agbench report results/rtx-5060-ti-16gb_20261016.jsonl results/rtx-3090-24gb_20261018.jsonl
```

This writes `results/compare_<date>_report/` with one column per GPU in every table and in
`comparison.csv`. When comparing, prefer the same suite, the same `agbench` version and the same
torch build (the report warns when torch builds differ).

## Privacy

- The benchmark uses **synthetic audio only**; no recordings, no dataset.
- The results contain the GPU model, driver, OS, CPU model, core count, RAM size, package versions
  and the measured numbers. They do **not** contain the user name, host name, IP or MAC addresses,
  e-mail addresses, tokens, environment dumps or the names of other programs using the GPU (the
  doctor shows those on the console only); home-directory paths are replaced by `~`.
- Nothing is uploaded anywhere. The only network access is the download of PyTorch and the public
  model weights (a Hugging Face token is optional and is never written to any file).

## Development

```bash
uv sync --locked --extra cpu       # --locked also checks that uv.lock matches pyproject.toml
uv run --no-sync pytest -q
```

The tests are CPU-only and offline: they use tiny random models, hand-made sample results in
`tests/fixtures/`, and run the real command line end to end in tiny mode
(`python -m agbench run --tiny --device cpu ...`). GitHub Actions runs them on Linux and Windows.
Design details: [SPEC.md](SPEC.md).

Maintainers can check the GPU code paths on a shared GPU without a full run:
`uv run --no-sync python -m agbench run --smoke --mem-cap-gib 2 --scenario gpu.matmul,train.w2v2b.6s.bs16.bf16`
uses the real weights but batch 2, clips of at most 2 s and short loops, under a memory cap. Test runs
(`--tiny`, `--smoke`) are written under their own name (`<gpu>-tiny_...`, `<gpu>-smoke_...`), are
flagged in the report and are never compared with the references. What has been run on a real GPU so
far is listed in [notes/VALIDATION.md](notes/VALIDATION.md).

## Credits and licence

- RawBoost: H. Tak, M. Kamble, J. Patino, M. Todisco, N. Evans, "RawBoost: A Raw Data Boosting and
  Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing", ICASSP 2022. This
  repository contains a re-implementation that follows the official MIT-licensed code (same
  operations, default parameters and parameter names; no lines copied); see [NOTICE](NOTICE) and
  [LICENSES/RawBoost-antispoofing-MIT.txt](LICENSES/RawBoost-antispoofing-MIT.txt).
- Models: wav2vec 2.0 and XLS-R (Meta AI), WavLM (Microsoft), downloaded from Hugging Face under
  their own licences (see each model card).
- ASVspoof 5 clip counts are used only to convert speed into minutes per epoch.

MIT licence, see [LICENSE](LICENSE).
