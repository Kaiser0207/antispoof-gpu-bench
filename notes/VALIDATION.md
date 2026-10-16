# Validation record

What has been checked before the first release, where, and what has **not** been run on a GPU yet.
Date: 2026-10-16. Code: agbench 0.1.0, the first commit, after the pre-release review
(notes/REVIEW.md). Rows 1-4, 6 and 10-13 were rerun after the review fixes; rows 5 and 7-9 and
section 2 are from before them.

The short version:

- Every test and the whole command line work on CPU (Linux), with the pinned torch 2.14.1 and with
  torch 2.11.0 (the `cu128` fallback build).
- The GPU code paths ran once on a real GPU: an **RTX A4500 (workstation)**, Ampere, Linux, torch
  2.14.1+cu130, in a 77-second smoke run that shared the GPU with another job. Real
  `facebook/wav2vec2-base` weights were used, at batch 2 and 2 s clips only.
- **Nothing has run on Blackwell (sm_120), on an RTX 5060 Ti, on Windows, or at the real batch 16 /
  6 s sizes on any GPU.** The friend's run is the first real test of those.

## 1. Checks and results

| # | Check | Where | Result |
|---|---|---|---|
| 1 | `pytest` (all tests, CPU-only, tiny random models) | Linux, torch 2.14.1+cpu | 227 passed, 0 skipped, with the maintainer denylist active and the opt-in real-weight checks switched on (`AGBENCH_TEST_REAL_MODELS=1`); 3 min 40 s |
| 2 | Privacy test with a maintainer denylist (`AGBENCH_PRIVACY_DENYLIST`) | Linux | passed, over every file a commit would publish |
| 3 | `python -m agbench quick --tiny --device cpu` (real CLI, real subprocesses) | Linux, 6-core desktop CPU, nice 19 | 16 ok, 2 skipped (`gpu.pcie`: cpu_device; `cpu.loader.w8.blas1`: workers_exceed_cores), exit 0, 2 min 28 s. Report built; all records pass `validate_record` and `results_schema.json`; no user name, host name or home path in results or logs. The `e2e.w2v2b` row ran in strict deterministic mode with flash SDPA off and recorded it. |
| 4 | `python -m agbench full --tiny --device cpu` in the same results folder (resume) | same | the 18 quick scenarios print `cached`; the other 31 scenarios: 28 ok, 3 skipped (`cpu.loader.w12/w16.blas1`, `w8.blasdefault`: workers_exceed_cores); exit 0, 5 min 6 s. All 53 records are valid; no private strings in the results. |
| 5 | `report` on several files (2 fixtures + the GPU smoke run) | Linux | comparison report with one column per GPU and all 5 figures; the smoke run is labelled "(smoke test)" and has no reference ratios |
| 6 | Whole test suite against torch **2.11.0** (the `cu128` fallback), CPU wheel | Linux | 226 passed, 1 skipped (privacy denylist not set for this run), real-weight checks on. torch 2.11 has every API the GPU paths use: `cuda.matmul.fp32_precision`, `cudnn.conv.fp32_precision`, `get_per_process_memory_fraction`, `OutOfMemoryError`, `amp.GradScaler`, `utils.deterministic.fill_uninitialized_memory`, the SDPA backend switches. |
| 7 | `uv sync --locked --extra cu130` with an **older uv (0.9.28)** | Linux GPU workstation | installs from the committed lock in 1.5 min; venv 5.6 GB; torch 2.14.1+cu130, transformers 5.20.0 |
| 8 | `fetch --models w2v2b` into an empty Hugging Face cache | Linux GPU workstation | downloaded at the pinned revision, size verified (380,267,417 bytes); `fetch --check` then reports it as cached |
| 9 | Capped GPU smoke | RTX A4500 (workstation) | section 2 |
| 10 | Real checkpoints on CPU: all four load with transformers 5.20.0 (0 missing encoder weights), and one forward + backward pass of the real W2V2 Base classifier and of the real fusion classifier (XLS-R 300M + WavLM Base + FusionHead) gives finite gradients | Linux, CPU only, nice 19 | 6 passed (`-k real_weights`), with torch 2.14.1 and 2.11.0. `facebook/wav2vec2-large` and `microsoft/wavlm-base` were downloaded for this (1.65 GB) |
| 11 | Pinned SHA-256 of all four `pytorch_model.bin` files | Hugging Face API at the pinned revisions; local files | the four LFS hashes in `agbench/models.py` match the API, and `fetch` verified the downloaded files against them |
| 12 | Real-size `cpu.loader` (6 s, batch 16, RawBoost p=0.5) with the new steady-state window | Linux, 6-core desktop CPU, nice 19 | w1 / w2 / w4 with `OPENBLAS_NUM_THREADS=1`: 37.5 / 73.9 / 135.8 clips/s (backlog 2 / 4 / 8 batches left out, 30-32 steady batches, none capped); w4 with default BLAS threads 44.1 clips/s. Same as before the change at these worker counts, as expected |
| 13 | Killed runner (SIGKILL, as a tool timeout does) | Linux, CPU | the scenario child exits at once; the next run stops a leftover child of an older run and finishes with exit 0 (`test_killed_runner_leaves_no_child_and_next_run_cleans_up`) |

## 2. GPU smoke run (RTX A4500, shared GPU)

**Setup.** RTX A4500, 20 GB class (torch reports 18.32 GiB), compute capability 8.6, driver
595.91.07, PCIe 4.0 x16. Software: Linux, Python 3.12.10, torch 2.14.1+cu130 (cuDNN 9.24),
transformers 5.20.0. Another training process used 14.9 GB of memory and about 97 % of the GPU for
the whole run. **The speed numbers are therefore meaningless; the run checks code paths only.**

**Safety limits.**

- Before the run, a probe measured how much memory a CUDA process uses outside PyTorch's allocator:
  232 MiB.
- From that, the allocator cap was set with `--mem-cap-gib 1.98`, so that the whole process stays
  under 2.5 GB.
- A watchdog polled `nvidia-smi` every 0.5 s. It was set to kill any of our processes above
  2,384 MiB; it never had to.
- Our largest process used 2,256 MiB.
- The other process held 14,930 MiB before and after the run. GPU utilisation was 96 % before and
  97 % after.
- No agbench process was left afterwards.
- Total GPU time: 77 s.

**Command** (all 11 scenarios ran at batch 2 and 2 s with real `facebook/wav2vec2-base` weights,
because of `--smoke`):

```
python -m agbench run --smoke --mem-cap-gib 1.98 --budget-min 0 --no-fetch --no-network --no-resume \
  --scenario gpu.matmul,gpu.membw,gpu.pcie,train.w2v2b.6s.bs16.bf16,train.w2v2b.6s.bs16.fp16,\
train.w2v2b.6s.bs16.bf16.ckpt,train.w2v2b.6s.bs16.bf16.det,infer.w2v2b.6s.bs16.bf16,\
frozen.w2v2b.6s.bs16.bf16,e2e.w2v2b.6s.bs16.bf16,sustain.w2v2b.6s.bs16.bf16
```

Result: 10 ok, 1 oom, exit 0, 61 s. The report was built, and every record is valid.

| Scenario (run as 2 s, batch 2) | Status | Peak allocated / reserved GiB | What it exercised on the GPU |
|---|---|---|---|
| `gpu.matmul` (n=1024) | ok | 0.03 / 0.04 | bf16, fp16, TF32 and strict fp32 matmuls (all results finite); CUDA-event batches; clock ramp-up |
| `gpu.membw` (64 MiB) | ok | 0.19 / 0.19 | copy, read (sum) and triad |
| `gpu.pcie` (32 MiB) | ok | 0.03 / 0.03 | pinned host-to-device 26.2 GB/s, pinned device-to-host 22.6 GB/s, pageable host-to-device 12.2 GB/s; NVML link read during the transfer (gen 4 x16 = 31.5 GB/s theoretical); round-trip copy check |
| `train.w2v2b...bf16` | ok | 1.80 / 1.98 | real weights at the pinned revision; the legacy checkpointing flag in the config was detected and switched off; SDPA attention; bf16 autocast; AdamW (foreach); gradient clipping; finite loss |
| `train.w2v2b...fp16` | ok | 1.80 / 1.98 | fp16 autocast with GradScaler (`unscale_`, clipping, `step`, `update`) |
| `train.w2v2b...bf16.ckpt` | ok | 1.80 / 1.96 | activation checkpointing switched on (`use_reentrant=False`) and confirmed |
| `train.w2v2b...bf16.det` | **oom** at 2x1, then 1x2 | (cap 1.98) | deterministic mode with `CUBLAS_WORKSPACE_CONFIG`. Forward and backward completed; the AdamW `_foreach_sqrt` temporary went over the 1.98 GiB cap. This also exercised a real allocator OOM, classified as `oom`, followed by the next rung in a fresh process. |
| `infer.w2v2b...bf16` | ok | 0.43 / 0.49 | non-blocking copies from pinned host memory; `inference_mode`; scores finite |
| `frozen.w2v2b...bf16` | ok | 0.49 / 0.60 | no-grad encoder with all 13 hidden states; probe-head set; AdamW on the heads only |
| `e2e.w2v2b...bf16` | ok | 1.80 / 1.98 | 2 spawn DataLoader workers decoding FLAC and feeding the GPU; loader-wait metrics |
| `sustain.w2v2b...bf16` (3 s, 1 s windows) | ok | 1.80 / 1.98 | per-window clips/s merged with NVML temperature, clock and power; telemetry series kept |

**Real out-of-memory test.** We ran a second command,
`run --smoke --mem-cap-gib 0.25 --scenario infer.w2v2b.6s.bs16.bf16`. The weights need 0.36 GiB, so
this cap is too small. The model load hit a real CUDA allocator OOM at 2x1, the runner tried 1x2 in a
fresh process, and that also ran out of memory. The status was `oom`, the exit code 0, and it took
10 s.

**Also exercised:**

- **Doctor, GPU branch, all checks:**
  - torch build (CUDA 13.0);
  - CUDA available;
  - arch list contains this GPU's architecture (`sm_86`);
  - kernel smoke: bf16 matmul, conv1d and SDPA on the GPU, compared with the CPU;
  - driver 595.91 >= 580 for CUDA 13;
  - VRAM, with a warning that listed the other GPU processes by name;
  - PCIe link through NVML;
  - idle power, clock and temperature.
- **Child process:**
  - memory cap through `set_per_process_memory_fraction`: the smaller of free VRAM minus 0.5 GiB and `--mem-cap-gib`;
  - peak memory reset and read;
  - the `near_limit` flag;
  - the NVML `GpuSampler`, which found the device by UUID;
  - `os._exit` after writing the attempt file.
- **Report:** built from GPU records, including the memory-fit and sustained-run figures.

**Observations:**

- At batch 2 and 2 s, W2V2 Base training needs about 1.8 GiB allocated, 2.0 GiB reserved. Most of
  that is weights, gradients and AdamW state (1.43 GiB), plus the full-size AdamW foreach
  temporary. The 0.5 GiB headroom that the cap leaves (free VRAM minus 0.5 GiB) is enough for the
  ~0.23-0.28 GiB that CUDA uses outside the allocator on this setup.
- At the time of this smoke run, `.det` used `use_deterministic_algorithms(True, warn_only=True)`.
  With SDPA, torch warned that the attention backward stays non-deterministic in warn-only mode, so
  the variant did **not** reproduce the reference runs' setting. After the review (notes/REVIEW.md,
  R1), `.det` and the two `e2e` rows that are compared with the training logs use strict mode, TF32
  off for cuDNN convolutions and flash attention off for 6 s clips, as the reference runs did. That
  strict path has run on CPU only (section 6).
- The memory cap at the time of this run was free VRAM minus 0.5 GiB on every OS; it is now 0.5 GiB
  on Windows and 0.1 GiB elsewhere.

## 3. Not exercised on any GPU yet

- **Blackwell and the target card.** Nothing ran on `sm_120`, an RTX 5060 Ti (16 GB or 8 GB), or
  Windows. Untested on Windows:
  - spawn DataLoader workers feeding a GPU;
  - the sysmem-fallback heuristics;
  - `CREATE_NEW_PROCESS_GROUP` timeouts on a GPU.

  The Windows CPU tests run only in CI, after the push.
- **The `cu128` build (torch 2.11.0) on a GPU.** It was tested on CPU only (section 1, check 6).
- **Real sizes.** Nothing ran at batch 16 with 4-10 s clips. XLS-R 300M, W2V2 Large, WavLM Base and
  the fusion were not loaded on a GPU: their weights were not downloaded there, and they do not fit
  under the 2.5 GB limit. Builder B loaded all four checkpoints on CPU. As a result:
  - whether batch 16 fits on a 16 GB card is unverified;
  - the full ladder (16x1 -> 8x2 -> 4x4 -> 2x8) is unverified.

  The ladder logic ran with simulated OOMs on CPU, and once with a real OOM in its tiny form (2x1 -> 1x2).
- **Reference numerics on a GPU:** strict `use_deterministic_algorithms(True)` with flash SDPA off
  and cuDNN conv TF32 off (`.det`, `e2e.w2v2b.6s.bs16.bf16`, `e2e.xlsr.6s.bs16.bf16`). If an op has no
  deterministic CUDA kernel in torch 2.14 / 2.11, the child falls back to warn-only and the report
  says so; this has only been tested with a stub. Not run on a GPU because the task rules allow the
  shared RTX A4500 for one capped smoke only, which was used before the review.
- **Process handling added after the review:** the Windows kill-on-close Job Object, the
  `SetThreadExecutionState` sleep inhibitor, and `systemd-inhibit` in a desktop session (on this
  host it is refused over SSH, which the code handles). Killing the runner was tested on Linux only.
- **Other paths not run on a GPU:**
  - TF32 training (now through the new `fp32_precision` API only);
  - `e2e...rawboost`: RawBoost in workers, `pin_memory=True`, persistent workers, `"auto"` worker count;
  - `e2e.xlsr`;
  - `frozen...h40` (40 heads);
  - batch-4 inference;
  - the batch-32 no-ladder row.
- **Full-size GPU basics:**
  - matmul at n=4096/8192;
  - 1 GiB `membw` buffers and the buffer shrink under a cap;
  - 256 MiB PCIe copies.
- **Long runs:**
  - the 300 s sustained run;
  - throttle detection under real heat;
  - the time-budget guard with real timings;
  - the actual quick/full wall-clock times. The 10-15 and 32-42 min figures are estimates (SPEC 6.3).
- **Telemetry fallback.** The `nvidia-smi` fallback was not used, because NVML was available. Builder C
  checked it read-only.
- **Timing accuracy.** It cannot be judged on a shared GPU.

## 4. Changes made during integration

1. **Test runs no longer mix with real results.** `--tiny` on a CUDA device used to write to the same
   `results/<gpu>_<date>.jsonl` as a real run. The report then treated the whole GPU as a test run,
   or let tiny records replace real ones. Test runs now get their own slug: `<gpu>-tiny` or
   `<gpu>-smoke` (`runner.run_slug`). That gives them their own results file, resume entries and
   report column. Tests: `test_run_slug_separates_test_runs`.
2. **New option `--smoke`** (maintainers only). It uses real weights with the tiny-mode shapes and
   loops (batch 2, at most 2 s, `MeasureCfg.tiny()`) and small GPU-basics buffers
   (`scenarios.SMOKE_BASICS`). Records carry `settings.smoke`. The report says "smoke test" in the
   title, the GPU label and the check list, and prints "smoke run, n/a" instead of reference ratios.
   AGENTS.md tells the agent not to use test options. Tests: `test_effective_smoke_transform`,
   `test_smoke_runs_are_flagged_and_not_compared`.
3. **Stale fixtures regenerated.** The `gpu.matmul` and `cpu.loader.*` rows of the fixtures used
   older parameter names and hashes, so they were regenerated from the current scenario matrix with
   `tests/fixtures/make_fixtures.py`.
4. **Repository files.**
   - `.gitignore`: added `/agbench_results*.zip` (AGENTS.md step 8 writes it).
   - `.gitattributes`: keeps `*.sh` at LF line endings.
   - SPEC.md: sections 7 and 8.5 now describe `--smoke` and the test-run slugs.

No defect was found in the GPU code paths themselves during the smoke run.

## 5. Known gaps left open

- `agbench/` has about 8,500 lines of Python against SPEC's target of under 4,000. Nothing is
  broken; it is just larger than planned.
- The tiny `full` suite takes 5-8 min on a loaded 6-core desktop CPU, against SPEC's 5-min CI
  target. CI runs only `pytest`, which takes about 4 min here, so CI is not affected.
- **On CPU, `--smoke` is impractical** on CPUs without native bf16. bf16 autocast on such a CPU is
  about 80x slower than fp32, so real-weight training steps time out. Use `--smoke` on a GPU, and
  `--tiny` on a CPU.

## 6. Changes after the pre-release review

All findings of the three reviews and what was done with each are in
[notes/REVIEW.md](REVIEW.md). In short: runs survive agent tool time limits (background start in
AGENTS.md, children die with a killed runner, leftovers are stopped), the rows compared with our
training logs now use the logs' numerics and a wall-clock rate, the `cpu.loader` rate leaves out
prefetched batches, model files are checked against pinned SHA-256 hashes, the doctor names the
right PyTorch build for the driver and no longer saves the names of other programs, and AGENTS.md
covers new shells per command, Git Bash, the `-open` Linux driver, several GPUs and `quick` then
`full`. Rows 1-4, 6 and 10-13 of section 1 were run after these changes; nothing new ran on a GPU.
