# Reference numbers

`reference_gpus.json` holds the numbers the report compares a new GPU against. There are two kinds,
and the report always says which one it is using.

## 1. Training-log references (`log_references`)

These come from **our real training runs**, not from this benchmark:

- PyTorch 2.2.2 and transformers 4.x (this benchmark uses torch 2.14 / transformers 5.20);
- data loading included (FLAC decode and crop in 2 spawn DataLoader workers, unless noted);
- bf16 autocast, AdamW, gradient clipping 1.0, LayerDrop 0, activation checkpointing off;
- strict deterministic algorithms (`torch.use_deterministic_algorithms(True)`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`), TF32 off for matmuls and cuDNN convolutions, and flash
  attention off for clips longer than 4 s (the 6 s runs).

The benchmark's GPU-only anchors run with PyTorch defaults (non-deterministic, flash attention on).
The rows compared with these logs, `e2e.w2v2b.6s.bs16.bf16` and `e2e.xlsr.6s.bs16.bf16`, use the
logs' numerics and the same loader setup, and their clips/s is clips divided by wall-clock time;
`train.w2v2b.6s.bs16.bf16.det` measures what these numerics cost on the GPU alone. The fusion
reference has no with-loader row, so its ratio uses the GPU-only rate (marked `*` in the report).

So a ratio against them is a good indication, not an exact comparison. The report draws them
hatched and labels them "reference: training log, data loading included, torch 2.2.2".

| GPU | Workload | Number | Provenance |
|---|---|---|---|
| RTX 3090 (workstation) | W2V2 Base, 6 s, batch 16, train | 101.8 clips/s (29.9 min per epoch), Dev 303 clips/s; peak 5.25 GiB allocated / 6.63 GiB reserved | median of 14 epoch records of one run; 2 workers, no augmentation |
| RTX 3090 (workstation) | XLS-R 300M, 6 s, batch 16, train | about 38 clips/s, **estimate** | partial runs and a loader emulation (36.9 clips/s without augmentation); not a completed epoch |
| RTX 3090 (workstation) | XLSR-53 + WavLM Base fusion, 4 s, batch 16, train | 40.6 clips/s (74.9 min per epoch); peak 13.6 GiB allocated / 14.9 GiB reserved | XLSR-53 has the same architecture as XLS-R 300M, which this benchmark uses |
| RTX A4500 (workstation) | W2V2 Base, 6 s, batch 16, train | 72.7 clips/s (41.8 min per epoch), Dev 236 clips/s (10.0 min) | pooling-baseline run, 2 workers |
| RTX A4500 (workstation) | XLS-R 300M, 6 s, batch 16, train | 28.7 clips/s (106 min per epoch) | steady-state rate of a running job, 2 workers; about 15 GB of 20 GB in use |
| A100 40 GB (Google Colab) | W2V2 Base, 6 s, Dev scoring | 450 clips/s | Dev pass inside a training run |
| A100 40 GB (Google Colab) | W2V2 Base, 6 s, batch 16, train with RawBoost | 91 clips/s, **CPU-bound** | limited by RawBoost on the CPU (observed 90.8 clips/s); not a GPU number, so the report shows "CPU-bound, not comparable" instead of a ratio |

`maps_to` lists the benchmark scenarios each number corresponds to. A Dev rate recorded with a
training run is compared with the matching batch-16 inference scenario (for example
`infer.w2v2b.6s.bs16.bf16`).

`estimates` holds the pre-run expectation for an RTX 5060 Ti 16 GB (0.55x an RTX 3090, range
0.45-0.67, from memory bandwidth and tensor throughput ratios; see SPEC.md section 6.3). The report
prints it next to the measured numbers for that GPU only.

## 2. Runs of this benchmark (`agbench_runs`)

Exact comparisons need this benchmark run on our own GPUs too. To add one:

1. Run the benchmark on that GPU (`quick` or `full`, same `agbench` version as the run you compare with).
2. Copy its results file into `references/runs/`, for example
   `references/runs/rtx-3090-24gb_20261020.jsonl`. Check it first: it must not contain private
   paths or names (`pytest tests/test_privacy.py` scans it).
3. List it under that GPU's entry in `reference_gpus.json`:

   ```json
   "agbench_runs": [
     {"file": "runs/rtx-3090-24gb_20261020.jsonl", "suite": "full", "date": "2026-10-20",
      "notes": "same settings as the published benchmark"}
   ]
   ```

The report then adds an `agbench_<gpu_key>` column to `comparison.csv`, draws that GPU as a solid
bar labelled "(this benchmark)", and computes the ratio columns from it (GPU only vs GPU only)
wherever it has the scenario, falling back to the training log otherwise.
