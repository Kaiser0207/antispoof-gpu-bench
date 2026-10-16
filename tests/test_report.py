"""Report generator on the hand-made fixtures (SPEC 10, 13.3): one GPU, two GPUs, a run with
failures, missing scenarios, missing references, agbench reference runs, and no torch import."""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agbench.report import LOG_REF_LABEL, build_report

FIX = Path(__file__).parent / "fixtures"
ONE = FIX / "rtx-5060-ti-16gb_20261016.jsonl"
OTHER = FIX / "rtx-3090-24gb_20261016.jsonl"
PROBLEMS = FIX / "rtx-5060-ti-8gb_20261017.jsonl"
REFS = Path(__file__).resolve().parents[1] / "references" / "reference_gpus.json"
ALL_FIGURES = ["fig1_training_throughput.png", "fig2_memory_fit.png", "fig3_length_scaling.png",
               "fig4_cpu_pipeline.png", "fig5_sustained.png"]


def _csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    return header, [dict(zip(header, r)) for r in rows[1:]]


def _row(rows: list[dict], sid: str, metric: str) -> dict:
    for r in rows:
        if r["scenario_id"] == sid and r["metric"] == metric:
            return r
    raise AssertionError(f"no comparison row {sid} / {metric}")


def _section(text: str, title: str) -> str:
    """Body of the '## N. <title>' section."""
    m = re.search(rf"^## \d+\. {re.escape(title)}\n", text, re.M)
    assert m, title
    return text[m.end():].split("\n## ", 1)[0]


def _figures_in(summary: str) -> list[str]:
    return re.findall(r"!\[[^\]]*\]\(figures/([^)]+)\)", summary)


@pytest.fixture(scope="module")
def one_report(tmp_path_factory):
    out = tmp_path_factory.mktemp("one")
    return build_report([ONE], out, references_path=REFS)


def test_one_gpu_writes_summary_csv_and_figures(one_report):
    out = one_report.parent
    assert one_report.name == "summary.md" and one_report.exists()
    assert (out / "comparison.csv").exists()
    for name in ALL_FIGURES:
        png = out / "figures" / name
        assert png.exists() and png.stat().st_size > 5000, name
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_summary_layout_and_labels(one_report):
    text = one_report.read_text(encoding="utf-8")
    assert text.startswith("# GPU benchmark: GeForce RTX 5060 Ti (16 GB)")
    first_lines = text.splitlines()[:4]
    assert any("driver 580.97" in ln and "torch 2.14.1+cu130" in ln and "49 scenarios" in ln for ln in first_lines)
    # Figures first in each section, each with a method line; every referenced figure exists.
    figs = _figures_in(text)
    assert figs == ALL_FIGURES
    for name in figs:
        assert (one_report.parent / "figures" / name).exists()
        after = text.split(f"(figures/{name})", 1)[1].lstrip().splitlines()[0]
        assert after.startswith("*Method:*"), name
    assert text.index("fig1_training_throughput.png") < text.index("What this means for ASVspoof 5")
    for heading in ("Training throughput vs our GPUs", "What this means for ASVspoof 5", "Memory fit",
                    "Inference", "Can the CPU keep up?", "GPU basics", "Sustained run",
                    "Scenarios that did not finish ok", "Environment and provenance"):
        assert re.search(rf"^## \d+\. {re.escape(heading)}", text, re.M), heading
    # Reference labelling: never presented as runs of this benchmark.
    assert "not from this benchmark" in text
    assert "training logs, not this benchmark" in text
    assert "CPU-bound, not comparable" in text
    assert "(estimate)" in text or "(ref est.)" in text
    refs_notes = json.loads(REFS.read_text(encoding="utf-8"))["notes"]
    assert refs_notes in text


def test_asv5_numbers(one_report):
    text = one_report.read_text(encoding="utf-8")
    row = next(ln for ln in text.splitlines() if ln.startswith("| W2V2 Base, 6 s |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    # workload, GPU-only, loader, fit, peak, epoch, dev, epoch+dev, eval, vs 3090, vs A4500
    assert cells[1] == "56.3" and cells[2] == "54.9" and cells[3] == "16x1"
    assert cells[5] == "54.0"                       # 182357 / 56.3 / 60
    assert cells[6] == "14.0"                       # 140950 / 168 / 60
    assert cells[8] == "67.5"                       # 680774 / 168 / 60
    assert cells[9] == "0.54x"                      # with-loader 54.9 / 101.8 (log includes data loading)
    assert cells[10] == "0.76x"                     # 54.9 / 72.7
    xlsr = next(ln for ln in text.splitlines() if ln.startswith("| XLS-R 300M, 6 s |"))
    assert "8x2 (OOM at 16x1 under a 14.5 GiB cap; 15.5 GiB were free)" in xlsr and "(ref est.)" in xlsr


def test_latest_record_wins_over_budget_skip(one_report):
    # cpu.loader.w8.blas1 was budget-skipped in the quick run and completed by the full run.
    _, rows = _csv(one_report.parent / "comparison.csv")
    assert _row(rows, "cpu.loader.w8.blas1", "status")["rtx-5060-ti-16gb"] == "ok"
    text = one_report.read_text(encoding="utf-8")
    problems = _section(text, "Scenarios that did not finish ok")
    assert "cpu.loader.w8.blas1" not in problems
    assert "cpu.loader.w16.blas1" in problems and "workers_exceed_cores" in problems


def test_comparison_csv(one_report):
    header, rows = _csv(one_report.parent / "comparison.csv")
    assert header[:4] == ["scenario_id", "metric", "unit", "rtx-5060-ti-16gb"]
    assert header[-1] == "ref_source"
    for key in ("rtx-3090-24gb", "rtx-a4500-20gb", "a100-40gb-colab"):
        assert f"ref_{key}" in header
    r = _row(rows, "train.w2v2b.6s.bs16.bf16", "clips_per_s")
    assert float(r["rtx-5060-ti-16gb"]) == pytest.approx(56.3)
    assert float(r["ref_rtx-3090-24gb"]) == pytest.approx(101.8)
    assert float(r["ref_rtx-a4500-20gb"]) == pytest.approx(72.7)
    assert "training log" in r["ref_source"]
    # Dev throughput logged with a training run maps onto the inference scenario.
    r = _row(rows, "infer.w2v2b.6s.bs16.bf16", "clips_per_s")
    assert float(r["ref_rtx-3090-24gb"]) == pytest.approx(303)
    assert float(r["ref_a100-40gb-colab"]) == pytest.approx(450)
    r = _row(rows, "e2e.w2v2b.6s.bs16.bf16.rawboost", "clips_per_s")
    assert float(r["ref_a100-40gb-colab"]) == pytest.approx(91)
    assert "CPU-bound, not comparable" in r["ref_source"]
    r = _row(rows, "train.xlsr.6s.bs16.bf16", "batch_used")
    assert r["rtx-5060-ti-16gb"] == "8x2 (OOM at 16x1 under a 14.5 GiB cap; 15.5 GiB were free)"
    assert float(_row(rows, "gpu.matmul", "tflops_bf16_8192")["rtx-5060-ti-16gb"]) == pytest.approx(46.8)
    assert _row(rows, "gpu.pcie", "link")["rtx-5060-ti-16gb"] == "gen5 x8"
    assert float(_row(rows, "cpu.rawboost.blas1", "stage_ms_lnl")["rtx-5060-ti-16gb"]) == pytest.approx(31.0)


def test_two_gpus(tmp_path):
    summary = build_report([ONE, OTHER], tmp_path / "cmp", references_path=REFS)
    text = summary.read_text(encoding="utf-8")
    assert text.startswith("# GPU benchmark comparison: RTX 5060 Ti 16 GB vs RTX 3090 24 GB")
    assert "| RTX 5060 Ti 16 GB | W2V2 Base, 6 s |" in text and "| RTX 3090 24 GB | W2V2 Base, 6 s |" in text
    header, rows = _csv(summary.parent / "comparison.csv")
    assert header[3:5] == ["rtx-5060-ti-16gb", "rtx-3090-24gb"]
    r = _row(rows, "train.w2v2b.6s.bs16.bf16", "clips_per_s")
    assert float(r["rtx-3090-24gb"]) == pytest.approx(104.6)
    # Scenarios only the first GPU ran keep an empty cell for the second.
    assert _row(rows, "train.wavlmb.6s.bs16.bf16", "clips_per_s")["rtx-3090-24gb"] == ""
    for name in ALL_FIGURES:
        assert (summary.parent / "figures" / name).exists(), name


def test_problem_run_without_sustain(tmp_path):
    summary = build_report([PROBLEMS], tmp_path / "bad", references_path=REFS)
    text = summary.read_text(encoding="utf-8")
    figs = sorted(p.name for p in (summary.parent / "figures").glob("*.png"))
    assert "fig5_sustained.png" not in figs and "fig3_length_scaling.png" not in figs
    assert "fig1_training_throughput.png" in figs and "fig2_memory_fit.png" in figs
    assert "Figure 3: Clip length scaling: no data" in text
    assert "Sustained run" not in text
    problems = _section(text, "Scenarios that did not finish ok")
    for sid, status in (("train.xlsr.6s.bs16.bf16", "oom"), ("infer.xlsr.6s.bs16.bf16", "error"),
                        ("train.w2v2l.6s.bs16.bf16", "crashed"), ("train.fusion.4s.bs16.bf16", "timeout"),
                        ("frozen.xlsr.6s.bs16.bf16", "skipped"), ("train.w2v2b.10s.bs16.bf16", "skipped")):
        line = next((ln for ln in problems.splitlines() if f"`{sid}`" in ln), None)
        assert line is not None and f"| {status} |" in line, sid
    assert "OOM at every split (16x1 to 2x8 under a " in text
    assert "model_missing" in problems and "budget" in problems
    check = text.split("**Check first:**", 1)[1].split("> **How to read", 1)[0]
    assert "3 scenario(s) failed" in check and "did not fit in GPU memory" in check


def test_default_output_dir(tmp_path):
    src = tmp_path / "results"
    src.mkdir()
    shutil.copy(ONE, src / ONE.name)
    summary = build_report([src / ONE.name], references_path=REFS)
    assert summary == src / "rtx-5060-ti-16gb_20261016_report" / "summary.md"
    shutil.copy(OTHER, src / OTHER.name)
    summary2 = build_report([src / ONE.name, src / OTHER.name], references_path=REFS)
    assert summary2.parent.parent == src and re.fullmatch(r"compare_\d{8}_report", summary2.parent.name)


def test_missing_references_file(tmp_path):
    summary = build_report([ONE], tmp_path / "r", references_path=tmp_path / "nope.json")
    text = summary.read_text(encoding="utf-8")
    assert "reference file not found" in text
    header, _ = _csv(summary.parent / "comparison.csv")
    assert not any(h.startswith("ref_") and h != "ref_source" for h in header)
    assert (summary.parent / "figures" / "fig1_training_throughput.png").exists()


def test_agbench_reference_runs_are_used(tmp_path):
    refdir = tmp_path / "references"
    (refdir / "runs").mkdir(parents=True)
    shutil.copy(OTHER, refdir / "runs" / OTHER.name)
    data = json.loads(REFS.read_text(encoding="utf-8"))
    for g in data["gpus"]:
        if g["gpu_key"] == "rtx-3090-24gb":
            g["agbench_runs"] = [{"file": f"runs/{OTHER.name}", "suite": "quick", "date": "2026-10-16"}]
    (refdir / "reference_gpus.json").write_text(json.dumps(data), encoding="utf-8")
    summary = build_report([ONE], tmp_path / "r", references_path=refdir / "reference_gpus.json")
    text = summary.read_text(encoding="utf-8")
    assert "RTX 3090 (workstation)'s run of this benchmark" in text
    header, rows = _csv(summary.parent / "comparison.csv")
    assert "agbench_rtx-3090-24gb" in header
    r = _row(rows, "train.w2v2b.6s.bs16.bf16", "clips_per_s")
    assert float(r["agbench_rtx-3090-24gb"]) == pytest.approx(104.6)
    # With an exact run available, the ratio compares GPU-only with GPU-only: 56.3 / 104.6.
    row = next(ln for ln in text.splitlines() if ln.startswith("| W2V2 Base, 6 s |"))
    assert "0.54x" in row


def test_minimal_and_odd_records_do_not_crash(tmp_path):
    p = tmp_path / "odd.jsonl"
    lines = [
        {"kind": "run_start", "schema_version": "1", "run_id": "r1", "suite": "run", "gpu_slug": "cpu",
         "env": {"os_family": "linux", "python": "3.12.0"}},
        {"kind": "scenario", "run_id": "r1", "gpu_slug": "cpu", "scenario_id": "train.w2v2b.6s.bs16.bf16",
         "status": "ok", "metrics": {"clips_per_s": 3.0}},
        {"kind": "scenario", "run_id": "r1", "gpu_slug": "cpu", "scenario_id": "train.xlsr.6s.bs16.bf16",
         "status": "oom", "fallback": None},
        {"kind": "scenario", "run_id": "r1", "scenario_id": "cpu.loader.w2.blas1", "status": "ok",
         "metrics": {"clips_per_s": "fast"}},
        {"kind": "scenario", "run_id": "r1", "gpu_slug": "cpu", "scenario_id": "something.new", "status": "error",
         "error": {"type": "X", "message": "a | b\nsecond line"}},
        {"kind": "mystery"},
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + '\n{"kind": "scen', encoding="utf-8")
    summary = build_report([p], tmp_path / "odd", references_path=REFS)
    text = summary.read_text(encoding="utf-8")
    assert "something.new" in text and "a \\| b" in text
    assert (summary.parent / "comparison.csv").exists()


def test_errors_for_unusable_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_report([tmp_path / "missing.jsonl"], tmp_path / "x", references_path=REFS)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        build_report([empty], tmp_path / "y", references_path=REFS)


def test_report_never_imports_torch(tmp_path):
    code = (
        "import sys; from pathlib import Path; from agbench.report import build_report; "
        f"build_report([Path({str(ONE)!r})], Path({str(tmp_path / 'nt')!r}), references_path=Path({str(REFS)!r})); "
        "bad = [m for m in ('torch', 'transformers') if m in sys.modules]; "
        "sys.exit('imported: ' + ', '.join(bad) if bad else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parents[1], timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_markdown_tables_are_well_formed(one_report):
    text = one_report.read_text(encoding="utf-8")
    # Tables must have a consistent number of cells per row.
    for block in re.findall(r"((?:^\|.*\|\n)+)", text, re.M):
        widths = {ln.count(" | ") for ln in block.strip().splitlines()}
        assert len(widths) == 1, block[:200]
    assert LOG_REF_LABEL.startswith("reference: training log")


def test_tiny_runs_are_flagged_and_not_compared(tmp_path):
    p = tmp_path / "cpu_20261016.jsonl"
    lines = []
    for line in ONE.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["kind"] == "run_start":
            rec["settings"]["tiny"] = True
        lines.append(json.dumps(rec))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = build_report([p], tmp_path / "tiny", references_path=REFS)
    text = summary.read_text(encoding="utf-8")
    assert "tiny / CPU test run" in text.split("> **How to read", 1)[0]
    row = next(ln for ln in text.splitlines() if ln.startswith("| W2V2 Base, 6 s |"))
    assert "tiny run, n/a" in row and "x |" not in row


def test_smoke_runs_are_flagged_and_not_compared(tmp_path):
    p = tmp_path / "rtx-a4500-20gb-smoke_20261016.jsonl"
    lines = []
    for line in ONE.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["kind"] == "run_start":
            rec["settings"]["smoke"] = True
        lines.append(json.dumps(rec))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = build_report([p], tmp_path / "smoke", references_path=REFS)
    text = summary.read_text(encoding="utf-8")
    assert "smoke test run" in text.split("> **How to read", 1)[0]
    row = next(ln for ln in text.splitlines() if ln.startswith("| W2V2 Base, 6 s |"))
    assert "smoke run, n/a" in row and "x |" not in row
