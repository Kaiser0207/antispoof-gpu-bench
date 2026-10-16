"""End to end through real subprocesses (SPEC 13.3): a tiny CPU run, resume, --no-resume, report.

Everything runs on the CPU with tiny random models; results, logs and the corpus go to tmp_path.
"""

from __future__ import annotations

from pathlib import Path

from agbench.schema import load_records, validate_record
from conftest import by_id, explain, scenario_records

SCENARIOS = ["train.w2v2b.6s.bs16.bf16", "cpu.decode"]
TINY = ["--tiny", "--device", "cpu"]


def _scenario_args(ids):
    out = []
    for sid in ids:
        out += ["--scenario", sid]
    return out


def test_list_works_without_running_anything(run_agbench):
    proc = run_agbench("list", "--suite", "full", timeout=120)
    assert proc.returncode == 0, explain(proc)
    for sid in SCENARIOS:
        assert sid in proc.stdout


def test_tiny_run_resume_no_resume_and_report(run_agbench, tmp_path):
    out = tmp_path / "results" / "tiny.jsonl"

    # 1) First run: everything runs and reaches a terminal status; the report is built at the end.
    p1 = run_agbench("run", *TINY, *_scenario_args(SCENARIOS), "--out", out)
    assert p1.returncode == 0, explain(p1, out)
    recs = load_records(out)
    assert recs and recs[0]["kind"] == "run_start" and recs[-1]["kind"] == "run_end"
    for rec in recs:
        assert validate_record(rec) == [], (rec.get("scenario_id"), validate_record(rec))
    first = by_id(scenario_records(out))
    assert set(first) == set(SCENARIOS)
    for sid in SCENARIOS:
        assert first[sid]["status"] == "ok", explain(p1, out)
    train = first["train.w2v2b.6s.bs16.bf16"]
    assert train["metrics"]["clips_per_s"] > 0 and train["metrics"]["loss_finite"] is True
    assert recs[0]["settings"]["tiny"] is True
    summaries = list((tmp_path / "results").glob("*_report/summary.md"))
    assert summaries, "no report was built at the end of the run:\n" + explain(p1)

    # 2) Same command again: resume finds both scenarios done and runs nothing.
    n_before = len(scenario_records(out))
    p2 = run_agbench("run", *TINY, *_scenario_args(SCENARIOS), "--out", out, "--no-report")
    assert p2.returncode == 0, explain(p2, out)
    assert "cached" in p2.stdout.lower(), explain(p2)
    assert len(scenario_records(out)) == n_before

    # 3) --no-resume runs the scenario again and appends a new record.
    p3 = run_agbench("run", *TINY, "--scenario", "cpu.decode", "--out", out, "--no-resume", "--no-report")
    assert p3.returncode == 0, explain(p3, out)
    after = scenario_records(out)
    assert len(after) == n_before + 1 and after[-1]["scenario_id"] == "cpu.decode"
    assert after[-1]["run_id"] != first["cpu.decode"]["run_id"]

    # 4) Explicit report into a chosen folder.
    rep = tmp_path / "rep"
    p4 = run_agbench("report", out, "--out", rep, timeout=300)
    assert p4.returncode == 0, explain(p4)
    assert (rep / "summary.md").exists() and (rep / "comparison.csv").exists()
    text = (rep / "summary.md").read_text(encoding="utf-8")
    assert "train.w2v2b.6s.bs16.bf16" in (rep / "comparison.csv").read_text(encoding="utf-8")
    assert "Environment and provenance" in text


def test_results_contain_no_home_path(run_agbench, tmp_path):
    out = tmp_path / "results" / "privacy.jsonl"
    proc = run_agbench("run", *TINY, "--scenario", "gpu.membw", "--out", out, "--no-report")
    assert proc.returncode == 0, explain(proc, out)
    home = str(Path.home())
    for path in [out, *(tmp_path / "results").rglob("*.log"), *(tmp_path / "results").rglob("*.json")]:
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(home) > 3:
            assert home not in text and home.replace("\\", "/") not in text, path.name
