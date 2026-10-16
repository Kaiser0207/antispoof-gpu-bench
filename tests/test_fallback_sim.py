"""OOM ladder and fault handling through the real runner (SPEC 8.3, 8.4, 13.3).

Uses the child's test hooks: AGBENCH_SIMULATE_OOM_MB_GE=<n> raises torch.OutOfMemoryError when the
attempt's micro-batch is >= n; AGBENCH_SIMULATE_FAULT=<scenario_id>:<crash|hang|error>.
In tiny mode the batch-16 ladder becomes 2x1 -> 1x2.
"""

from __future__ import annotations

import pytest

from agbench.schema import load_records
from conftest import by_id, explain, scenario_records

LADDERED = "infer.w2v2b.6s.bs16.bf16"
TINY = ["--tiny", "--device", "cpu", "--no-report"]


def test_oom_then_fits_at_the_next_rung(run_agbench, tmp_path):
    out = tmp_path / "results" / "oom1.jsonl"
    proc = run_agbench("run", *TINY, "--scenario", LADDERED, "--out", out,
                       env={"AGBENCH_SIMULATE_OOM_MB_GE": "2"})
    assert proc.returncode == 0, explain(proc, out)
    rec = by_id(scenario_records(out))[LADDERED]
    assert rec["status"] == "ok", explain(proc, out)
    fb = rec["fallback"]
    assert fb["fits_requested"] is False
    assert [(a["micro_batch"], a["accum"], a["status"]) for a in fb["attempts"]] == [(2, 1, "oom"), (1, 2, "ok")]
    assert fb["used"]["micro_batch"] == 1 and fb["used"]["accum"] == 2
    assert rec["metrics"]["effective_batch"] == 2
    assert rec["metrics"]["clips_per_s"] > 0


def test_oom_at_every_rung_is_a_result_not_a_failure(run_agbench, tmp_path):
    out = tmp_path / "results" / "oom2.jsonl"
    proc = run_agbench("run", *TINY, "--scenario", LADDERED, "--out", out,
                       env={"AGBENCH_SIMULATE_OOM_MB_GE": "1"})
    assert proc.returncode == 0, explain(proc, out)   # oom is a terminal, valid status
    rec = by_id(scenario_records(out))[LADDERED]
    assert rec["status"] == "oom"
    assert [(a["micro_batch"], a["accum"], a["status"]) for a in rec["fallback"]["attempts"]] == \
        [(2, 1, "oom"), (1, 2, "oom")]
    assert rec["metrics"] == {}


@pytest.mark.parametrize("kind,expected", [("crash", "crashed"), ("error", "error"), ("hang", "timeout")])
def test_faults_are_contained_and_the_suite_continues(run_agbench, tmp_path, kind, expected):
    out = tmp_path / "results" / f"fault_{kind}.jsonl"
    faulty, nxt = "gpu.matmul", "gpu.membw"
    # The hang case runs alone with short timeouts (120 s minimum x 0.1 = 12 s); the other two cases
    # check that the suite goes on to the next scenario after the fault.
    ids = [faulty] if kind == "hang" else [faulty, nxt]
    extra = ["--timeout-scale", "0.1"] if kind == "hang" else []
    args = [a for sid in ids for a in ("--scenario", sid)]
    proc = run_agbench("run", *TINY, *args, "--out", out, *extra,
                       env={"AGBENCH_SIMULATE_FAULT": f"{faulty}:{kind}"}, timeout=600)
    assert proc.returncode == 1, explain(proc, out)   # at least one error/timeout/crashed
    recs = by_id(scenario_records(out))
    assert recs[faulty]["status"] == expected, explain(proc, out)
    assert isinstance(recs[faulty]["error"], dict) and recs[faulty]["error"]["type"]
    assert recs[faulty]["metrics"] == {}
    end = [r for r in load_records(out) if r["kind"] == "run_end"][-1]
    assert end["counts"][expected] == 1 and end["interrupted"] is False
    if nxt in ids:
        assert recs[nxt]["status"] == "ok", explain(proc, out)
        assert end["counts"]["ok"] == 1
