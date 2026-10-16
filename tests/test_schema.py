"""Results JSONL schema: the hand-made fixtures validate, broken records are caught with useful
messages, and the JSON Schema file agrees with validate_record (SPEC 9, 13.3)."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from agbench.scenarios import SCENARIOS, SUITES
from agbench.schema import SCHEMA_VERSION, load_records, schema_path, validate_record

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.jsonl"))
FULL_FIXTURE = Path(__file__).parent / "fixtures" / "rtx-5060-ti-16gb_20261016.jsonl"
PROBLEM_FIXTURE = Path(__file__).parent / "fixtures" / "rtx-5060-ti-8gb_20261017.jsonl"


def _first(path: Path, kind: str, status: str | None = None) -> dict:
    for r in load_records(path):
        if r.get("kind") == kind and (status is None or r.get("status") == status):
            return copy.deepcopy(r)
    raise AssertionError(f"no {kind}/{status} record in {path.name}")


def test_fixtures_exist():
    names = {p.name for p in FIXTURES}
    assert {"rtx-5060-ti-16gb_20261016.jsonl", "rtx-3090-24gb_20261016.jsonl",
            "rtx-5060-ti-8gb_20261017.jsonl"} <= names


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_records_validate(path):
    recs = load_records(path)
    assert recs, "empty fixture"
    assert recs[0]["kind"] == "run_start" and recs[-1]["kind"] == "run_end"
    for i, rec in enumerate(recs):
        assert validate_record(rec) == [], f"{path.name} line {i + 1}: {validate_record(rec)}"
        assert rec["schema_version"] == SCHEMA_VERSION


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_scenarios_match_the_matrix(path):
    for rec in load_records(path):
        if rec["kind"] != "scenario":
            continue
        s = SCENARIOS.get(rec["scenario_id"])
        assert s is not None, f"unknown scenario id {rec['scenario_id']}"
        assert rec["category"] == s.category and rec["workload"] == s.workload


def test_full_fixture_covers_every_scenario():
    ids = {r["scenario_id"] for r in load_records(FULL_FIXTURE) if r["kind"] == "scenario"}
    assert ids == set(SUITES["full"])


def test_fixtures_cover_every_status_and_category():
    statuses, categories = set(), set()
    for path in FIXTURES:
        for r in load_records(path):
            if r["kind"] == "scenario":
                statuses.add(r["status"])
                categories.add(r["category"])
    assert statuses == {"ok", "oom", "error", "timeout", "crashed", "skipped"}
    assert categories == {"basics", "train", "frozen", "infer", "e2e", "cpu", "sustain"}


def _broken_cases() -> list[tuple[str, dict, str]]:
    ok = _first(FULL_FIXTURE, "scenario", "ok")
    oom = _first(PROBLEM_FIXTURE, "scenario", "oom")
    err = _first(PROBLEM_FIXTURE, "scenario", "error")
    start = _first(FULL_FIXTURE, "run_start")
    end = _first(FULL_FIXTURE, "run_end")
    cases = []

    def case(name, rec, expect, mutate):
        rec = copy.deepcopy(rec)
        mutate(rec)
        cases.append((name, rec, expect))

    case("missing status", ok, "status", lambda r: r.pop("status"))
    case("unknown status", ok, "status", lambda r: r.update(status="great"))
    case("unknown category", ok, "category", lambda r: r.update(category="gpu"))
    case("wall_s as string", ok, "wall_s", lambda r: r.update(wall_s="12"))
    train_ok = next(r for r in load_records(FULL_FIXTURE)
                    if r["kind"] == "scenario" and r["category"] == "train" and r["status"] == "ok")
    case("ok without clips_per_s", train_ok, "clips_per_s", lambda r: r["metrics"].pop("clips_per_s"))
    case("oom with metrics", oom, "metrics", lambda r: r.update(metrics={"clips_per_s": 1.0}))
    case("error without error object", err, "error", lambda r: r.update(error=None))
    case("unknown kind", ok, "kind", lambda r: r.update(kind="banana"))
    case("wrong schema version", ok, "schema_version", lambda r: r.update(schema_version="2"))
    case("attempts not a list", oom, "attempts", lambda r: r["fallback"].update(attempts="16x1"))
    case("unknown telemetry source", ok, "telemetry", lambda r: r["telemetry"].update(source="magic"))
    case("run_start without env", start, "env", lambda r: r.pop("env"))
    case("run_end without counts.ok", end, "counts", lambda r: r["counts"].pop("ok"))
    return cases


@pytest.mark.parametrize("name,rec,expect", _broken_cases(), ids=[c[0] for c in _broken_cases()])
def test_broken_records_are_reported(name, rec, expect):
    problems = validate_record(rec)
    assert problems, f"{name}: validate_record found nothing"
    assert all(isinstance(p, str) and p for p in problems)
    assert any(expect in p for p in problems), f"{name}: expected a message about {expect!r}, got {problems}"


def test_validate_record_rejects_non_objects():
    assert validate_record([]) != []  # type: ignore[arg-type]


def test_json_schema_agrees_with_validate_record():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(schema_path().read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    for path in FIXTURES:
        for i, rec in enumerate(load_records(path)):
            errs = [e.message for e in validator.iter_errors(rec)]
            assert errs == [], f"{path.name} line {i + 1}: JSON Schema says {errs[:3]}"
    # Structural breakage must be caught by both validators.
    structural = {"missing status", "unknown status", "unknown category", "wall_s as string", "unknown kind",
                  "wrong schema version", "attempts not a list", "run_start without env"}
    for name, rec, _ in _broken_cases():
        if name in structural:
            assert validate_record(rec), name
            assert not validator.is_valid(rec), f"JSON Schema accepted a broken record: {name}"


def test_load_records_ignores_a_truncated_last_line(tmp_path):
    p = tmp_path / FULL_FIXTURE.name
    shutil.copy(FULL_FIXTURE, p)
    n = len(load_records(p))
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"kind": "scenario", "schema_version": "1", "run_id": "2026')
    recs = load_records(p)
    assert len(recs) == n
    assert [r.get("scenario_id") for r in recs] == [r.get("scenario_id") for r in load_records(FULL_FIXTURE)]
