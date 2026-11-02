"""Offline checks for the grading harness, never calling real APIs."""
from __future__ import annotations
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE = Path(__file__).resolve().parents[1] / "evals" / "run_benchmark.py"
spec = importlib.util.spec_from_file_location("encore_eval_runner", MODULE)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_assignment_case_counts_and_ground_truth():
    cases = benchmark.validate_manifest(benchmark.load_json(benchmark.MANIFEST))
    assert len(cases) == 30
    assert {c["category"] for c in cases} == {"happy", "boundary", "failure"}
    assert sum(bool(c.get("repeat")) for c in cases) == 5
    assert all(c["expected"] and c["review"] in {"mechanical", "manual"} for c in cases)


def test_memory_case_count_and_independent_prompts():
    scenarios = benchmark.load_json(benchmark.SCENARIOS)["scenarios"]
    assert len(scenarios) == 5
    assert len(set(s["id"] for s in scenarios)) == 5
    assert all(s["session1"] != s["session2"] for s in scenarios)


def test_missing_trace_is_not_graded_as_correct():
    case = next(c for c in benchmark.load_json(benchmark.MANIFEST)["requests"] if c["id"] == "F09")
    data = benchmark.interpret("Status: answered\n", [], 2.2)
    checks = benchmark.automatic_checks(case, data, events_present=False)
    assert checks["checks_passed"] is False
    assert "Missing readable JSONL trace" in checks["issues"]


def test_trace_metrics_include_summary_calls_and_retries(tmp_path):
    path = tmp_path / "session-example.jsonl"
    events = [
        {"event":"llm_call","tokens_in":500,"tokens_out":50},
        {"event":"tool_attempt","tool":"get_listen_count","attempt":1},
        {"event":"tool_attempt","tool":"get_listen_count","attempt":2},
        {"event":"retry","tool":"get_listen_count"},
        {"event":"tool_result","tool":"get_listen_count","success":True},
        {"event":"memory_read","tool":"search_past_sessions","ok":True,"matches":1},
        {"event":"compress","method":"llm_summary"},
        {"event":"llm_call","phase":"episodic_summary","tokens_in":90,"tokens_out":20},
        {"event":"llm_call","phase":"episodic_quality","tokens_in":100,"tokens_out":40},
        {"event":"final","status":"answered","steps":2},
    ]
    path.write_text("\n".join(json.dumps(e) for e in events))
    result = benchmark.interpret("Status: stuck", benchmark.load_events(path), 5.5)
    assert result["status"] == "answered"  # persistent trace wins
    assert result["tool_calls"] == 1  # retries aren't unique tool actions
    assert result["llm_calls"] == 3
    assert result["retries"] == 1
    assert result["memory_reads"] == 1
    assert result["compressions"] == 1
    assert result["total_tokens_in"] == 690
    assert result["total_tokens_out"] == 110
    assert result["peak_tokens_in"] == 500
    assert result["est_cost_usd"] == "0.000000"


def test_bad_or_forbidden_write_is_not_marked_pass():
    case = next(c for c in benchmark.load_json(benchmark.MANIFEST)["requests"] if c["id"] == "B06")
    result = benchmark.interpret("Status: answered", [
        {"event":"tool_attempt","tool":"create_playlist","attempt":1},
        {"event":"final","status":"answered"},
    ], 3)
    checks = benchmark.automatic_checks(case, result, events_present=True)
    assert not checks["checks_passed"]
    assert "Forbidden tool attempted" in "; ".join(checks["issues"])


def test_session_start_memory_counts_as_required_tool():
    case = next(c for c in benchmark.load_json(benchmark.MANIFEST)["requests"] if c["id"] == "B05")
    result = benchmark.interpret("Status: answered", [
        {"event":"memory_read","tool":"search_past_sessions","ok":True,"matches":1},
        {"event":"final","status":"answered"},
    ], 3)
    checks = benchmark.automatic_checks(case, result, events_present=True)
    assert checks["checks_passed"]


def test_isolated_env_does_not_change_real_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("LISTENBRAINZ_USER", "any-user")
    monkeypatch.setenv("LISTENBRAINZ_TOKEN", "secret-must-not-be-written")
    env = benchmark.isolated_env(tmp_path, "M01")
    assert env["ENCORE_DB"].startswith(str(tmp_path))
    assert env["ENCORE_TRACE_DIR"].startswith(str(tmp_path))
    assert env["ENCORE_USER_ID"] == "encore_evaluation_M01"
    assert env["LISTENBRAINZ_TOKEN"] == "secret-must-not-be-written"


def test_execute_never_auto_approves_and_redacts_stdout(monkeypatch, tmp_path):
    fake_trace = tmp_path / "trace.jsonl"
    fake_trace.write_text(json.dumps({"event":"final","status":"policy_halt","steps":1}) + "\n")
    called = {}
    def fake_run(command, **kwargs):
        called["command"] = command
        called["input"] = kwargs["input"]
        return SimpleNamespace(stdout=f"Secret topSECRET\nSaved trace: {fake_trace}\n",
                               stderr="topSECRET error", returncode=0)
    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    result = benchmark.execute("Delete a playlist", env={"LISTENBRAINZ_TOKEN":"topSECRET"}, timeout=5)
    assert called["input"] == "NO\n"
    assert "topSECRET" not in result["stdout"]
    assert "topSECRET" not in result["stderr_tail"]
    assert result["metrics"]["status"] == "policy_halt"


def test_variance_report_does_not_treat_blank_as_correct():
    row = {key:"0" for key in benchmark.OUTPUT_FIELDS}
    row.update(request_id="H02",run="1",category="happy",status="answered",correct="",
               latency_s="5.0",total_tokens_in="100",tool_calls="1")
    second = dict(row,run="2",correct="0",latency_s="10.0")
    third = dict(row,run="3",correct="1",latency_s="15.0")
    report = benchmark.stats_report([row,second,third], details=[
        {"request_id":"H02", "tools":["get_listen_count"]},
        {"request_id":"H02", "tools":["get_listen_count", "search_knowledge"]},
    ])
    assert report["graded"] == 2 and report["correct"] == 1
    assert report["variance"]["H02"]["n"] == 3
    assert report["variance"]["H02"]["distinct_tool_sequences"] == 2
    assert report["variance"]["H02"]["latency_stddev_s"] == 5.0


def test_predeclared_faults_reproducible():
    cases = benchmark.load_json(benchmark.MANIFEST)["requests"]
    for case in cases:
        if case.get("inject"):
            data = benchmark.load_json(benchmark.ROOT / case["inject"])
            assert isinstance(data["seed"], int)
            assert 0 <= data["transient_rate"] <= 1


def test_csv_has_exact_assignment_columns(tmp_path):
    f = tmp_path / "results.csv"
    benchmark.write_csv(f, benchmark.OUTPUT_FIELDS, [])
    with f.open(newline="") as fd:
        reader = csv.reader(fd)
        assert next(reader) == benchmark.OUTPUT_FIELDS
        assert list(reader) == []
