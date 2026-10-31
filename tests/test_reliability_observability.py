"""Offline compliance tests. All tests avoid remote ListenBrainz writes."""
import json
from uuid import uuid4

import pytest
import requests

from encore.agent import run_agent
from encore.clients import Http, ServiceError
from encore.faults import FaultInjector
from encore.memory import Memory
from encore.observability import SessionTrace, safe_user_dir
from encore.tools import Tools


def test_injection_repeatable_and_pre_network(tmp_path):
    cfg = tmp_path / "faults.json"
    cfg.write_text('{"transient_rate": 1.0, "seed": 5, "max_steps": 4}')
    failures = []
    inj = FaultInjector.load(cfg, reporter=lambda event, **kwargs: failures.append((event, kwargs)))
    http = Http("test", injector=inj, mb_min_interval=0, lb_min_interval=0)
    class MustNotCall:
        def request(self, *a, **kw):
            raise AssertionError("A synthetic fault reached the network")
    http.session = MustNotCall()
    for url in ["https://api.listenbrainz.org/1/user/a/listens", "https://musicbrainz.org/ws/2/recording"]:
        with pytest.raises(ServiceError, match="Injected HTTP") as exc:
            http.request("GET", url, mb="musicbrainz.org" in url)
        assert exc.value.kind == "transient"
    assert len(failures) == 2
    assert [x[1]["http_status"] for x in failures] == [429, 503]
    assert inj.max_steps == 4


def test_fault_config_rejects_unknown_fields(tmp_path):
    config = tmp_path / "inject.json"
    config.write_text('{"transient_rate": 0.3, "seed": 1, "strange": true}')
    with pytest.raises(ValueError, match="Unknown injection"):
        FaultInjector.load(config)


def test_http_read_timeout_and_uncertain_write():
    http = Http("test", mb_min_interval=0, lb_min_interval=0)
    calls = []
    class TimeoutSession:
        def request(self, *args, **kwargs):
            calls.append(kwargs)
            raise requests.Timeout()
    http.session = TimeoutSession()
    with pytest.raises(ServiceError) as err:
        http.request("GET", "https://api.listenbrainz.org/1/recording")
    assert err.value.kind == "transient" and calls[-1]["timeout"] == (5, 5)
    with pytest.raises(ServiceError) as err:
        http.request("POST", "https://api.listenbrainz.org/1/playlist/create", token="TEST")
    assert err.value.kind == "uncertain_write" and calls[-1]["timeout"] == (5, 15)
    assert len(calls) == 2  # no internal retries, even on GET


def test_credential_redaction_and_user_scoping(tmp_path):
    token = "SERIOUSLY_PRIVATE_TOKEN_123"
    sid = str(uuid4())
    with SessionTrace(tmp_path, "userA", sid, secrets=(token,)) as log:
        log.emit("tool_attempt", args={"token": token, "authorization": "Token " + token,
                                       "query": "something " + token},
                 headers={"Authorization": token})
    data = log.path.read_text()
    assert token not in data
    assert "[REDACTED]" in data
    assert "userA" not in str(log.path)
    assert safe_user_dir(tmp_path, "userA") != safe_user_dir(tmp_path, "userB")
    assert json.loads(data)["event"] == "tool_attempt"


class CountLB:
    token = ""
    def __init__(self): self.calls = 0
    def count(self):
        self.calls += 1
        if self.calls < 3:
            raise ServiceError("transient", "Mock HTTP 503")
        return {"count": 42}


class CountModel:
    def __init__(self): self.calls = 0
    def chat(self, messages, specs):
        self.calls += 1
        if self.calls == 1:
            return {"message": {"tool_calls": [{"function": {"name": "get_listen_count", "arguments": {}}}]},
                    "prompt_eval_count": 40, "eval_count": 8}
        return {"message": {"tool_calls": [{"function": {"name": "final_answer", "arguments": {
            "answer": "You have 42 listens", "status": "answered"}}}]},
                "prompt_eval_count": 45, "eval_count": 12}


def test_loop_retries_transient_reads_without_wasting_model_steps(tmp_path, monkeypatch):
    monkeypatch.setattr("encore.agent.time.sleep", lambda _: None)
    mem = Memory(tmp_path / "db.sqlite")
    lb = CountLB()
    try:
        with SessionTrace(tmp_path / "traces", "user", str(uuid4())) as log:
            tools = Tools(lb, None, mem, approve=lambda _:False, current_goal="How many listens?",
                          session_id=log.session_id)
            result = run_agent("How many listens?", tools, CountModel(),
                               event_log=log, max_read_attempts=3)
        assert lb.calls == 3
        assert result.status == "answered"
        assert result.steps == 2
        assert result.tool_trace[0]["attempts"] == 3
        events = [json.loads(x) for x in log.path.read_text().splitlines()]
        assert [x["event"] for x in events].count("retry") == 2
        assert [x["event"] for x in events].count("llm_call") == 2
        assert [x["event"] for x in events].count("final") == 1
        assert events[0]["event"] == "route"
        assert events[-1]["status"] == "answered"
    finally:
        mem.close()


class WriteModel:
    def chat(self, messages, specs):
        return {"message": {"tool_calls": [{"function": {"name": "create_playlist",
                "arguments": {"title": "[Encore] Test", "mbids": ["57bdf688-6d28-4eaf-8779-40d5ca8837f9"]}}}]}}


class WriteFaultTools:
    def __init__(self, mem):
        self.mem = mem
        self.session_id = str(uuid4())
        self.calls = 0
    def run(self, name, args):
        from encore.tools import ToolResult
        self.calls += 1
        return ToolResult(ok=False, error_type="transient", message="Injected HTTP 503")


def test_write_never_retried_even_when_fault_is_transient(tmp_path, monkeypatch):
    monkeypatch.setattr("encore.agent.time.sleep", lambda _: None)
    mem = Memory(tmp_path / "db.sqlite")
    try:
        tools = WriteFaultTools(mem)
        result = run_agent("Create a playlist called Test", tools, WriteModel(), max_steps=3)
        assert tools.calls == 1
        assert result.status == "stuck"
        assert result.tool_trace[0]["tool"] == "create_playlist"
        assert "attempts" not in result.tool_trace[0]
    finally:
        mem.close()


def test_rate_limit_remaining_zero_delays_next_request(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr("encore.clients.time.sleep", lambda duration: sleep_calls.append(duration))
    http = Http("test", lb_min_interval=0, mb_min_interval=0)
    class Response:
        ok = True
        status_code = 200
        content = b"{}"
        headers = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset-In": "4"}
        def json(self): return {}
    class Session:
        def request(self, *args, **kwargs): return Response()
    http.session = Session()
    http.request("GET", "https://api.listenbrainz.org/1/user/a/listens")
    http.request("GET", "https://api.listenbrainz.org/1/user/a/listens")
    assert any(duration > 3 for duration in sleep_calls)
