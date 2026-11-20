"""Offline tests for Encore's terminal trace. No account, Ollama, or network needed."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from rich.console import Console

# Run these standalone against the files bundled in the ZIP.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from encore.observability import SessionTrace
from encore.terminal_ui import TerminalUI, ObservableOllama


def consoles():
    out, err = io.StringIO(), io.StringIO()
    return out, err, Console(file=err, width=105, color_system=None), Console(file=out, width=105, color_system=None)


def test_events_are_live_and_durable_and_redacted(tmp_path):
    out, err, progress, output = consoles()
    ui = TerminalUI(console=progress, output=output)
    ui.start("Find five recordings")
    with SessionTrace(tmp_path, "test-user", "session-1", secrets=("TOPSECRET",), on_event=ui.on_event) as trace:
        trace.emit("session_start", model="qwen", max_steps=12)
        trace.emit("route", domains=["catalog"], allowed_tools=["search_recordings", "final_answer"])
        trace.emit("tool_attempt", step=1, tool="search_recordings", attempt=1, args={"query": "TOPSECRET"})
        trace.emit("tool_result", step=1, tool="search_recordings", success=True)
        trace.emit("tool_call", step=1, tool="search_recordings", ok=True, returned_recordings=5)
        trace.emit("session_end", status="answered", latency_s=2.3)
        path = trace.path
    ui.result(SimpleNamespace(answer="Here are 5 songs", status="answered", confidence="high",
                              steps=1, seconds=2.3, sources=["musicbrainz"],
                              prompt_tokens=23, generated_tokens=10), path)
    text = err.getvalue()
    assert "Request classified" in text
    assert "search_recordings succeeded" in text
    assert "returned recordings: 5" in text
    assert "Session complete" in text
    assert "Here are 5 songs" in out.getvalue()
    saved = path.read_text()
    assert "TOPSECRET" not in saved
    assert "[REDACTED]" in saved
    assert "TOPSECRET" not in text
    assert len(saved.splitlines()) == 6


def test_quiet_only_answer(tmp_path):
    out, err, progress, output = consoles()
    ui = TerminalUI(quiet=True, console=progress, output=output)
    ui.start("Quiet request")
    with SessionTrace(tmp_path, "test-user", "session-1", on_event=ui.on_event) as trace:
        trace.emit("tool_attempt", step=1, tool="search_recordings", args={"query": "hello"})
        trace.emit("tool_result", step=1, tool="search_recordings", success=True)
        path = trace.path
    ui.result(SimpleNamespace(answer="JUST THE ANSWER", sources=[]), path)
    assert err.getvalue() == ""
    assert out.getvalue().strip() == "JUST THE ANSWER"
    assert path.exists()


def test_debug_names_only_not_raw_argument_values(tmp_path):
    out, err, progress, output = consoles()
    ui = TerminalUI(debug=True, console=progress, output=output)
    with SessionTrace(tmp_path, "test-user", "session-1", on_event=ui.on_event) as trace:
        trace.emit("tool_attempt", step=2, tool="search_recordings", args={"query": "highly private song title"})
    assert "Argument fields" in err.getvalue()
    assert "query" in err.getvalue()
    assert "highly private song title" not in err.getvalue()


def test_renderer_failure_does_not_prevent_jsonl_write(tmp_path):
    def broken(record):
        raise RuntimeError("terminal unplugged")
    with SessionTrace(tmp_path, "test-user", "session-1", on_event=broken) as trace:
        trace.emit("tool_result", tool="get_recent_listens", success=True)
        path = trace.path
    assert json.loads(path.read_text().splitlines()[0])["event"] == "tool_result"


def test_spinner_events_and_no_leakage(tmp_path):
    out, err, progress, output = consoles()
    ui = TerminalUI(console=progress, output=output)
    ui.on_event({"event": "model_start", "phase": "interpreting request"})
    ui.on_event({"event": "model_end", "phase": "interpreting request"})
    ui.close()
    assert ui._spinner is None


def test_observable_ollama_is_real_subclass_and_emits_stage_events(monkeypatch):
    # Install an isolated stub without importing the real application or network.
    stub = ModuleType("encore.ollama_client")
    class Ollama:
        def __init__(self, host, model, *args, **kwargs):
            self.model = model
        def chat(self, messages, tools, *, tier="cheap"):
            return {"message": "ok"}
        def extract_intent(self, goal):
            return {"action": "read"}
        def route_domains(self, goal):
            return (("catalog",), {})
        def review_final(self, goal, proposed, evidence):
            return proposed
        def summarize_working_memory(self, goal, turns):
            return ("summary", 0, 0)
    stub.Ollama = Ollama
    monkeypatch.setitem(sys.modules, "encore.ollama_client", stub)
    events = []
    model = ObservableOllama("http://localhost:11434", "qwen3:8b",
                             on_model_event=lambda event, **fields: events.append(event))
    assert isinstance(model, Ollama)
    assert model.extract_intent("hello") == {"action": "read"}
    assert model.chat([], [], tier="cheap") == {"message": "ok"}
    assert model.review_final("", {}, []) == {}
    assert events == ["model_start", "model_end"] * 3


def test_cli_ask_uses_live_trace_and_keeps_approval_hook(tmp_path, monkeypatch, capsys):
    import importlib.util

    def fake_module(name, **items):
        module = ModuleType(name)
        module.__dict__.update(items)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    class Memory:
        def __init__(self, path):
            self.path = path
        def prune(self, **kwargs):
            return 0
        def enable_semantic(self, store):
            self.semantic = store
        def enable_episodic(self, store):
            self.episodic = store
        def add_structured_session(self, *args):
            return {"quality_flag": "good", "merged": False}
        def close(self):
            pass

    class EpisodicStore:
        def __init__(self, *args):
            pass
        def prune(self):
            pass

    class OllamaEmbedder:
        def __init__(self, *args):
            pass

    class SessionSummarizer:
        def __init__(self, *args, **kwargs):
            pass
        def summarize(self, *args):
            return ({"decisions": [], "preferences": [], "open_threads": [],
                     "key_facts": [], "errors": [], "topics": []},
                    {"score": 1.0, "consistent": True, "issues": []})

    class Http:
        def __init__(self, *args, **kwargs):
            pass

    class ListenBrainz:
        def __init__(self, *args, **kwargs):
            pass

    class MusicBrainz:
        def __init__(self, *args, **kwargs):
            pass

    class Tools:
        def __init__(self, *args, **kwargs):
            self.approve = kwargs["approve"]

    def fake_run(goal, tools, model, *, event_log, **kwargs):
        # Real model method is invoked, exercising ObservableOllama's subclass.
        model.chat([], [], tier="cheap")
        event_log.emit("route", domains=["catalog"], allowed_tools=["search_recordings"])
        event_log.emit("tool_attempt", step=1, tool="search_recordings", attempt=1, args={"query": "Benny"})
        event_log.emit("tool_result", step=1, tool="search_recordings", success=True)
        event_log.emit("final", status="answered", steps=1)
        return SimpleNamespace(answer="A TEST ANSWER", status="answered", confidence="high",
                               steps=1, seconds=1.25, sources=[],
                               prompt_tokens=10, generated_tokens=5, tool_trace=[])

    class Ollama:
        def __init__(self, host, model, *args, **kwargs):
            self.model = model
        def chat(self, messages, tools, *, tier="cheap"):
            return {"message": "ok"}

    class FaultInjector:
        @staticmethod
        def load(path):
            raise AssertionError("not used")

    config = SimpleNamespace(username="fake", token="secret", mb_contact="fake@example.com",
                             model="qwen3:8b", ollama_host="http://localhost:11434",
                             db=tmp_path / "memory.sqlite3", timezone="UTC", user_id="fake",
                             episodic_path=tmp_path / "chroma", embedding_model="embed",
                             summary_model="qwen3:8b", cheap_model="qwen3:8b",
                             final_model="qwen3:8b", router_model="qwen3:8b")
    fake_module("encore.agent", Ollama=Ollama, run_agent=fake_run)
    fake_module("encore.ollama_client", Ollama=Ollama)
    fake_module("encore.clients", Http=Http, ListenBrainz=ListenBrainz, MusicBrainz=MusicBrainz)
    fake_module("encore.config", get_config=lambda: config)
    fake_module("encore.memory", Memory=Memory)
    fake_module("encore.episodic", EpisodicStore=EpisodicStore,
                OllamaEmbedder=OllamaEmbedder, SessionSummarizer=SessionSummarizer,
                SUMMARY_FIELDS=("decisions", "preferences", "open_threads", "key_facts", "errors", "topics"),
                _strip_secrets=lambda text, secrets=(): text.replace("secret", "[REDACTED]"))
    fake_module("encore.retrieval_eval", evaluate=lambda *args: {})
    fake_module("encore.semantic", SemanticStore=lambda *args: object())
    fake_module("encore.tools", Tools=Tools)
    fake_module("encore.faults", FaultInjector=FaultInjector)

    # Imported by isolated path to avoid keeping a modified CLI in sys.modules.
    spec = importlib.util.spec_from_file_location("encore.cli", Path(__file__).resolve().parents[1] / "src/encore/cli.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("ENCORE_TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.setattr(sys, "argv", ["encore", "ask", "Find", "five", "songs"])
    module.main()
    captured = capsys.readouterr()
    assert "A TEST ANSWER" in captured.out
    assert "search_recordings succeeded" in captured.err
    assert "Tool trace:" not in captured.out
    assert len(list((tmp_path / "traces").rglob("*.jsonl"))) == 1

    monkeypatch.setattr(sys, "argv", ["encore", "ask", "Find", "five", "songs", "--quiet"])
    module.main()
    captured = capsys.readouterr()
    assert captured.out.strip() == "A TEST ANSWER"
    assert captured.err == ""
