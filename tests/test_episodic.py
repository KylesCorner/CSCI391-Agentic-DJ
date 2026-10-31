"""Offline episodic-store tests. No Ollama/Chroma server required here."""
from datetime import datetime, timedelta, timezone
import json
import math

import pytest

from encore.episodic import EpisodicStore, SessionSummarizer, SUMMARY_FIELDS
from encore.memory import Memory
from encore.agent import run_agent
from encore.tools import Tools


class Embed:
    model = "test-emb:v1"
    def embed(self, text):
        tokens = text.casefold().split()
        # Reproducible 12-D vector with a small category-level signal.
        v = [0.0] * 12
        for token in tokens:
            v[sum(token.encode()) % 12] += 1.0
        magnitude = math.sqrt(sum(n * n for n in v)) or 1.0
        return [n / magnitude for n in v]


class Collection:
    def __init__(self):
        self.rows = {}

    def get(self, *, where=None, include=None):
        matching = [(i, row) for i, row in self.rows.items()
                    if not where or all(row["meta"].get(k) == v for k, v in where.items())]
        return {"ids": [i for i, row in matching],
                "metadatas": [row["meta"] for i, row in matching],
                "documents": [row["doc"] for i, row in matching]}

    def upsert(self, *, ids, documents, metadatas, embeddings):
        for i, doc, meta, vec in zip(ids, documents, metadatas, embeddings):
            self.rows[i] = {"doc": doc, "meta": meta, "vec": vec}

    def delete(self, *, ids):
        for i in ids:
            self.rows.pop(i, None)

    def query(self, *, query_embeddings, n_results, where, include):
        q = query_embeddings[0]
        matching = [(i, row, 1 - sum(x*y for x, y in zip(q, row["vec"])))
                    for i, row in self.rows.items()
                    if row["meta"].get("user_id") == where["user_id"]]
        matching.sort(key=lambda x: x[2])
        matching = matching[:n_results]
        return {"ids": [[i for i, row, d in matching]],
                "distances": [[d for i, row, d in matching]],
                "metadatas": [[row["meta"] for i, row, d in matching]],
                "documents": [[row["doc"] for i, row, d in matching]]}


class FakeChroma:
    def __init__(self):
        self.collection = Collection()
    def get_or_create_collection(self, **kwargs):
        assert kwargs["embedding_function"] is None
        return self.collection


def summary(**overrides):
    result = {key: [] for key in SUMMARY_FIELDS}
    result.update(overrides)
    return result


GOOD = {"consistent": True, "score": 0.95, "issues": []}
BAD = {"consistent": False, "score": 0.2, "issues": ["unsupported fact"]}


def make(client, user="kyle", **kwargs):
    return EpisodicStore("unused", user, Embed(), client=client, **kwargs)


def test_dedup_merges_repeated_session_in_chroma():
    client = FakeChroma()
    store = make(client)
    ep = summary(decisions=["Created a swing playlist"], topics=["swing jazz"])
    r1 = store.add("Create swing playlist", ep, "answered", GOOD)
    r2 = store.add("Create swing playlist", ep, "answered", GOOD)
    assert r1["merged"] is False
    assert r2["merged"] is True
    assert r1["id"] == r2["id"]
    assert client.collection.get(where={"user_id": "kyle"})["metadatas"][0]["seen_count"] == 2


def test_user_isolation_and_forget_only_deletes_self():
    client = FakeChroma()
    a = make(client, "user-a")
    b = make(client, "user-b")
    ep = summary(preferences=["I prefer instrumental swing jazz"], topics=["study music"])
    a.add("study music", ep, "answered", GOOD)
    b.add("study music", ep, "answered", GOOD)
    assert len(a.search("study music")) == 1
    assert len(b.search("study music")) == 1
    assert a.delete_user() == 1
    assert a.search("study music") == []
    assert len(b.search("study music")) == 1


def test_bad_quality_is_flagged_and_lower_ranked():
    client = FakeChroma()
    store = make(client)
    store.add("study preference", summary(preferences=["No vocals"], topics=["study"]), "answered", BAD)
    found = store.search("study preference")
    assert found[0]["quality_flag"] == "verify_before_using"
    assert "outdated" in found[0]["stale_warning"]


def test_prune_ttl_and_capacity_limit():
    client = FakeChroma()
    now = datetime(2026, 10, 31, tzinfo=timezone.utc)
    store = make(client, ttl_days=90, max_per_user=2, threshold=1.0)
    store.add("old music", summary(topics=["old"]), "answered", GOOD,
              now=now - timedelta(days=100))
    store.add("recent swing", summary(topics=["swing"]), "answered", GOOD,
              now=now - timedelta(days=2))
    store.add("recent bebop", summary(topics=["bebop"]), "answered", GOOD,
              now=now - timedelta(days=1))
    store.add("new bossa", summary(topics=["bossa nova"]), "answered", GOOD, now=now)
    assert len(store._get()) == 2
    assert all(item["meta"]["updated_ts"] >= (now-timedelta(days=2)).timestamp() for item in store._get())


def test_embedding_model_mismatch_refused():
    client = FakeChroma()
    store = make(client)
    store.add("study jazz", summary(topics=["jazz"]), "answered", GOOD)
    different = Embed()
    different.model = "some-other-model"
    with pytest.raises(RuntimeError, match="embedding model changed"):
        EpisodicStore("unused", "kyle", different, client=client)


def test_summary_review_is_separate_call_and_catches_false_write_claim(monkeypatch):
    s = SessionSummarizer("http://fake:11434", "local-model")
    calls = []
    ep = summary(decisions=["Created playlist Swing Study"], topics=["swing"])
    def fake_ask(system, payload, schema):
        calls.append((system, payload))
        return ep if len(calls) == 1 else dict(GOOD)
    monkeypatch.setattr(s, "_ask", fake_ask)
    actual, review = s.summarize("Create a playlist", "Failed", "stuck",
                                 [{"tool": "create_playlist", "ok": False, "message": "HTTP 400"}])
    assert len(calls) == 2
    assert review["consistent"] is False
    assert review["score"] <= 0.25
    assert actual["decisions"]


def test_run_agent_session_start_injects_episdodic_memory(tmp_path):
    client = FakeChroma()
    store = make(client)
    store.add("Made Swing Test playlist last time",
              summary(decisions=["Created Swing Test playlist"], topics=["playlist"]),
              "answered", GOOD)
    memory = Memory(tmp_path / "memory.db")
    memory.enable_episodic(store)

    class FakeModel:
        context = None
        def chat(self, messages, specs):
            self.context = messages
            return {"message": {"tool_calls": [{"function": {"name": "final_answer", "arguments": {
                "status": "answered", "answer": "We previously created Swing Test."
            }}}]}}

    try:
        goal = "Which playlist was mentioned earlier?"
        model = FakeModel()
        toolset = Tools(type("LB", (), {"token": ""})(), None, memory,
                        approve=lambda _: False, current_goal=goal)
        result = run_agent(goal, toolset, model)
        assert result.status == "answered"
        assert result.steps == 1  # Prefetch does not use up the ReAct budget.
        assert result.tool_trace[0]["tool"] == "search_past_sessions"
        assert result.tool_trace[0]["phase"] == "session_start"
        assert "Swing Test playlist" in json.dumps(model.context)
        assert "may be outdated" in json.dumps(model.context)
    finally:
        memory.close()


def test_sqlite_deletion_cascades_to_chroma(tmp_path):
    client = FakeChroma()
    store = make(client)
    store.add("Made Swing Test", summary(topics=["playlist"]), "answered", GOOD)
    memory = Memory(tmp_path / "m.db")
    memory.enable_episodic(store)
    try:
        memory.set_preference("jazz", "instrumental", "I prefer instrumental jazz")
        result = memory.delete_personal()
        assert result["episodic_chroma"] == 1
        assert result["preferences"] == 1
        assert store._get() == []
    finally:
        memory.close()
