"""Regressions for top-artist question, missing aggregate statistics, and calendar ranges."""
from datetime import datetime, timezone
import pytest
from encore.clients import Http, ListenBrainz, ServiceError
from encore.memory import Memory
from encore.agent import run_agent
from encore.tools import Tools


def make_client():
    lb = ListenBrainz(Http("test"), "testuser", timezone_name="UTC")
    log = []
    now = int(datetime.now(timezone.utc).timestamp())
    tracks = [
        {"listened_at": now - 10, "track_metadata": {"artist_name": "Stevie Wonder", "track_name": "Higher Ground"}},
        {"listened_at": now - 20, "track_metadata": {"artist_name": "Stevie Wonder", "track_name": "Sir Duke"}},
        {"listened_at": now - 30, "track_metadata": {"artist_name": "Johnny Cash", "track_name": "Get Rhythm"}},
    ]

    def mock_get(path, params=None):
        log.append((path, params))
        if path.startswith("stats/"):
            return {}  # HTTP 204 normalized by Http.request
        if path.endswith("/listens"):
            return {"payload": {"listens": tracks}}
        raise AssertionError(path)

    lb.get = mock_get
    return lb, log


def test_top_artists_fallback_on_stats_204():
    lb, calls = make_client()
    data = lb.top("artists", count=2, span="this_month")
    assert data["source"] == "raw_scrobbles_fallback"
    assert data["complete"] is True
    assert data["range"] == "this_month"
    assert data["artists"] == [
        {"artist_name": "Stevie Wonder", "listen_count": 2},
        {"artist_name": "Johnny Cash", "listen_count": 1},
    ]
    assert calls[0][1]["range"] == "this_month"
    assert "max_ts" in calls[1][1]


def test_month_does_not_silently_mean_this_month():
    lb, _ = make_client()
    with pytest.raises(ServiceError, match="unavailable for month"):
        lb.top("artists", count=10, span="month")


def test_top_recordings_fallback():
    lb, _ = make_client()
    data = lb.top("recordings", count=10, span="this_month")
    assert data["source"] == "raw_scrobbles_fallback"
    assert len(data["recordings"]) == 3
    assert data["recordings"][0]["recording_name"]


def test_partial_scan_is_marked_incomplete():
    lb, _ = make_client()
    lb._recent_window = lambda from_ts: {"listens": [{"artist": "A", "title": "B"}], "complete": False}
    data = lb.top("artists", span="this_month")
    assert data["complete"] is False
    assert "PARTIAL" in data["note"]


def test_precomputed_stats_accepted_without_fallback():
    lb, log = make_client()
    lb.get = lambda path, params=None: {"payload": {"artists": [
        {"artist_name": "A", "listen_count": 5}], "last_updated": 123}}
    data = lb.top("artists", span="this_month")
    assert data["source"] == "listenbrainz_statistics"
    assert data["artists"][0]["listen_count"] == 5


class RepeatModel:
    def __init__(self):
        self.n = 0

    def chat(self, messages, specs):
        self.n += 1
        if self.n in (1, 2):
            return {"message": {"tool_calls": [
                {"function": {"name": "get_top_artists", "arguments": {"range": "this_month", "count": 5}}}
            ]}}
        return {"message": {"tool_calls": [
            {"function": {"name": "final_answer", "arguments": {
                "answer": "Stevie Wonder leads with 2 listens", "status": "answered"}}}
        ]}}


def test_agent_prevents_duplicate_top_artists_calls(tmp_path):
    lb, log = make_client()
    memory = Memory(tmp_path / "e.sqlite")
    try:
        tools = Tools(lb, None, memory, approve=lambda _: False,
                      current_goal="My top artists this month")
        result = run_agent("My top artists this month", tools, RepeatModel())
        assert result.status == "answered"
        assert result.tool_trace[0]["source"] == "raw_scrobbles_fallback"
        assert result.tool_trace[1]["error_type"] == "duplicate_read"
        assert sum(1 for path, _ in log if path.startswith("stats/")) == 1
    finally:
        memory.close()


def test_router_narrows_top_artist_query():
    from encore.agent import route
    names = {x["function"]["name"] for x in route("What are my top artists this month?")}
    assert "get_top_artists" in names
    assert "final_answer" in names
    assert "create_playlist" not in names
