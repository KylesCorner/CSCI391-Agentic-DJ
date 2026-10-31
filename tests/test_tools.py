from encore.memory import Memory
from encore.tools import Tools
from encore.clients import ListenBrainz, mbids_from_track

MBID = "f0c20a72-9a7a-4fb3-9d22-7a3b17814b11"


class FakeLB:
    token = "testtoken"
    def __init__(self):
        self.called = 0

    def create_playlist(self, title, mbids):
        self.called += 1
        return {"playlist_mbid": "123", "title": title, "count": len(mbids)}

    def listens_since(self, days, max_pages=10):
        return {"complete": True, "listens": [], "days": days}


class FakeMB:
    def verify(self, mbids):
        return {"valid": [{"mbid": x, "title": "Tune", "artist_credit": []} for x in mbids], "invalid": []}


def test_no_write_without_approval(tmp_path):
    mem = Memory(tmp_path / "m.db")
    lb = FakeLB()
    try:
        tools = Tools(lb, FakeMB(), mem, approve=lambda _: False,
                      current_goal="Make a playlist")
        assert tools.run("verify_recordings", {"mbids": [MBID]}).ok
        result = tools.run("create_playlist", {"title": "test", "mbids": [MBID]})
        assert not result.ok and result.error_type == "policy_halt"
        assert lb.called == 0
        # A second attempt with identical payload cannot accidentally create after denial.
        assert not tools.run("create_playlist", {"title": "test", "mbids": [MBID]}).ok
    finally:
        mem.close()


def test_no_unverified_write_and_idempotent_attempt(tmp_path):
    mem = Memory(tmp_path / "m.db")
    lb = FakeLB()
    tools = Tools(lb, FakeMB(), mem, approve=lambda _: True, current_goal="Make a playlist")
    try:
        result = tools.run("create_playlist", {"title": "test", "mbids": [MBID]})
        assert not result.ok and lb.called == 0
        tools.run("verify_recordings", {"mbids": [MBID]})
        assert tools.run("create_playlist", {"title": "test", "mbids": [MBID]}).ok
        assert lb.called == 1
        assert not tools.run("create_playlist", {"title": "test", "mbids": [MBID]}).ok
        assert lb.called == 1
    finally:
        mem.close()


def test_preference_pollution_prevented(tmp_path):
    mem = Memory(tmp_path / "m.db")
    try:
        tools = Tools(FakeLB(), FakeMB(), mem, approve=lambda _: True,
                      current_goal="Tell me my top artists")
        result = tools.run("remember_preference", {
            "key": "jazz", "value": "loves jazz", "evidence": "I like jazz"})
        assert result.error_type == "policy_halt"
        assert mem.preferences() == []
    finally:
        mem.close()


def test_track_ids_are_extracted():
    assert mbids_from_track({"identifier": f"https://musicbrainz.org/recording/{MBID}"}) == [MBID]
    assert mbids_from_track({"identifier": "https://musicbrainz.org/artist/" + MBID}) == []


def test_jspf_shape():
    payload = ListenBrainz.jspf("Focus", [MBID])
    item = payload["playlist"]["track"][0]
    assert MBID in item["identifier"]
    assert payload["playlist"]["extension"]["https://musicbrainz.org/doc/jspf#playlist"]["public"] is False


def test_exclusion_never_claims_complete_if_window_truncated(tmp_path):
    mem = Memory(tmp_path / "m.db")
    lb = FakeLB()
    lb.listens_since = lambda days, max_pages=10: {"complete": False, "listens": [], "days": days}
    tools = Tools(lb, FakeMB(), mem, approve=lambda _: True, current_goal="Make playlist")
    try:
        assert tools.run("verify_recordings", {"mbids": [MBID]}).ok
        result = tools.run("create_playlist", {"title": "Focus", "mbids": [MBID], "exclude_days": 7})
        assert result.error_type == "resource"
        assert lb.called == 0
    finally:
        mem.close()


def test_playlist_duration_optimizer_never_guesses(tmp_path):
    mem = Memory(tmp_path / "m.db")
    second = "ff4ae62f-42c8-46c2-b1b5-447f69a312f1"
    tools = Tools(FakeLB(), FakeMB(), mem, approve=lambda _: True, current_goal="40 minute playlist")
    try:
        tools.verified = {
            MBID: {"duration_ms": 1200000},
            second: {"duration_ms": 1200000},
        }
        res = tools.run("select_playlist_duration", {"mbids": [MBID, second], "target_minutes": 40})
        assert res.ok and res.data["minutes"] == 40
        tools.verified[second]["duration_ms"] = None
        res = tools.run("select_playlist_duration", {"mbids": [MBID, second], "target_minutes": 40})
        assert res.ok and second in res.data["omitted_unknown_length"]
    finally:
        mem.close()
