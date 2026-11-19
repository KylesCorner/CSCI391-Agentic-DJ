"""Offline unit tests: no network, Wikipedia server, Ollama, or account required."""
from encore.on_this_day import (
    birth_candidates, credited_to, date_title, event_day, match_birthday,
    unpack_mcp_result,
)


SAMPLE = [
    {"title": "Events", "text": "1989 – Nothing musical", "sections": []},
    {"title": "Births", "text": "", "sections": [
        {"title": "20th century", "text": (
            "1989 – Taylor Swift, American singer-songwriter\n"
            "1900 – Someone Else, American mathematician\n"
            "1981 – Amy Lee, American singer and pianist"
        ), "sections": []},
    ]},
]


class FakeMB:
    def search_artist(self, name, limit=5):
        return {"artists": [{"name": "Taylor Swift", "id": "artist-123",
                             "life-span": {"begin": "1989-12-13"}}]}

    def search(self, query, limit=8):
        assert query == "arid:artist-123"
        return {"recordings": [{"id": "recording-456", "title": "Test Track",
                                "artist-credit": [{"artist": {"id": "artist-123"}}]}]}

    def lookup(self, mbid):
        return {"id": mbid, "title": "Test Track",
                "artist-credit": [{"artist": {"id": "artist-123"}}]}


def test_date_and_extraction():
    assert event_day("12-13", "UTC") == (12, 13)
    assert date_title(12, 13) == "December 13"
    found = birth_candidates(SAMPLE)
    assert [v["artist"] for v in found] == ["Taylor Swift", "Amy Lee"]
    assert found[0]["birth_year"] == 1989


def test_musicbrainz_birthdate_and_recording_verification():
    candidate = birth_candidates(SAMPLE)[0]
    found = match_birthday(FakeMB(), candidate, 12, 13)
    assert found["artist"] == "Taylor Swift"
    assert found["recording_mbid"] == "recording-456"
    assert match_birthday(FakeMB(), candidate, 12, 12) is None
    assert not credited_to({"artist-credit": []}, "artist-123")


def test_failed_tool_cannot_silently_look_successful():
    class Block:
        text = '{"title": "December 13", "sections": []}'

    class Result:
        isError = True
        content = [Block()]
        structuredContent = None

    try:
        unpack_mcp_result(Result())
    except ValueError as exc:
        assert "error" in str(exc)
    else:
        assert False, "MCP error was swallowed"
