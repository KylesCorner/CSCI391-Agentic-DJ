"""On This Day tests: mock Ollama's structured response, not the old regex parser."""
from types import SimpleNamespace
import json

from encore.on_this_day import (
    birth_candidates, credited_to, date_title, event_day, match_birthday,
    unpack_mcp_result,
)

SAMPLE = [
    {"title": "Events", "text": "1989 – Nothing musical", "sections": []},
    {"title": "Births", "text": "", "sections": [{
        "title": "20th century",
        "text": ("1989 – Taylor Swift, American singer-songwriter\n"
                 "1900 – Someone Else, American mathematician\n"
                 "1981 – Amy Lee, American singer and pianist"),
        "sections": [],
    }]},
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


class Response:
    def __init__(self, data):
        self.data = data
    def raise_for_status(self):
        pass
    def json(self):
        return {'message': {'content': json.dumps(self.data)}}


def mock_ollama(monkeypatch, rows):
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs['json'])
        return Response({'births': rows})
    monkeypatch.setattr('encore.on_this_day.requests.post', post)
    return calls


def test_date_and_structured_extraction(monkeypatch):
    assert event_day('12-13', 'UTC') == (12, 13)
    assert date_title(12, 13) == 'December 13'
    rows = [
        {'artist': 'Taylor Swift', 'birth_year': 1989,
         'statement': '1989 – Taylor Swift, American singer-songwriter'},
        {'artist': 'Amy Lee', 'birth_year': 1981,
         'statement': '1981 – Amy Lee, American singer and pianist'},
    ]
    calls = mock_ollama(monkeypatch, rows)
    cfg = SimpleNamespace(ollama_host='http://localhost:11434', model='test-model')
    found = birth_candidates(SAMPLE, cfg=cfg)
    assert [v['artist'] for v in found] == ['Taylor Swift', 'Amy Lee']
    assert found[0]['birth_year'] == 1989
    assert calls and calls[0]['format']['type'] == 'object'
    assert calls[0]['stream'] is False


def test_musicbrainz_birthdate_and_recording_verification(monkeypatch):
    mock_ollama(monkeypatch, [{
        'artist': 'Taylor Swift', 'birth_year': 1989,
        'statement': '1989 – Taylor Swift, American singer-songwriter',
    }])
    cfg = SimpleNamespace(ollama_host='http://localhost:11434', model='test-model')
    candidate = birth_candidates(SAMPLE, cfg=cfg)[0]
    found = match_birthday(FakeMB(), candidate, 12, 13)
    assert found['artist'] == 'Taylor Swift'
    assert found['recording_mbid'] == 'recording-456'
    assert match_birthday(FakeMB(), candidate, 12, 12) is None
    assert not credited_to({'artist-credit': []}, 'artist-123')


def test_model_cannot_invent_unlisted_birthdays(monkeypatch):
    mock_ollama(monkeypatch, [
        {'artist': 'Invented Celebrity', 'birth_year': 2000,
         'statement': '2000 – Invented Celebrity, pianist'},
    ])
    cfg = SimpleNamespace(ollama_host='http://localhost:11434', model='test-model')
    assert birth_candidates(SAMPLE, cfg=cfg) == []


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
        assert 'error' in str(exc)
    else:
        assert False, 'MCP error was swallowed'
