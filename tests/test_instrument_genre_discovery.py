"""Regression tests for musical descriptors confused with artist/title entities."""
from __future__ import annotations

import ast
from pathlib import Path

from encore.request_intent import Intent, SCHEMA
from encore.tools import Tools


def payload(**updates):
    value = {
        'action': 'recommend', 'domains': ['catalog', 'discovery'],
        'artist': None, 'track_title': None, 'playlist_name': None,
        'track_count': 7, 'duration_minutes': None, 'distinct': True,
        'position': 'none', 'position_index': None, 'scope_all': False,
        'feedback_score': None, 'needs_preferences': False,
        'explicit_preference': False, 'instrumental_only': False, 'explicit': False,
    }
    return value | updates


class Catalog:
    def __init__(self):
        self.queries = []
        self.artists = []
        self.artist_name = 'Sample Ensemble'

    def search(self, query):
        self.queries.append(query)
        return {'recordings': [{
            'id': 'f0c20a72-9a7a-4fb3-9d22-7a3b17814b11',
            'title': 'Swing Sample',
            'artist-credit': [{'name': self.artist_name}],
            'length': 185000,
        }]}

    def search_artist(self, query):
        self.artists.append(query)
        raise AssertionError('Instrument is not a MusicBrainz artist!')


def make(intent=None):
    cat = Catalog()
    tool = Tools(type('LB', (), {'token': ''})(), cat, object(),
                 approve=lambda _: False, current_goal='clarinet swing jazz')
    if intent:
        tool.intent = intent
    return tool, cat


def test_ollama_fields_include_genre_and_instrument():
    assert 'instrument' in SCHEMA['properties']
    assert 'genre' in SCHEMA['properties']


def test_bad_model_interpretation_is_corrected():
    parsed = Intent.from_dict(payload(artist='clarinet', track_title='swing jazz'))
    assert parsed.artist is None and parsed.track_title is None
    assert parsed.instrument == 'clarinet'
    assert parsed.genre == 'swing jazz'
    assert parsed.track_count == 7
    assert not parsed.permitted_writes


def test_recover_when_small_model_calls_it_read():
    parsed = Intent.from_dict(payload(action='read', artist='clarinet',
                                      track_title='swing jazz'))
    assert parsed.action == 'recommend'
    assert parsed.track_count == 7


def test_artist_name_remains_artist():
    parsed = Intent.from_dict(payload(artist='Benny Goodman', track_title=None))
    assert parsed.artist == 'Benny Goodman'
    assert parsed.instrument is None


def test_tag_query_replaces_malformed_artist_and_recording_query():
    parsed = Intent.from_dict(payload(artist='clarinet', track_title='swing jazz'))
    tool, cat = make(parsed)
    result = tool.run('search_recordings', {
        'query': 'artist:"clarinet" AND recording:"swing jazz"'})
    assert result.ok, result.message
    assert cat.queries == ['tag:"swing" AND tag:"clarinet"']
    assert result.data[0]['evidence_note'].startswith('MusicBrainz search-tag')


def test_an_instrument_is_rejected_from_artist_search_before_http():
    tool, cat = make(Intent.from_dict(payload(artist='clarinet', track_title='swing jazz')))
    from encore.clients import ServiceError
    import pytest
    with pytest.raises(ServiceError) as err:
        tool.tool_search_artists('clarinet')
    assert err.value.kind == 'resource'
    assert cat.artists == []


def test_fallback_rewrites_bad_query_even_without_valid_intent():
    tool, cat = make()
    result = tool.run('search_recordings', {
        'query': 'artist:"clarinet" AND recording:"swing jazz"'})
    assert result.ok, result.message
    assert cat.queries == ['tag:"swing" AND tag:"clarinet"']


def test_normal_artist_query_unchanged():
    parsed = Intent.from_dict(payload(artist='Benny Goodman'))
    tool, cat = make(parsed)
    cat.artist_name = 'Benny Goodman'
    result = tool.run('search_recordings', {'query': 'Benny Goodman'})
    assert result.ok
    assert cat.queries == ['artist:"Benny Goodman"']


def test_no_regex_in_updated_modules():
    root = Path(__file__).parents[1] / 'src' / 'encore'
    for module in ('request_intent.py', 'ollama_client.py', 'tools.py', 'agent.py'):
        syntax = ast.parse((root / module).read_text())
        assert not any(isinstance(node, ast.Import) and any(a.name == 're' for a in node.names)
                       or isinstance(node, ast.ImportFrom) and node.module == 're'
                       for node in ast.walk(syntax))


def test_http_503_uses_cooldown_without_retrying_any_write(monkeypatch):
    from encore.agent import run_agent
    from encore.tools import ToolResult
    delays = []
    monkeypatch.setattr('encore.agent.time.sleep', delays.append)

    class StubMemory:
        def search_sessions(self, goal, limit=3):
            return []

    class StubTools:
        mem = StubMemory()
        session_id = 'test-session'
        verified = {}
        def __init__(self):
            self.calls = 0
        def run(self, name, args):
            self.calls += 1
            return ToolResult(ok=False, error_type='transient',
                              message='External API returned HTTP 503')

    class StubModel:
        model = 'fake-ollama'
        def extract_intent(self, goal):
            return payload()
        def route_domains(self, goal):
            return ('catalog', 'discovery'), {'model': 'mock-router', 'tokens_in': 1, 'tokens_out': 1}
        def chat(self, messages, specs):
            return {'message': {'tool_calls': [{
                'function': {'name': 'search_recordings', 'arguments': {'query': 'tag:"swing"'}}
            }]}}

    tools = StubTools()
    result = run_agent('Find clarinet swing jazz', tools, StubModel(), max_steps=3)
    assert tools.calls == 3
    assert len(delays) == 2
    assert delays[0] >= 5
    assert delays[1] >= 10
    assert result.status == 'stuck'


def test_agent_does_not_offer_artist_search_for_instrument_request():
    from encore.agent import run_agent

    class StubMemory:
        def search_sessions(self, goal, limit=3):
            return []

    class StubTools:
        mem = StubMemory()
        verified = {}
        session_id = 'test'

    class StubModel:
        model = 'fake-ollama'
        seen = None
        def extract_intent(self, goal):
            # Reproduce the actual bad model interpretation from the user log.
            return payload(artist='clarinet', track_title='swing jazz')
        def route_domains(self, goal):
            return ('catalog', 'discovery'), {'model': 'mock-router', 'tokens_in': 1, 'tokens_out': 1}
        def chat(self, messages, specs):
            if self.seen is None:
                self.seen = {s['function']['name'] for s in specs}
            return {'message': {'tool_calls': [{
                'function': {'name': 'final_answer', 'arguments': {
                    'status': 'answered_low_confidence', 'answer': 'No verified recordings yet',
                    'confidence': 'low', 'sources': [],
                }}
            }]}}

    model = StubModel()
    run_agent('Can you find 7 clarinet swing jazz recordings?', StubTools(), model)
    assert 'search_artists' not in model.seen
    assert 'search_recordings' in model.seen
