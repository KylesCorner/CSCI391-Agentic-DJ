"""Offline regression coverage for schema routing + typed write-authorization.

No live Ollama calls and no external account writes. Each test supplies its
own structured intent explicitly instead of invoking the old text parser.
"""
from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

import pytest
from encore.agent import run_agent
from encore.agent_policies import route
from encore.memory import Memory
from encore.request_intent import Intent, WRITE_TOOLS
from encore.router import DOMAIN_SCHEMA, validate_domains, tools_for_domains
from encore.tools import Tools


def intent_payload(**changed):
    payload = {
        "action": "read", "domains": ["listening"], "artist": None,
        "track_title": None, "playlist_name": None, "track_count": None,
        "duration_minutes": None, "distinct": False, "position": "none",
        "position_index": None, "scope_all": False, "feedback_score": None,
        "needs_preferences": False, "explicit_preference": False,
        "instrumental_only": False, "explicit": False,
    }
    return payload | changed


class LB:
    token = 'test-token'
    def count(self):
        return {"count": 42}


class Model:
    def __init__(self, *, domains=('listening',), fail_route=False, intent=None):
        self.domains = domains
        self.fail_route = fail_route
        self.intent_payload = intent or intent_payload()
        self.step = 0
        self.seen = []
        self.router_calls = 0

    def extract_intent(self, goal):
        return deepcopy(self.intent_payload)

    def route_domains(self, goal):
        self.router_calls += 1
        if self.fail_route:
            raise RuntimeError('Ollama unavailable')
        return self.domains, {"model": "test-router", "tokens_in": 17, "tokens_out": 4}

    def chat(self, messages, specs):
        available = {sp['function']['name'] for sp in specs}
        self.seen.append(available)
        self.step += 1
        if self.step == 1 and 'get_listen_count' in available:
            name, arguments = 'get_listen_count', {}
        else:
            name, arguments = 'final_answer', {
                'status': 'answered', 'answer': '42 listens were observed',
                'confidence': 'high', 'sources': ['tool:get_listen_count'],
            }
        assert name in available
        return {'message': {'tool_calls': [
            {'function': {'name': name, 'arguments': arguments}}
        ]}, 'prompt_eval_count': 15, 'eval_count': 4}


def _run(tmp_path, goal, model):
    mem = Memory(tmp_path / 'test.db')
    log = []
    class Events:
        def emit(self, event, **fields):
            log.append((event, fields))
    try:
        tools = Tools(LB(), None, mem, approve=lambda _: False,
                      current_goal=goal, session_id=str(uuid4()))
        result = run_agent(goal, tools, model, event_log=Events())
        return result, model, log, tools.intent
    finally:
        mem.close()


def test_schema_router_is_called_and_token_usage_is_accounted(tmp_path):
    result, model, events, extracted = _run(tmp_path, 'How many listens?', Model())
    assert model.router_calls == 1
    assert result.status == 'answered'
    assert result.steps == 2
    assert result.prompt_tokens >= 17
    routes = [fields for name, fields in events if name == 'route']
    assert len(routes) == 1
    assert routes[0]['routing_strategy'] == 'ollama_schema'
    assert routes[0]['domains'] == ['listening']
    assert routes[0]['tokens_in'] == 17
    assert any(name == 'llm_call' and fields.get('phase') == 'routing'
               and fields.get('tokens_in') == 17 for name, fields in events)
    assert 'get_listen_count' in model.seen[0]
    assert 'create_playlist' not in model.seen[0]


def test_failed_router_falls_back_to_reads_without_writes(tmp_path):
    result, model, events, _ = _run(tmp_path, 'How many listens?', Model(fail_route=True))
    routes = [f for event, f in events if event == 'route']
    assert routes[0]['router_fallback'] is True
    assert routes[0]['routing_strategy'] == 'all_domains_fallback_safety_scoped'
    assert model.router_calls == 1
    assert result.status == 'answered'
    assert all(not (names & {tool for group in WRITE_TOOLS.values() for tool in group})
               for names in model.seen)


def test_explicit_feedback_is_available_if_domain_router_misses_library(tmp_path):
    structured = intent_payload(action='set_feedback', domains=['catalog'],
                                explicit=True, feedback_score=1)
    model = Model(domains=('catalog',), intent=structured)
    _result, model, events, intent = _run(tmp_path, 'Love this recording', model)
    assert intent.permitted_writes == {'set_feedback'}
    assert 'set_feedback' in model.seen[0]
    assert 'create_playlist' not in model.seen[0]
    assert 'library' in [f for e,f in events if e=='route'][0]['domains']


def test_no_unverified_writes_even_when_router_requests_library(tmp_path):
    model = Model(domains=('library', 'catalog'), intent=intent_payload())
    _, model, _, _ = _run(tmp_path, 'How do I make a playlist?', model)
    assert 'create_playlist' not in model.seen[0]
    assert 'delete_playlist' not in model.seen[0]
    assert 'final_answer' in model.seen[0]


def test_safety_route_accepts_typed_intent_not_raw_string():
    p = Intent.from_dict(intent_payload(action='add_playlist_tracks',
                                      playlist_name='Swing Test',
                                      explicit=True))
    routed = route(p)
    names = {x['function']['name'] for x in routed}
    assert {'add_playlist_tracks', 'add_playlist_items', 'final_answer'} <= names
    assert 'create_playlist' not in names
    assert 'delete_playlist' not in names
    narrowed = tools_for_domains(routed, ('library', 'catalog'))
    assert 'add_playlist_items' in {x['function']['name'] for x in narrowed}


def test_invalid_router_schema_falls_back_without_write_leak(tmp_path):
    model = Model(domains=('library', 'library'))  # duplicate => invalid
    _, model, events, _ = _run(tmp_path, 'How many listens?', model)
    assert [f for n,f in events if n=='route'][0]['router_fallback'] is True
    assert 'create_playlist' not in model.seen[0]


def test_none_domain_only_allows_final_answer(tmp_path):
    model = Model(domains=('none',), intent=intent_payload(domains=[]))
    _, model, events, _ = _run(tmp_path, 'Explain this?', model)
    assert model.seen[0] == {'final_answer'}
    assert [f for n,f in events if n=='route'][0]['domains'] == ['none']


def test_intent_extraction_failure_never_authorizes_playlist_creation(tmp_path):
    class MissingIntent(Model):
        def extract_intent(self, goal):
            raise ValueError('bad JSON')
    model = MissingIntent(domains=('library',))
    _, model, _, _ = _run(tmp_path, 'Create a playlist called Test', model)
    assert 'create_playlist' not in model.seen[0]


def test_structured_read_intent_recovers_from_domain_false_negative(tmp_path):
    model = Model(domains=('none',), intent=intent_payload(domains=['listening']))
    _, model, events, _ = _run(tmp_path, 'How many listens?', model)
    assert 'get_listen_count' in model.seen[0]
    assert [f for event, f in events if event=='route'][0]['domains'] == ['listening']
