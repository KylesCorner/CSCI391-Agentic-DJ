"""Assignment contract: enum router, tier usage, final schema and source validation.
No external network calls or account writes.
"""
import json
from uuid import uuid4

import pytest
from jsonschema import ValidationError

from encore.agent import Ollama, run_agent
from encore.memory import Memory
from encore.router import (
    DOMAINS, DOMAIN_SCHEMA, TOOL_DOMAIN, validate_domains, tools_for_domains,
)
from encore.tools import INDEX, SPECS, Tools


class Response:
    def __init__(self, data):
        self.data = data
    def raise_for_status(self):
        pass
    def json(self):
        return self.data


def test_all_tools_classified_and_router_is_strict():
    assert set(INDEX) <= set(TOOL_DOMAIN)
    assert validate_domains({'domains': ['catalog', 'memory']}) == ('catalog', 'memory')
    for malformed in [
        {'domains': ['catalog', 'catalog']},
        {'domains': ['other']},
        {'domains': ['none', 'library']},
        {'domains': []},
        {'domains': ['catalog'], 'extra': True},
    ]:
        with pytest.raises(ValidationError):
            validate_domains(malformed)


def test_routed_domains_never_broaden_write_permissions():
    from encore.agent import route
    allowed = route('Add a recording to my Swing Test playlist')
    result = tools_for_domains(allowed, ('library', 'catalog'))
    names = {s['function']['name'] for s in result}
    assert 'create_playlist' not in names
    assert 'delete_playlist' not in names
    assert 'add_playlist_items' in names
    assert 'final_answer' in names


def test_ollama_router_uses_schema_constrained_json(monkeypatch):
    calls = []
    def post(url, **kw):
        calls.append(kw['json'])
        return Response({'message': {'content': '{"domains": ["catalog", "discovery"]}'},
                         'prompt_eval_count': 37, 'eval_count': 8})
    monkeypatch.setattr('encore.agent.requests.post', post)
    llm = Ollama('http://127.0.0.1:11434', 'tool-model',
                 cheap_model='cheap-model', final_model='expensive-model',
                 router_model='router-model')
    domains, usage = llm.route_domains('Find swing jazz songs')
    assert domains == ('catalog', 'discovery')
    assert usage == {'model': 'router-model', 'tokens_in': 37, 'tokens_out': 8}
    assert calls[0]['format'] == DOMAIN_SCHEMA
    assert calls[0]['model'] == 'router-model'
    assert calls[0]['think'] is False


def test_ollama_finalizer_uses_expensive_model_and_output_schema(monkeypatch):
    calls = []
    final = {'answer': 'There are 42 listens', 'status': 'answered',
             'confidence': 'high', 'sources': []}
    def post(url, **kw):
        calls.append(kw['json'])
        return Response({'message': {'content': json.dumps(final)},
                         'prompt_eval_count': 13, 'eval_count': 5})
    monkeypatch.setattr('encore.agent.requests.post', post)
    llm = Ollama('http://localhost:11434', 'cheap', final_model='expensive')
    reviewed, usage = llm.review_final('How many listens?', final, [])
    assert reviewed == final
    assert calls[0]['model'] == 'expensive'
    assert calls[0]['format'] == INDEX['final_answer']['function']['parameters']
    assert usage['tokens_in'] == 13


class CountLB:
    token = ''
    def count(self):
        return {'count': 42}


class RoutedMock:
    def __init__(self, domains=('listening',), fail=False):
        self.domains = domains
        self.fail = fail
        self.step = 0
    def route_domains(self, goal):
        if self.fail:
            raise ValueError('invalid routing payload')
        return self.domains, {'model': 'cheap-route', 'tokens_in': 17, 'tokens_out': 3}
    def chat(self, messages, tools):
        self.step += 1
        names = {s['function']['name'] for s in tools}
        if self.step == 1:
            assert 'get_listen_count' in names
            return {'message': {'tool_calls': [{'function': {'name': 'get_listen_count', 'arguments': {}}}]},
                    'prompt_eval_count': 30, 'eval_count': 10}
        return {'message': {'tool_calls': [{'function': {
            'name': 'final_answer',
            'arguments': {'answer': '42 listens were observed', 'status': 'answered',
                          'confidence': 'high', 'sources': []},
        }}]}, 'prompt_eval_count': 30, 'eval_count': 10}


@pytest.mark.parametrize('fail', [False, True])
def test_run_agent_uses_model_router_and_safe_fallback(tmp_path, fail):
    mem = Memory(tmp_path / 'db.sqlite')
    events = []
    class EventLog:
        def emit(self, event, **kw):
            events.append((event, kw))
    try:
        tools = Tools(CountLB(), None, mem, approve=lambda _: False,
                      current_goal='How many listens?', session_id=str(uuid4()))
        result = run_agent('How many listens?', tools, RoutedMock(fail=fail),
                           event_log=EventLog())
        assert result.status == 'answered'
        assert result.steps == 2
        routes = [x for k, x in events if k == 'route']
        assert len(routes) == 1
        assert routes[0]['routing_strategy'] == (
            'all_domains_fallback_safety_scoped' if fail else 'ollama_schema')
        assert result.prompt_tokens == (60 if fail else 77)
    finally:
        mem.close()


def test_legacy_final_tool_call_gets_required_metadata(tmp_path):
    class LegacyModel:
        def chat(self, messages, tools):
            return {'message': {'tool_calls': [{'function': {'name': 'final_answer',
                'arguments': {'status': 'answered', 'answer': 'Hello'}}}]}}
    mem = Memory(tmp_path / 'db.sqlite')
    try:
        tools = Tools(CountLB(), None, mem, approve=lambda _: False,
                      current_goal='Say hello', session_id=str(uuid4()))
        result = run_agent('Say hello', tools, LegacyModel(), max_steps=2)
        assert result.status == 'answered'
        assert result.confidence == 'medium'
        assert result.sources == []
        assert set(INDEX['final_answer']['function']['parameters']['required']) == {
            'answer', 'status', 'confidence', 'sources'
        }
    finally:
        mem.close()


def test_ollama_chat_switches_model_by_tier(monkeypatch):
    posts = []
    def post(url, **kwargs):
        posts.append(kwargs['json'])
        return Response({'message': {'tool_calls': []}})
    monkeypatch.setattr('encore.agent.requests.post', post)
    llm = Ollama('http://localhost:11434', 'default',
                 cheap_model='cheap-qwen', final_model='expensive-qwen')
    llm.chat([{'role': 'user', 'content': 'test'}], [], tier='cheap')
    llm.chat([{'role': 'user', 'content': 'test'}], [], tier='expensive')
    assert [x['model'] for x in posts] == ['cheap-qwen', 'expensive-qwen']


def test_feedback_write_remains_accessible_if_router_misses_library(tmp_path):
    from encore.agent import route
    class Misclassified(RoutedMock):
        def route_domains(self, goal):
            return ('catalog',), {'model': 'misrouter', 'tokens_in': 1, 'tokens_out': 1}
        def chat(self, messages, specs):
            names = {sp['function']['name'] for sp in specs}
            assert 'set_feedback' in names
            return {'message': {'tool_calls': [{'function': {
                'name': 'final_answer',
                'arguments': {'status': 'answered', 'answer': 'I loved the recording',
                              'confidence': 'high', 'sources': []},
            }}]}}
    mem = Memory(tmp_path/'db.sqlite')
    try:
        tools = Tools(CountLB(), None, mem, approve=lambda _: False,
                      current_goal='Love this recording', session_id=str(uuid4()))
        result = run_agent('Love this recording', tools, Misclassified())
        assert result.status == 'stuck'
    finally:
        mem.close()
