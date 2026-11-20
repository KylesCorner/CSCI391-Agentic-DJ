from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path
import ast
import json
import pytest

from encore.request_intent import Intent, interpret, ALL_DOMAINS
from encore.agent_policies import route, _answer_grounding_error, requested_discovery_count, _normalize
from encore.tools import Tools
from encore.observability import sanitize
from encore.episodic import _strip_secrets
from encore.semantic import _slug, _terms, load_sections
from encore.memory import fts_query


def payload(**kwargs):
    value = {'action':'read', 'domains':['catalog'], 'artist':None, 'track_title':None,
       'playlist_name':None, 'track_count':None, 'duration_minutes':None,
       'distinct':False, 'position':'none','position_index':None,'scope_all':False,
       'feedback_score':None,'needs_preferences':False,'explicit_preference':False,
       'instrumental_only':False,'explicit':False}
    return value | kwargs


def test_ambiguous_request_never_allows_writes():
    p = Intent.from_dict(payload())
    assert not p.permitted_writes
    assert 'create_playlist' not in {s['function']['name'] for s in route(p)}


def test_readonly_model_failure_fails_closed():
    class Broken:
        def extract_intent(self,goal):raise ValueError('model offline')
    assert interpret('delete all playlists',Broken()).permitted_writes == frozenset()


def test_create_explicit_and_artist_constraint():
    p=Intent.from_dict(payload(action='create_playlist',explicit=True,artist='Benny Goodman',track_count=5))
    assert p.permitted_writes == {'create_playlist'}
    assert 'catalog' in p.allowed_domains and 'library' in p.allowed_domains
    assert 'delete_playlist' not in {s['function']['name'] for s in route(p)}


def test_unsafe_write_requires_explicit():
    with pytest.raises(ValueError):
        Intent.from_dict(payload(action='delete_playlist',playlist_name='Mix'))


def test_delete_cannot_target_an_unnamed_playlist():
    with pytest.raises(ValueError):
        Intent.from_dict(payload(action='delete_playlist',explicit=True,scope_all=True))


def test_feedback_requires_explicit_score():
    with pytest.raises(ValueError):
        Intent.from_dict(payload(action='set_feedback',explicit=True))


def test_discovery_count_and_grounded_answer_validation():
    p=Intent.from_dict(payload(action='recommend', track_count=2))
    assert requested_discovery_count(p)==2
    cand={('goodman','sing'): {'artist':'Benny Goodman','title':'Sing Sing Sing','mbids':['a']}}
    result=_answer_grounding_error(p,'Sing Sing Sing — Benny Goodman','answered',cand,set())
    assert 'requested 2 songs' in result
    r=Intent.from_dict(payload())
    assert _answer_grounding_error(r,'anything','answered',cand,set()) is None


def test_song_title_normalization():
    assert _normalize('BÉNNY — Goodman') == 'benny goodman'
    assert _normalize('Sing, Sing, Sing') == 'sing sing sing'


def test_tool_create_denied_before_network():
    lb=SimpleNamespace(token='token')
    t=Tools(lb,None,SimpleNamespace(),approve=lambda x:True,current_goal='explain how to create a playlist')
    outcome=t.run('create_playlist',{'title':'No','mbids':[]})
    assert not outcome.ok


def test_sensitive_trace_redaction():
    assert sanitize({'Authorization':'secret','value':'Bearer xyz'})=={'Authorization':'[REDACTED]','value':'[REDACTED AUTH STRING]'}
    assert 'secret' not in _strip_secrets('Authorization: token secret\nsafe', ('secret',))


def test_slug_and_terms_no_pattern_engine():
    assert _slug('MusicBrainz API!')=='musicbrainz-api'
    assert 'musicbrainz' in _terms('MusicBrainz API!')
    assert 'goodman' in fts_query('Benny Goodman').lower()


def test_no_regex_calls_in_any_replaced_module():
    root = Path(__file__).resolve().parents[1] / 'src' / 'encore'
    for p in root.glob('*.py'):
        if p.name in {'tool_specs.py','clients.py','router.py','working_memory.py','__init__.py'}:continue
        tree=ast.parse(p.read_text(encoding='utf8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):assert all(a.name != 're' for a in node.names), p
            if isinstance(node,ast.ImportFrom):assert node.module not in ('re','regex'), p
            if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name):
                assert node.value.id not in ('re','regex'),p


def test_ollama_returns_schema_constrained_intent(monkeypatch):
    from encore.ollama_client import Ollama
    import encore.ollama_client as ollama_client
    seen={}
    class Response:
        def raise_for_status(self):pass
        def json(self):return {'message':{'content':json.dumps(payload(action='recommend',track_count=5,artist='Benny Goodman'))}}
    def fake_post(url, **kwargs):
        seen['url']=url
        seen['args']=kwargs['json']
        return Response()
    monkeypatch.setattr(ollama_client.requests,'post',fake_post)
    model=Ollama('http://localhost:11434','qwen3:8b')
    raw=model.extract_intent('Find five Benny Goodman songs')
    value=Intent.from_dict(raw)
    assert value.artist=='Benny Goodman'
    assert value.track_count==5
    assert value.action=='recommend'
    assert seen['args']['format']['type']=='object'
    assert seen['args']['stream'] is False


def test_on_this_day_extraction_uses_exact_wikipedia_evidence():
    # Extract just the pure candidate function to avoid launching an MCP server.
    import ast
    from types import SimpleNamespace
    from unittest.mock import Mock
    module=ast.parse((Path(__file__).resolve().parents[1] / 'src' / 'encore' / 'on_this_day.py').read_text())
    func=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='birth_candidates')
    func.decorator_list=[]
    data={'births':[
        {'artist':'Jane Doe','birth_year':1972,'statement':'1972 – Jane Doe, pianist'},
        {'artist':'Unlisted Person','birth_year':1989,'statement':'1989 – Unlisted Person, singer'}]}
    fake_request=SimpleNamespace(post=Mock(return_value=SimpleNamespace(
        raise_for_status=lambda:None,
        json=lambda:{'message':{'content':json.dumps(data)}})))
    scope={'requests':fake_request,'json':json,'normalize': lambda v: ''.join(ch for ch in v.casefold() if ch.isalnum())}
    exec(compile(ast.Module(body=[func], type_ignores=[]),'test_births.py','exec'),scope)
    sections=[{'title':'Births','text':'1972 – Jane Doe, pianist\n2000 – Other, actor','sections':[]}]
    results=scope['birth_candidates'](sections,cfg=SimpleNamespace(ollama_host='http://localhost:11434',model='qwen3:8b'))
    assert len(results)==1
    assert results[0]['artist']=='Jane Doe'
