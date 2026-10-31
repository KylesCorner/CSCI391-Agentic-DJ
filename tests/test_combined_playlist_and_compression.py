"""Regression coverage for the October 31 10 tracks / 30 minutes failure."""
import json
from pathlib import Path

from encore.agent import run_agent
from encore.memory import Memory
from encore.tools import Tools, ToolResult
from encore.episodic import SessionSummarizer

DURATIONS = [
    ("Benny Goodman", 130333), ("Moten Swing", 155776),
    ("Swing Angel", 190000), ("Organ Grinder's Swing", 190800),
    ("Life Goes to a Party", 252893), ("Solo Flight", 169466),
    ("King Porter Stomp", 189986), ("Avalon", 375000),
    ("One O'Clock Jump", 394773), ("Sensation Rag", 79626),
    ("Lullaby in Rhythm", 213000), ("Dizzy Spells", 337506),
]


def make(tmp_path, goal):
    mem = Memory(tmp_path / 'memory.db')
    tools = Tools(type('LB', (), {'token': 'fake'})(), None, mem,
                  approve=lambda _: False, current_goal=goal)
    ids = []
    for n, (title, length) in enumerate(DURATIONS, start=1):
        mid = f'{n:08x}-0000-4000-8000-000000000000'
        ids.append(mid)
        tools.verified[mid] = {"mbid": mid, "title": title, "duration_ms": length}
        tools.searched_recordings[mid] = {"title": title, "artist": "Benny Goodman"}
    return mem, tools, ids


def test_trace_ten_songs_duration_conflict(tmp_path):
    goal = 'Find 10 Benny Goodman recordings, verify them, and build a playlist close to 30 minutes.'
    mem, tools, ids = make(tmp_path, goal)
    try:
        result = tools.run('select_playlist_duration', {'mbids': ids, 'target_minutes': 30})
        assert result.ok, result.message
        data = result.data
        assert data['required_track_count'] == 10
        assert data['selected_track_count'] == 10
        assert len(data['mbids']) == 10
        assert data['ready'] is False
        assert abs(data['minutes'] - 31.82) < 0.05
        assert 'NOT READY' in data['note']
        # Even when model ignores the tool's ready=False, the write is blocked.
        outcome = tools.run('create_playlist', {'title': '30 Minute Test', 'mbids': data['mbids']})
        assert not outcome.ok and outcome.error_type == 'resource'
        assert 'Duration mismatch' in outcome.message
    finally:
        mem.close()


def test_extra_short_song_can_satisfy_both(tmp_path):
    goal = 'Find 10 Benny Goodman recordings, verify them, and build a playlist close to 30 minutes.'
    mem, tools, ids = make(tmp_path, goal)
    try:
        mid = '77777777-0000-4000-8000-000000000000'
        tools.verified[mid] = {'mbid': mid, 'title': 'Additional short recording', 'duration_ms': 60000}
        data = tools.run('select_playlist_duration', {'mbids': ids + [mid], 'target_minutes': 30}).data
        assert data['ready'] and data['selected_track_count'] == 10
        assert abs(data['difference_seconds']) <= 60
    finally:
        mem.close()


def test_model_cannot_change_duration_target(tmp_path):
    mem, tools, ids = make(tmp_path, 'Build a 30-minute playlist with 10 Benny Goodman recordings')
    try:
        result = tools.run('select_playlist_duration', {'mbids': ids, 'target_minutes': 32})
        assert not result.ok and result.error_type == 'invalid'
        assert 'requested 30 minutes' in result.message
    finally:
        mem.close()


class FakeModel:
    def __init__(self):
        self.n = 0
        self.summarized = 0
        self.contexts = []

    def summarize_working_memory(self, goal, text):
        self.summarized += 1
        assert goal == 'Tell me about music in my collection'
        return 'Earlier tools returned facts; no write occurred.', 120, 30

    def chat(self, messages, specs):
        self.n += 1
        self.contexts.append(messages)
        tasks = ['get_listen_count', 'get_playing_now', 'get_recent_listens',
                 'get_top_artists', 'get_top_recordings', 'search_artists']
        if self.n <= len(tasks):
            return {'message': {'tool_calls': [{'function': {'name': tasks[self.n - 1], 'arguments': {}}}]}}
        return {'message': {'tool_calls': [{'function': {'name': 'final_answer',
                    'arguments': {'answer': 'All observations collected.', 'status': 'answered'}}}]}}


def test_rolling_compression_and_user_scoped_archive_deletion(tmp_path):
    goal = 'Tell me about music in my collection'
    mem, tools, ids = make(tmp_path, goal)
    try:
        # Model chooses tools; executor returns large observations for each.
        tools.run = lambda name, args: ToolResult(True, data={'text': 'X' * 2600})
        model = FakeModel()
        result = run_agent(goal, tools, model, context_budget_tokens=1300)
        assert result.status == 'answered'
        events = [x for x in result.tool_trace if x['tool'] == 'compress']
        assert events and any(x['method'] == 'llm_summary' for x in events)
        assert result.steps == 7  # Compressor is NOT counted as an agent action
        assert model.summarized
        archives = list((tmp_path / 'working_archives').rglob('*.jsonl'))
        assert len(archives) == 1
        assert json.loads(archives[0].read_text().splitlines()[0])['messages']
        assert 'COMPRESSED WORKING MEMORY' in json.dumps(model.contexts)
        counts = mem.delete_personal()
        assert counts['working_archives'] == 1
        assert not archives[0].exists()
    finally:
        mem.close()


def test_malformed_json_summary_is_retried_once(monkeypatch):
    attempts = []
    class Response:
        def __init__(self, text): self._text = text
        def raise_for_status(self): pass
        def json(self): return {'message': {'content': self._text}}
    def post(url, json, timeout):
        attempts.append(json)
        if len(attempts) == 1:
            return Response('{"field":')
        return Response('{"field":"safe"}')
    monkeypatch.setattr('encore.episodic.requests.post', post)
    summarizer = SessionSummarizer('http://local', 'small-model')
    schema = {'type': 'object', 'properties': {'field': {'type': 'string'}}, 'required': ['field']}
    assert summarizer._ask('Instructions', {'goal': 'study'}, schema) == {'field': 'safe'}
    assert len(attempts) == 2 and all(x['think'] is False for x in attempts)


def test_artist_scope_for_find_n_recordings_then_build_playlist(tmp_path):
    goal = 'Find 10 Benny Goodman recordings, verify them, and build a playlist close to 30 minutes.'
    mem, tools, ids = make(tmp_path, goal)
    try:
        assert tools._requested_playlist_artist() == 'Benny Goodman'
        class MB:
            def __init__(self): self.queries = []
            def search(self, query):
                self.queries.append(query)
                return {'recordings': [{
                    'id': ids[0], 'title': 'Benny Goodman',
                    'artist-credit': [{'name': 'Benny Goodman'}], 'length': 130333,
                }]}
        mb = MB()
        tools.mb = mb
        data = tools.run('search_recordings', {'query': 'artist:Benny Goodman'})
        assert data.ok, data.message
        assert mb.queries[0] == 'artist:"Benny Goodman"'
    finally:
        mem.close()
