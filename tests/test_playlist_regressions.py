"""Playlist search and recovery regressions using the structured intent contract."""
from encore.agent import run_agent
from encore.memory import Memory
from encore.tools import Tools
from encore.request_intent import Intent
from test_playlist_creation import LB, IDS
from test_router_intent_integration import intent_payload

GOAL = (
    'Create a private playlist called Swing Test with 5 distinct Benny Goodman '
    'recordings. Verify the recordings before writing. Do not export.'
)


def intended_create():
    return intent_payload(action='create_playlist', explicit=True,
                          domains=['catalog', 'library'],
                          playlist_name='Swing Test', artist='Benny Goodman',
                          track_count=5, distinct=True)


class Catalog:
    def __init__(self):
        self.queries = []
    def search(self, query):
        self.queries.append(query)
        assert query == 'artist:"Benny Goodman"'
        rows = [
            {'id': IDS[0], 'title': 'Sing, Sing, Sing',
             'artist-credit': [{'name': 'Benny Goodman'}]},
            {'id': '462dc36e-318e-4f87-9131-693f7aa87617',
             'title': 'Sing, Sing, Sing',
             'artist-credit': [{'name': 'Benny Goodman'}]},
            {'id': '229e2bb4-c9b2-4162-860c-ff90137bbfcc',
             'title': 'Goodman', 'artist-credit': [{'name': 'Max B'}]},
        ]
        rows.extend({'id': mbid, 'title': title,
                     'artist-credit': [{'name': 'Benny Goodman'}]}
                    for mbid, title in zip(IDS[1:], [
                        "Don't Be That Way", "Stompin' at the Savoy",
                        'King Porter Stomp', 'Air Mail Special',
                    ]))
        return {'recordings': rows}
    def verify(self, mbids):
        return {'valid': [{'mbid': mbid, 'title': 'Title', 'artist_credit': []}
                          for mbid in mbids], 'invalid': []}


class ScriptedModel:
    def __init__(self):
        self.n = 0
    def extract_intent(self, goal):
        return intended_create()
    def route_domains(self, goal):
        return ('catalog', 'library'), {'model': 'test', 'tokens_in': 0, 'tokens_out': 0}
    def chat(self, messages, specs):
        sequence = [
            ('search_recordings', {'query': 'Benny Goodman'}),
            ('verify_recordings', {'mbids': IDS}),
            ('create_playlist', {'mbids': IDS}),
            ('create_playlist', {'title': 'Swing Test', 'mbids': IDS}),
            ('final_answer', {'status': 'answered',
                              'answer': 'Created Swing Test with 5 verified tracks.',
                              'confidence': 'high', 'sources': []}),
        ]
        name, arguments = sequence[self.n]
        self.n += 1
        assert name in {s['function']['name'] for s in specs}
        return {'message': {'tool_calls': [
            {'function': {'name': name, 'arguments': arguments}}
        ]}}


def make(tmp_path, approve):
    mem = Memory(tmp_path / 'test.db')
    lb, catalog = LB(), Catalog()
    tools = Tools(lb, catalog, mem, approve=approve, current_goal=GOAL)
    tools.intent = Intent.from_dict(intended_create())
    return mem, lb, catalog, tools


def test_artist_target_search_rewrites_bad_model_query(tmp_path):
    mem, lb, catalog, tools = make(tmp_path, lambda _: True)
    try:
        result = tools.run('search_recordings', {'query': 'Benny Goodman'})
        assert result.ok, result.message
        assert len(result.data) == 5
        assert all('Benny Goodman' in row['artist'] for row in result.data)
        assert len({row['title'].casefold() for row in result.data}) == 5
        assert catalog.queries == ['artist:"Benny Goodman"']
    finally: mem.close()


def test_reject_duplicate_titles_or_wrong_artist_before_approval(tmp_path):
    approvals = []
    mem, lb, catalog, tools = make(tmp_path,
                                    lambda desc: approvals.append(desc) or True)
    try:
        assert tools.run('search_recordings', {'query': 'Benny Goodman'}).ok
        assert tools.run('verify_recordings', {'mbids': IDS}).ok
        tools.searched_recordings[IDS[1]] = {'artist': 'Max B', 'title': "Don't Be That Way"}
        assert tools.run('create_playlist', {'title': 'Swing Test', 'mbids': IDS}).error_type == 'invalid'
        assert not approvals and lb.created == 0
        tools.searched_recordings[IDS[1]] = {'artist': 'Benny Goodman', 'title': 'Sing, Sing, Sing'}
        assert tools.run('create_playlist', {'title': 'Swing Test', 'mbids': IDS}).error_type == 'invalid'
        assert not approvals and lb.created == 0
    finally: mem.close()


def test_agent_recovers_missing_playlist_title_without_fabrication(tmp_path):
    approvals = []
    mem, lb, catalog, tools = make(tmp_path,
                                    lambda desc: approvals.append(desc) or True)
    try:
        result = run_agent(GOAL, tools, ScriptedModel())
        assert result.status == 'answered', result.answer
        assert [row['tool'] for row in result.tool_trace] == [
            'search_recordings', 'verify_recordings', 'create_playlist',
            'create_playlist', 'final_answer',
        ]
        assert result.tool_trace[2]['error_type'] == 'invalid'
        assert result.tool_trace[3]['confirmed'] is True
        assert len(approvals) == 1 and lb.created == 1
    finally: mem.close()
