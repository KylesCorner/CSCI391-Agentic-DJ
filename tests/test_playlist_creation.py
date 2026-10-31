from encore.clients import mbids_from_track
from encore.memory import Memory
from encore.tools import Tools
from encore.agent import run_agent

IDS = [
    'f0c20a72-9a7a-4fb3-9d22-7a3b17814b11',
    'f0c20a72-9a7a-4fb3-9d22-7a3b17814b12',
    'f0c20a72-9a7a-4fb3-9d22-7a3b17814b13',
    'f0c20a72-9a7a-4fb3-9d22-7a3b17814b14',
    'f0c20a72-9a7a-4fb3-9d22-7a3b17814b15',
]
PLAYLIST_ID='99b985f9-41e9-4ac0-8d7e-568a83213c54'

class LB:
    token = 'fake-token'
    def __init__(self):
        self.created=0
        self.remote=[]
        self.mismatch=False
    def playlists(self,count=20,offset=0):
        return {'playlists': [{'playlist': {'title':x}} for x in self.remote[offset:offset+count]]}
    def create_playlist(self,title,mbids):
        self.created+=1
        self.remote.append(title)
        self.tracks=list(mbids)
        return {'playlist_mbid':PLAYLIST_ID}
    def playlist(self,playlist_mbid):
        tracks = self.tracks[:-1] if self.mismatch else self.tracks
        return {'playlist': {'title':self.remote[-1], 'track':[{'identifier': f'https://musicbrainz.org/recording/{mid}'} for mid in tracks]}}

class MB:
    def verify(self,mbids):
        return {'valid':[{'mbid':x,'title':f'Track-{i}', 'artist_credit':[{'name':'Benny Goodman'}]} for i,x in enumerate(mbids)],'invalid':[]}

def setup(tmp_path,lb,approve):
    m=Memory(tmp_path/'db.sqlite')
    goal='Create a private playlist called Swing Test with 5 Benny Goodman recordings'
    t=Tools(lb,MB(),m,approve=approve,current_goal=goal)
    assert t.run('verify_recordings',{'mbids':IDS}).ok
    return m,t

def test_write_creates_private_tagged_playlist_and_checks_tracks(tmp_path):
    lb=LB(); approvals=[]
    m,t=setup(tmp_path,lb,lambda text: approvals.append(text) or True)
    try:
        result=t.run('create_playlist',{'title':'Swing Test','mbids':IDS})
        assert result.ok, result.message
        assert result.data['confirmed'] is True
        assert result.data['playlist_mbid']==PLAYLIST_ID
        assert result.data['url'].endswith(PLAYLIST_ID)
        assert result.data['title']=='[Encore] Swing Test'
        assert len(approvals)==1 and 'Benny Goodman' in approvals[0]
        assert lb.created==1
    finally: m.close()

def test_denial_makes_zero_writes(tmp_path):
    lb=LB();m,t=setup(tmp_path,lb,lambda _:False)
    try:
        result=t.run('create_playlist',{'title':'Swing Test','mbids':IDS})
        assert not result.ok and result.error_type=='policy_halt'
        assert lb.created==0
    finally: m.close()

def test_existing_playlist_blocked_before_approval(tmp_path):
    lb=LB();lb.remote=['[Encore] Swing Test']
    m,t=setup(tmp_path,lb,lambda _: (_ for _ in ()).throw(AssertionError('approval should not be called')))
    try:
        result=t.run('create_playlist',{'title':'Swing Test','mbids':IDS})
        assert not result.ok and result.error_type=='resource'
        assert lb.created==0
    finally:m.close()

def test_count_checked_before_mutation(tmp_path):
    lb=LB();m,t=setup(tmp_path,lb,lambda _:True)
    try:
        result=t.run('create_playlist',{'title':'Swing Test','mbids':IDS[:3]})
        assert not result.ok and result.error_type=='invalid'
        assert lb.created==0
    finally:m.close()

def test_readback_mismatch_not_reported_as_confirmed(tmp_path):
    lb=LB();lb.mismatch=True;m,t=setup(tmp_path,lb,lambda _:True)
    try:
        result=t.run('create_playlist',{'title':'Swing Test','mbids':IDS})
        assert result.ok and result.data['created'] is True and result.data['confirmed'] is False
        assert lb.created==1
    finally:m.close()

class FinalModel:
    def chat(self, messages, specs):
        return {'message':{'tool_calls':[{'function':{'name':'final_answer','arguments':{'status':'answered','answer':'Successfully created a playlist'}}}]}}

def test_final_answer_cannot_claim_creation_if_no_write(tmp_path):
    lb=LB();m,t=setup(tmp_path,lb,lambda _:True)
    try:
        result=run_agent(t.goal,t,FinalModel(),max_steps=3)
        assert result.status == 'stuck'
        assert lb.created == 0
    finally:m.close()

def test_playlist_router_narrows_scope():
    from encore.agent import route
    allowed={s['function']['name'] for s in route('Create a private playlist with 5 Benny Goodman recordings')}
    assert {'create_playlist','search_recordings','verify_recordings','list_playlists','final_answer'}<=allowed
    assert 'remove_playlist_tracks' not in allowed
    assert 'select_playlist_duration' not in allowed

class ScriptedModel:
    def __init__(self):
        self.n=0
    def chat(self,messages,specs):
        allowed = {s['function']['name'] for s in specs}
        steps = [
            ('search_recordings', {'query': 'artist:"Benny Goodman"'}),
            ('verify_recordings', {'mbids': IDS}),
            ('create_playlist', {'title':'Swing Test','mbids':IDS}),
            ('final_answer', {'status':'answered', 'answer':'Created [Encore] Swing Test with 5 confirmed tracks.'}),
        ]
        name,args=steps[self.n]
        self.n+=1
        assert name in allowed
        return {'message':{'tool_calls':[{'function':{'name':name,'arguments':args}}]}}

class SearchMB(MB):
    def search(self,query):
        return {'recordings':[{'id':v,'title':f'Track-{i}','artist-credit':[{'name':'Benny Goodman'}]} for i,v in enumerate(IDS)]}

def test_end_to_end_agent_create(tmp_path):
    lb=LB(); mem=Memory(tmp_path/'m.db')
    goal='Create a private playlist called Swing Test with 5 Benny Goodman recordings'
    tools=Tools(lb,SearchMB(),mem,approve=lambda _:True,current_goal=goal)
    try:
        result=run_agent(goal,tools,ScriptedModel())
        assert result.status=='answered', result.answer
        assert result.steps==4
        assert result.tool_trace[2]['tool']=='create_playlist'
        assert result.tool_trace[2]['confirmed'] is True
        assert lb.created==1
    finally:
        mem.close()
