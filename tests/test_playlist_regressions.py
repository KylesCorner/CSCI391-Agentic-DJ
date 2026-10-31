"""Regression tests matching the user's Oct 31 playlist creation failure."""
from encore.agent import run_agent
from encore.memory import Memory
from encore.tools import Tools
from test_playlist_creation import LB, IDS

GOAL = (
    "Create a private playlist called Swing Test with 5 distinct Benny Goodman "
    "recordings. Verify the recordings before writing. Do not export."
)

class Catalog:
    def __init__(self):
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        assert query == 'artist:"Benny Goodman"'
        # Multiple versions of the same track, and unrelated results.
        rows = [
            {"id": IDS[0], "title": "Sing, Sing, Sing", "artist-credit": [{"name": "Benny Goodman"}]},
            {"id": "462dc36e-318e-4f87-9131-693f7aa87617", "title": "Sing, Sing, Sing", "artist-credit": [{"name": "Benny Goodman"}]},
            {"id": "229e2bb4-c9b2-4162-860c-ff90137bbfcc", "title": "Goodman", "artist-credit": [{"name": "Max B"}]},
        ]
        rows += [
            {"id": mbid, "title": title, "artist-credit": [{"name": "Benny Goodman"}]}
            for mbid, title in zip(IDS[1:], [
                "Don't Be That Way", "Stompin' at the Savoy",
                "King Porter Stomp", "Air Mail Special",
            ])
        ]
        return {"recordings": rows}

    def verify(self, mbids):
        return {"valid": [{"mbid": mbid, "title": "Title", "artist_credit": []} for mbid in mbids], "invalid": []}

class ScriptedModel:
    def __init__(self): self.n = 0
    def chat(self, messages, specs):
        sequence = [
            ("search_recordings", {"query": "Benny Goodman"}),
            ("verify_recordings", {"mbids": IDS}),
            ("create_playlist", {"mbids": IDS}), # reproduce missing required title
            ("create_playlist", {"title": "Swing Test", "mbids": IDS}),
            ("final_answer", {"status": "answered", "answer": "Created Swing Test with 5 verified tracks."}),
        ]
        name, arguments = sequence[self.n]
        self.n += 1
        return {"message": {"tool_calls": [{"function": {"name": name, "arguments": arguments}}]}}


def test_artist_target_search_rewrites_bad_model_query(tmp_path):
    mem = Memory(tmp_path / "test.db")
    try:
        catalog = Catalog()
        tools = Tools(LB(), catalog, mem, approve=lambda _: True, current_goal=GOAL)
        result = tools.run("search_recordings", {"query": "Benny Goodman"})
        assert result.ok, result.message
        assert len(result.data) == 5
        assert all("Benny Goodman" in row["artist"] for row in result.data)
        assert len({row["title"].casefold() for row in result.data}) == 5
        assert catalog.queries == ['artist:"Benny Goodman"']
    finally: mem.close()


def test_reject_duplicate_titles_or_wrong_artist_before_approval(tmp_path):
    mem = Memory(tmp_path / "test.db")
    try:
        lb, catalog = LB(), Catalog()
        approvals = []
        tools = Tools(lb, catalog, mem, approve=lambda desc: approvals.append(desc) or True, current_goal=GOAL)
        assert tools.run("search_recordings", {"query": "Benny Goodman"}).ok
        assert tools.run("verify_recordings", {"mbids": IDS}).ok
        tools.searched_recordings[IDS[1]] = {"artist": "Max B", "title": "Don't Be That Way"}
        assert tools.run("create_playlist", {"title":"Swing Test", "mbids": IDS}).error_type == "invalid"
        assert not approvals and lb.created == 0
        tools.searched_recordings[IDS[1]] = {"artist": "Benny Goodman", "title": "Sing, Sing, Sing"}
        assert tools.run("create_playlist", {"title":"Swing Test", "mbids": IDS}).error_type == "invalid"
        assert not approvals and lb.created == 0
    finally: mem.close()


def test_agent_recovers_missing_playlist_title_without_fabrication(tmp_path):
    mem = Memory(tmp_path / "test.db")
    try:
        lb, catalog = LB(), Catalog()
        approvals = []
        tools = Tools(lb, catalog, mem, approve=lambda desc: approvals.append(desc) or True, current_goal=GOAL)
        output = run_agent(GOAL, tools, ScriptedModel())
        assert output.status == "answered", output.answer
        assert [row["tool"] for row in output.tool_trace] == [
            "search_recordings", "verify_recordings", "create_playlist", "create_playlist", "final_answer"
        ]
        assert output.tool_trace[2]["error_type"] == "invalid"
        assert output.tool_trace[3]["confirmed"] is True
        assert len(approvals) == 1 and lb.created == 1
    finally: mem.close()
