from pathlib import Path
from encore.memory import Memory, chunks, fts_query
from encore.retrieval_eval import evaluate


def test_memory_search_retention_and_deletion(tmp_path):
    mem = Memory(tmp_path / "encore.db")
    try:
        mem.add_session("Make me a focus playlist", "instrumental ambient", "answered")
        results = mem.search_sessions("focus playlist")
        assert results and results[0]["outcome"] == "answered"
        mem.set_preference("vocals", "avoid vocals", "I prefer no vocals")
        assert len(mem.preferences()) == 1
        mem.ingest("api-doc", "Use recording MBIDs to create ListenBrainz playlists")
        assert "recording" in mem.search_knowledge("recording playlist")[0]["text"]
        assert mem.forget_preference("vocals") == 1
        mem.delete_personal()
        assert mem.preferences() == []
        assert mem.search_sessions("focus") == []
        assert mem.search_knowledge("recording") == []
    finally:
        mem.close()


def test_chunks_and_quoted_query():
    assert chunks("a b c d e f", size=4, overlap=2) == ["a b c d", "c d e f", "e f"]
    assert '"playlist"' in fts_query('playlist" OR dangerous*')
