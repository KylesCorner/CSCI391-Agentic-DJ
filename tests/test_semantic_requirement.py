"""Offline unit tests for semantic-memory rubric; no Chroma/Ollama network needed."""
from pathlib import Path
import json
import re

from encore.semantic import load_sections, chunk_documents, SemanticStore
from encore.retrieval_eval import evaluate
from encore.memory import Memory
from encore.tools import INDEX, Tools
from encore.agent import route

ROOT = Path(__file__).resolve().parents[1]


class LocalEmbedder:
    model = "fake-embedding-test-v1"
    def embed(self, text):
        words = re.findall(r"[a-z0-9]+",text.lower())
        return [float(sum(w.startswith('m') for w in words)),
                float(sum(w.startswith('l') for w in words)),
                float(len(words) or 1)]
    def embed_many(self, texts):
        return [self.embed(t) for t in texts]


class LocalCollection:
    def __init__(self, metadata):
        self.metadata=metadata
        self.rows={}
    def get(self,include=None):
        return {'ids':list(self.rows),'documents':[r[0] for r in self.rows.values()],
                'metadatas':[r[1] for r in self.rows.values()]}
    def count(self):return len(self.rows)
    def delete(self, ids):
        for i in ids: self.rows.pop(i,None)
    def upsert(self,ids,documents,metadatas,embeddings):
        self.rows.update((i,(d,m,e)) for i,d,m,e in zip(ids,documents,metadatas,embeddings))
    def query(self,query_embeddings,n_results,include=None):
        q=query_embeddings[0]
        def distance(v):
            return sum((x-y)**2 for x,y in zip(q,v))**0.5 / 1000
        ranked=sorted(self.rows.items(),key=lambda r:distance(r[1][2]))[:n_results]
        return {'ids':[[i for i,_ in ranked]],
                'documents':[[r[0] for _,r in ranked]],
                'metadatas':[[r[1] for _,r in ranked]],
                'distances':[[distance(r[2]) for _,r in ranked]]}


class LocalClient:
    def __init__(self):self.db={}
    def get_collection(self,name,embedding_function=None):
        if name not in self.db:
            raise ValueError('not found')
        return self.db[name]
    def get_or_create_collection(self,name,embedding_function=None,metadata=None):
        return self.db.setdefault(name,LocalCollection(metadata))


def make_store(tmp_path):
    return SemanticStore(tmp_path,LocalEmbedder(),client=LocalClient())


def test_corpus_includes_required_35_glossary_entries_and_sources():
    sections=load_sections(ROOT/'knowledge')
    assert len([s for s in sections if s.source=='glossary.md']) >=30
    assert len([s for s in sections if s.source.startswith('listenbrainz_')]) >=10
    assert len([s for s in sections if s.source=='musicbrainz_api.md']) >=6


def test_two_distinct_chunkers_respect_section_labels():
    sections=load_sections(ROOT/'knowledge')
    fixed=chunk_documents(sections,'fixed')
    section=chunk_documents(sections,'section')
    assert fixed and section and len(fixed)!=len(section)
    assert all(c['metadata']['section_id'] for c in fixed+section)
    assert all(len(c['document'].split())<=160 for c in section)
    assert all(len(c['document'].split())<=96 for c in fixed)


def test_index_search_and_rebuild(tmp_path):
    s=make_store(tmp_path)
    result=s.index(ROOT/'knowledge')
    assert result['sections']>=50 and result['glossary_entries']>=30
    assert result['strategies']['fixed']>0
    assert result['strategies']['section']>0
    assert len(s.search('What is gypsy jazz?',limit=3,mode='hybrid'))==3
    assert s.search('recording MBID',limit=3)[0]['source']
    assert s.index(ROOT/'knowledge')['strategies']==result['strategies']


def test_retrieval_benchmark_has_gt_20_ground_truthed_queries():
    examples=json.loads((ROOT/'evals'/'retrieval_questions.json').read_text())
    valid={section.section_id for section in load_sections(ROOT/'knowledge')}
    assert len(examples)>=20
    assert all(q['expected_section'] in valid for q in examples)
    assert len({q['id'] for q in examples})==len(examples)


def test_retrieval_three_way_metrics(tmp_path):
    s=make_store(tmp_path)
    s.index(ROOT/'knowledge')
    mem=Memory(tmp_path/'encore.db')
    try:
        mem.enable_semantic(s)
        result=evaluate(mem,ROOT/'evals'/'retrieval_questions.json',k=3)
        assert result['question_count']>=20
        assert len(result['strategies'])==3
        assert all(0<=r['recall_at_k']<=1 for r in result['strategies'].values())
        assert all(r['n']==result['question_count'] for r in result['strategies'].values())
    finally: mem.close()


def test_required_tool_is_routed_and_calls_semantic_store(tmp_path):
    s=make_store(tmp_path)
    s.index(ROOT/'knowledge')
    mem=Memory(tmp_path/'x.db')
    try:
        mem.enable_semantic(s)
        tool=object.__new__(Tools)
        tool.mem=mem
        answer=tool.tool_search_music_knowledge('MusicBrainz User-Agent')
        assert answer and 'source_url' in answer[0]
        assert 'search_music_knowledge' in INDEX
        assert 'search_music_knowledge' in {sp['function']['name'] for sp in route('How does ListenBrainz export a playlist?')}
    finally: mem.close()


def test_embedding_change_must_not_silently_reuse_vectors(tmp_path):
    c=LocalClient()
    SemanticStore(tmp_path,LocalEmbedder(),client=c)
    class Changed(LocalEmbedder):
        model='changed-model'
    import pytest
    with pytest.raises(RuntimeError,match='mismatch'):
        SemanticStore(tmp_path,Changed(),client=c)
