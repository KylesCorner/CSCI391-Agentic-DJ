"""Assignment Part 5: document-only semantic memory in a separate ChromaDB collection.

Two independently indexed chunking strategies and a measured hybrid BM25/vector
retriever. Each embedding is explicit, produced by the user's local Ollama model.
No ListenBrainz personal data or tokens are indexed here.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from math import log
from pathlib import Path
import json
import re

import requests


@dataclass(frozen=True)
class Section:
    section_id: str
    title: str
    source: str
    source_url: str
    text: str


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _words(text: str) -> list[str]:
    return text.split()


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-_/][a-z0-9]+)*", text.casefold())


def load_sections(directory: Path) -> list[Section]:
    """Split trusted Markdown files into explicitly labelled sections."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Knowledge corpus not found: {directory}")
    sections = []
    for path in sorted(directory.glob("*.md")):
        section_title = None
        lines: list[str] = []
        def flush():
            if section_title is None or not lines:
                return
            body = "\n".join(lines).strip()
            if not body:
                return
            url_match = re.search(r"(?m)^Source:\s*(https?://\S+)", body)
            url = url_match.group(1) if url_match else ""
            sid = f"{path.stem}:{_slug(section_title)}"
            sections.append(Section(sid, section_title, path.name, url, body))
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                flush()
                section_title = line[3:].strip()
                lines = []
            elif section_title is not None:
                lines.append(line)
        flush()
    if not sections:
        raise ValueError("Knowledge corpus contains no Markdown ## sections")
    ids = [s.section_id for s in sections]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate knowledge section IDs")
    return sections


def _windows(words: list[str], size: int, overlap: int):
    if not (0 <= overlap < size):
        raise ValueError("Need 0 <= overlap < size")
    if not words:
        return
    start = 0
    while start < len(words):
        stop = min(start + size, len(words))
        yield start, stop
        if stop == len(words):
            break
        start += size - overlap


def chunk_documents(sections: list[Section], strategy: str) -> list[dict]:
    """Compare 96-word windows (16 overlap) with heading-aware 160-word sections.

    The implementation uses *whitespace words*, not exact model tokens. This
    approximation is documented so evaluation does not claim exact token sizes.
    """
    if strategy not in {"fixed", "section"}:
        raise ValueError("strategy must be fixed or section")
    output = []
    if strategy == "section":
        for section in sections:
            tokens = _words(f"{section.title}\n{section.text}")
            for index, (a,b) in enumerate(_windows(tokens, 160, 16)):
                output.append({
                    "id": f"section:{section.section_id}:{index}",
                    "document": " ".join(tokens[a:b]),
                    "metadata": {
                        "section_id": section.section_id,
                        "section_ids": section.section_id,
                        "section_title": section.title,
                        "source": section.source,
                        "source_url": section.source_url,
                        "strategy": "section",
                    },
                })
        return output

    grouped: dict[str, list[Section]] = {}
    for section in sections:
        grouped.setdefault(section.source, []).append(section)
    for source, items in grouped.items():
        words: list[tuple[str,str]] = []
        for section in items:
            words += [(w, section.section_id) for w in _words(f"{section.title}\n{section.text}")]
        url_by_id = {item.section_id:item.source_url for item in items}
        title_by_id = {item.section_id:item.title for item in items}
        for index, (a,b) in enumerate(_windows(words, 96, 16)):
            window = words[a:b]
            relevant = list(dict.fromkeys(sid for _,sid in window))
            output.append({
                "id": f"fixed:{source}:{index}",
                "document": " ".join(word for word,_ in window),
                "metadata": {
                    "section_id": relevant[0],
                    "section_ids": "|".join(relevant),
                    "section_title": title_by_id[relevant[0]],
                    "source": source,
                    "source_url": url_by_id[relevant[0]],
                    "strategy": "fixed",
                },
            })
    return output


def _bm25_ranking(query: str, rows: list[dict]) -> list[str]:
    tokens = _terms(query)
    if not rows or not tokens:
        return []
    counts = [Counter(_terms(row["document"])) for row in rows]
    df = Counter()
    for c in counts:
        df.update(c.keys())
    avg_len = sum(sum(c.values()) for c in counts) / max(len(rows),1)
    result = []
    for row, c in zip(rows, counts):
        score = 0.0
        length = sum(c.values())
        for term in set(tokens):
            f = c[term]
            if not f:
                continue
            idf = log(1 + (len(rows) - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (f * 2.2) / (f + 1.2 * (0.25 + 0.75 * length / max(avg_len, 1)))
        if score:
            result.append((row["id"],score))
    result.sort(key=lambda item: (-item[1],item[0]))
    return [key for key,_ in result]


class SemanticStore:
    """Public documents only. User-scoped episodic memories live in a different collection."""
    def __init__(self, path: Path, embedder, *, client=None):
        self.embedder = embedder
        if not getattr(embedder, "model", ""):
            raise ValueError("Embedding model must have an explicit name")
        if client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise RuntimeError("ChromaDB missing: uv add 'chromadb>=1,<2'") from exc
            path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(path))
        self.collections = {}
        for strategy in ("fixed", "section"):
            name = f"encore_semantic_{strategy}_v1"
            # Existing collection metadata must be checked before indexing/querying.
            try:
                collection = client.get_collection(name=name, embedding_function=None)
            except Exception as exc:
                if exc.__class__.__name__ not in {"NotFoundError", "InvalidCollectionException", "ValueError"}:
                    raise
                collection = client.get_or_create_collection(
                    name=name,
                    embedding_function=None,
                    metadata={"hnsw:space":"cosine", "embedding_model":embedder.model,
                              "chunking_strategy":strategy,"corpus_version":"1"},
                )
            meta = collection.metadata or {}
            if meta.get("embedding_model") != embedder.model:
                raise RuntimeError(
                    f"Semantic embedding model mismatch for {name}. Re-index explicitly; "
                    "do not mix embedding model versions."
                )
            if meta.get("chunking_strategy") != strategy:
                raise RuntimeError(f"Corrupt semantic collection metadata: {name}")
            self.collections[strategy] = collection

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        if hasattr(self.embedder, "embed_many"):
            return self.embedder.embed_many(texts)
        # Batch Ollama's documented /api/embed API where supported.
        if hasattr(self.embedder, "host") and hasattr(self.embedder, "model"):
            response = requests.post(
                f"{self.embedder.host}/api/embed",
                json={"model": self.embedder.model, "input": texts, "truncate": True},
                timeout=(5, 180),
            )
            response.raise_for_status()
            vectors = response.json().get("embeddings") or []
            if len(vectors) != len(texts) or any(not v for v in vectors):
                raise RuntimeError("Ollama embedding batch result was incomplete")
            return [[float(v) for v in vector] for vector in vectors]
        return [self.embedder.embed(text) for text in texts]

    def _rows(self, strategy: str) -> list[dict]:
        data = self.collections[strategy].get(include=["documents","metadatas"])
        return [{"id": i, "document": doc, "metadata": meta or {}}
                for i,doc,meta in zip(data.get("ids") or [],
                                       data.get("documents") or [],
                                       data.get("metadatas") or [])]

    def index(self, corpus: Path) -> dict:
        sections = load_sections(corpus)
        report = {"sections":len(sections),"glossary_entries":sum(s.source=="glossary.md" for s in sections),
                  "embedding_model":self.embedder.model,"strategies":{}}
        for strategy, collection in self.collections.items():
            chunks = chunk_documents(sections, strategy)
            # Compute ALL embeddings before mutating the existing collection;
            # an Ollama failure cannot replace it with a half-built index.
            vectors = []
            for i in range(0,len(chunks),32):
                vectors.extend(self._embed_many([c["document"] for c in chunks[i:i+32]]))
            old = collection.get().get("ids") or []
            if old:
                collection.delete(ids=old)
            for i in range(0,len(chunks),50):
                batch = chunks[i:i+50]
                collection.upsert(
                    ids=[x["id"] for x in batch],
                    documents=[x["document"] for x in batch],
                    metadatas=[x["metadata"] for x in batch],
                    embeddings=vectors[i:i+50],
                )
            report["strategies"][strategy] = len(chunks)
        return report

    def search(self, query: str, limit: int = 5, *, strategy: str = "section",
               mode: str = "hybrid") -> list[dict]:
        if strategy not in self.collections or mode not in {"hybrid","vector"}:
            raise ValueError("Unsupported strategy or retrieval mode")
        collection = self.collections[strategy]
        count = collection.count()
        if not query.strip() or count == 0:
            return []
        vector = self.embedder.embed(query)
        result = collection.query(
            query_embeddings=[vector], n_results=min(count,max(15,limit*5)),
            include=["documents","metadatas","distances"],
        )
        vector_ids = (result.get("ids") or [[]])[0]
        if mode == "vector":
            vector_rows = zip(vector_ids,
                              (result.get("documents") or [[]])[0],
                              (result.get("metadatas") or [[]])[0],
                              (result.get("distances") or [[]])[0])
            return [self._format_row(key,doc,meta, 1 - float(dist))
                    for key,doc,meta,dist in vector_rows][:limit]
        corpus = self._rows(strategy)
        lexical_ids = _bm25_ranking(query, corpus)
        vector_rank = {key:rank for rank,key in enumerate(vector_ids,1)}
        lexical_rank = {key:rank for rank,key in enumerate(lexical_ids,1)}
        # Reciprocal rank fusion: robust to incomparable cosine and BM25 scales.
        fused = {row["id"]: (1/(60+vector_rank[row["id"]]) if row["id"] in vector_rank else 0)
                 + (1/(60+lexical_rank[row["id"]]) if row["id"] in lexical_rank else 0)
                 for row in corpus}
        ranked = sorted(corpus,key=lambda r:(-fused[r["id"]],r["id"]))
        return [self._format_row(row["id"],row["document"],row["metadata"],fused[row["id"]])
                for row in ranked[:limit] if fused[row["id"]]>0]

    @staticmethod
    def _format_row(row_id: str, text: str, meta: dict, score: float) -> dict:
        return {
            "id":row_id,
            "source":meta.get("source", ""),
            "section_id":meta.get("section_id", ""),
            "section_ids":(meta.get("section_ids") or meta.get("section_id", "")).split("|"),
            "section_title":meta.get("section_title", ""),
            "source_url":meta.get("source_url", ""),
            "text":text,
            "score":round(float(score), 5),
        }
