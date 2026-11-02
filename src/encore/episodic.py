"""User-scoped episodic memory in ChromaDB with local Ollama embeddings.

No external credentials or cloud embeddings are used. The collection is
initialized with explicit vectors to avoid Chroma's implicit model download.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
from uuid import uuid4

import requests
import time
from jsonschema import validate, ValidationError


SUMMARY_FIELDS = (
    "decisions", "preferences", "open_threads", "key_facts", "errors", "topics"
)
SUMMARY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": list(SUMMARY_FIELDS),
    "properties": {
        key: {"type": "array", "items": {"type": "string", "maxLength": 500}, "maxItems": 12}
        for key in SUMMARY_FIELDS
    },
}
QUALITY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["consistent", "score", "issues"],
    "properties": {
        "consistent": {"type": "boolean"},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "issues": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
    },
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _prompt(filename: str, fallback: str) -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / filename
    return path.read_text(encoding="utf-8") if path.is_file() else fallback


def _text(summary: dict) -> str:
    return "\n".join(f"{key}: {'; '.join(summary.get(key, []))}" for key in SUMMARY_FIELDS)


def _strip_secrets(s: str, secrets: tuple[str, ...] = ()) -> str:
    for secret in secrets:
        if secret:
            s = s.replace(secret, "[REDACTED]")
    s = re.sub(r"(?i)(authorization\s*:\s*token\s+)\S+", r"\1[REDACTED]", s)
    return s


class OllamaEmbedder:
    def __init__(self, host: str, model: str = "nomic-embed-text"):
        self.host = host.rstrip("/")
        self.model = model

    def embed(self, text: str) -> list[float]:
        response = requests.post(
            f"{self.host}/api/embed",
            json={"model": self.model, "input": text, "truncate": True},
            timeout=(5, 90),
        )
        response.raise_for_status()
        vectors = response.json().get("embeddings") or []
        if not vectors or not vectors[0]:
            raise RuntimeError("Ollama returned no embedding vector")
        return [float(x) for x in vectors[0]]


class SessionSummarizer:
    """Two independent structured LLM calls: summary, then quality review."""

    def __init__(self, host: str, model: str, *, secrets: tuple[str, ...] = (), reporter=None):
        self.host = host.rstrip("/")
        self.model = model
        self.secrets = secrets
        self.reporter = reporter

    def _ask(self, system: str, payload: dict, schema: dict, *, phase: str = "episodic") -> dict:
        safe_payload = _strip_secrets(json.dumps(payload, ensure_ascii=False), self.secrets)
        last_error = None
        for attempt in range(2):
            call_started = time.monotonic()
            response = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system + "\nReturn ONLY schema-compliant JSON."},
                        {"role": "user", "content": safe_payload},
                    ],
                    "stream": False,
                    "think": False,
                    "format": schema,
                    "options": {"temperature": 0, "num_predict": 2048 if attempt == 0 else 3072},
                },
                timeout=(5, 180),
            )
            response.raise_for_status()
            payload_json = response.json()
            content = payload_json.get("message", {}).get("content") or ""
            try:
                value = json.loads(content)
                validate(value, schema)
                valid = True
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                valid = False
            if self.reporter is not None:
                self.reporter(
                    "llm_call", phase=getattr(self, "_phase", phase), attempt=attempt + 1,
                    model=self.model, tier="cheap", success=valid,
                    tokens_in=int(payload_json.get("prompt_eval_count") or 0),
                    tokens_out=int(payload_json.get("eval_count") or 0),
                    latency_ms=round((time.monotonic()-call_started)*1000),
                    error_category=None if valid else type(last_error).__name__,
                )
            if valid:
                return value
        raise ValueError(
            f"Ollama did not return valid structured JSON after two attempts: "
            f"{type(last_error).__name__}"
        )

    def summarize(self, goal: str, answer: str, status: str, tool_trace: list[dict]) -> tuple[dict, dict]:
        # Only verified facts are forwarded. Never include API headers, credentials,
        # or arbitrary observation_preview / model_content_preview fields.
        safe_events = [
            {key: event[key] for key in (
                "tool", "ok", "error_type", "confirmed", "playlist_mbid", "track_count_observed",
                "phase", "message"
            ) if key in event}
            for event in tool_trace
        ]
        evidence = {"goal": goal, "answer": answer, "status": status, "events": safe_events}
        self._phase = "episodic_summary"
        summary = self._ask(
            _prompt("episodic_summary_v1.md", "Summarize one session factually. Never fabricate."),
            evidence, SUMMARY_SCHEMA,
        )
        self._phase = "episodic_quality"
        review = self._ask(
            _prompt("episodic_quality_v1.md", "Review a session memory against supplied evidence."),
            {"evidence": evidence, "proposed_summary": summary}, QUALITY_SCHEMA,
        )
        # Deterministic second line of defense for false claims of successful writes.
        successful_writes = any(
            item.get("tool") in {
                "create_playlist", "add_playlist_tracks", "remove_playlist_tracks", "export_playlist"
            }
            and item.get("ok") and item.get("confirmed")
            for item in safe_events
        )
        any_write_assertion = any(re.search(
            r"\b(created|added|removed|exported|deleted)\b.{0,100}\bplaylist\b|"
            r"\bplaylist\b.{0,100}\b(created|added|removed|exported|deleted)\b",
            statement, re.IGNORECASE
        ) for statement in summary["decisions"])
        if any_write_assertion and not successful_writes:
            review = {"consistent": False, "score": min(float(review["score"]), 0.25),
                      "issues": list(review["issues"]) + ["No confirmed matching playlist write in tool trace"]}
        explicitly_stated_preference = re.search(
            r"\b(i prefer|i like|i love|i dislike|i hate|remember that i|"
            r"don't recommend|never recommend)\b", goal, re.IGNORECASE,
        )
        if summary["preferences"] and not explicitly_stated_preference:
            review = {"consistent": False, "score": min(float(review["score"]), 0.25),
                      "issues": list(review["issues"]) + ["User did not explicitly state a new preference"]}
        return summary, review


class EpisodicStore:
    """Real ChromaDB store; client/embedder injection supports isolated unit tests."""

    def __init__(
        self, path: Path, user_id: str, embedder,
        *, client=None, threshold: float = 0.93, ttl_days: int = 90,
        max_per_user: int = 50, recency_weight: float = 0.2,
    ):
        if not user_id.strip():
            raise ValueError("ENCORE_USER_ID must not be empty")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("dedup threshold must be 0..1")
        self.user_id = user_id
        self.embedder = embedder
        self.threshold = threshold
        self.ttl_days = ttl_days
        self.max_per_user = max_per_user
        self.recency_weight = recency_weight
        if client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise RuntimeError("ChromaDB is required. Run: uv add 'chromadb>=1,<2'") from exc
            path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(path))
        self.collection = client.get_or_create_collection(
            name="encore_episodic_v1",
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        # Refuse mixed embeddings, even if they share the same dimensionality.
        rows = self.collection.get(where={"user_id": user_id}, include=["metadatas"])
        for metadata in rows.get("metadatas") or []:
            if metadata and metadata.get("embedding_model") != self.embedder.model:
                raise RuntimeError(
                    "Episodic embedding model changed. Do not mix vectors. "
                    "Export/re-index or delete the old user collection before switching."
                )

    def _get(self) -> list[dict]:
        data = self.collection.get(where={"user_id": self.user_id}, include=["metadatas"])
        return [{"id": i, "meta": m or {}} for i, m in zip(
            data.get("ids") or [], data.get("metadatas") or []
        )]

    def prune(self, *, now: datetime | None = None) -> int:
        current = now or _now()
        cutoff = (current - timedelta(days=self.ttl_days)).timestamp()
        docs = self._get()
        expired = [x["id"] for x in docs if float(x["meta"].get("updated_ts", 0)) < cutoff]
        survivors = sorted((x for x in docs if x["id"] not in set(expired)),
                           key=lambda x: float(x["meta"].get("updated_ts", 0)), reverse=True)
        expired.extend(x["id"] for x in survivors[self.max_per_user:])
        if expired:
            self.collection.delete(ids=list(set(expired)))
        return len(set(expired))

    def add(self, goal: str, summary: dict, outcome: str, quality: dict, *, now: datetime | None = None) -> dict:
        validate(summary, SUMMARY_SCHEMA)
        validate(quality, QUALITY_SCHEMA)
        self.prune(now=now)
        current = now or _now()
        # Include the original goal for reliable retrieval of phrase references.
        document = f"Original request: {goal[:500]}\n{_text(summary)}"
        vector = self.embedder.embed(document)
        duplicate = None
        if self._get():
            results = self.collection.query(
                query_embeddings=[vector], n_results=5,
                where={"user_id": self.user_id}, include=["distances", "metadatas", "documents"],
            )
            ids = (results.get("ids") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            metadatas = (results.get("metadatas") or [[]])[0]
            documents = (results.get("documents") or [[]])[0]
            for i, distance, meta, original_doc in zip(ids, distances, metadatas, documents):
                if 1.0 - float(distance) < self.threshold:
                    continue
                old_quality_good = meta.get("quality_flag") == "good"
                new_quality_good = bool(quality["consistent"]) and float(quality["score"]) >= 0.8
                if old_quality_good != new_quality_good:
                    continue  # Never merge a suspect episode into a trusted one.
                old_summary = json.loads(meta["summary_json"])
                old_decisions = set(old_summary.get("decisions", []))
                new_decisions = set(summary.get("decisions", []))
                if old_decisions and new_decisions and old_decisions.isdisjoint(new_decisions):
                    continue  # Similar topics do not imply the same completed action.
                duplicate = (i, meta, original_doc)
                break
        record_id = duplicate[0] if duplicate else str(uuid4())
        if duplicate:
            old_meta = duplicate[1]
            old_summary = json.loads(old_meta["summary_json"])
            # Merge distinct facts instead of discarding earlier memory.
            summary = {
                key: list(dict.fromkeys(old_summary[key] + summary[key]))[-12:]
                for key in SUMMARY_FIELDS
            }
            goal = f"{old_meta.get('original_goal', '')} | {goal}"[:500]
            document = f"Original requests: {goal}\n{_text(summary)}"
            vector = self.embedder.embed(document)
            created_ts = float(old_meta.get("created_ts", current.timestamp()))
            seen_count = int(old_meta.get("seen_count", 1)) + 1
        else:
            created_ts, seen_count = current.timestamp(), 1
        good = bool(quality["consistent"]) and float(quality["score"]) >= 0.8
        metadata = {
            "user_id": self.user_id,
            "original_goal": goal[:500],
            "created_ts": created_ts,
            "updated_ts": current.timestamp(),
            "updated_at": current.isoformat(),
            "outcome": outcome,
            "topics_json": json.dumps(summary["topics"], ensure_ascii=False),
            "summary_json": json.dumps(summary, ensure_ascii=False),
            "quality_score": float(quality["score"]),
            "quality_flag": "good" if good else "verify_before_using",
            "quality_issues_json": json.dumps(quality["issues"], ensure_ascii=False),
            "embedding_model": self.embedder.model,
            "seen_count": seen_count,
        }
        self.collection.upsert(ids=[record_id], documents=[document], metadatas=[metadata], embeddings=[vector])
        self.prune(now=now)
        return {"id": record_id, "merged": bool(duplicate), "quality_flag": metadata["quality_flag"]}

    def search(self, query: str, *, limit: int = 3, now: datetime | None = None) -> list[dict]:
        if not query.strip() or not self._get():
            return []
        current = now or _now()
        cutoff = (current - timedelta(days=self.ttl_days)).timestamp()
        response = self.collection.query(
            query_embeddings=[self.embedder.embed(query)],
            n_results=min(self.max_per_user, max(limit * 7, 10)),
            where={"user_id": self.user_id},
            include=["distances", "metadatas", "documents"],
        )
        ids = (response.get("ids") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        ranked = []
        for uid, distance, meta in zip(ids, distances, metadatas):
            updated = float(meta.get("updated_ts", 0))
            if updated < cutoff or meta.get("embedding_model") != self.embedder.model:
                continue
            similarity = max(0.0, min(1.0, 1.0 - float(distance)))
            age_days = max(0.0, (current.timestamp() - updated) / 86400)
            recency = math.exp(-age_days / 30.0)
            score = (1.0 - self.recency_weight) * similarity + self.recency_weight * recency
            if meta.get("quality_flag") != "good":
                score *= 0.65
            ranked.append({
                "id": uid, "created_at": meta.get("updated_at"),
                "goal": meta.get("original_goal"), "summary": json.loads(meta["summary_json"]),
                "outcome": meta.get("outcome"),
                "topics": json.loads(meta.get("topics_json", "[]")),
                "quality_flag": meta.get("quality_flag"),
                "score": round(score, 4), "similarity": round(similarity, 4),
                "stale_warning": "Historical context may be outdated; verify conflicts with current request.",
            })
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:limit]

    def delete_user(self) -> int:
        ids = [item["id"] for item in self._get()]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)
