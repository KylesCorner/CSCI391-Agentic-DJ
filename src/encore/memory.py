"""Distinct episodic/semantic/preference stores with retention and deletion."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json
import re
import sqlite3
import uuid
import shutil


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def chunks(text: str, size: int = 110, overlap: int = 20) -> list[str]:
    if size <= overlap or overlap < 0:
        raise ValueError("size must be > overlap >= 0")
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step) if words[i:i + size]]


def fts_query(text: str) -> str:
    # Quoted terms avoid exposing raw FTS syntax to untrusted user input.
    terms = re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)[:12]
    return " OR ".join(f'"{term}"' for term in terms if len(term) >= 3)


class Memory:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = db_path
        self.db = sqlite3.connect(str(db_path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS sessions (
          id TEXT PRIMARY KEY, created_at TEXT NOT NULL, goal TEXT NOT NULL,
          summary TEXT NOT NULL, outcome TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
          goal, summary, content='sessions', content_rowid='rowid'
        );
        CREATE TRIGGER IF NOT EXISTS session_ai AFTER INSERT ON sessions BEGIN
          INSERT INTO sessions_fts(rowid,goal,summary) VALUES(new.rowid,new.goal,new.summary);
        END;
        CREATE TRIGGER IF NOT EXISTS session_ad AFTER DELETE ON sessions BEGIN
          INSERT INTO sessions_fts(sessions_fts,rowid,goal,summary)
          VALUES('delete',old.rowid,old.goal,old.summary);
        END;
        CREATE TABLE IF NOT EXISTS preferences (
          key TEXT PRIMARY KEY, value TEXT NOT NULL, evidence TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS knowledge (
          id TEXT PRIMARY KEY, source TEXT NOT NULL, text TEXT NOT NULL,
          source_hash TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
          text, content='knowledge', content_rowid='rowid'
        );
        CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge BEGIN
          INSERT INTO knowledge_fts(rowid,text) VALUES(new.rowid,new.text);
        END;
        CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge BEGIN
          INSERT INTO knowledge_fts(knowledge_fts,rowid,text) VALUES('delete',old.rowid,old.text);
        END;
        CREATE TABLE IF NOT EXISTS writes (
          op_id TEXT PRIMARY KEY, action TEXT NOT NULL, state TEXT NOT NULL,
          response_json TEXT, created_at TEXT NOT NULL
        );
        """)
        self.db.commit()

    def enable_episodic(self, store):
        """Opt in to Chroma-backed episodic retrieval; leave other SQLite stores intact."""
        self.episodic = store

    def add_structured_session(self, goal: str, summary: dict, outcome: str, quality: dict) -> dict:
        if not getattr(self, "episodic", None):
            raise RuntimeError("Episodic Chroma store was not initialized")
        return self.episodic.add(goal, summary, outcome, quality)

    def _working_archive_dir(self) -> Path:
        user = getattr(getattr(self, "episodic", None), "user_id", "local")
        # User-scoped directory, without embedding the account name in a path.
        return self.path.parent / "working_archives" / hashlib.sha256(user.encode()).hexdigest()[:20]

    def archive_working_messages(self, session_id: str, messages: list[dict]) -> None:
        directory = self._working_archive_dir()
        directory.mkdir(parents=True, exist_ok=True)
        # IDs are generated locally by the tool executor, not by the LLM.
        filename = directory / (str(uuid.UUID(session_id)) + ".jsonl")
        with filename.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": utc_now(), "messages": messages},
                               ensure_ascii=False, default=str) + "\n")

    def close(self):
        self.db.close()

    def prune(self, days=90) -> int:
        older = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cur = self.db.execute("DELETE FROM sessions WHERE created_at < ?", (older,))
        self.db.commit()
        return cur.rowcount

    def add_session(self, goal: str, summary: str, outcome: str):
        self.db.execute("INSERT INTO sessions VALUES(?,?,?,?,?)",
                        (str(uuid.uuid4()), utc_now(), goal[:2000], summary[:4000], outcome))
        self.db.commit()

    def search_sessions(self, query: str, limit=5) -> list[dict]:
        if getattr(self, "episodic", None) is not None:
            return self.episodic.search(query, limit=limit)
        q = fts_query(query)
        if not q:
            return []
        rows = self.db.execute("""
          SELECT s.id,s.created_at,s.goal,s.summary,s.outcome
          FROM sessions_fts f JOIN sessions s ON s.rowid=f.rowid
          WHERE sessions_fts MATCH ? ORDER BY bm25(sessions_fts) LIMIT ?
          """, (q, limit)).fetchall()
        return [dict(x) for x in rows]

    def set_preference(self, key: str, value: str, evidence: str):
        self.db.execute("""INSERT INTO preferences VALUES(?,?,?,?)
         ON CONFLICT(key) DO UPDATE SET value=excluded.value,evidence=excluded.evidence,
         updated_at=excluded.updated_at""", (key[:80], value[:400], evidence[:1000], utc_now()))
        self.db.commit()

    def preferences(self) -> list[dict]:
        return [dict(x) for x in self.db.execute("SELECT * FROM preferences ORDER BY key")]

    def forget_preference(self, key: str) -> int:
        cur = self.db.execute("DELETE FROM preferences WHERE key=?", (key,))
        self.db.commit()
        return cur.rowcount

    def ingest(self, source: str, text: str, size=110, overlap=20) -> int:
        digest = hashlib.sha256(text.encode()).hexdigest()
        # Replace all chunks for this source; old documentation is never left mixed in.
        self.db.execute("DELETE FROM knowledge WHERE source=?", (source,))
        pieces = chunks(text, size=size, overlap=overlap)
        for index, piece in enumerate(pieces):
            self.db.execute("INSERT INTO knowledge VALUES(?,?,?,?,?)",
                            (f"{source}:{index}", source, piece, digest, utc_now()))
        self.db.commit()
        return len(pieces)

    def search_knowledge(self, query: str, limit=5) -> list[dict]:
        q = fts_query(query)
        if not q:
            return []
        rows = self.db.execute("""
          SELECT k.source,k.text,k.source_hash,k.updated_at
          FROM knowledge_fts f JOIN knowledge k ON k.rowid=f.rowid
          WHERE knowledge_fts MATCH ? ORDER BY bm25(knowledge_fts) LIMIT ?
          """, (q, limit)).fetchall()
        return [dict(x) for x in rows]

    def reserve_write(self, op_id: str, action: str) -> tuple[bool, dict | None]:
        """Persistent no-retry ledger. Uncertain writes must be reconciled manually."""
        row = self.db.execute("SELECT state,response_json FROM writes WHERE op_id=?", (op_id,)).fetchone()
        if row:
            response = json.loads(row["response_json"]) if row["response_json"] else None
            return False, {"state": row["state"], "response": response}
        self.db.execute("INSERT INTO writes VALUES(?,?,?,?,?)", (op_id, action, "started", None, utc_now()))
        self.db.commit()
        return True, None

    def finish_write(self, op_id: str, response: dict):
        self.db.execute("UPDATE writes SET state='done',response_json=? WHERE op_id=?",
                        (json.dumps(response)[:10000], op_id))
        self.db.commit()

    def mark_uncertain(self, op_id: str):
        self.db.execute("UPDATE writes SET state='uncertain' WHERE op_id=?", (op_id,))
        self.db.commit()

    def delete_personal(self):
        """Delete local SQLite data and the current user's Chroma episodes."""
        deleted = {}
        if getattr(self, "episodic", None) is not None:
            deleted["episodic_chroma"] = self.episodic.delete_user()
        for table in ("sessions", "preferences", "knowledge", "writes"):
            deleted[table] = self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            self.db.execute(f"DELETE FROM {table}")
        self.db.commit()
        archive_dir = self._working_archive_dir()
        deleted["working_archives"] = len(list(archive_dir.glob("*.jsonl"))) if archive_dir.exists() else 0
        if archive_dir.exists():
            shutil.rmtree(archive_dir)
        return deleted
