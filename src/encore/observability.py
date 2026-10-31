"""Append-only, user-scoped JSONL traces for Encore sessions.

Trace files never contain authorization headers or access tokens. Raw model
messages/tool results are intentionally omitted; only bounded event metadata
and sanitized arguments are recorded.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

SENSITIVE_KEYS = frozenset({
    "token", "authorization", "access_token", "refresh_token", "password",
    "api_key", "secret", "headers", "cookie", "set-cookie", "credentials",
})


def safe_user_dir(root: str | Path, user_id: str) -> Path:
    """Deterministic opaque directory name, without the user's public username."""
    return Path(root) / sha256(user_id.encode("utf-8")).hexdigest()[:24]


def sanitize(value: Any, secrets: tuple[str, ...] = (), depth: int = 0) -> Any:
    """Allow bounded JSON primitives with conservative secret redaction."""
    if depth > 8:
        return "[truncated]"
    if isinstance(value, dict):
        return {
            str(k)[:90]: ("[REDACTED]" if str(k).casefold().replace("-", "_") in SENSITIVE_KEYS
                          else sanitize(v, secrets, depth + 1))
            for k, v in list(value.items())[:50]
        }
    if isinstance(value, (list, tuple)):
        return [sanitize(v, secrets, depth + 1) for v in value[:80]]
    if isinstance(value, str):
        value = value[:4000]
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"(?i)(authorization\s*[:=]\s*token\s+)\S+", r"\1[REDACTED]", value)
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(type(value).__name__)


class SessionTrace:
    """One durable trace file per session; write-through for crash resilience."""
    def __init__(self, root: str | Path, user_id: str, session_id: str,
                 *, secrets: tuple[str, ...] = ()):
        if not user_id or not session_id:
            raise ValueError("Trace user_id and session_id are required")
        self.session_id = session_id
        self.secrets = secrets
        directory = safe_user_dir(root, user_id)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = directory / f"session-{session_id}.jsonl"
        self._file = self.path.open("a", encoding="utf-8")
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def emit(self, event: str, **fields: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "event": event,
            **fields,
        }
        self._file.write(json.dumps(sanitize(record, self.secrets), ensure_ascii=False, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
