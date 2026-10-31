from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    username: str
    token: str
    mb_contact: str
    model: str
    ollama_host: str
    db: Path
    timezone: str


def get_config() -> Config:
    load_dotenv()
    return Config(
        username=os.getenv("LISTENBRAINZ_USER", "").strip(),
        token=os.getenv("LISTENBRAINZ_TOKEN", "").strip(),
        mb_contact=os.getenv("MB_CONTACT", "").strip(),
        model=os.getenv("OLLAMA_MODEL", "qwen3-coder:30b").strip(),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
        db=Path(os.getenv("ENCORE_DB", "./data/encore.sqlite3")),
        timezone=os.getenv("ENCORE_TIMEZONE", "UTC").strip(),
    )
