"""One validated, model-interpreted representation of a natural-language request.

No free-text lexical matching is used for authorizing changes to a remote account.
An invalid, ambiguous, or unavailable interpretation is read-only.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from jsonschema import validate

ACTIONS = (
    "read", "recommend", "create_playlist", "add_playlist_tracks",
    "remove_playlist_tracks", "delete_playlist", "export_playlist",
    "set_feedback", "remember_preference", "forget_preference",
)
WRITE_TOOLS = {
    "create_playlist": frozenset({"create_playlist"}),
    "add_playlist_tracks": frozenset({"add_playlist_tracks", "add_playlist_items"}),
    "remove_playlist_tracks": frozenset({"remove_playlist_tracks", "remove_playlist_items"}),
    "delete_playlist": frozenset({"delete_playlist"}),
    "export_playlist": frozenset({"export_playlist"}),
    "set_feedback": frozenset({"set_feedback"}),
    "remember_preference": frozenset({"remember_preference", "note_preference"}),
    "forget_preference": frozenset({"forget_preference"}),
}
ALL_DOMAINS = ("listening", "catalog", "discovery", "library", "memory")

# Keep the schema small enough for local Ollama models to produce reliably.
SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "domains": {"type": "array", "items": {"type": "string", "enum": list(ALL_DOMAINS)},
                    "uniqueItems": True},
        "artist": {"type": ["string", "null"]},
        "track_title": {"type": ["string", "null"]},
        "instrument": {"type": ["string", "null"]},
        "genre": {"type": ["string", "null"]},
        "playlist_name": {"type": ["string", "null"]},
        "track_count": {"type": ["integer", "null"], "minimum": 1, "maximum": 100},
        "duration_minutes": {"type": ["integer", "null"], "minimum": 1, "maximum": 480},
        "distinct": {"type": "boolean"},
        "position": {"type": "string", "enum": ["none", "first", "last", "index"]},
        "position_index": {"type": ["integer", "null"], "minimum": 0},
        "scope_all": {"type": "boolean"},
        "feedback_score": {"type": ["integer", "null"], "enum": [1, 0, -1, None]},
        "needs_preferences": {"type": "boolean"},
        "explicit_preference": {"type": "boolean"},
        "instrumental_only": {"type": "boolean"},
        "explicit": {"type": "boolean"},
    },
    "required": ["action", "domains", "artist", "track_title", "playlist_name",
                 "track_count", "duration_minutes", "distinct", "position", "position_index",
                 "scope_all", "feedback_score", "needs_preferences", "explicit_preference",
                 "instrumental_only", "explicit"],
}


# Controlled vocabulary is for category disambiguation, not for extracting user
# intent or approving account changes. No pattern matching or regex engines.
INSTRUMENT_NAMES = {
    'clarinet': 'clarinet', 'clarinets': 'clarinet', 'bass clarinet': 'bass clarinet',
    'saxophone': 'saxophone', 'sax': 'saxophone', 'alto sax': 'alto saxophone',
    'tenor sax': 'tenor saxophone', 'trumpet': 'trumpet', 'trombone': 'trombone',
    'piano': 'piano', 'guitar': 'guitar', 'violin': 'violin', 'flute': 'flute',
    'drums': 'drums', 'drum': 'drums', 'bass': 'bass', 'double bass': 'double bass',
    'accordion': 'accordion', 'banjo': 'banjo', 'vibraphone': 'vibraphone',
}
GENRE_NAMES = {
    'swing': 'swing', 'swing jazz': 'swing', 'jazz swing': 'swing',
    'big band swing': 'swing', 'big band jazz': 'big band',
    'jazz': 'jazz', 'gypsy jazz': 'gypsy jazz', 'bebop': 'bebop',
    'dixieland': 'dixieland', 'traditional jazz': 'traditional jazz',
    'cool jazz': 'cool jazz', 'blues': 'blues', 'bossa nova': 'bossa nova',
    'ragtime': 'ragtime', 'fusion jazz': 'jazz fusion',
}


def _descriptor(value: str | None) -> str:
    return ' '.join(''.join(ch if ch.isalnum() else ' ' for ch in
                            (value or '').casefold()).split())


@dataclass(frozen=True)
class Intent:
    action: str = "read"
    domains: tuple[str, ...] = ALL_DOMAINS
    artist: str | None = None
    track_title: str | None = None
    instrument: str | None = None
    genre: str | None = None
    playlist_name: str | None = None
    track_count: int | None = None
    duration_minutes: int | None = None
    distinct: bool = False
    position: str = "none"
    position_index: int | None = None
    scope_all: bool = False
    feedback_score: int | None = None
    needs_preferences: bool = False
    explicit_preference: bool = False
    instrumental_only: bool = False
    explicit: bool = False

    @classmethod
    def from_dict(cls, value: dict) -> "Intent":
        validate(value, SCHEMA)
        payload = {**value, 'domains': tuple(value['domains'])}
        # Small models sometimes classify an instrument as an artist and a
        # genre as a track title. Do not allow that category error to become a
        # MusicBrainz artist:/recording: query. Keep explicit artist names.
        if payload['action'] in {'read', 'recommend'}:
            artist_label = _descriptor(payload.get('artist'))
            title_label = _descriptor(payload.get('track_title'))
            if artist_label in INSTRUMENT_NAMES:
                payload['instrument'] = payload.get('instrument') or INSTRUMENT_NAMES[artist_label]
                payload['artist'] = None
            elif artist_label in GENRE_NAMES:
                payload['genre'] = payload.get('genre') or artist_label
                payload['artist'] = None
            if title_label in GENRE_NAMES:
                payload['genre'] = payload.get('genre') or title_label
                payload['track_title'] = None
            elif title_label in INSTRUMENT_NAMES:
                payload['instrument'] = payload.get('instrument') or INSTRUMENT_NAMES[title_label]
                payload['track_title'] = None
            if payload.get('instrument') and payload.get('genre') and payload.get('track_count'):
                payload['action'] = 'recommend'
        result = cls(**payload)
        if result.action in WRITE_TOOLS and not result.explicit:
            raise ValueError("Write request is not unambiguously explicit")
        if result.action in {"create_playlist", "add_playlist_tracks", "remove_playlist_tracks",
                             "delete_playlist", "export_playlist"} and not result.playlist_name:
            # A new playlist may have an automatically generated title, but must
            # never invent the identity of an existing playlist to edit/delete.
            if result.action != "create_playlist":
                raise ValueError("Existing playlist must be named explicitly")
        if result.action == "remove_playlist_tracks" and result.position == "none":
            raise ValueError("Track removal requires a specified position")
        if result.action == "set_feedback" and result.feedback_score is None:
            raise ValueError("Feedback change requires an explicit score")
        return result

    @property
    def permitted_writes(self) -> frozenset[str]:
        return WRITE_TOOLS.get(self.action, frozenset()) if self.explicit else frozenset()

    @property
    def allowed_domains(self) -> tuple[str, ...]:
        selected = set(self.domains)
        if self.action in WRITE_TOOLS:
            selected.add("library")
        if self.action in {"create_playlist", "add_playlist_tracks", "set_feedback"}:
            selected.add("catalog")
        if self.needs_preferences:
            selected.add("memory")
        return tuple(d for d in ALL_DOMAINS if d in selected) or ALL_DOMAINS


def interpret(goal: str, model: Any) -> Intent:
    """Interpret the request once; fail closed to read-only on any parse failure."""
    if not hasattr(model, "extract_intent"):
        return Intent()
    try:
        raw = model.extract_intent(goal)
        if isinstance(raw, str):
            raw = json.loads(raw)
        return Intent.from_dict(raw)
    except Exception:
        # Schema rejection, model downtime, and malformed JSON all fail closed.
        return Intent()
