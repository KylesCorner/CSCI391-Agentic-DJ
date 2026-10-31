"""Strictly validated, enum-based domain router for the Encore ReAct agent.

Ollama uses the JSON-schema `format` parameter for constrained routing output.
The API does not provide a portable native tool_choice equivalent; routing is
validated here, and the existing intent-specific safety gate remains in place.
"""
from __future__ import annotations

from jsonschema import validate, ValidationError

DOMAINS = ("listening", "catalog", "discovery", "library", "memory", "none")
DOMAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "domains": {
            "type": "array", "items": {"type": "string", "enum": list(DOMAINS)},
            "minItems": 1, "maxItems": len(DOMAINS), "uniqueItems": True,
        },
    },
    "required": ["domains"], "additionalProperties": False,
}

# The mapping is exhaustive for the current toolset, including compatibility aliases.
TOOL_DOMAIN = {
    "get_recent_listens": "listening", "get_listens_since": "listening",
    "get_listen_count": "listening", "get_top_artists": "listening",
    "get_top_recordings": "listening", "get_playing_now": "listening",
    "get_feedback": "listening", "get_fresh_releases": "listening",
    "set_feedback": "library",
    "search_recordings": "catalog", "search_artists": "catalog",
    "lookup_recording": "catalog", "lookup_artist": "catalog",
    "get_top_recordings_for_artist": "catalog", "verify_recordings": "catalog",
    "lb_radio": "discovery", "get_recommendations": "discovery",
    "select_playlist_duration": "discovery", "filter_recent_candidates": "discovery",
    "find_playlists": "library", "list_playlists": "library",
    "get_playlist": "library", "create_playlist": "library",
    "add_playlist_tracks": "library", "add_playlist_items": "library",
    "remove_playlist_tracks": "library", "remove_playlist_items": "library",
    "export_playlist": "library", "delete_playlist": "library",
    "search_past_sessions": "memory", "search_knowledge": "memory",
    "search_music_knowledge": "memory", "recall_preferences": "memory",
    "remember_preference": "memory", "note_preference": "memory",
    "forget_preference": "memory", "final_answer": "none",
}


def validate_domains(data: dict) -> tuple[str, ...]:
    validate(instance=data, schema=DOMAIN_SCHEMA)
    domains = tuple(data["domains"])
    # "none" is for answering without domain tools. Never mix it with others.
    if "none" in domains and len(domains) > 1:
        raise ValidationError("none cannot be combined with other domains")
    return domains


def tools_for_domains(specs: list[dict], selected: tuple[str, ...]) -> list[dict]:
    """Filter already-authorized tools; never expand write permissions."""
    if selected == ("none",):
        return [s for s in specs if s["function"]["name"] == "final_answer"]
    chosen = set(selected)
    return [s for s in specs if s["function"]["name"] == "final_answer"
            or TOOL_DOMAIN.get(s["function"]["name"]) in chosen]
