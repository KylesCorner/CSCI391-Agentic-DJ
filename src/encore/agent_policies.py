"""Tool routing, observation compaction, discovery, and answer validation.

Read-only policies moved verbatim from agent.py. The original import paths
remain usable for callers and tests.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import unicodedata

from .tool_specs import SPECS, WRITE_ACTIONS
from .request_intent import Intent, WRITE_TOOLS

# ============================================================
# TOOL ROUTER
# ============================================================

def route(intent: Intent) -> list[dict]:
    """Offer tools by validated intent and let the model choose read actions.

    No natural-language keywords can grant a write permission.
    """
    selected = set(intent.allowed_domains)
    from .router import TOOL_DOMAIN
    return [tool for tool in SPECS if (
        tool['function']['name'] == 'final_answer' or
        (tool['function']['name'] not in WRITE_ACTIONS
         and tool['function']['name'] not in {name for names in WRITE_TOOLS.values() for name in names}
         and TOOL_DOMAIN.get(tool['function']['name']) in selected) or
        tool['function']['name'] in intent.permitted_writes
    )]


# ============================================================
# TOOL RESULT COMPRESSION
# ============================================================

def compact_observation(name: str, outcome) -> str:
    """
    Bound tool observation size without truncating JSON.

    LB Radio can return many candidates, so only the first
    20 distinct tracks are sent back into the model context.
    """

    envelope = asdict(outcome)
    data = envelope.get("data")

    if (
        name == "lb_radio"
        and isinstance(data, dict)
        and isinstance(data.get("tracks"), list)
    ):
        tracks = data["tracks"]

        distinct = []
        seen = set()

        for track in tracks:
            if not isinstance(track, dict):
                continue

            key = (
                str(track.get("artist", "")).casefold(),
                str(track.get("title", "")).casefold(),
            )

            if key in seen or not all(key):
                continue

            seen.add(key)
            distinct.append(track)

            if len(distinct) >= 20:
                break

        envelope["data"] = {
            "title": data.get("title"),
            "tracks": distinct,
            "total_tracks": len(tracks),
            "note": (
                "Showing up to 20 distinct candidates."
            ),
        }

    return json.dumps(
        envelope,
        ensure_ascii=False,
        default=str,
    )


# ============================================================
# REQUEST PARSING
# ============================================================

def requested_discovery_count(intent: Intent) -> int | None:
    """Get an optional explicit count from the interpreted request."""
    return intent.track_count if intent.action == 'recommend' else None


def _normalize(value: str) -> str:
    """Normalize identifiers and display strings without patterns."""
    decomposed = unicodedata.normalize('NFKD', str(value)).casefold()
    normalized = ''.join(ch for ch in decomposed if not unicodedata.category(ch).startswith('M'))
    return ' '.join(''.join(ch if ch.isalnum() else ' ' for ch in normalized).split())


# ============================================================
# CANDIDATE TRACKING
# ============================================================

def _remember_candidates(
    name: str,
    data,
    candidates: dict[tuple[str, str], dict],
):
    """
    Preserve candidate recordings from real tool results.

    This allows the agent loop to validate final answers
    against actual API observations.
    """

    if name == "lb_radio" and isinstance(data, dict):
        rows = data.get("tracks", [])

    elif name in {
        "search_recordings",
        "get_recommendations",
        "get_top_recordings_for_artist",
    }:
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            rows = data.get("recordings", [])
        else:
            rows = []

    elif name == "lookup_recording":
        rows = [data]

    else:
        return

    if not isinstance(rows, list):
        return

    for row in rows:
        if not isinstance(row, dict):
            continue

        title = row.get("title")
        artist = row.get("artist")

        if (
            not artist
            and isinstance(row.get("artist_credit"), list)
        ):
            artist = "".join(
                item.get("name", "")
                + item.get("joinphrase", "")
                for item in row["artist_credit"]
                if isinstance(item, dict)
            ).strip()

        if not (
            isinstance(title, str)
            and isinstance(artist, str)
            and title.strip()
            and artist.strip()
        ):
            continue

        ids = row.get("recording_mbids") or (
            [row["mbid"]]
            if row.get("mbid")
            else []
        )

        key = (
            _normalize(artist),
            _normalize(title),
        )

        candidates.setdefault(
            key,
            {
                "artist": artist,
                "title": title,
                "mbids": [
                    mbid
                    for mbid in ids
                    if isinstance(mbid, str)
                ],
            },
        )


def grounded_popularity_answer(intent: Intent, data: dict) -> tuple[str, list[str]] | None:
    """Generic discovery follows the tool-driven path, not a template matcher."""
    return None


# ============================================================
# FINAL ANSWER GROUNDING
# ============================================================

def _answer_grounding_error(
    intent: Intent,
    answer: str,
    status: str,
    candidates: dict,
    verified: set[str],
    claimed_verification: bool = False,
) -> str | None:
    """
    Validate discovery answers against observed API data.

    A response such as "All recordings were verified"
    does not satisfy "Find 5 songs".
    """

    count = requested_discovery_count(intent)

    if (
        status != "answered"
        or count is None
        or count < 1
    ):
        return None

    normalized_answer = _normalize(answer)

    matches = [
        entry
        for entry in candidates.values()
        if (
            _normalize(entry["title"]) in normalized_answer
            and _normalize(entry["artist"]) in normalized_answer
        )
    ]

    if len(matches) < count:
        choices = list(candidates.values())[:12]

        examples = "; ".join(
            f'{item["title"]} — {item["artist"]}'
            for item in choices
        )

        return (
            f"The user requested {count} songs, "
            f"but your answer names only {len(matches)} "
            "distinct artist/title pairs from tool results. "
            f"Respond with {count} real, distinct "
            "SONG — ARTIST entries, not a verification report. "
            "Do not invent songs. "
            f"Candidates observed: {examples or 'none'}. "
            "If there are not enough suitable tracks, "
            "use answered_low_confidence and explain."
        )

    if claimed_verification:
        unverified = [
            item
            for item in matches
            if not (
                set(item["mbids"]) & verified
            )
        ]

        if unverified:
            return (
                "Your final answer claims MusicBrainz "
                "verification for songs that were not "
                "among successful verify_recordings results. "
                "Verify those MBIDs or explicitly state "
                "which songs were not verified."
            )

    return None


def positive_verification_claim(text: str) -> bool:
    """Conservative plain-token check for unsupported verification claims.

    Prefer structured proof based on verified MBIDs for authoritative checks.
    """
    words = ''.join(ch.casefold() if ch.isalpha() else ' ' for ch in text).split()
    for i, word in enumerate(words):
        if word not in {'verified', 'validated'}:
            continue
        if not any(term in {'not','never','without','cannot','unverified'}
                   for term in words[max(0,i-5):i]):
            return True
    return False
