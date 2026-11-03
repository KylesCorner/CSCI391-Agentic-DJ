"""Tool routing, observation compaction, discovery, and answer validation.

Read-only policies moved verbatim from agent.py. The original import paths
remain usable for callers and tests.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import re
import unicodedata

from .tool_specs import SPECS

# ============================================================
# TOOL ROUTER
# ============================================================

def route(goal: str) -> list[dict]:
    """
    Narrow the available tools for a request.

    The model still chooses its own next action.
    This does not hardcode an execution sequence.
    """

    low = goal.casefold()

    # Explanations of API behavior must never be routed to real account writes.
    # The user saying "How does ... export ...?" is not authorization to export.
    if re.search(r"^\s*(?:how does|how do|what is|what does|why does|explain|define)\b", low) and any(
        term in low for term in (
            "listenbrainz", "musicbrainz", "api", "jspf", "lb radio",
            "rate limit", "scrobbl", "music glossary", "genre", "export",
        )
    ):
        names = {"search_music_knowledge", "search_knowledge", "final_answer"}
        return [sp for sp in SPECS if sp["function"]["name"] in names]

    # Assignment-required read/write capabilities. Keep named operations
    # isolated so the model cannot turn a read into an account mutation.
    if re.search(r"\b(?:fresh|new|newest|latest|upcoming|recent)\s+releases?\b", low):
        names = {"get_fresh_releases", "final_answer"}
        return [sp for sp in SPECS if sp["function"]["name"] in names]

    if re.search(r"\b(?:members?|membership|who(?:'s| is) in|played with|collaborat|lineup)\b", low) and        re.search(r"\b(?:band|quartet|trio|artist|musician|group|who)\b", low):
        names = {"search_artists", "lookup_artist", "get_top_recordings_for_artist", "final_answer"}
        return [sp for sp in SPECS if sp["function"]["name"] in names]

    if re.search(r"\b(?:most popular|top recordings|top tracks|biggest songs)\b", low) and        re.search(r"\b(?:artist|by|for)\b", low):
        names = {"search_artists", "get_top_recordings_for_artist", "final_answer"}
        return [sp for sp in SPECS if sp["function"]["name"] in names]

    if re.search(r"\b(?:feedback|loved|hated|liked|disliked)\b", low) and        re.search(r"\b(?:what|show|which|list|read|my)\b", low) and not re.search(r"\b(?:set|mark|clear|remove|change)\b", low):
        names = {"get_feedback", "final_answer"}
        return [sp for sp in SPECS if sp["function"]["name"] in names]

    if re.search(r"\b(?:mark|set|love|hate|unlove|unhate|clear|remove|rate|thumbs up|thumbs down)\b", low) and        re.search(r"\b(?:song|track|recording|music|feedback|this)\b", low) and not "playlist" in low:
        names = {"get_feedback", "search_recordings", "lookup_recording", "verify_recordings",
                 "set_feedback", "final_answer"}
        return [sp for sp in SPECS if sp["function"]["name"] in names]

    # Whole-playlist deletion must never accidentally route to item removal.
    if re.search(r"\b(?:delete|remove)\b", low) and re.search(r"\bplaylists?\b", low) and        not re.search(r"\b(?:song|track|recording|item)\b", low):
        names = {"find_playlists", "get_playlist", "delete_playlist", "final_answer"}
        return [sp for sp in SPECS if sp["function"]["name"] in names]

    write_intent = (
        any(
            word in low
            for word in (
                "create",
                "make",
                "build",
                "save",
                "add",
                "remove",
                "delete",
                "export",
                "spotify",
                "apple music",
                "put on",
            )
        )
        and any(
            word in low
            for word in (
                "playlist",
                "spotify",
                "apple music",
            )
        )
    )

    personalized = any(
        word in low
        for word in (
            "my listening",
            "my plays",
            "my favorite",
            "my favourite",
            "my listens",
            "my history",
            "i've played",
            "i have played",
            "i listened",
            "scrobbl",
            "last week",
            "recently played",
        )
    )

    discovery = (
        any(
            word in low
            for word in (
                "similar",
                "find",
                "discover",
                "suggest",
                "recommend",
            )
        )
        and any(
            word in low
            for word in (
                "song",
                "track",
                "artist",
                "jazz",
                "music",
                "similar",
                "recommend",
                "suggest",
            )
        )
    )

    memory_intent = any(term in low for term in (
        "preference", "previously", "prior session", "last time",
        "told you", "what i like", "studying", "study music",
    ))

    # Route mutations by the action the user actually requested. The model
    # may choose tools within a domain but cannot choose a different write.
    if "playlist" in low and re.search(
        r"\b(?:add|adding|append|appending|insert|inserting)\b", low,
    ):
        names = {"list_playlists", "get_playlist", "search_recordings",
                 "search_artists", "verify_recordings", "lb_radio",
                 "add_playlist_tracks", "add_playlist_items", "find_playlists", "final_answer"}
        return [sp for sp in SPECS if sp["function"]["name"] in names]

    if "playlist" in low and re.search(r"\b(remove|delete)\b", low):
        names = {"list_playlists", "find_playlists", "get_playlist", "remove_playlist_tracks",
                 "remove_playlist_items", "final_answer"}
        return [sp for sp in SPECS if sp["function"]["name"] in names]

    # Limit first-time playlist creation to the domains actually needed.
    # This is tool scoping, not a scripted tool sequence.
    if "playlist" in low and re.search(r"\b(create|make|build|save)\b", low):
        names = {
            "search_recordings", "search_artists", "lb_radio",
            "verify_recordings", "get_recommendations",
            "list_playlists", "find_playlists", "get_playlist", "create_playlist",
            "final_answer",
        }
        if memory_intent:
            names |= {"recall_preferences", "search_past_sessions"}
        if any(word in low for word in ("minute", "hour", "duration")):
            names.add("select_playlist_duration")
        if any(word in low for word in ("haven't played", "last 7 days", "recently played", "heard this week")):
            names |= {"get_listens_since", "filter_recent_candidates"}
        return [sp for sp in SPECS if sp["function"]["name"] in names]

    if "playlist" in low and re.search(r"\bexport\b", low) and not re.search(r"\b(?:do not|don.t|without)\s+export", low):
        names = {"list_playlists", "get_playlist", "export_playlist",
                 "final_answer"}
        return [sp for sp in SPECS if sp["function"]["name"] in names]

    # Preferences and music discovery must be routed together. Previously,
    # the discovery-only route hid recall_preferences from the model.
    if memory_intent:
        names = {
            "recall_preferences", "search_past_sessions",
            "remember_preference", "note_preference", "forget_preference", "final_answer",
        }
        if discovery or "music" in low or "songs" in low or "tracks" in low:
            names |= {
                "lb_radio", "search_artists", "search_recordings",
                "lookup_recording", "verify_recordings",
                "get_recommendations", "search_knowledge", "search_music_knowledge",
            }
        if write_intent:
            names |= {
                "list_playlists", "find_playlists", "get_playlist", "create_playlist",
                "add_playlist_tracks", "add_playlist_items", "remove_playlist_tracks", "remove_playlist_items",
                "export_playlist", "select_playlist_duration",
            }
        return [sp for sp in SPECS if sp["function"]["name"] in names]

    # Focused read-only discovery.
    if (
        discovery
        and not write_intent
        and not personalized
        and "playlist" not in low
    ):
        names = {
            "search_artists",
            "search_recordings",
            "lookup_artist", "get_top_recordings_for_artist",
            "lookup_recording",
            "verify_recordings",
            "lb_radio",
            "search_knowledge", "search_music_knowledge",
            "final_answer",
        }

        return [
            spec
            for spec in SPECS
            if spec["function"]["name"] in names
        ]

    # Playlist workflows may need several domains.
    if any(
        word in low
        for word in (
            "playlist",
            "recommend",
            "suggest",
            "similar",
            "export",
            "spotify",
            "apple music",
            "discover",
        )
    ):
        return SPECS

    # Listening statistics.
    if any(
        word in low
        for word in (
            "how many",
            "most listened",
            "most played",
            "top artist",
            "top song",
            "top recording",
            "recent",
            "played",
            "listening",
        )
    ):
        names = {
            "get_recent_listens",
            "get_listens_since",
            "get_listen_count",
            "get_top_artists",
            "get_top_recordings",
            "get_playing_now", "get_feedback", "get_fresh_releases",
            "search_past_sessions",
            "recall_preferences",
            "search_knowledge", "search_music_knowledge",
            "final_answer",
        }

        return [
            spec
            for spec in SPECS
            if spec["function"]["name"] in names
        ]

    return SPECS


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

def requested_discovery_count(goal: str) -> int | None:
    """
    Extract an explicitly requested number of songs.

    Example:
        Find 5 gypsy jazz songs
        -> 5
    """

    low = goal.casefold()

    # Playlist creation needs its own write postcondition, not the
    # read-only recommendation formatter. Also, "playlist" contains
    # the substring "list", which must not trigger discovery checks.
    if "playlist" in low and re.search(r"\b(create|make|build|save)\b", low):
        return None

    if not re.search(r"\b(?:find|recommend|suggest|give me|list|discover)\b", low):
        return None

    match = re.search(
        r"\b(\d{1,2})\s+"
        r"(?:[\w-]+\s+){0,5}"
        r"(?:songs?|tracks?|recordings?)\b",
        low,
    )

    return int(match.group(1)) if match else None


def _normalize(value: str) -> str:
    """
    Normalize artist/title text for comparison.
    """

    decomposed = unicodedata.normalize(
        "NFKD",
        str(value),
    ).casefold()

    return " ".join(
        re.findall(r"[\w]+", decomposed)
    ).strip()


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


# ============================================================
# FINAL ANSWER GROUNDING
# ============================================================

def _answer_grounding_error(
    goal: str,
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

    count = requested_discovery_count(goal)

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
    """
    Detect affirmative verification claims while allowing
    statements such as 'not independently verified'.
    """

    for match in re.finditer(
        r"\b(?:verified|validated)\b",
        text,
        re.IGNORECASE,
    ):
        preceding = text[
            max(0, match.start() - 45):match.start()
        ]

        negation = re.search(
            r"\b(?:not|never|without|cannot|can't|unverified)"
            r"\b.{0,30}$",
            preceding,
            re.IGNORECASE,
        )

        if not negation:
            return True

    return False
