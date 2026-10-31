"""Typed result envelope, tool schemas, execution, and safe mutation policy."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from hashlib import sha256
from typing import Any, Callable
import json
import re
from uuid import uuid4
from jsonschema import validate, ValidationError
from .clients import ListenBrainz, MusicBrainz, ServiceError, check_uuid
from .memory import Memory


def spec(name: str, desc: str, properties: dict, required: tuple[str, ...] = ()) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc,
             "parameters": {"type": "object", "properties": properties,
                            "required": list(required), "additionalProperties": False}}}


def s(desc: str) -> dict:
    return {"type": "string", "description": desc}


def n(desc: str, minimum=1, maximum=100) -> dict:
    return {"type": "integer", "minimum": minimum, "maximum": maximum, "description": desc}


def arr(desc: str, mx=40) -> dict:
    return {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": mx, "uniqueItems": True, "description": desc}


SPECS = [
    spec("get_recent_listens", "Get latest scrobbled tracks", {"count": n("How many listens", 1, 100)}),
    spec("get_listens_since", "Retrieve the FULL recent-days exclusion window, if bounded scan completes", {"days": n("Days", 1, 30)}),
    spec("get_listen_count", "Get total scrobbles", {}),
    spec("get_top_artists", "Top artists; this_month means CURRENT calendar month, month means PREVIOUS month. Falls back to counting current-month scrobbles if aggregated stats are unavailable.", {"count": n("Max results", 1, 50), "range": {"type": "string", "enum": ["this_week", "this_month", "this_year", "week", "month", "quarter", "year", "half_yearly", "all_time"]}}),
    spec("get_top_recordings", "Top songs; this_month means CURRENT calendar month, month means PREVIOUS month. Falls back to counting current-month scrobbles if aggregated stats are unavailable.", {"count": n("Max results", 1, 50), "range": {"type": "string", "enum": ["this_week", "this_month", "this_year", "week", "month", "quarter", "year", "half_yearly", "all_time"]}}),
    spec("get_playing_now", "Currently playing track", {}),

    spec(
        "search_recordings",
        "Search MusicBrainz recordings. "
        'To find songs BY an artist, use artist:"Benny Goodman". '
        'To find a specific song TITLE, use recording:"Sing, Sing, Sing". '
        "Use AND to combine artist and title. "
        "Do not confuse performer names with song titles. "
        "Returns real MusicBrainz recording IDs.",
        {
            "query": s(
                "MusicBrainz Lucene query. "
                "Use artist: for performer, "
                "recording: for song title."
            )
        },
        ("query",),
    ),

    spec("search_artists", "Find a NAMED MusicBrainz artist; do not use for styles such as instrumental, swing, or study music", {"query": s("Artist name")}, ("query",)),
    spec("lookup_recording", "Look up one recording MBID, including length in ms", {"mbid": s("Recording UUID")}, ("mbid",)),
    spec("verify_recordings", "Resolve MBIDs in MusicBrainz before writing a playlist", {"mbids": arr("Recording UUIDs")}, ("mbids",)),
    spec("lb_radio", "Discover related recordings via LB Radio; use well-formed artist:(Benny Goodman) or tag:(swing) expressions, NOT natural-language questions", {"prompt": s("LB Radio prompt"), "mode": {"type": "string", "enum": ["easy", "medium", "hard"]}}, ("prompt",)),
    spec("get_recommendations", "Retrieve ListenBrainz collaborative recommendations", {"count": n("Number", 1, 100)}),
    spec("select_playlist_duration", "Select subset of verified recordings closest to a target length; uses MusicBrainz duration in milliseconds", {"mbids": arr("Verified recording MBIDs"), "target_minutes": n("Target duration in minutes", 1, 180)}, ("mbids", "target_minutes")),
    spec("list_playlists", "List own ListenBrainz playlists", {}),
    spec("get_playlist", "Read a playlist by MBID", {"playlist_mbid": s("Playlist UUID")}, ("playlist_mbid",)),
    spec("filter_recent_candidates", "Filter latest LB Radio candidates against scrobbles from last N days, using MBID or artist/title; only succeeds if history complete", {"days": n("Days to exclude", 1, 30)}, ("days",)),
    spec("create_playlist", "CREATE in ListenBrainz. User approval REQUIRED, all recording IDs must first be verified", {"title": s("Title"), "mbids": arr("Verified recording MBIDs"), "exclude_days": n("Optional days to exclude", 0, 30)}, ("title", "mbids")),
    spec("add_playlist_tracks", "ADD tracks to an existing playlist with user approval", {"playlist_mbid": s("Playlist UUID"), "mbids": arr("Verified recording MBIDs")}, ("playlist_mbid", "mbids")),
    spec("remove_playlist_tracks", "REMOVE tracks by index/count with user approval", {"playlist_mbid": s("Playlist UUID"), "index": n("Zero-based index", 0, 100000), "count": n("Tracks to remove", 1, 40)}, ("playlist_mbid", "index", "count")),
    spec("export_playlist", "EXPORT to connected Spotify/Apple Music, with user approval", {"playlist_mbid": s("Playlist UUID"), "service": {"type": "string", "enum": ["spotify", "apple_music"]}}, ("playlist_mbid", "service")),
    spec("search_past_sessions", "Retrieve episodic memory for prior requests", {"query": s("Past-goal search")}, ("query",)),
    spec("search_knowledge", "Retrieve indexed ListenBrainz/MusicBrainz API documentation", {"query": s("Documentation search")}, ("query",)),
    spec("recall_preferences", "Read actual previously expressed user preferences. Use BEFORE recommendations based on prior preferences; never infer genre solely from a recording title", {}),
    spec("remember_preference", "Remember only an EXPLICIT user preference quoted exactly from CURRENT request", {"key": s("Short key"), "value": s("Preference"), "evidence": s("Exact quote from this user request")}, ("key", "value", "evidence")),
    spec("forget_preference", "Forget one saved preference", {"key": s("Saved preference key")}, ("key",)),
    spec("final_answer", "Finish. Give truthful answer and status. Call only when done or unable to proceed", {"answer": s("Final explanation"), "status": {"type": "string", "enum": ["answered", "answered_low_confidence", "stuck"]}}, ("answer", "status")),
]
INDEX = {x["function"]["name"]: x for x in SPECS}
WRITE_ACTIONS = frozenset({"create_playlist", "add_playlist_tracks", "remove_playlist_tracks", "export_playlist"})


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error_type: str | None = None
    message: str | None = None

    def json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)


class Tools:
    def __init__(self, lb: ListenBrainz, mb: MusicBrainz | None, memory: Memory,
                 *, approve: Callable[[str], bool], current_goal: str,
                 session_id: str | None = None):
        self.lb, self.mb, self.mem = lb, mb, memory
        self.approve = approve
        self.goal = current_goal
        self.session_id = session_id or str(uuid4())
        self.verified: dict[str, dict] = {}
        self.last_candidates: list[dict] = []
        self.recent_cache: dict[int, dict] = {}

    def _recent(self, days: int) -> dict:
        if days not in self.recent_cache:
            self.recent_cache[days] = self.lb.listens_since(days, max_pages=10)
        return self.recent_cache[days]

    @staticmethod
    def _key(artist: str | None, title: str | None) -> tuple[str, str]:
        def norm(x):
            return re.sub(r"[^\w]+", "", (x or "").casefold())
        return norm(artist), norm(title)

    def _exclude(self, candidates: list[dict], days: int) -> list[dict]:
        history = self._recent(days)
        if not history["complete"]:
            raise ServiceError("resource", "Scrobble window truncated; cannot guarantee exclusions")
        heard_ids = {x["recording_mbid"] for x in history["listens"] if x.get("recording_mbid")}
        heard_names = {self._key(x["artist"], x["title"]) for x in history["listens"]}
        return [c for c in candidates if not (
            set(c.get("recording_mbids", [])) & heard_ids
            or self._key(c.get("artist"), c.get("title")) in heard_names
        )]

    def _write(self, action: str, args: dict, callback) -> dict:
        if not self.lb.token:
            raise ServiceError("auth", "Set LISTENBRAINZ_TOKEN for account writes")
        fingerprint = sha256(json.dumps([self.session_id, action, args], sort_keys=True).encode()).hexdigest()
        # Always request approval before mutating real account; avoid approving duplicate retries.
        authorized, old = self.mem.reserve_write(fingerprint, action)
        if not authorized:
            raise ServiceError("resource", f"Write already attempted; never retry blindly. Local ledger: {old['state']}")
        descriptor = f"{action}: {json.dumps(args, ensure_ascii=False)}"
        if not self.approve(descriptor):
            self.mem.mark_uncertain(fingerprint)  # blocks replay after denial
            raise ServiceError("policy_halt", "User denied (or no interactive approval)")
        try:
            response = callback()
        except Exception:
            self.mem.mark_uncertain(fingerprint)
            raise
        self.mem.finish_write(fingerprint, response)
        return response

    def _catalog(self) -> MusicBrainz:
        if self.mb is None:
            raise ServiceError("resource", "Set MB_CONTACT for MusicBrainz tools")
        return self.mb

    def _verified_mbids(self, mbids: list[str]) -> list[str]:
        cleaned = [check_uuid(x) for x in mbids]
        missing = [x for x in cleaned if x not in self.verified]
        if missing:
            raise ServiceError("invalid", f"Verify recording MBIDs first: {missing}")
        return cleaned

    def run(self, name: str, args: dict) -> ToolResult:
        if name not in INDEX or name == "final_answer":
            return ToolResult(False, error_type="invalid", message=f"Unknown/non-executable tool {name}")
        try:
            validate(instance=args, schema=INDEX[name]["function"]["parameters"])
            method = getattr(self, f"tool_{name}")
            return ToolResult(True, data=method(**args))
        except ValidationError as exc:
            return ToolResult(False, error_type="invalid", message=f"Tool argument error: {exc.message}")
        except ServiceError as exc:
            return ToolResult(False, error_type=exc.kind, message=str(exc))
        except (ValueError, TypeError) as exc:
            return ToolResult(False, error_type="invalid", message=str(exc))
        except Exception as exc:
            # Preserve error taxonomy without dumping tokens/headers/user data.
            return ToolResult(False, error_type="fatal", message=f"Unexpected {type(exc).__name__}")

    def tool_get_recent_listens(self, count=20): return self.lb.recent(count)
    def tool_get_listens_since(self, days=7): return self._recent(days)
    def tool_get_listen_count(self): return self.lb.count()
    def tool_get_top_artists(self, count=10, range="this_month"): return self.lb.top("artists", count, range)
    def tool_get_top_recordings(self, count=10, range="this_month"): return self.lb.top("recordings", count, range)
    def tool_get_playing_now(self): return self.lb.playing_now()


    @staticmethod
    def _is_generic_music_descriptor(query: str) -> bool:
        """Descriptors are not artist identities; reject ambiguous seed searches."""
        clean = re.sub(r"[\W_]+", " ", query.casefold()).strip()
        return clean in {
            "instrumental", "instrumental jazz", "swing", "swing jazz",
            "jazz", "study", "study music", "focus music", "relaxing",
            "background music", "music for studying", "acoustic", "no vocals",
        }

    def tool_search_recordings(self, query):
        """
        Search MusicBrainz recordings.

        Prevents an explicit request for recordings BY an
        artist from becoming a search for a recording
        TITLED with that artist's name.
        """

        # The model has repeatedly interpreted "instrumental" (an adjective)
        # as a MusicBrainz artist. Do not waste an API call on that guess.
        generic_artist = re.search(r'\b(?:artist|artistname):\s*"?([^"\)]+)', query, re.I)
        if (
            self._is_generic_music_descriptor(query)
            or (generic_artist and self._is_generic_music_descriptor(generic_artist.group(1)))
        ) and not re.search(r'\b(?:recordings?|songs?|tracks?)\s+by\s+instrumental\b', self.goal, re.I):
            raise ServiceError(
                "resource",
                "This is a musical descriptor, not a named artist. "
                "Read saved preferences and search an actual artist name, "
                "or use LB Radio tag:(swing).",
            )

        artist_match = re.search(
            r"\b(?:recordings?|songs?|tracks?)\s+by\s+"
            r"(.+?)(?=[.!?]|$)",
            self.goal,
            flags=re.IGNORECASE,
        )

        if artist_match and not re.search(
            r"\b(?:artist|artistname|arid):",
            query,
            re.IGNORECASE,
        ):
            artist = artist_match.group(1).strip()
            escaped_artist = artist.replace(
                "\\", "\\\\"
            ).replace('"', '\\"')

            artist_clause = f'artist:"{escaped_artist}"'

            if re.search(
                r"\brecording:", query, re.IGNORECASE
            ):
                query = f"{query} AND {artist_clause}"
            else:
                plain = query.strip().strip('"').casefold()

                if (
                    plain in artist.casefold()
                    or artist.casefold() in plain
                ):
                    query = artist_clause
                else:
                    escaped_title = query.replace(
                        "\\", "\\\\"
                    ).replace('"', '\\"')

                    query = (
                        f'recording:"{escaped_title}" '
                        f'AND {artist_clause}'
                    )

        data = self._catalog().search(query)

        recordings = []

        for item in data.get("recordings", [])[:10]:
            artist = "".join(
                part.get("name", "")
                + part.get("joinphrase", "")
                for part in item.get("artist-credit", [])
                if isinstance(part, dict)
            ).strip()

            recordings.append({
                "mbid": item.get("id"),
                "title": item.get("title"),
                "artist": artist,
                "score": item.get("score"),
                "length_ms": item.get("length"),
            })

        if not recordings:
            raise ServiceError(
                "resource",
                "No matching MusicBrainz recordings. "
                "Try another query.",
            )

        return recordings

    def tool_search_artists(self, query):
        if self._is_generic_music_descriptor(query) and not re.search(
            r'\b(?:artist|musician|band)\s+(?:named|called)\s+', self.goal, re.I
        ):
            raise ServiceError(
                "resource",
                "Do not search for 'instrumental', 'swing jazz', or other "
                "style descriptors as artist names. Use a specific artist or "
                "a genre/tag discovery tool.",
            )
        data = self._catalog().search_artist(query)
        return [{"mbid": x.get("id"), "name": x.get("name"), "score": x.get("score")}
                for x in data.get("artists", [])[:10]]

    def tool_lookup_recording(self, mbid):
        item = self._catalog().lookup(mbid)
        return {"mbid": item.get("id"), "title": item.get("title"),
                "length_ms": item.get("length"), "artist_credit": item.get("artist-credit", [])}

    def tool_verify_recordings(self, mbids):
        data = self._catalog().verify(mbids)
        for item in data["valid"]:
            self.verified[check_uuid(item["mbid"])] = item
        return data

    def tool_lb_radio(self, prompt, mode="easy"):
        data = self.lb.radio(prompt, mode)
        self.last_candidates = data["tracks"]
        return data

    def tool_get_recommendations(self, count=20): return self.lb.recommendations(count)

    def tool_select_playlist_duration(self, mbids, target_minutes):
        cleaned = self._verified_mbids(mbids)
        duration_seconds = {}
        omitted = []
        for mbid in cleaned:
            length = self.verified[mbid].get("duration_ms")
            if not isinstance(length, int) or length <= 0:
                omitted.append(mbid)
            else:
                duration_seconds[mbid] = max(1, round(length / 1000))
        if not duration_seconds:
            raise ServiceError("resource", "No verified tracks have known durations")
        target = target_minutes * 60
        # Deterministic bounded subset-sum; never substitute guessed durations.
        states = {0: ()}
        for mbid, seconds in duration_seconds.items():
            for total, subset in list(states.items()):
                new_total = total + seconds
                if new_total <= target + 180 and new_total not in states:
                    states[new_total] = subset + (mbid,)
        best = min((x for x in states if x > 0), key=lambda x: (abs(target - x), x > target))
        return {"mbids": list(states[best]), "minutes": round(best / 60, 2),
                "target_minutes": target_minutes, "difference_seconds": best - target,
                "omitted_unknown_length": omitted,
                "note": "Instrumental status is not validated by recording duration."}

    def tool_list_playlists(self): return self.lb.playlists()
    def tool_get_playlist(self, playlist_mbid): return self.lb.playlist(playlist_mbid)

    def tool_filter_recent_candidates(self, days):
        candidates = self._exclude(self.last_candidates, days)
        return {"tracks": candidates, "excluded": len(self.last_candidates) - len(candidates),
                "history_complete": True}

    def tool_create_playlist(self, title, mbids, exclude_days=0):
        cleaned = self._verified_mbids(mbids)
        if exclude_days:
            candidates = [{"recording_mbids": [mbid],
                           "title": self.verified[mbid].get("title"),
                           "artist": "".join(x.get("name", "") if isinstance(x, dict) else "" for x in self.verified[mbid].get("artist_credit", []))}
                          for mbid in cleaned]
            filtered = self._exclude(candidates, exclude_days)
            if len(filtered) != len(candidates):
                raise ServiceError("invalid", "A selected track is in the exclusion window")
        return self._write("create_playlist", {"title": title, "mbids": cleaned, "exclude_days": exclude_days},
                           lambda: self.lb.create_playlist(title, cleaned))

    def tool_add_playlist_tracks(self, playlist_mbid, mbids):
        cleaned = self._verified_mbids(mbids)
        return self._write("add_playlist_tracks", {"playlist_mbid": playlist_mbid, "mbids": cleaned},
                           lambda: self.lb.add_tracks(playlist_mbid, cleaned))

    def tool_remove_playlist_tracks(self, playlist_mbid, index, count):
        return self._write("remove_playlist_tracks", {"playlist_mbid": playlist_mbid, "index": index, "count": count},
                           lambda: self.lb.remove_tracks(playlist_mbid, index, count))

    def tool_export_playlist(self, playlist_mbid, service):
        return self._write("export_playlist", {"playlist_mbid": playlist_mbid, "service": service},
                           lambda: self.lb.export(playlist_mbid, service))

    def tool_search_past_sessions(self, query): return self.mem.search_sessions(query)
    def tool_search_knowledge(self, query): return self.mem.search_knowledge(query)
    def tool_recall_preferences(self): return self.mem.preferences()

    def tool_remember_preference(self, key, value, evidence):
        if not evidence or evidence.casefold() not in self.goal.casefold():
            raise ServiceError("policy_halt", "Evidence must occur verbatim in current user message")
        prefix = self.goal.casefold()
        if not any(marker in prefix for marker in ("i like", "i prefer", "i dislike", "remember", "don't recommend", "never recommend", "i love", "i hate")):
            raise ServiceError("policy_halt", "No explicit stable preference in user message")
        self.mem.set_preference(key, value, evidence)
        return {"saved": key}

    def tool_forget_preference(self, key): return {"deleted": self.mem.forget_preference(key)}