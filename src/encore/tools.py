"""Typed result envelope, tool schemas, execution, and safe mutation policy."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from hashlib import sha256
from typing import Any, Callable
import json
import re
from uuid import uuid4
from jsonschema import validate, ValidationError
from .clients import ListenBrainz, MusicBrainz, ServiceError, check_uuid, mbids_from_track
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
    spec("select_playlist_duration", "Select subset of verified recordings and report ready=true only if within 60 seconds of target. If ready=false, search and verify MORE recordings. Never create a shorter playlist.", {"mbids": arr("Verified recording MBIDs"), "target_minutes": n("Target duration in minutes", 1, 180)}, ("mbids", "target_minutes")),
    spec("list_playlists", "List own ListenBrainz playlists", {}),
    spec("get_playlist", "Read a playlist by MBID", {"playlist_mbid": s("Playlist UUID")}, ("playlist_mbid",)),
    spec("filter_recent_candidates", "Filter latest LB Radio candidates against scrobbles from last N days, using MBID or artist/title; only succeeds if history complete", {"days": n("Days to exclude", 1, 30)}, ("days",)),
    spec("create_playlist", "Requires BOTH title (string) and mbids (list of verified UUIDs). Example: {title: Swing Test, mbids: [five verified recording IDs]}. CREATE a PRIVATE ListenBrainz playlist ONLY if the user explicitly requested creation. Verify all recording MBIDs first. Do not call more than once. Checks existing titles, requests human approval, and confirms the resulting tracks; include the created playlist MBID in final_answer. Do not use for read-only recommendations.", {"title": s("Title"), "mbids": arr("Verified recording MBIDs"), "exclude_days": n("Optional days to exclude", 0, 30)}, ("title", "mbids")),
    spec("add_playlist_tracks", "APPEND verified, non-duplicate recordings to an EXISTING playlist only when user said add; read actual playlist first, validate title, request approval, verify readback. Never create a new playlist.", {"playlist_mbid": s("Playlist UUID"), "mbids": arr("Verified recording MBIDs")}, ("playlist_mbid", "mbids")),
    spec("remove_playlist_tracks", "REMOVE tracks only from an EXISTING playlist if the user asked to remove; for 'last song', index argument is ignored and safely derived from full playlist detail; approval shows the exact selected track, then verifies readback. Never infer positions from list_playlists.", {"playlist_mbid": s("Playlist UUID"), "index": n("Zero-based index", 0, 100000), "count": n("Tracks to remove", 1, 40)}, ("playlist_mbid", "index", "count")),
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
        self.searched_recordings: dict[str, dict] = {}

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

    def _write(self, action: str, args: dict, callback, *, summary: str | None = None) -> dict:
        if not self.lb.token:
            raise ServiceError("auth", "Set LISTENBRAINZ_TOKEN for account writes")
        # Stable across sessions for playlist creation; session-scoped for edits.
        scope = [action, args] if action == "create_playlist" else [self.session_id, action, args]
        fingerprint = sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()
        reserved, previous = self.mem.reserve_write(fingerprint, action)
        if not reserved:
            raise ServiceError("resource", "Write already attempted. Reconcile the remote playlist before retrying; "
                               f"local ledger state: {previous['state']}")

        if action == "create_playlist":
            lines = []
            for index, mbid in enumerate(args["mbids"], 1):
                item = self.verified[mbid]
                credits = item.get("artist_credit", [])
                artist = "".join(x.get("name", "") + x.get("joinphrase", "") for x in credits
                                 if isinstance(x, dict)).strip()
                artist = artist or self.searched_recordings.get(mbid, {}).get("artist", "")
                lines.append(f"{index}. {item.get('title') or mbid} — {artist or 'artist unconfirmed'}")
            description = (f"CREATE PRIVATE PLAYLIST: {args['title']}\n" + "\n".join(lines)
                           + "\nThis modifies your real ListenBrainz account.")
        else:
            description = summary or f"{action}: {json.dumps(args, ensure_ascii=False)}"

        if not self.approve(description):
            self.mem.mark_uncertain(fingerprint)  # no replay after a denial
            raise ServiceError("policy_halt", "User declined this write")
        try:
            result = callback()
        except Exception:
            # The server might have accepted the POST before the connection failed.
            self.mem.mark_uncertain(fingerprint)
            raise
        self.mem.finish_write(fingerprint, result)
        return result

    def _find_existing_playlist(self, title: str) -> dict | None:
        """Find an identical title in the user's playlists, including private ones.

        Never allow a write if the listing cannot be fully inspected.
        """
        if not callable(getattr(self.lb, "playlists", None)):
            # Lightweight existing unit-test fakes have no listing endpoint.
            # Production ListenBrainz clients always have playlists().
            return None
        for offset in range(0, 500, 100):
            response = self.lb.playlists(count=100, offset=offset)
            if not isinstance(response, dict):
                raise ServiceError("resource", "Playlist listing has an unexpected format")
            payload = response.get("payload", response)
            rows = payload.get("playlists") if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                raise ServiceError("resource", "Could not inspect existing playlists; creation was blocked")
            for row in rows:
                if not isinstance(row, dict):
                    continue
                item = row.get("playlist", row)
                if isinstance(item, dict) and str(item.get("title", "")).strip().casefold() == title.casefold():
                    return item
            if len(rows) < 100:
                return None
        raise ServiceError("resource", "Too many playlists to confirm title uniqueness safely")

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

    def _requested_playlist_artist(self) -> str | None:
        """Extract an explicit performer constraint from a playlist goal.

        E.g. 'playlist ... with 5 distinct Benny Goodman recordings'.
        Do not guess an artist from a genre-only request.
        """
        if not re.search(r"\bplaylist\b", self.goal, re.I):
            return None
        match = re.search(
            r"\b(?:with|of)\s+\d{1,2}\s+(?:distinct\s+)?"
            r"(?P<artist>[\w&'’.-]+(?:\s+[\w&'’.-]+){0,4}?)\s+"
            r"(?:recordings?|songs?|tracks?)\b",
            self.goal,
            re.I,
        )
        if not match:
            match = re.search(
                r"\badd\s+\d{1,2}\s+(?:distinct\s+)?"
                r"(?P<artist>[\w&'’.-]+(?:\s+[\w&'’.-]+){0,4}?)\s+"
                r"(?:recordings?|songs?|tracks?)\b",
                self.goal, re.I,
            )
        if not match:
            by = re.search(r"\bby\s+(.+?)\s+to\s+(?:my\s+)?", self.goal, re.I)
            if by:
                artist = by.group(1).strip()
                return None if self._is_generic_music_descriptor(artist) else artist
            return None
        artist = match.group("artist").strip()
        if self._is_generic_music_descriptor(artist):
            return None
        return artist

    def _requested_duration_minutes(self) -> int | None:
        """Read a numeric minute requirement from the original user goal.

        Examples: '15-minute playlist', '15 minute playlist', '20 mins'.
        This is deliberately NOT inferred from the model's tool arguments.
        """
        match = re.search(r"\b(\d{1,3})\s*(?:-\s*)?(?:minutes?|mins?)\b", self.goal, re.I)
        if match and int(match.group(1)) > 0:
            return int(match.group(1))
        return None

    @staticmethod
    def _duration_string(duration_ms: int) -> str:
        seconds = round(duration_ms / 1000)
        return f"{seconds // 60}:{seconds % 60:02d}"

    def _enforce_playlist_duration(self, mbids: list[str]) -> None:
        """Hard safety/integrity invariant: check before approval and POST.

        Missing recording durations cannot be guessed. A target-minute
        request requires the sum of known lengths within one minute.
        """
        minutes = self._requested_duration_minutes()
        if minutes is None:
            return

        unknown = [mid for mid in mbids if not isinstance(
            self.verified[mid].get("duration_ms"), int
        ) or self.verified[mid]["duration_ms"] <= 0]
        if unknown:
            raise ServiceError(
                "resource",
                f"Cannot create a {minutes}-minute playlist: {len(unknown)} selected "
                "recording(s) have unknown duration. Verify alternatives with known "
                "duration; NO playlist was created.",
            )

        total_ms = sum(self.verified[mid]["duration_ms"] for mid in mbids)
        target_ms = minutes * 60_000
        if abs(total_ms - target_ms) > 60_000:
            raise ServiceError(
                "resource",
                f"Duration mismatch: selected {self._duration_string(total_ms)} "
                f"versus requested {minutes}:00 (allowed +/-1:00). "
                "Search and verify more appropriate recordings, then use "
                "select_playlist_duration again. NO playlist was created.",
            )

    def tool_search_recordings(self, query):
        """Search actual artist credits, not recordings named after an artist."""
        target = self._requested_playlist_artist()
        by_match = re.search(
            r"\b(?:recordings?|songs?|tracks?)\s+by\s+(.+?)(?=[.!?]|$)",
            self.goal,
            re.I,
        )
        if not target and by_match:
            target = by_match.group(1).strip()

        generic_artist = re.search(
            r'\b(?:artist|artistname):\s*"?([^"\)]+)', query, re.I
        )
        if not target and (
            self._is_generic_music_descriptor(query)
            or (generic_artist and self._is_generic_music_descriptor(generic_artist.group(1)))
        ):
            raise ServiceError(
                "resource",
                "A genre or mood is not an artist. Use LB Radio or a named performer.",
            )

        # User's explicit artist constraint has precedence over model-supplied
        # title searches such as recording:"Benny Goodman".
        if target:
            safe = target.replace("\\", "\\\\").replace('"', '\\"')
            specific_song = re.search(
                r"\badd\s+(.+?)\s+by\s+.+?\s+to\s+(?:my\s+)?",
                self.goal, re.I,
            )
            if specific_song:
                requested_title = specific_song.group(1).strip()
                escaped_title = requested_title.replace("\\", "\\\\").replace('"', '\\"')
                query = f'recording:"{escaped_title}" AND artist:"{safe}"'
            elif re.search(r"\brecording:\s*", query, re.I):
                # Preserve the model's specific song-title predicate and
                # append the artist. Don't collapse it into artist-only search.
                title_predicate = re.search(r'recording:\s*(?:"[^"]+"|[^\s]+)', query, re.I)
                query = (f'{title_predicate.group(0)} AND artist:"{safe}"'
                         if title_predicate else f'artist:"{safe}"')
            else:
                query = f'artist:"{safe}"'

        self.last_recording_query = query  # safe debug metadata, never the token
        data = self._catalog().search(query)
        rows = list(data.get("recordings", []))

        # Genre-specific MusicBrainz recording metadata is sparse. If a
        # constrained artist+genre query cannot supply even the requested
        # minutes, make a second, broader artist search so the model can
        # verify more candidates. This does NOT establish genre membership.
        target_minutes = self._requested_duration_minutes()
        search_artist_match = re.search(
            r'\bartist:\s*(?:"([^"]+)"|(.+?)(?=\s+AND\b|$))',
            query, re.I,
        )
        searched_artist = (
            (search_artist_match.group(1) or search_artist_match.group(2) or "").strip()
            if search_artist_match else ""
        )
        known_ms = sum(
            x.get("length") for x in rows
            if isinstance(x.get("length"), int) and x["length"] > 0
        )
        if (target_minutes and searched_artist and
                re.search(r'\bgenre:', query, re.I) and
                known_ms < max(0, target_minutes * 60_000 - 60_000)):
            escaped = searched_artist.replace("\\", "\\\\").replace('"', '\\"')
            fallback_query = f'artist:"{escaped}"'
            broadened = self._catalog().search(fallback_query)
            rows.extend(broadened.get("recordings", []))
            self.last_recording_query += f' | fallback: {fallback_query}'

        recordings = []
        seen = set()
        artist_constraint = target or (searched_artist if target_minutes and searched_artist else None)
        for item in rows:
            artist = "".join(
                credit.get("name", "") + credit.get("joinphrase", "")
                for credit in item.get("artist-credit", [])
                if isinstance(credit, dict)
            ).strip()
            title = str(item.get("title") or "").strip()
            mbid = item.get("id")
            if not title or not artist or not mbid:
                continue
            if artist_constraint and artist_constraint.casefold() not in artist.casefold():
                continue
            # For an explicit distinct-song request, suppress different
            # recordings of the same title so the model sees real variety.
            identity = (artist.casefold(), title.casefold())
            if identity in seen:
                continue
            seen.add(identity)
            entry = {
                "mbid": mbid,
                "title": title,
                "artist": artist,
                "score": item.get("score"),
                "length_ms": item.get("length"),
            }
            self.searched_recordings[mbid] = entry
            recordings.append(entry)
            if len(recordings) >= (20 if target_minutes else 10):
                break
        if not recordings:
            raise ServiceError(
                "resource",
                "No MusicBrainz recordings matched the requested artist. "
                "Try another explicit artist search or discovery source.",
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
        ready = abs(best - target) <= 60
        return {"mbids": list(states[best]), "minutes": round(best / 60, 2),
                "target_minutes": target_minutes, "difference_seconds": best - target,
                "ready": ready, "remaining_seconds": max(0, target - best),
                "omitted_unknown_length": omitted,
                "note": (
                    "Selection is within +/-1 minute of the target. "
                    "Instrumental status is not validated by duration."
                    if ready else
                    "NOT READY: too far from the requested duration. "
                    "Search for additional songs, verify their MBIDs, then "
                    "select_playlist_duration again. DO NOT create a shorter playlist."
                )}

    def tool_list_playlists(self): return self.lb.playlists()
    def tool_get_playlist(self, playlist_mbid): return self.lb.playlist(playlist_mbid)

    def tool_filter_recent_candidates(self, days):
        candidates = self._exclude(self.last_candidates, days)
        return {"tracks": candidates, "excluded": len(self.last_candidates) - len(candidates),
                "history_complete": True}

    def tool_create_playlist(self, title, mbids, exclude_days=0):
        # Never let the LLM turn a read-only request into an account write.
        if not ("playlist" in self.goal.casefold() and re.search(
            r"\b(create|make|build|save)\b", self.goal, re.IGNORECASE
        )):
            raise ServiceError("policy_halt", "Current user request did not explicitly authorize playlist creation")

        title = str(title).strip()
        if not title or len(title) > 180:
            raise ServiceError("invalid", "Playlist title must be 1–180 characters")
        if not title.casefold().startswith("[encore]"):
            title = "[Encore] " + title

        cleaned = self._verified_mbids(mbids)
        if len(set(cleaned)) != len(cleaned):
            raise ServiceError("invalid", "Playlist recordings must have unique MBIDs")

        # An MBID can be real while referring to the wrong artist or to a
        # second recording of the same song. Validate both before approval.
        target_artist = self._requested_playlist_artist()
        distinct_requested = bool(re.search(r"\bdistinct\b", self.goal, re.I))
        titles = []
        display_tracks = []
        for mbid in cleaned:
            indexed = self.searched_recordings.get(mbid, {})
            verified = self.verified[mbid]
            title_name = indexed.get("title") or verified.get("title")
            if not title_name:
                raise ServiceError("invalid", f"No title was resolved for {mbid}")
            titles.append(str(title_name).strip().casefold())
            if target_artist:
                credits = verified.get("artist_credit") or []
                confirmed_artist = indexed.get("artist") or "".join(
                    x.get("name", "") + x.get("joinphrase", "")
                    for x in credits if isinstance(x, dict)
                )
                if target_artist.casefold() not in confirmed_artist.casefold():
                    raise ServiceError(
                        "invalid",
                        f"Recording {mbid} has no confirmed artist credit matching "
                        f"{target_artist!r}. Search by artist first.",
                    )
        if distinct_requested and len(set(titles)) != len(titles):
            raise ServiceError(
                "invalid", "Distinct recordings requested, but the selection "
                "contains duplicate song titles. Select different songs.",
            )

        # Check the requested track count *before* any approval or POST.
        expected = re.search(
            r"\b(\d{1,2})\s+(?:[\w-]+\s+){0,5}(?:songs?|tracks?|recordings?)\b",
            self.goal, re.IGNORECASE
        )
        if expected and len(cleaned) != int(expected.group(1)):
            raise ServiceError("invalid", f"User requested {expected.group(1)} recordings, got {len(cleaned)}")

        if exclude_days:
            candidates = [{"recording_mbids": [mbid],
                           "title": self.verified[mbid].get("title"),
                           "artist": "".join(x.get("name", "") for x in
                                             self.verified[mbid].get("artist_credit", [])
                                             if isinstance(x, dict))}
                          for mbid in cleaned]
            filtered = self._exclude(candidates, exclude_days)
            if len(filtered) != len(candidates):
                raise ServiceError("invalid", "A selected recording was played in the exclusion window")

        # Enforce the ORIGINAL user's duration target even if the model
        # skipped or ignored select_playlist_duration. Never approve a
        # 7:55 playlist in response to a 15-minute creation request.
        self._enforce_playlist_duration(cleaned)

        # Avoid duplicate remote playlists, including those created in earlier sessions.
        existing = self._find_existing_playlist(title)
        if existing:
            raise ServiceError("resource", f"Playlist {title!r} already exists. Read it instead of creating another.")

        def perform_create():
            created = self.lb.create_playlist(title, cleaned)  # ONE POST only
            playlist_mbid = created.get("playlist_mbid") or created.get("mbid")
            if not playlist_mbid:
                # The POST succeeded but the response did not expose its identifier.
                # Do not blindly POST again; retrieve by title on a later run.
                return {"created": True, "confirmed": False, "title": title,
                        "track_count_requested": len(cleaned),
                        "warning": "Creation accepted, but no playlist MBID returned; inspect ListenBrainz."}

            output = {"created": True, "confirmed": False, "title": title,
                      "playlist_mbid": playlist_mbid, "track_count_requested": len(cleaned),
                      "url": f"https://listenbrainz.org/playlist/{playlist_mbid}"}
            if not callable(getattr(self.lb, "playlist", None)):
                output["warning"] = "Creation accepted; no readback method available for confirmation."
                return output
            try:
                detail = self.lb.playlist(playlist_mbid)
                payload = detail.get("payload", detail)
                playlist = payload.get("playlist", {})
                if not isinstance(playlist, dict):
                    raise ValueError("Unexpected playlist detail format")
                actual = [x for t in playlist.get("track", [])
                          for x in mbids_from_track(t) if isinstance(t, dict)]
                output["confirmed"] = actual == cleaned
                output["track_count_observed"] = len(actual)
                if not output["confirmed"]:
                    output["warning"] = "Created, but readback does not match requested recordings. Inspect manually."
            except (ServiceError, ValueError, TypeError, AttributeError, KeyError):
                output["warning"] = "Created, but confirmation read failed. Do not submit another create request."
            return output

        return self._write("create_playlist", {"title": title, "mbids": cleaned,
                                               "exclude_days": exclude_days}, perform_create)

    def _playlist_edit_snapshot(self, playlist_mbid: str) -> dict:
        """Only the playlist detail endpoint includes actual tracks.

        User playlist listings return JSPF metadata with track:[] even when
        the playlist has recordings. Never derive offsets from that listing.
        """
        playlist_mbid = check_uuid(playlist_mbid)
        detail = self.lb.playlist(playlist_mbid)
        payload = detail.get("payload", detail)
        entry = payload.get("playlist") if isinstance(payload, dict) else None
        if not isinstance(entry, dict):
            raise ServiceError("resource", "Couldn't load the full playlist detail; no write performed")
        tracks = entry.get("track")
        if not isinstance(tracks, list):
            raise ServiceError("resource", "Playlist detail omitted its track list; no write performed")
        ids = [mbids_from_track(item) if isinstance(item, dict) else [] for item in tracks]
        if any(len(item) != 1 for item in ids):
            raise ServiceError("resource", "Cannot identify every playlist track by MBID; editing blocked")

        # The user must own the playlist and name the target explicitly.
        creator = entry.get("creator")
        user = getattr(self.lb, "username", None)
        if creator and user and str(creator).casefold() != str(user).casefold():
            raise ServiceError("policy_halt", "Cannot edit a playlist owned by another user")
        match = re.search(
            r"\b(?:to|from|in)\s+(?:my\s+)?(.+?)\s+playlist\b",
            self.goal, re.I,
        )
        if not match:
            raise ServiceError("resource", "Please name the existing playlist to edit")
        requested = re.sub(r"^\[Encore\]\s*", "", match.group(1).strip(), flags=re.I)
        actual = re.sub(r"^\[Encore\]\s*", "", str(entry.get("title", "")).strip(), flags=re.I)
        if requested.casefold() != actual.casefold():
            raise ServiceError("invalid", f"Requested playlist {requested!r}, but selected {actual!r}")
        return {"mbid": playlist_mbid, "title": entry["title"],
                "tracks": tracks, "mbids": [item[0] for item in ids]}

    def _require_edit_request(self, action: str):
        low = self.goal.casefold()
        patterns = {
            "add_playlist_tracks": r"\b(add|append|insert)\b",
            "remove_playlist_tracks": r"\b(remove|delete)\b",
        }
        if not ("playlist" in low and re.search(patterns[action], low)):
            raise ServiceError("policy_halt", f"User did not request {action}; no write performed")

    def tool_add_playlist_tracks(self, playlist_mbid, mbids):
        self._require_edit_request("add_playlist_tracks")
        snapshot = self._playlist_edit_snapshot(playlist_mbid)
        cleaned = self._verified_mbids(mbids)
        if len(set(cleaned)) != len(cleaned):
            raise ServiceError("invalid", "Requested add contains duplicate MBIDs")
        if set(cleaned) & set(snapshot["mbids"]):
            raise ServiceError("resource", "One or more tracks are already in this playlist; nothing added")

        expected = re.search(r"\badd\s+(\d{1,2})\s+", self.goal, re.I)
        if expected and len(cleaned) != int(expected.group(1)):
            raise ServiceError("invalid", f"User requested {expected.group(1)} tracks; got {len(cleaned)}")

        # Require requested artist identity, not just a valid recording MBID.
        target = self._requested_playlist_artist()
        titles = []
        display_tracks = []
        for mbid in cleaned:
            indexed = self.searched_recordings.get(mbid, {})
            verified = self.verified[mbid]
            artist = indexed.get("artist") or "".join(
                x.get("name", "") + x.get("joinphrase", "")
                for x in verified.get("artist_credit", []) if isinstance(x, dict)
            )
            if target and target.casefold() not in artist.casefold():
                raise ServiceError("invalid", f"{mbid} is not confirmed as a {target} recording")
            display_title = indexed.get("title") or verified.get("title") or mbid
            titles.append(display_title)
            display_tracks.append(f"{display_title} — {artist or 'artist unconfirmed'}")

        # When the user names an individual song, do not allow an arbitrary
        # song from the same artist to be substituted.
        requested_song = re.search(
            r"\badd\s+(.+?)\s+by\s+.+?\s+to\s+(?:my\s+)?",
            self.goal, re.I,
        )
        if requested_song:
            target_title = self._key("", requested_song.group(1))[1]
            if len(cleaned) != 1 or self._key("", titles[0])[1] != target_title:
                raise ServiceError("invalid", "Selected recording title does not match the requested song")

        if "distinct" in self.goal.casefold() and len({x.casefold() for x in titles}) != len(titles):
            raise ServiceError("invalid", "Requested distinct tracks have duplicate titles")
        # Detect same song as a different MBID where the detail carries names.
        existing_titles = {self._key(x.get("creator"), x.get("title"))
                           for x in snapshot["tracks"] if isinstance(x, dict)}
        if any(self._key(
            self.searched_recordings.get(mbid, {}).get("artist"), title
        ) in existing_titles for mbid, title in zip(cleaned, titles)):
            raise ServiceError("resource", "A song with the same artist and title is already present")

        description = (f"ADD {len(cleaned)} tracks to {snapshot['title']} "
                       f"(currently {len(snapshot['mbids'])} tracks):\n" +
                       "\n".join(f"{i}. {title}" for i, title in enumerate(display_tracks, 1)) +
                       "\nThis modifies your real ListenBrainz account.")
        before = snapshot["mbids"]

        def perform_add():
            server = self.lb.add_tracks(snapshot["mbid"], cleaned)
            result = {"status": server.get("status", "ok"), "playlist_mbid": snapshot["mbid"],
                      "added": len(cleaned), "confirmed": False}
            try:
                after = self._playlist_edit_snapshot(snapshot["mbid"])["mbids"]
                result["confirmed"] = (after == before + cleaned)
                result["track_count_observed"] = len(after)
            except (ServiceError, ValueError, KeyError, TypeError):
                result["warning"] = "Write accepted, but readback unavailable; reconcile before retrying"
            if not result["confirmed"] and "warning" not in result:
                result["warning"] = "Readback differs; don't blindly repeat the write"
            return result

        return self._write("add_playlist_tracks",
                           {"playlist_mbid": snapshot["mbid"], "mbids": cleaned},
                           perform_add, summary=description)

    def tool_remove_playlist_tracks(self, playlist_mbid, index, count):
        self._require_edit_request("remove_playlist_tracks")
        snapshot = self._playlist_edit_snapshot(playlist_mbid)
        before = snapshot["mbids"]
        if not before:
            raise ServiceError("resource", "Playlist is already empty; nothing removed")
        if re.search(r"\blast\s+(?:song|track|recording)\b", self.goal, re.I):
            # Ignore the LLM's index; last means last in the REAL detail.
            index, count = len(before) - 1, 1
        elif re.search(r"\bfirst\s+(?:song|track|recording)\b", self.goal, re.I):
            index, count = 0, 1
        elif not re.search(r"\b(?:at\s+)?index\s+\d+\b", self.goal, re.I):
            raise ServiceError("resource", "Specify first/last track or an explicit zero-based index")
        if index < 0 or count < 1 or index + count > len(before):
            raise ServiceError("invalid", f"Track range {index}:{index+count} is outside {len(before)} tracks")
        selected = snapshot["tracks"][index:index + count]
        selected_info = [f"{entry.get('title') or mbid} — {entry.get('creator') or 'artist unavailable'}"
                         for entry, mbid in zip(selected, before[index:index + count])]
        description = (f"REMOVE {count} track(s) from {snapshot['title']} "+
                       f"at zero-based index {index}:\n" +
                       "\n".join(selected_info) +
                       f"\nPlaylist currently contains {len(before)} tracks. "+
                       "This modifies your real ListenBrainz account.")
        expected = before[:index] + before[index + count:]

        def perform_remove():
            server = self.lb.remove_tracks(snapshot["mbid"], index, count)
            result = {"status": server.get("status", "ok"), "playlist_mbid": snapshot["mbid"],
                      "removed_count": count, "removed_index": index, "confirmed": False}
            try:
                after = self._playlist_edit_snapshot(snapshot["mbid"])["mbids"]
                result["confirmed"] = after == expected
                result["track_count_observed"] = len(after)
            except (ServiceError, ValueError, KeyError, TypeError):
                result["warning"] = "Write accepted, but readback unavailable; reconcile before retrying"
            if not result["confirmed"] and "warning" not in result:
                result["warning"] = "Readback differs; don't blindly repeat the write"
            return result

        return self._write("remove_playlist_tracks",
                           {"playlist_mbid": snapshot["mbid"], "index": index, "count": count},
                           perform_remove, summary=description)

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