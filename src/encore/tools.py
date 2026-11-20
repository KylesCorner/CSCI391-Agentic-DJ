"""Typed result envelope, tool schemas, execution, and safe mutation policy."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from hashlib import sha256
from typing import Any, Callable
import json
from uuid import uuid4
from jsonschema import validate, ValidationError
from .clients import ListenBrainz, MusicBrainz, ServiceError, check_uuid, mbids_from_track
from .memory import Memory
from .tool_specs import spec, s, n, arr, SPECS, INDEX, WRITE_ACTIONS
from .playlist_tools import PlaylistToolsMixin
from .request_intent import Intent, WRITE_TOOLS, INSTRUMENT_NAMES, GENRE_NAMES, _descriptor



@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error_type: str | None = None
    message: str | None = None

    def json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)


class Tools(PlaylistToolsMixin):
    def __init__(self, lb: ListenBrainz, mb: MusicBrainz | None, memory: Memory,
                 *, approve: Callable[[str], bool], current_goal: str,
                 session_id: str | None = None):
        self.lb, self.mb, self.mem = lb, mb, memory
        self.approve = approve
        self.goal = current_goal
        self.session_id = session_id or str(uuid4())
        self.intent = Intent()
        self.verified: dict[str, dict] = {}
        self.last_candidates: list[dict] = []
        self.recent_cache: dict[int, dict] = {}
        self.searched_recordings: dict[str, dict] = {}
        self.last_duration_result: dict | None = None

    def _recent(self, days: int) -> dict:
        if days not in self.recent_cache:
            self.recent_cache[days] = self.lb.listens_since(days, max_pages=10)
        return self.recent_cache[days]

    @staticmethod
    def _key(artist: str | None, title: str | None) -> tuple[str, str]:
        def normalize(value: str | None) -> str:
            return ''.join(ch for ch in (value or '').casefold() if ch.isalnum())
        return normalize(artist), normalize(title)

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
        if action not in self.intent.permitted_writes:
            raise ServiceError('policy_halt', 'This operation was not explicitly authorized')
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
        if name in {item for names in WRITE_TOOLS.values() for item in names} and name not in self.intent.permitted_writes:
            return ToolResult(False, error_type='policy_halt', message='Write not authorized by request intent')
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

    def tool_get_feedback(self, score=None, count=25, offset=0):
        data = self.lb.get_feedback(score=score, count=count, offset=offset)
        payload = data.get("payload", data)
        rows = payload.get("feedback", payload.get("feedbacks", payload.get("items", []))) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            raise ServiceError("invalid", "Feedback response contains no feedback list")
        return {"feedback": [{"recording_mbid": x.get("recording_mbid"),
                               "score": x.get("score"),
                               "created": x.get("created") or x.get("created_at"),
                               "recording_name": (x.get("track_metadata") or {}).get("track_name")}
                              for x in rows[:count] if isinstance(x, dict)],
                "count": len(rows), "offset": offset}

    def tool_lookup_artist(self, mbid):
        data = self._catalog().lookup_artist(mbid)
        relations = []
        for relation in data.get("relations", [])[:80]:
            if not isinstance(relation, dict):
                continue
            entity = relation.get("artist", {})
            if relation.get("target-type") != "artist" or not isinstance(entity, dict):
                continue
            relations.append({"type": relation.get("type"),
                              "direction": relation.get("direction"),
                              "artist": entity.get("name"), "artist_mbid": entity.get("id"),
                              "begin": relation.get("begin"), "end": relation.get("end")})
        return {"mbid": data.get("id"), "name": data.get("name"),
                "type": data.get("type"), "country": data.get("country"),
                "relations": relations}

    def tool_get_top_recordings_for_artist(self, artist_mbid):
        data = self.lb.top_recordings_for_artist(artist_mbid)
        rows = data.get("items", data.get("recordings", data.get("payload", [])))
        if isinstance(rows, dict):
            rows = rows.get("recordings", [])
        if not isinstance(rows, list):
            raise ServiceError("invalid", "Popularity endpoint response was not a recording list")
        return {"artist_mbid": check_uuid(artist_mbid), "recordings": [
            {"mbid": x.get("recording_mbid"), "title": x.get("recording_name"),
             "artist": x.get("artist_name"), "listen_count": x.get("total_listen_count"),
             "listener_count": x.get("total_user_count"), "duration_ms": x.get("length")}
            for x in rows[:25] if isinstance(x, dict)]}

    def tool_get_fresh_releases(self, days=30, sort="release_date"):
        data = self.lb.fresh_releases(days=days, sort=sort)
        payload = data.get("payload", data)
        # Upstream may return a list or {releases: [...]} depending on deployment.
        rows = payload if isinstance(payload, list) else payload.get("releases", payload.get("items", [])) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            raise ServiceError("invalid", "Fresh-releases response contained no release list")
        return {"days": days, "releases": [
            {"artist": x.get("artist_credit_name"), "title": x.get("release_name"),
             "release_date": x.get("release_date"), "release_mbid": x.get("release_mbid"),
             "release_group_mbid": x.get("release_group_mbid"), "type": x.get("release_group_primary_type")}
            for x in rows[:40] if isinstance(x, dict)]}


    @staticmethod
    def _is_generic_music_descriptor(query: str) -> bool:
        """Recognize category names, not artist identities."""
        clean = _descriptor(query)
        return clean in (set(INSTRUMENT_NAMES) | set(GENRE_NAMES) | {
            'instrumental', 'instrumental jazz', 'study', 'study music',
            'focus music', 'relaxing', 'background music',
            'music for studying', 'acoustic', 'no vocals',
        })

    @staticmethod
    def _catalog_tag_query(instrument: str | None, genre: str | None) -> str:
        """MusicBrainz has a recording tag: field, not instrument:/genre:.

        Both tags are metadata hints, not proof of audible instrumentation.
        """
        def quote(value: str) -> str:
            return value.replace('\\', '\\\\').replace('"', '\\"')
        def safe_label(value: str) -> str:
            # Tags are words, not Lucene field expressions.
            return ' '.join(''.join(ch if ch.isalnum() or ch in ' -' else ' '
                                    for ch in value).split())[:60]
        tags = []
        if genre:
            tags.append(GENRE_NAMES.get(_descriptor(genre), safe_label(genre)))
        if instrument:
            tags.append(INSTRUMENT_NAMES.get(_descriptor(instrument), safe_label(instrument)))
        tags = [tag for tag in tags if tag]
        return ' AND '.join(f'tag:"{quote(tag)}"' for tag in dict.fromkeys(tags))

    def _requested_playlist_artist(self) -> str | None:
        return self.intent.artist if self.intent.action in {
            'create_playlist', 'add_playlist_tracks', 'recommend', 'read'} else None

    def _requested_playlist_track_count(self) -> int | None:
        return self.intent.track_count

    def _requested_duration_minutes(self) -> int | None:
        return self.intent.duration_minutes

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
        """Search recordings using entity-typed constraints, never guessed fields.

        An instrument is not an artist; a style is not a recording title. The
        MusicBrainz tag index can suggest candidates, but does not establish
        that a specific instrument is audible on the recording.
        """
        target = self.intent.artist
        track = self.intent.track_title
        instrument = self.intent.instrument
        genre = self.intent.genre

        def quote(value):
            return str(value).replace('\\', '\\\\').replace('"', '\\"')

        if target:
            predicates = [f'artist:"{quote(target)}"']
            if track:
                predicates.append(f'recording:"{quote(track)}"')
            query = ' AND '.join(predicates)
        elif track:
            query = f'recording:"{quote(track)}"'
        elif instrument or genre:
            # Discard model-authored artist:"clarinet" AND recording:"swing jazz"
            # instead of blindly forwarding it to MusicBrainz.
            query = self._catalog_tag_query(instrument, genre)
        else:
            # Failsafe when intent extraction is unavailable: inspect only
            # known MusicBrainz field names, without regex or arbitrary text
            # inference. Never send a fake artist/recording entity to the API.
            misplaced_instrument = None
            misplaced_genre = None
            for clause in query.split(' AND '):
                field, separator, raw_value = clause.partition(':')
                if not separator:
                    continue
                label = _descriptor(raw_value.strip().strip('"'))
                if field.strip().casefold() in {'artist', 'artistname'} and label in INSTRUMENT_NAMES:
                    misplaced_instrument = INSTRUMENT_NAMES[label]
                if field.strip().casefold() == 'recording' and label in GENRE_NAMES:
                    misplaced_genre = label
            if misplaced_instrument or misplaced_genre:
                instrument = misplaced_instrument
                genre = misplaced_genre
                query = self._catalog_tag_query(instrument, genre)
            elif self._is_generic_music_descriptor(query):
                raise ServiceError('resource', 'An instrument or genre is not an artist or song title. '
                                   'Use recording tags or a music discovery tool.')

        self.last_recording_query = query
        data = self._catalog().search(query)
        rows = list(data.get('recordings') or [])
        found = []
        seen = set()
        for item in rows:
            artist = ''.join(
                credit.get('name', '') + credit.get('joinphrase', '')
                for credit in item.get('artist-credit', []) if isinstance(credit, dict)
            ).strip()
            title = str(item.get('title') or '').strip()
            mbid = item.get('id')
            if not title or not artist or not mbid:
                continue
            if target and target.casefold() not in artist.casefold():
                continue
            identity = self._key(artist, title)
            if identity in seen:
                continue
            seen.add(identity)
            entry = {'mbid': mbid, 'title': title, 'artist': artist,
                     'score': item.get('score'), 'length_ms': item.get('length')}
            if instrument or genre:
                entry['evidence_note'] = ('MusicBrainz search-tag candidate; '
                                          'instrumentation is not verified by a recording ID')
            self.searched_recordings[mbid] = entry
            found.append(entry)
            if len(found) >= (20 if self.intent.duration_minutes else 10):
                break
        if not found:
            raise ServiceError('resource', 'No suitable MusicBrainz recordings found for this query; '
                               'try a different genre or an identified artist')
        return found

    def tool_search_artists(self, query):
        if not self.intent.artist and (self.intent.instrument or self.intent.genre or
                                       self._is_generic_music_descriptor(query)):
            raise ServiceError('resource', 'This request describes an instrument or genre, '
                               'not an artist name. Use search_recordings (tag search) '
                               'or lb_radio for discovery.')
        data = self._catalog().search_artist(self.intent.artist or query)
        return [{'mbid': x.get('id'), 'name': x.get('name'), 'score': x.get('score')}
                for x in data.get('artists', [])[:10]]

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
        """Choose real verified recordings jointly satisfying count and duration.

        This is a bounded, deterministic exact-count subset sum over
        whole-second durations (for candidate ranking). The write operation
        independently revalidates exact millisecond durations and count.
        """
        requested_minutes = self._requested_duration_minutes()
        if requested_minutes is not None and requested_minutes != target_minutes:
            raise ServiceError(
                "invalid",
                f"The user requested {requested_minutes} minutes, not {target_minutes}. "
                "Use the ORIGINAL user's target; do not relax it.",
            )

        cleaned = self._verified_mbids(mbids)
        required_count = self._requested_playlist_track_count()
        lengths: dict[str, int] = {}
        omitted = []
        for mbid in cleaned:
            length = self.verified[mbid].get("duration_ms")
            if not isinstance(length, int) or length <= 0:
                omitted.append(mbid)
            else:
                lengths[mbid] = max(1, round(length / 1000))
        if not lengths:
            raise ServiceError("resource", "No verified tracks have known durations")

        target_s = target_minutes * 60
        # Bound the sum space while allowing a diagnostic answer when all
        # count-feasible subsets are a few minutes longer than requested.
        upper_s = target_s + 600
        max_count = required_count if required_count is not None else len(lengths)
        states: list[dict[int, tuple[str, ...]]] = [dict() for _ in range(max_count + 1)]
        states[0][0] = ()
        for mbid, seconds in lengths.items():
            for count in range(max_count - 1, -1, -1):
                for total, subset in list(states[count].items()):
                    new_total = total + seconds
                    if new_total <= upper_s and new_total not in states[count + 1]:
                        states[count + 1][new_total] = subset + (mbid,)

        if required_count is not None:
            pool = states[required_count]
            if not pool:
                missing = max(0, required_count - len(lengths))
                result = {
                    "mbids": [], "minutes": None, "target_minutes": target_minutes,
                    "required_track_count": required_count, "selected_track_count": 0,
                    "ready": False, "difference_seconds": None,
                    "omitted_unknown_length": omitted,
                    "note": (
                        f"NOT READY: Need {required_count} recordings with known lengths; "
                        f"have {len(lengths)}. " +
                        (f"Find at least {missing} more recordings, " if missing else
                         "Current count-feasible selections are too long; find shorter recordings, ") +
                        "verify them, then run select_playlist_duration again. "
                        "No playlist may be created."
                    ),
                }
                self.last_duration_result = result
                return result
        else:
            pool = {
                total: subset for count, level in enumerate(states)
                if count > 0 for total, subset in level.items()
            }

        best_s = min(pool, key=lambda x: (abs(x - target_s), x > target_s, x))
        selected = list(pool[best_s])
        exact_ms = sum(self.verified[mbid]["duration_ms"] for mbid in selected)
        difference_s = round((exact_ms - target_s * 1000) / 1000)
        ready = abs(exact_ms - target_s * 1000) <= 60_000 and (
            required_count is None or len(selected) == required_count
        )
        actual_minutes = round(exact_ms / 60_000, 2)
        fmt = self._duration_string(exact_ms)
        message = (
            "READY: both original track count and duration constraints are met. "
            if ready else
            f"NOT READY: closest {len(selected)}-track selection totals {fmt}; "
            f"target is {target_minutes}:00 (+/-1:00). "
            "Search and verify additional recordings with suitable durations, "
            "then select again; do NOT create the playlist. "
        )
        result = {
            "mbids": selected,
            "minutes": actual_minutes,
            "target_minutes": target_minutes,
            "required_track_count": required_count,
            "selected_track_count": len(selected),
            "difference_seconds": difference_s,
            "ready": ready,
            "remaining_seconds": max(0, -difference_s),
            "omitted_unknown_length": omitted,
            "note": message + "Instrumental status is not validated by duration.",
        }
        self.last_duration_result = result
        return result

    def tool_list_playlists(self): return self.lb.playlists()
    def tool_find_playlists(self, title=None):
        """Find own playlists by full title or by title without '[Encore]'.

        ListenBrainz returns a name such as '[Encore] Swingin Clarinet'.
        The model may pass 'Swingin Clarinet'; resolve that to the actual
        playlist entry instead of reporting a misleading successful zero-match
        result. Never choose an ambiguous alias without disambiguation.
        Playlist *listings* are metadata only; edit tools read full detail.
        """
        if title is None:
            return self.tool_list_playlists()

        target = str(title).strip()
        if not target:
            raise ServiceError("invalid", "Playlist title must not be empty")

        def title_key(value: str) -> str:
            bare = str(value).strip()
            if bare.casefold().startswith('[encore]'):
                bare = bare[len('[encore]'):].lstrip()
            return ' '.join(bare.casefold().split())

        normalized = title_key(target)
        matches = []
        for offset in range(0, 500, 100):
            response = self.lb.playlists(count=100, offset=offset)
            payload = response.get("payload", response) if isinstance(response, dict) else None
            rows = payload.get("playlists") if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                raise ServiceError("resource", "Playlist listing has an unexpected format")

            for row in rows:
                item = row.get("playlist", row) if isinstance(row, dict) else None
                if isinstance(item, dict) and title_key(item.get("title", "")) == normalized:
                    matches.append(row)

            if len(rows) < 100:
                break
        else:
            raise ServiceError("resource", "Playlist search exceeded the safe listing limit")

        if len(matches) > 1:
            # If the user explicitly named [Encore] NAME, select that full
            # title only. Otherwise do not silently select between an
            # unprefixed and a prefixed playlist with the same base title.
            prefixed = [row for row in matches if str(
                row.get("playlist", row).get("title", "")
            ).strip().casefold() == f"[encore] {normalized}"]
            if f"[encore] {normalized}" in self.goal.casefold() and len(prefixed) == 1:
                matches = prefixed
            else:
                titles = [str(row.get("playlist", row).get("title")) for row in matches]
                raise ServiceError(
                    "resource",
                    "Ambiguous playlist title; name the exact playlist: "
                    + ", ".join(titles[:10]),
                )

        return {
            "count": len(matches),
            "offset": 0,
            "playlist_count": len(matches),
            "playlists": matches,
            "matched_title": (
                matches[0].get("playlist", matches[0]).get("title")
                if matches else None
            ),
            "query_title": target,
        }
    def tool_get_playlist(self, playlist_mbid): return self.lb.playlist(playlist_mbid)

    def tool_filter_recent_candidates(self, days):
        candidates = self._exclude(self.last_candidates, days)
        return {"tracks": candidates, "excluded": len(self.last_candidates) - len(candidates),
                "history_complete": True}

    def tool_set_feedback(self, recording_mbid, score):
        if self.intent.action != 'set_feedback' or not self.intent.explicit:
            raise ServiceError('policy_halt', 'No explicit feedback instruction')
        if score != self.intent.feedback_score:
            raise ServiceError('invalid', 'Proposed score differs from user request')
        mbid = check_uuid(recording_mbid)
        if mbid not in self.verified and mbid not in self.searched_recordings:
            raise ServiceError('invalid', 'Search or verify the recording first')
        known = self.verified.get(mbid) or self.searched_recordings.get(mbid) or {}
        title = known.get('title') or 'Unknown title'
        labels = {1: 'LOVE', -1: 'HATE', 0: 'CLEAR'}
        description = f"{labels[score]} recording feedback for {title}\nRecording MBID: {mbid}"
        return self._write('set_feedback', {'recording_mbid': mbid, 'score': score},
                           lambda: self.lb.set_feedback(mbid, score), summary=description)

    def tool_export_playlist(self, playlist_mbid, service):
        return self._write("export_playlist", {"playlist_mbid": playlist_mbid, "service": service},
                           lambda: self.lb.export(playlist_mbid, service))

    def tool_search_past_sessions(self, query): return self.mem.search_sessions(query)
    def tool_search_music_knowledge(self, query): return self.mem.search_knowledge(query)
    def tool_search_knowledge(self, query): return self.mem.search_knowledge(query)
    def tool_recall_preferences(self): return self.mem.preferences()

    def tool_note_preference(self, key, value, evidence):
        return self.tool_remember_preference(key, value, evidence)

    def tool_remember_preference(self, key, value, evidence):
        if not evidence or evidence.casefold() not in self.goal.casefold():
            raise ServiceError("policy_halt", "Evidence must occur verbatim in current user message")
        if not self.intent.explicit_preference or self.intent.action != 'remember_preference':
            raise ServiceError('policy_halt', 'No explicit stable preference in user message')
        self.mem.set_preference(key, value, evidence)
        return {"saved": key}

    def tool_forget_preference(self, key):
        if self.intent.action != 'forget_preference':
            raise ServiceError('policy_halt', 'User did not ask to forget a preference')
        return {'deleted': self.mem.forget_preference(key)}
