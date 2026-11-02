"""Network clients. GET may retry transient errors; writes NEVER retry automatically."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from collections import Counter
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import time
from uuid import UUID
from urllib.parse import quote
import requests

from .transport import Http, ServiceError, _response_json

LB = "https://api.listenbrainz.org/1"
MB = "https://musicbrainz.org/ws/2"


def check_uuid(value: str) -> str:
    try:
        return str(UUID(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"Invalid MBID UUID: {value!r}") from exc


def mbids_from_track(track: dict) -> list[str]:
    value = track.get("identifier", [])
    items = value if isinstance(value, list) else [value]
    result = []
    for item in items:
        if isinstance(item, str) and "/recording/" in item:
            try:
                result.append(check_uuid(item.rsplit("/", 1)[-1]))
            except ValueError:
                pass
    return result


class ListenBrainz:
    def __init__(self, http: Http, username: str, token: str = "", timezone_name: str = "UTC"):
        self.http, self.username, self.token = http, username, token
        if not username:
            raise ValueError("Set LISTENBRAINZ_USER in .env")
        try:
            self.timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown ENCORE_TIMEZONE: {timezone_name}") from exc
        self.user = quote(username, safe="")

    def get(self, path: str, params=None) -> dict:
        # Listening-history endpoints are measured slow reads (15s allowance).
        timeout_s = 15 if "/listens" in f"/{path.lstrip('/')}" else 5
        return self.http.request("GET", f"{LB}/{path.lstrip('/')}",
                                 token=self.token, params=params, timeout_s=timeout_s)

    def post(self, path: str, body=None, params=None) -> dict:
        if not self.token:
            raise ServiceError("auth", "LISTENBRAINZ_TOKEN required for writes")
        return self.http.request("POST", f"{LB}/{path.lstrip('/')}", token=self.token, params=params, body=body)

    def recent(self, count=20) -> list[dict]:
        raw = self.get(f"user/{self.user}/listens", {"count": min(100, max(1, count))})
        return [self._listen(x) for x in raw.get("payload", {}).get("listens", [])]

    @staticmethod
    def _listen(item: dict) -> dict:
        track = item.get("track_metadata") or {}
        info = track.get("additional_info") or {}
        return {
            "artist": track.get("artist_name", ""),
            "title": track.get("track_name", ""),
            "timestamp": item.get("listened_at"),
            "recording_mbid": info.get("recording_mbid"),
        }

    def listens_since(self, days=7, max_pages=10) -> dict:
        """Bounded recent history. Explicitly reports if the window was incomplete."""
        now = int(datetime.now(timezone.utc).timestamp())
        cutoff = now - int(days * 86400)
        before = now + 1
        found = []
        complete = False
        for _ in range(max_pages):
            raw = self.get(f"user/{self.user}/listens", {"count": 100, "max_ts": before})
            batch = [self._listen(x) for x in raw.get("payload", {}).get("listens", [])]
            if not batch:
                complete = True
                break
            times = [x["timestamp"] for x in batch if isinstance(x["timestamp"], int)]
            found.extend(x for x in batch if isinstance(x["timestamp"], int) and x["timestamp"] >= cutoff)
            if len(batch) < 100 or (times and min(times) < cutoff):
                complete = True
                break
            if not times or min(times) >= before:
                break
            before = min(times)
        return {"listens": found, "complete": complete, "days": days, "cutoff_utc": cutoff,
                "note": "Do NOT claim exclusions are complete" if not complete else "Window fully checked"}

    def count(self) -> dict:
        return self.get(f"user/{self.user}/listen-count").get("payload", {})

    def _recent_window(self, from_ts: int, *, max_pages: int = 10) -> dict:
        """Bounded scan from a timestamp. Never silently present truncated rankings as complete."""
        before = int(datetime.now(timezone.utc).timestamp()) + 1
        listens = []
        for _ in range(max_pages):
            raw = self.get(f"user/{self.user}/listens", {"count": 100, "max_ts": before})
            batch = [self._listen(x) for x in (raw.get("payload", {}).get("listens") or [])]
            if not batch:
                return {"listens": listens, "complete": True}
            times = [x["timestamp"] for x in batch if isinstance(x.get("timestamp"), int)]
            if not times:
                raise ServiceError("invalid", "Listens response has no usable timestamps")
            listens.extend(x for x in batch if x["timestamp"] >= from_ts)
            if len(batch) < 100 or min(times) < from_ts:
                return {"listens": listens, "complete": True}
            # max_ts is exclusive; always advance backwards to prevent a stalled scan.
            if min(times) >= before:
                raise ServiceError("resource", "Listen pagination did not advance")
            before = min(times)
        return {"listens": listens, "complete": False}

    def _fallback_top_this_month(self, kind: str, count: int) -> dict:
        now = datetime.now(self.timezone)
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        from_ts = int(start.timestamp())
        history = self._recent_window(from_ts)
        listens = history["listens"]
        if kind == "artists":
            tally = Counter(x["artist"] for x in listens if x["artist"])
            entries = [{"artist_name": name, "listen_count": total}
                       for name, total in tally.most_common(count)]
        else:
            tally = Counter((x["artist"], x["title"]) for x in listens
                            if x["artist"] and x["title"])
            entries = [{"artist_name": artist, "recording_name": title, "listen_count": total}
                       for (artist, title), total in tally.most_common(count)]
        return {
            kind: entries,
            "range": "this_month",
            "source": "raw_scrobbles_fallback",
            "timezone": str(self.timezone),
            "from_ts": from_ts,
            "to_ts": int(now.timestamp()),
            "listens_counted": len(listens),
            "complete": history["complete"],
            "note": ("Counted actual scrobbles for this calendar month."
                     if history["complete"] else
                     "PARTIAL: bounded scan stopped before reaching start of month; rankings may change."),
        }

    def top(self, kind="artists", count=10, span="this_month") -> dict:
        allowed = {"this_week", "this_month", "this_year", "week", "month",
                   "quarter", "year", "half_yearly", "all_time"}
        if kind not in {"artists", "recordings"}:
            raise ValueError("Unsupported statistics type")
        if span not in allowed:
            raise ValueError(f"Invalid ListenBrainz range: {span!r}")
        payload = self.get(f"stats/user/{self.user}/{kind}",
                           {"count": count, "range": span}).get("payload")
        if isinstance(payload, dict) and payload.get(kind):
            return {**payload, "source": "listenbrainz_statistics",
                    "note": "Precomputed ListenBrainz statistics; may lag recent scrobbles."}
        # 204 No Content is normalized to {}, so missing/empty payload is not a success.
        if span == "this_month":
            return self._fallback_top_this_month(kind, count)
        raise ServiceError("resource", f"ListenBrainz {kind} statistics unavailable for {span}; "
                           "select this_month for recent-listen fallback or try later")

    def playing_now(self) -> dict:
        return self.get(f"user/{self.user}/playing-now").get("payload", {})

    def recommendations(self, count=20) -> dict:
        return self.get(f"cf/recommendation/user/{self.user}/recording", {"count": count}).get("payload", {})

    def radio(self, prompt: str, mode: str) -> dict:
        payload = self.get("explore/lb-radio", {"prompt": prompt, "mode": mode}).get("payload", {})
        jspf = payload.get("jspf", {})
        playlist = jspf.get("playlist", {})
        tracks = playlist.get("track", [])
        return {"title": playlist.get("title"), "tracks": [
            {"artist": t.get("creator"), "title": t.get("title"), "duration_ms": t.get("duration"),
             "recording_mbids": mbids_from_track(t)} for t in tracks[:100]
        ], "feedback": payload.get("feedback", [])}

    def get_feedback(self, score: int | None = None, count: int = 25, offset: int = 0) -> dict:
        """Read explicit ListenBrainz love/hate feedback, not recommendation feedback."""
        if score not in (None, -1, 1):
            raise ValueError("Feedback filter must be -1, 1, or omitted")
        params = {"count": count, "offset": offset, "metadata": "true"}
        if score is not None:
            params["score"] = score
        return self.get(f"feedback/user/{self.user}/get-feedback", params)

    def set_feedback(self, recording_mbid: str, score: int) -> dict:
        if score not in (-1, 0, 1):
            raise ValueError("Feedback score must be -1, 0, or 1")
        return self.post("feedback/recording-feedback", {
            "recording_mbid": check_uuid(recording_mbid), "score": score,
        })

    def fresh_releases(self, days: int = 30, sort: str = "release_date") -> dict:
        if not 1 <= days <= 90:
            raise ValueError("days must be 1..90")
        if sort not in {"release_date", "artist_credit_name", "release_name", "confidence"}:
            raise ValueError("Invalid release sorting field")
        return self.get(f"user/{self.user}/fresh_releases", {
            "days": days, "sort": sort, "past": "true", "future": "true",
        })

    def top_recordings_for_artist(self, artist_mbid: str) -> dict:
        return self.get(f"popularity/top-recordings-for-artist/{check_uuid(artist_mbid)}")

    def delete_playlist(self, playlist_mbid: str) -> dict:
        # Endpoint is POST /playlist/{mbid}/delete, not HTTP DELETE.
        return self.post(f"playlist/{check_uuid(playlist_mbid)}/delete")

    def playlists(self, count=20, offset=0) -> dict:
        return self.get(f"user/{self.user}/playlists", {"count": count, "offset": offset})

    def playlist(self, mbid: str) -> dict:
        return self.get(f"playlist/{check_uuid(mbid)}")

    @staticmethod
    def jspf(title: str, mbids: list[str], *, public: bool = False) -> dict:
        return {"playlist": {
            "title": title,
            "extension": {"https://musicbrainz.org/doc/jspf#playlist": {"public": public}},
            "track": [{"identifier": f"https://musicbrainz.org/recording/{check_uuid(x)}"} for x in mbids],
        }}

    def create_playlist(self, title: str, mbids: list[str]) -> dict:
        return self.post("playlist/create", self.jspf(title, mbids, public=False))

    def add_tracks(self, playlist_mbid: str, mbids: list[str]) -> dict:
        """Append verified recording MBIDs using JSPF without an empty title."""
        playlist_id = check_uuid(playlist_mbid)
        if not mbids:
            raise ValueError("Cannot add an empty list of recordings")
        body = {
            "playlist": {
                "track": [
                    {"identifier": [f"https://musicbrainz.org/recording/{check_uuid(mbid)}"]}
                    for mbid in mbids
                ]
            }
        }
        return self.post(f"playlist/{playlist_id}/item/add", body)

    def remove_tracks(self, playlist_mbid: str, index: int, count: int) -> dict:
        return self.post(f"playlist/{check_uuid(playlist_mbid)}/item/delete", {"index": index, "count": count})

    def export(self, playlist_mbid: str, service: str) -> dict:
        if service not in ("spotify", "apple_music"):
            raise ValueError("service must be spotify or apple_music")
        return self.post(f"playlist/{check_uuid(playlist_mbid)}/export/{service}", params={"is_public": "false"})


class MusicBrainz:
    def __init__(self, http: Http, contact: str):
        if not contact:
            raise ValueError("Set MB_CONTACT in .env (email or project URL) before catalog lookups")
        self.http = http
        self.agent = f"Encore-Educational/0.1 ({contact})"

    def get(self, path: str, params=None) -> dict:
        old = self.http.user_agent
        try:
            self.http.user_agent = self.agent
            return self.http.request("GET", f"{MB}/{path.lstrip('/')}",
                                     params={**(params or {}), "fmt": "json"}, mb=True)
        finally:
            self.http.user_agent = old

    def search(self, query: str, limit=30) -> dict:
        return self.get("recording", {"query": query, "limit": limit})

    def search_artist(self, query: str, limit=10) -> dict:
        return self.get("artist", {"query": query, "limit": limit})

    def lookup_artist(self, mbid: str) -> dict:
        # artist-rels retrieves member-of-band / collaboration relationships.
        return self.get(f"artist/{check_uuid(mbid)}", {"inc": "artist-rels"})

    def lookup(self, mbid: str) -> dict:
        return self.get(f"recording/{check_uuid(mbid)}")

    def verify(self, mbids: list[str]) -> dict:
        valid, invalid = [], []
        for value in dict.fromkeys(mbids):
            try:
                item = self.lookup(value)
                valid.append({"mbid": value, "title": item.get("title"),
                              "duration_ms": item.get("length"),
                              "artist_credit": item.get("artist-credit", [])})
            except ValueError as exc:
                invalid.append({"mbid": value, "reason": str(exc)})
            except ServiceError as exc:
                if exc.kind != "not_found":
                    # An outage/auth failure never proves a recording is invalid.
                    raise
                invalid.append({"mbid": value, "reason": str(exc)})
        return {"valid": valid, "invalid": invalid}
