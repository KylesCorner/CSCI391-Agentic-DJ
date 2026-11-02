"""Playlist creation, editing, readback and deletion safeguards.

Keep approval, idempotency, artist verification, duplicate checks and
ambiguous-write handling unchanged from the original Tools class.
"""
from __future__ import annotations

import re

from .clients import ServiceError, check_uuid, mbids_from_track


class PlaylistToolsMixin:

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
        expected_count = self._requested_playlist_track_count()
        if expected_count is not None and len(cleaned) != expected_count:
            raise ServiceError(
                "invalid",
                f"User requested {expected_count} recordings, got {len(cleaned)}. "
                "Do not create a playlist until BOTH track count and duration match.",
            )

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

    def _playlist_edit_snapshot(self, playlist_mbid: str, *, require_named_match: bool = True) -> dict:
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
        if require_named_match:
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

    def tool_add_playlist_items(self, playlist_mbid, mbids):
        return self.tool_add_playlist_tracks(playlist_mbid, mbids)

    def tool_remove_playlist_items(self, playlist_mbid, index, count):
        return self.tool_remove_playlist_tracks(playlist_mbid, index, count)

    def tool_delete_playlist(self, playlist_mbid):
        """Delete an explicitly named Encore-owned playlist, with human approval."""
        goal = self.goal.casefold()
        if not re.search(r"\b(delete|remove)\b", goal) or not re.search(r"\bplaylists?\b", goal):
            raise ServiceError("policy_halt", "User did not request whole-playlist deletion")
        if re.search(r"\b(?:including|even|also)\b.{0,50}\b(?:not\s+(?:made|created|owned)\s+by\s+encore|non[- ]?encore|other\s+people|everyone)\b", goal):
            raise ServiceError("policy_halt", "Deletion beyond Encore-created playlists is forbidden")
        playlist_mbid = check_uuid(playlist_mbid)
        raw = self.lb.playlist(playlist_mbid)
        entry = raw.get("payload", raw).get("playlist") if isinstance(raw, dict) else None
        if not isinstance(entry, dict):
            raise ServiceError("resource", "Could not inspect actual playlist; no deletion attempted")
        title = str(entry.get("title", "")).strip()
        if not title.startswith("[Encore] "):
            raise ServiceError("policy_halt", "Only Encore-created playlists may be deleted")
        # A named user request must identify the actual playlist. For an 'all'
        # request, all candidate playlists must still individually pass owner
        # and marker checks, each with its own approval.
        if not re.search(r"\b(all|every|each)\s+(?:my\s+)?playlists?\b", goal):
            cleaned_title = title.removeprefix("[Encore] ").casefold().strip()
            if cleaned_title and cleaned_title not in goal and title.casefold() not in goal:
                raise ServiceError("invalid", "Requested playlist name does not match the target")
        owner = str(entry.get("creator", "")).strip()
        user = str(getattr(self.lb, "username", "")).strip()
        if not owner or not user or owner.casefold() != user.casefold():
            raise ServiceError("policy_halt", "Playlist ownership could not be confirmed")
        tracks = entry.get("track")
        if not isinstance(tracks, list):
            raise ServiceError("resource", "Playlist detail omitted track count; deletion blocked")
        description = (f"DELETE ENTIRE PLAYLIST: {title}\n"
                       f"Owner: {owner}\nTracks removed from playlist: {len(tracks)}\n"
                       "This permanently deletes the real ListenBrainz playlist (not the recordings themselves).")
        def perform_delete():
            self.lb.delete_playlist(playlist_mbid)
            result = {"playlist_mbid": playlist_mbid, "title": title,
                      "deleted": True, "confirmed": False}
            try:
                self.lb.playlist(playlist_mbid)
                result["warning"] = "Delete accepted, but playlist still readable; reconcile before retrying"
            except ServiceError as exc:
                if exc.kind == "not_found":
                    result["confirmed"] = True
                else:
                    result["warning"] = "Delete accepted but readback unavailable; check account before retrying"
            return result
        return self._write("delete_playlist", {"playlist_mbid": playlist_mbid},
                           perform_delete, summary=description)
