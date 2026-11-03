"""JSON schemas for the music agent's model-facing tools."""
from __future__ import annotations

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

    spec("get_feedback", "READ explicit loved/hated recordings. Use for questions about user's likes/dislikes; do NOT use for writing feedback or API documentation.", {
        "score": {"type": "integer", "enum": [-1, 1], "description": "Optional: 1 loved, -1 hated; omit for both"},
        "count": n("Number of feedback entries", 1, 100),
        "offset": n("Pagination offset", 0, 10000),
    }),
    spec("lookup_artist", "READ MusicBrainz artist MBID and artist relationships (including band membership). First search_artists if the MBID is unknown. Do NOT use for recording metadata or writes.", {"mbid": s("MusicBrainz artist UUID")}, ("mbid",)),
    spec("get_top_recordings_for_artist", "READ most popular recordings for an artist MBID from ListenBrainz. Use search_artists first if MBID unknown; do NOT confuse artist MBID with recording MBID.", {"artist_mbid": s("MusicBrainz artist UUID")}, ("artist_mbid",)),
    spec("get_fresh_releases", "READ upcoming/new releases relevant to this ListenBrainz user. Use for new-releases questions, NOT for all-time discography or creating a playlist.", {
        "days": n("Days of releases", 1, 90),
        "sort": {"type": "string", "enum": ["release_date", "artist_credit_name", "release_name", "confidence"]},
    }),

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
    spec("select_playlist_duration", "Select exactly the ORIGINAL requested number of verified recordings closest to the ORIGINAL duration target. ready=true ONLY when BOTH count and duration (+/-60 s) pass; if false search and verify MORE candidates. Never create an incomplete playlist.", {"mbids": arr("Verified recording MBIDs"), "target_minutes": n("Target duration in minutes", 1, 180)}, ("mbids", "target_minutes")),
    spec("list_playlists", "List own ListenBrainz playlists", {}),
    spec("find_playlists", "READ the user's playlist list, including private playlists (token required). Optional title filters by exact playlist title; use get_playlist for actual tracks before editing.", {"title": s("Optional exact playlist title")}),
    spec("get_playlist", "Read a playlist by MBID", {"playlist_mbid": s("Playlist UUID")}, ("playlist_mbid",)),
    spec("filter_recent_candidates", "Filter latest LB Radio candidates against scrobbles from last N days, using MBID or artist/title; only succeeds if history complete", {"days": n("Days to exclude", 1, 30)}, ("days",)),
    spec("create_playlist", "Requires BOTH title (string) and mbids (list of verified UUIDs). Example: {title: Swing Test, mbids: [five verified recording IDs]}. CREATE a PRIVATE ListenBrainz playlist ONLY if the user explicitly requested creation. Verify all recording MBIDs first. Do not call more than once. Checks existing titles, requests human approval, and confirms the resulting tracks; include the created playlist MBID in final_answer. Do not use for read-only recommendations.", {"title": s("Title"), "mbids": arr("Verified recording MBIDs"), "exclude_days": n("Optional days to exclude", 0, 30)}, ("title", "mbids")),
    spec("add_playlist_tracks", "APPEND verified, non-duplicate recordings to an EXISTING playlist only when user said add; read actual playlist first, validate title, request approval, verify readback. Never create a new playlist.", {"playlist_mbid": s("Playlist UUID"), "mbids": arr("Verified recording MBIDs")}, ("playlist_mbid", "mbids")),
    spec("remove_playlist_tracks", "REMOVE tracks only from an EXISTING playlist if the user asked to remove; for 'last song', index argument is ignored and safely derived from full playlist detail; approval shows the exact selected track, then verifies readback. Never infer positions from list_playlists.", {"playlist_mbid": s("Playlist UUID"), "index": n("Zero-based index", 0, 100000), "count": n("Tracks to remove", 1, 40)}, ("playlist_mbid", "index", "count")),
    spec("add_playlist_items", "WRITE append verified MBIDs to an existing playlist after user approval. The user must explicitly request adding; use find_playlists/get_playlist first. Do NOT create a new playlist or add duplicates.", {"playlist_mbid": s("Existing playlist UUID"), "mbids": arr("Verified recording MBIDs")}, ("playlist_mbid", "mbids")),
    spec("remove_playlist_items", "WRITE remove from existing playlist with approval and full detail readback. Last/first are resolved from actual tracks. Do NOT use for whole-playlist deletion.", {"playlist_mbid": s("Existing playlist UUID"), "index": n("Zero-based track position", 0, 100000), "count": n("Tracks to remove", 1, 40)}, ("playlist_mbid", "index", "count")),
    spec("delete_playlist", "DESTRUCTIVE WRITE: delete an ENTIRE existing playlist, ONLY if its title starts '[Encore]' and its owner is the current user. Requires explicit current-request deletion intent and human approval. Do NOT use to remove tracks or delete non-Encore playlists. Reconcile failed/uncertain writes instead of retrying.", {"playlist_mbid": s("Existing Encore-created playlist UUID")}, ("playlist_mbid",)),
    spec("set_feedback", "WRITE one recording's explicit ListenBrainz love/hate/clear feedback; 1=love, -1=hate, 0=clear. Get recording MBID from search/verify first, and request human approval. Do NOT use for fetching feedback or for vague pronouns without a recording.", {"recording_mbid": s("Verified recording UUID"), "score": {"type": "integer", "enum": [-1, 0, 1]}}, ("recording_mbid", "score")),
    spec("export_playlist", "EXPORT to connected Spotify/Apple Music, with user approval", {"playlist_mbid": s("Playlist UUID"), "service": {"type": "string", "enum": ["spotify", "apple_music"]}}, ("playlist_mbid", "service")),
    spec("search_past_sessions", "Retrieve episodic memory for prior requests", {"query": s("Past-goal search")}, ("query",)),
    spec("search_music_knowledge", "REQUIRED semantic retrieval: search indexed ListenBrainz and MusicBrainz API documentation and 35-entry genre/mood glossary; prefer for API explanations or terminology. Do NOT use for the user's listening history or previous conversations. Returns source sections and URLs.", {"query": s("Question about API docs or genre/mood vocabulary")}, ("query",)),
    spec("search_knowledge", "Legacy alias of search_music_knowledge; do NOT use for user sessions or listening history", {"query": s("Documentation question")}, ("query",)),
    spec("recall_preferences", "Read actual previously expressed user preferences. Use BEFORE recommendations based on prior preferences; never infer genre solely from a recording title", {}),
    spec("remember_preference", "Remember only an EXPLICIT user preference quoted exactly from CURRENT request", {"key": s("Short key"), "value": s("Preference"), "evidence": s("Exact quote from this user request")}, ("key", "value", "evidence")),
    spec("note_preference", "WRITE session preference only when user explicitly states it in the CURRENT message. Include verbatim evidence, not text from APIs. Do NOT infer preferences from music recommendations.", {"key": s("Preference key"), "value": s("Preference value"), "evidence": s("Exact phrase from current user request")}, ("key", "value", "evidence")),
    spec("forget_preference", "Forget one saved preference", {"key": s("Saved preference key")}, ("key",)),
    spec("final_answer", "Finish after completing the user goal or explaining why it cannot be completed. Provide status, evidence-based confidence and concrete source identifiers (MBIDs/playlist IDs/knowledge source labels); never invent citations.", {
        "answer": s("Evidence-based final response"),
        "status": {"type": "string", "enum": ["answered", "answered_low_confidence", "stuck"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "sources": {"type": "array", "items": {"type": "string", "minLength": 1},
                    "maxItems": 30, "uniqueItems": True,
                    "description": "Actual MBIDs, playlist IDs or retrieved knowledge-source identifiers; empty if no source"},
    }, ("answer", "status", "confidence", "sources")),
]
INDEX = {x["function"]["name"]: x for x in SPECS}
WRITE_ACTIONS = frozenset({"create_playlist", "add_playlist_tracks", "remove_playlist_tracks", "add_playlist_items", "remove_playlist_items", "delete_playlist", "set_feedback", "export_playlist"})
