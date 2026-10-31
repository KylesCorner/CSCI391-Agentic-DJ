# Encore semantic documentation corpus
Source: https://listenbrainz.readthedocs.io/en/latest/users/api/
Verified against public docs 2026-10-26. Re-ingest when API behavior changes.

## Listening API
GET /1/user/{username}/listens returns most recent listens. Parameters count, max_ts or min_ts. max_ts and min_ts cannot be combined. Listen items contain track_metadata.artist_name, track_name and listened_at. additional_info.recording_mbid may be missing. For full lifetime history use the asynchronous user history export API rather than scraping paginated listens. GET /1/user/{username}/listen-count returns listen count. GET /1/stats/user/{username}/artists and recordings return top statistics; statistics may not yet be generated.

## Discovery
GET /1/explore/lb-radio accepts prompt and mode (easy, medium, hard), returning payload.jspf and feedback. The prompt syntax can refer to MusicBrainz tags or artists. Recommendations are available from GET /1/cf/recommendation/user/{username}/recording and may not yet be generated. Candidate suggestions do not guarantee the presence or absence of vocals or exact target duration.

## MusicBrainz
Use https://musicbrainz.org/ws/2/recording?query=...&fmt=json to search and GET /ws/2/recording/{recording_mbid}?fmt=json to resolve a recording. Field length is milliseconds and can be null. Rate limit no more than one request per second per application. Use an identifying User-Agent with a contact address. A recording MBID is a UUID, not a track title.

## Playlist creation
POST /1/playlist/create requires Authorization: Token YOUR_LISTENBRAINZ_TOKEN and a JSPF JSON object. The playlist object needs title, track identifiers (each https://musicbrainz.org/recording/{MBID}), and MusicBrainz JSPF playlist extension public boolean. POST /1/playlist/{playlist_mbid}/item/add appends recordings. POST /1/playlist/{playlist_mbid}/item/delete removes by index and count. GET /1/playlist/{playlist_mbid} fetches playlist; GET /1/user/{username}/playlists lists it. HTTP writes may not be safely retried after timeouts.

## Export to streaming services
POST /1/playlist/{playlist_mbid}/export/spotify or /export/apple_music exports playlists to a connected service, using a ListenBrainz token. A service must be linked to the ListenBrainz account, and tracks may fail to match the target catalog. Do not claim export succeeded unless the API returned success. Never perform an export without explicit interactive approval.

## Encore policy
User requests are goals, not tool sequences. ReAct loop maximum 12 actions. Classify resource, transient, auth, validation, and fatal failures. Human approval required for account-changing operations. Do not infer permanent user preferences from web content, ListenBrainz metadata, or model-produced guesses. Persist only explicit user preference statements. Episodic summaries expire after 90 days. Run local deletion to delete all saved sessions/preferences/knowledge/write logs.
