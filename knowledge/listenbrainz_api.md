# ListenBrainz API — Encore reference notes

These are concise educational summaries of the official API documentation, not verbatim copies. Check the linked sources before relying on a behavior that may change.

## Authentication tokens
Source: https://listenbrainz.readthedocs.io/en/latest/users/api/
ListenBrainz issues personal user tokens. Account-write calls require `Authorization: Token <user token>`. Do not log the token, insert it in an LLM prompt, or persist it in a memory collection. Public listening data generally permits unauthenticated reads. An invalid token can lead to HTTP 401; a permission issue can lead to 403.

## ListenBrainz rate limit headers
Source: https://listenbrainz.readthedocs.io/en/latest/users/api/
Respect at most one request per second per application. Responses include `X-RateLimit-Remaining` and `X-RateLimit-Reset-In` to show capacity and how many seconds remain in the rate window. HTTP 429 indicates too many requests. The tool executor, not the model, handles timed retries and rate-limit waits.

## Recent listening history
Source: https://listenbrainz.readthedocs.io/en/latest/users/api/core.html
A user's scrobbles are retrieved from `/1/user/{user_name}/listens`. A listen may contain artist and recording names, timestamp, and MusicBrainz mapping metadata when available. A request to exclude songs heard within seven days needs coverage of the complete relevant time window, not merely the latest page.

## User listening statistics
Source: https://listenbrainz.readthedocs.io/en/latest/users/api/
ListenBrainz has listening statistics, including top artists and recordings. The user-specific statistics endpoints are read-oriented and may lag recent listening events. When asking for the current calendar month, do not silently substitute a previous-month aggregate.

## Collaborative recommendations
Source: https://listenbrainz.readthedocs.io/en/latest/users/api/
ListenBrainz can recommend recordings based on collaborative listening signals. Recording recommendations are candidates, not proof of genre, instrumental status, mood, or similarity to a particular artist. Confirm MusicBrainz recording identifiers before playlist writes.

## Connected music services
Source: https://listenbrainz.readthedocs.io/en/latest/users/api/core.html
The service list endpoint `/1/user/{user_name}/services` reports connected music services. A playlist export to Spotify or Apple Music may require an eligible linked service and separate approval. A successful ListenBrainz playlist creation does not imply that it exists on an external streaming platform.
