# MusicBrainz API — Encore reference notes

## API root and formats
Source: https://musicbrainz.org/doc/MusicBrainz_API
The MusicBrainz web service root is `https://musicbrainz.org/ws/2/`. It has search, lookup, and browse operations for artists, recordings, releases, release groups, works, and related entities. Use `fmt=json` to request JSON rather than the default XML.

## MusicBrainz rate limit and User-Agent
Source: https://musicbrainz.org/doc/MusicBrainz_API/Rate_Limiting
A client should make no more than about one request per second on average per public IP, and identify its application and contact in the User-Agent. Requests above the current rate can yield HTTP 503; follow bounded backoff in the executor rather than letting the LLM retry rapidly.

## Recording search artist field
Source: https://musicbrainz.org/doc/MusicBrainz_API/Search
MusicBrainz searches use Lucene-style fielded syntax. In the recording search index, `artist:"Benny Goodman"` targets the credited performer; `artistname` matches individual artist names, and `arid` matches an artist MBID. A bare `Benny Goodman` is likely to search recording names instead of guaranteeing the performer.

## Recording title search field
Source: https://musicbrainz.org/doc/MusicBrainz_API/Search
Use `recording:"Sing, Sing, Sing"` to search for a particular recording title. Combine fields with `AND`, for example `artist:"Benny Goodman" AND recording:"Sing, Sing, Sing"`. Lucene special characters may require escaping beyond URL encoding.

## Lookup a recording by MBID
Source: https://musicbrainz.org/doc/MusicBrainz_API
`GET /ws/2/recording/{recording_mbid}?fmt=json` looks up a known recording. A valid MBID is a UUID. The lookup may provide title, length in milliseconds, and artist credit; an unknown length must not be guessed to satisfy a playlist duration target.

## Artist relationships and artist-rels
Source: https://musicbrainz.org/doc/MusicBrainz_API
`GET /ws/2/artist/{artist_mbid}?inc=artist-rels&fmt=json` includes artist-to-artist relationships that may identify memberships and collaborations. Read relationship types carefully; searching artists by name alone does not establish historical band membership.

## Releases and release-groups
Source: https://musicbrainz.org/doc/MusicBrainz_API
A release is a particular issued edition; a release-group collects related editions of an album or other issue. An artist lookup can include `release-groups`. Release-group metadata is useful for discographies but does not alone establish the most recent streaming-platform availability.

## Browse versus search versus lookup
Source: https://musicbrainz.org/doc/MusicBrainz_API
Search accepts a textual Lucene query. Lookup uses a known entity MBID. Browse enumerates entities directly linked to another known entity and supports limit and offset pagination. Do not confuse a recording title search with a browse of an artist's recordings.

## Tags and genres
Source: https://musicbrainz.org/doc/MusicBrainz_API
MusicBrainz can include genres and tags through supported `inc=` parameters. Tags can reflect community labeling, and sparse or missing tags should not be treated as a definitive negative classification. Metadata must be interpreted as evidence, not as a guarantee of audio properties.
