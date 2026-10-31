# ListenBrainz playlists — Encore reference notes

## Create playlist in JSPF
Source: https://listenbrainz.readthedocs.io/en/latest/users/api/playlist.html
`POST /1/playlist/create` creates a playlist. Its body is a JSPF playlist containing a nonempty title and a `track` array. Track identifiers refer to MusicBrainz recording UUIDs via `https://musicbrainz.org/recording/{MBID}`. Account authorization is required. Encore must obtain human approval before a real write.

## Add items to playlist
Source: https://listenbrainz.readthedocs.io/en/latest/users/api/playlist.html
`POST /1/playlist/{playlist_mbid}/item/add` appends one or more JSPF tracks to an existing playlist. The optional offset path variant inserts at a specified position. A track identifier must designate a MusicBrainz recording; an empty playlist title is not required in the add-items body. Reject duplicate recordings when the user's request prohibits duplicates.

## Remove items from playlist
Source: https://listenbrainz.readthedocs.io/en/latest/users/api/playlist.html
ListenBrainz offers playlist item removal. Encore should first fetch full playlist detail to resolve positional references such as 'last song'. An array in a user's playlist list response is not reliable evidence of track contents. Show the exact targeted recording before the approval prompt and confirm the change with a detail readback.

## Retrieve full playlist detail
Source: https://listenbrainz.readthedocs.io/en/latest/users/api/playlist.html
`GET /1/playlist/{playlist_mbid}` retrieves playlist content, including its track list in JSPF format. Listing a user's playlists may include lightweight metadata without embedded recording detail. Use the detail endpoint to calculate playlist position, track count, and post-write verification.

## Export playlist
Source: https://listenbrainz.readthedocs.io/en/latest/users/api/playlist.html
`POST /1/playlist/{playlist_mbid}/export/{service}` requests export to a connected music service. Such a write requires valid account authorization, an eligible connected service, and explicit human approval. A failed export should not be presented as success or retried blindly.

## Playlist ownership and privacy
Source: https://listenbrainz.readthedocs.io/en/latest/users/api/playlist.html
A playlist can have public or private status. Editing requires authorization from its owner or an eligible collaborator. Encore's `[Encore]` prefix acts as a project ownership marker; mass deletion must never target playlists outside that marker. Private status alone is not an explanation for all HTTP 400 errors.
