# ListenBrainz discovery and radio — Encore reference notes

## LB Radio explore endpoint
Source: https://listenbrainz.readthedocs.io/en/latest/users/api/misc.html
`GET /1/explore/lb-radio` generates a JSPF playlist and feedback from a prompt. It accepts a required `prompt` and a `mode` parameter. A 400 response indicates a missing or invalid parameter, but may also reflect a prompt the service cannot resolve. Inspect error details rather than retrying arbitrary phrases.

## LB Radio difficulty modes
Source: https://listenbrainz.readthedocs.io/en/latest/users/api/misc.html
LB Radio's mode enum accepts `easy`, `medium`, and `hard`. Easy mode generally stays closer to familiar or broadly matched music; hard mode permits more adventurous discovery. Treat these as discovery settings rather than objective proof of match quality.

## LB Radio tag recording pool
Source: https://listenbrainz.readthedocs.io/en/latest/users/api/core.html
`GET /1/lb-radio/tags` retrieves recordings for MusicBrainz tags. One or more tag parameters may be supplied. For multiple tags the operator parameter controls AND/OR combination. Tag matching provides candidate metadata, not independent proof that every recommendation satisfies an exact musical subgenre.

## Genre matching is not recording verification
Source: https://musicbrainz.org/doc/MusicBrainz_API/Search
A successful MusicBrainz MBID lookup verifies that a recording identity exists. It does not certify that the recording is swing, gypsy jazz, instrumental, relaxing, or appropriate for studying. Genre and mood constraints require additional grounded evidence and uncertainty handling.
