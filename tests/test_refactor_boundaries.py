"""Check public compatibility after splitting implementation across modules.

These tests make no network calls or account writes.
"""
from encore.agent import (
    Ollama, route, compact_observation, requested_discovery_count,
    positive_verification_claim,
)
from encore.ollama_client import Ollama as ModelClient
from encore.agent_policies import route as policy_route
from encore.clients import Http, ServiceError, ListenBrainz, MusicBrainz
from encore.transport import Http as TransportHttp, ServiceError as TransportError
from encore.tools import Tools, SPECS, INDEX, WRITE_ACTIONS
from encore.tool_specs import SPECS as RegistrySpecs, INDEX as RegistryIndex
from encore.playlist_tools import PlaylistToolsMixin


def test_public_import_paths_still_work():
    assert Ollama is ModelClient
    assert route is policy_route
    assert Http is TransportHttp
    assert ServiceError is TransportError
    assert SPECS is RegistrySpecs
    assert INDEX is RegistryIndex
    assert WRITE_ACTIONS
    assert callable(compact_observation)
    assert callable(requested_discovery_count)
    assert callable(positive_verification_claim)
    assert ListenBrainz and MusicBrainz


def test_playlist_methods_remain_on_tools():
    assert issubclass(Tools, PlaylistToolsMixin)
    for name in (
        "create_playlist", "add_playlist_tracks", "remove_playlist_tracks",
        "delete_playlist", "add_playlist_items", "remove_playlist_items",
    ):
        assert callable(getattr(Tools, f"tool_{name}"))
