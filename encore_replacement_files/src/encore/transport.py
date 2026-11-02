"""Shared HTTP transport: throttling, failure categories and trace reporting.

Read failures can be retried by the agent; write timeouts remain uncertain.
"""
from __future__ import annotations

import time
import requests
from .faults import FaultInjector

class ServiceError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def _response_json(r: requests.Response) -> dict:
    try:
        obj = r.json()
        return obj if isinstance(obj, dict) else {"items": obj}
    except ValueError as exc:
        raise ServiceError("fatal", "Non-JSON response from external API") from exc


class Http:
    """Single-attempt HTTP transport. The ReAct loop owns transient retries.

    Read timeouts: 5 s by default; listening history gets 15 s. Writes: 15 s.
    A write never retries automatically because it may have succeeded remotely.
    """
    def __init__(self, user_agent: str, *, injector: FaultInjector | None = None,
                 reporter=None, mb_min_interval: float = 1.1,
                 lb_min_interval: float = 1.0):
        self.session = requests.Session()
        self.user_agent = user_agent
        self.last_mb_request = 0.0
        self.last_lb_request = 0.0
        self.lb_wait_until = 0.0
        self.mb_min_interval = mb_min_interval
        self.lb_min_interval = lb_min_interval
        self.injector = injector
        self.reporter = reporter

    def request(self, method: str, url: str, *, token: str = "", params=None,
                body=None, mb=False, timeout_s: float | None = None) -> dict:
        method = method.upper()
        is_write = method not in {"GET", "HEAD"}
        service = "musicbrainz" if mb else "listenbrainz"
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Token {token}"

        # A failed injected request happens BEFORE the network, including writes.
        # Never retry a write, even after a synthetic failure.
        if self.injector:
            injected = self.injector.before_request(method, url)
            if injected is not None:
                raise ServiceError("transient", f"Injected HTTP {injected} ({service}; no network call)")

        now = time.monotonic()
        if mb:
            remaining = self.last_mb_request + self.mb_min_interval - now
            if remaining > 0:
                time.sleep(remaining)
            self.last_mb_request = time.monotonic()
        else:
            wait_until = max(self.lb_wait_until, self.last_lb_request + self.lb_min_interval)
            remaining = wait_until - now
            if remaining > 0:
                time.sleep(remaining)
            self.last_lb_request = time.monotonic()

        effective_timeout = timeout_s if timeout_s is not None else (15 if is_write else 5)
        start = time.monotonic()
        try:
            response = self.session.request(
                method, url, params=params, json=body, headers=headers,
                timeout=(5, effective_timeout),
            )
        except requests.Timeout as exc:
            if self.reporter:
                self.reporter("http_request", service=service, method=method,
                              success=False, error_category="uncertain_write" if is_write else "transient",
                              latency_ms=round((time.monotonic() - start)*1000))
            raise ServiceError("uncertain_write" if is_write else "transient",
                               "Network timeout; write outcome uncertain" if is_write else "Network timeout") from exc
        except requests.RequestException as exc:
            if self.reporter:
                self.reporter("http_request", service=service, method=method,
                              success=False, error_category="uncertain_write" if is_write else "transient",
                              latency_ms=round((time.monotonic() - start)*1000))
            raise ServiceError("uncertain_write" if is_write else "transient", "Network connection failed") from exc

        if not mb:
            # Honor ListenBrainz's window reset and server Retry-After; no busy retry.
            if response.headers.get("X-RateLimit-Remaining") == "0":
                try:
                    reset = float(response.headers.get("X-RateLimit-Reset-In", "1"))
                    self.lb_wait_until = max(self.lb_wait_until, time.monotonic()+max(0,reset))
                except (ValueError, TypeError):
                    self.lb_wait_until = max(self.lb_wait_until, time.monotonic()+1)
            if response.status_code == 429 and response.headers.get("Retry-After"):
                try:
                    self.lb_wait_until = max(self.lb_wait_until,
                                             time.monotonic()+max(0,float(response.headers["Retry-After"])))
                except (TypeError, ValueError):
                    pass

        if self.reporter:
            self.reporter("http_request", service=service, method=method,
                          http_status=response.status_code, success=bool(response.ok),
                          latency_ms=round((time.monotonic() - start)*1000))
        if not response.ok:
            if response.status_code in (401,403):
                kind="auth"
            elif response.status_code == 404:
                kind="not_found"
            elif response.status_code == 429 or response.status_code >= 500:
                kind="uncertain_write" if is_write else "transient"
            else:
                kind="invalid"
            # Do not log/return remote response bodies: they may contain personal data.
            raise ServiceError(kind, f"External API returned HTTP {response.status_code}")
        if response.status_code == 204 or not response.content:
            return {}
        return _response_json(response)
