"""Reproducible, pre-network API fault injection for evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import random

class FaultInjector:
    def __init__(self, transient_rate: float, *, seed: int = 0,
                 max_steps: int | None = None, reporter=None):
        if isinstance(transient_rate, bool) or not 0 <= transient_rate <= 1:
            raise ValueError("transient_rate must be in [0, 1]")
        if max_steps is not None and (isinstance(max_steps, bool) or not 1 <= max_steps <= 12):
            raise ValueError("max_steps must be an integer in 1..12")
        self.transient_rate = float(transient_rate)
        self.seed = seed
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self._n = 0
        self.reporter = reporter

    @classmethod
    def load(cls, path: str | Path, *, reporter=None) -> "FaultInjector":
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Injection config must be a JSON object")
        if set(config) - {"transient_rate", "seed", "max_steps"}:
            raise ValueError("Unknown injection config fields: " + ", ".join(sorted(set(config) - {"transient_rate", "seed", "max_steps"})))
        seed = config.get("seed", 0)
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        return cls(config.get("transient_rate", 0.0), seed=seed,
                   max_steps=config.get("max_steps"), reporter=reporter)

    def before_request(self, method: str, url: str) -> int | None:
        """Return HTTP status to inject, or None. Never access the network."""
        if not url.startswith(("https://api.listenbrainz.org/1/", "https://musicbrainz.org/ws/2/")):
            return None
        self._n += 1
        if self._rng.random() >= self.transient_rate:
            return None
        status = 429 if self._n % 2 else 503
        if self.reporter:
            self.reporter("fault_injected", method=method.upper(),
                          service="musicbrainz" if "musicbrainz.org" in url else "listenbrainz",
                          http_status=status, attempt=self._n, seed=self.seed)
        return status
