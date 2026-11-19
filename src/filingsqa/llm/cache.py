"""Disk cache for LLM responses: identical request + identical model settings -> stored reply.

Purpose: keep development and repeated eval runs cheap and reproducible. The key covers
everything that can change the output (provider, model, effort, prompts, schema), so editing
the prompt automatically misses the cache. See ADR-013.
"""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from filingsqa.llm.base import LLMClient, LLMRequest, LLMResponse


def cache_key(identity: dict[str, str], request: LLMRequest) -> str:
    payload = json.dumps({"llm": identity, "request": request.cache_payload()}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CachingLLM:
    def __init__(self, inner: LLMClient, cache_dir: Path) -> None:
        self._inner = inner
        self._dir = cache_dir
        self.hits = 0
        self.misses = 0

    @property
    def identity(self) -> dict[str, str]:
        return self._inner.identity

    def complete(self, request: LLMRequest) -> LLMResponse:
        key = cache_key(self.identity, request)
        path = self._dir / key[:2] / f"{key}.json"
        if path.exists():
            self.hits += 1
            data = json.loads(path.read_text(encoding="utf-8"))
            return replace(LLMResponse(**data), cached=True)
        self.misses += 1
        response = self._inner.complete(request)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(response), indent=2), encoding="utf-8")
        return response
