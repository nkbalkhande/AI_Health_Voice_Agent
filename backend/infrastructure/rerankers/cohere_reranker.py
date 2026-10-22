import asyncio
import time
from collections import deque

import litellm
from litellm import arerank

from backend.core.config import settings
from backend.core.logger import logger
from backend.domain.entities.chunk import Chunk
from backend.domain.interfaces.reranker_provider import RerankerProvider

litellm.suppress_debug_info = True
log = logger.getChild("infrastructure.rerankers.cohere")

_WINDOW_SECONDS = 60.0
_RERANK_MAX_ATTEMPTS = 3


class _RateLimiter:
    """Sliding-window limiter: at most ``max_calls`` per 60s, process-wide.

    Each rerank request counts as one Cohere API call. When the window is full
    we sleep until the oldest call ages out, so calls are spaced out instead of
    being batched together.
    """

    def __init__(self, max_calls: int) -> None:
        self._max_calls = max(1, max_calls)
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= _WINDOW_SECONDS:
                    self._calls.popleft()

                if len(self._calls) < self._max_calls:
                    self._calls.append(now)
                    return

                wait = _WINDOW_SECONDS - (now - self._calls[0])
                log.info(
                    "Cohere rerank rate limit reached (%d/min); waiting %.1fs",
                    self._max_calls,
                    wait,
                )
                await asyncio.sleep(max(wait, 0.0))


class CohereReranker(RerankerProvider):
    _rate_limiter = _RateLimiter(settings.reranker_rate_limit_per_min)

    def __init__(self) -> None:
        self._model = settings.reranker_model
        self._api_key = settings.reranker_api_key

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def api_key(self) -> str:
        return self._api_key

    async def rank(
        self,
        query: str,
        chunks: list[Chunk],
        *,
        top_k: int | None = None,
    ) -> list[Chunk]:
        if not chunks:
            return []

        chunks_content = [chunk.content for chunk in chunks]

        # R-41: one 429/timeout used to propagate and kill the whole query
        # (and its eval item). The in-process rate limiter only guards our own
        # budget; it cannot prevent provider-side throttling.
        await self._rate_limiter.acquire()
        responses = None
        last_exc: Exception | None = None
        for attempt in range(_RERANK_MAX_ATTEMPTS):
            try:
                responses = await arerank(
                    model=self._model,
                    query=query,
                    documents=chunks_content,
                    top_n=top_k,
                    api_key=self._api_key,
                )
                break
            except Exception as exc:  # noqa: BLE001 - 429 / 5xx / timeout
                last_exc = exc
                log.warning(
                    "Cohere rerank attempt %d/%d failed: %s",
                    attempt + 1,
                    _RERANK_MAX_ATTEMPTS,
                    exc,
                )
                if attempt < _RERANK_MAX_ATTEMPTS - 1:
                    await asyncio.sleep(2**attempt)
        if responses is None:
            raise last_exc  # type: ignore[misc]
        ranked_chunks = []
        for response in responses['results']:
            relevance = float(response['relevance_score'])
            # With fusion enabled the identical relevance gate runs post-rank
            # in fuse_retrieval_and_rerank; skip here so it lives in one place.
            if (
                not settings.score_fusion_enabled
                and settings.reranker_threshold is not None
                and relevance < settings.reranker_threshold
            ):
                continue
            chunk = chunks[response['index']]
            chunk.score = relevance
            ranked_chunks.append(chunk)
        return ranked_chunks
