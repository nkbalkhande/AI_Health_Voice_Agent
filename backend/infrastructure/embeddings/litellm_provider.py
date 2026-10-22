import asyncio
import time
from typing import Any

import litellm
from litellm import aembedding
from litellm import embedding as sync_embedding

from backend.core.config import settings
from backend.core.logger import logger
from backend.domain.interfaces.embedding_provider import EmbeddingProvider
from backend.infrastructure.litellm.model_resolver import (
    azure_kwargs,
    resolve_embedding_model,
)

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken optional / offline
    _ENCODER = None


log = logger.getChild("infrastructure.embeddings.litellm")

# R-71: this provider runs on every query (embed_query) and on every semantic
# chunking batch at ingest, and had no retry at all - one transient 429 aborted
# the query and its eval item. Mirrors _TRANSIENT_EXCEPTIONS in
# app/infrastructure/llm/litellm_provider.py.
_TRANSIENT_EXCEPTIONS = (
    litellm.RateLimitError,
    litellm.Timeout,
    litellm.APIConnectionError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
)
_EMBED_MAX_ATTEMPTS = 3


def _count_tokens(text: str) -> int:
    if _ENCODER is not None:
        return len(_ENCODER.encode(text, disallowed_special=()))
    return max(1, len(text) // 4)


def _vectors_from_response(response: Any) -> list[list[float]]:
    data = sorted(response.data, key=lambda item: item["index"])
    return [item["embedding"] for item in data]


class LiteLLMEmbedding(EmbeddingProvider):
    """
    Unified embeddings via LiteLLM.

    Set embedding.model to a full LiteLLM id (e.g. openai/text-embedding-3-small).
    """

    def __init__(self) -> None:
        self._model = resolve_embedding_model(
            settings.embedding_provider, settings.embedding_model
        )
        self._extra_kwargs = settings.litellm_embedding_kwargs
        self._max_tokens_per_request = settings.embedding_max_tokens_per_request
        self._max_batch_size = settings.embedding_max_batch_size
        self._timeout = settings.embedding_timeout

    @property
    def model_name(self) -> str:
        return self._model

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.aembed_documents([text])
        return vectors[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.aembed_documents(texts)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await self._aembed(texts)

    def embed_query_sync(self, text: str) -> list[float]:
        return self.embed_documents_sync([text])[0]

    def embed_documents_sync(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed_sync(texts)

    def _build_batches(self, texts: list[str]) -> list[list[str]]:
        batches: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0
        for text in texts:
            tokens = _count_tokens(text)
            exceeds_tokens = (
                current and current_tokens + tokens > self._max_tokens_per_request
            )
            exceeds_count = len(current) >= self._max_batch_size
            if exceeds_tokens or exceeds_count:
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(text)
            current_tokens += tokens
        if current:
            batches.append(current)
        return batches

    def _request_kwargs(self, batch: list[str]) -> dict[str, Any]:
        # R-87: litellm.request_timeout defaults to 6000s, so a stalled Azure
        # connection froze semantic chunking indefinitely with no log line and
        # never reached the retry below.
        return {
            "model": self._model,
            "input": batch,
            "timeout": self._timeout,
            **azure_kwargs(self._model),
            **self._extra_kwargs,
        }

    async def _aembed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for batch in self._build_batches(texts):
            response = None
            for attempt in range(_EMBED_MAX_ATTEMPTS):
                try:
                    response = await aembedding(**self._request_kwargs(batch))
                    break
                except _TRANSIENT_EXCEPTIONS as exc:
                    if attempt == _EMBED_MAX_ATTEMPTS - 1:
                        raise
                    delay = 0.5 * 2**attempt
                    log.warning(
                        "Transient embedding error (%s); retry %d/%d in %.1fs.",
                        exc,
                        attempt + 1,
                        _EMBED_MAX_ATTEMPTS - 1,
                        delay,
                    )
                    await asyncio.sleep(delay)
            embeddings.extend(_vectors_from_response(response))
        return embeddings

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Sync path for LangChain callers running inside worker threads."""
        embeddings: list[list[float]] = []
        for batch in self._build_batches(texts):
            response = None
            for attempt in range(_EMBED_MAX_ATTEMPTS):
                try:
                    response = sync_embedding(**self._request_kwargs(batch))
                    break
                except _TRANSIENT_EXCEPTIONS as exc:
                    if attempt == _EMBED_MAX_ATTEMPTS - 1:
                        raise
                    delay = 0.5 * 2**attempt
                    log.warning(
                        "Transient embedding error (%s); retry %d/%d in %.1fs.",
                        exc,
                        attempt + 1,
                        _EMBED_MAX_ATTEMPTS - 1,
                        delay,
                    )
                    time.sleep(delay)
            embeddings.extend(_vectors_from_response(response))
        return embeddings
