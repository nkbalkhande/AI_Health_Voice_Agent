"""The retrieval half of a RAG turn, in exactly one place.

R-65: ``QueryUseCase`` and ``ChatUseCase`` each carried their own copy of
retrieve → preprocess → rerank → fuse → authority-reorder, and they had already
drifted: chat skipped context compression entirely, never recorded stage
latencies, and did not capture authority metadata. So a question asked through
/chat and the same question asked through /query saw different context — while
every eval number came from the /query path only.

Tuning now happens once, and chat inherits compression, timings and authority
metadata by construction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from backend.core.config import settings
from backend.core.logger import logger
from backend.domain.entities.chunk import Chunk
from backend.domain.interfaces.context_compressor import ContextCompressor
from backend.domain.interfaces.reranker_provider import RerankerProvider
from backend.domain.interfaces.retriever import Retriever
from backend.rag.observability import snapshot_retrieval
from backend.rag.retrieval.authority import (
    authority_reorder_metadata,
    detect_authority,
    reorder_by_authority,
)
from backend.rag.retrieval.score_fusion import fuse_retrieval_and_rerank


def backfill_by_retrieval(
    admitted: list[Chunk],
    candidates: list[Chunk],
    retrieval_by_id: dict[str, dict[str, Any]],
    minimum: int,
) -> list[Chunk]:
    """Top ``admitted`` up to ``minimum`` chunks from the best retrieval scores.

    Admitted chunks keep their place first. Backfilled chunks failed the
    relevance gate (or were never reranked), so they are flagged in metadata
    and score with their retrieval score.
    """
    if len(admitted) >= minimum:
        return admitted
    taken = {c.id for c in admitted}

    def retrieval_score(chunk: Chunk) -> float:
        return float((retrieval_by_id.get(chunk.id) or {}).get("score") or 0.0)

    extra: list[Chunk] = []
    for chunk in sorted(candidates, key=retrieval_score, reverse=True):
        if len(admitted) + len(extra) >= minimum:
            break
        if chunk.id in taken:
            continue
        taken.add(chunk.id)
        chunk.metadata = {**(chunk.metadata or {}), "backfilled": True}
        chunk.score = retrieval_score(chunk)
        extra.append(chunk)
    if extra:
        logger.info(
            "Only %d chunk(s) passed the relevance gate; backfilled %d by retrieval score.",
            len(admitted),
            len(extra),
        )
    return [*admitted, *extra]


@dataclass
class RetrievalOutcome:
    """Everything a use case needs from the retrieval half of a turn."""

    retrieved: list[Chunk]
    context_chunks: list[Chunk]
    retrieval_by_id: dict[str, dict[str, Any]]
    authority_meta: dict[str, Any] = field(default_factory=dict)
    preprocess_status: str | None = None
    retrieval_latency_ms: float = 0.0
    preprocess_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    compression_latency_ms: float = 0.0
    # R-15 observability: True when compression returned nothing and the
    # pipeline fell back to the reranked chunks.
    compression_fallback: bool = False


class RetrievalPipeline:
    """Retrieve, dedup, rerank, fuse, authority-reorder and compress."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        reranker: RerankerProvider,
        preprocessor: ContextCompressor | None = None,
        compressor: ContextCompressor | None = None,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._preprocessor = preprocessor
        self._compressor = compressor

    async def run(
        self,
        query: str,
        *,
        filters: dict | None = None,
        compress: bool = True,
    ) -> RetrievalOutcome:
        """``compress=False`` skips the LLM extractor; live voice turns use it
        because the extractor costs seconds of time-to-first-audio."""
        t0 = time.perf_counter()
        retrieved = await self._retriever.retrieve(
            query, top_k=settings.retrieval_top_k, filters=filters
        )
        retrieval_latency_ms = (time.perf_counter() - t0) * 1000
        # Snapshot before rerank overwrites Chunk.score with relevance scores.
        retrieval_by_id = snapshot_retrieval(retrieved)

        t0 = time.perf_counter()
        preprocessed = retrieved
        preprocess_status = None
        if self._preprocessor is not None:
            preprocessed = await self._preprocessor.process(query, retrieved)
            preprocess_status = getattr(self._preprocessor, "last_status", None)
        preprocess_latency_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        reranked = await self._reranker.rank(
            query, preprocessed, top_k=settings.rerank_top_k
        )
        if settings.score_fusion_enabled:
            reranked = fuse_retrieval_and_rerank(
                reranked,
                retrieval_by_id,
                threshold=settings.reranker_threshold,
                retrieval_weight=settings.score_fusion_retrieval_weight,
                rerank_weight=settings.score_fusion_rerank_weight,
            )
        rerank_latency_ms = (time.perf_counter() - t0) * 1000
        reranked = backfill_by_retrieval(
            reranked, preprocessed, retrieval_by_id, settings.min_context_chunks
        )

        query_authority = (
            detect_authority(query) if settings.authority_reordering_enabled else None
        )
        if settings.authority_reordering_enabled:
            reranked = reorder_by_authority(query_authority, reranked)
        authority_meta = authority_reorder_metadata(query_authority, reranked)

        t0 = time.perf_counter()
        context_chunks = reranked
        compression_fallback = False
        if self._compressor is not None and compress:
            compressed = await self._compressor.process(query, reranked)
            if compressed:
                # Compression may drop whole chunks; keep the floor.
                context_chunks = backfill_by_retrieval(
                    compressed, reranked, retrieval_by_id, settings.min_context_chunks
                )
            elif reranked:
                # R-15: compression is an optimization — never drop all
                # context. The compressor already guards this internally; this
                # is defense in depth against any compressor implementation.
                logger.warning(
                    "Context compression returned 0 chunks for %d reranked "
                    "chunk(s); falling back to reranked context.",
                    len(reranked),
                )
                compression_fallback = True
        compression_latency_ms = (time.perf_counter() - t0) * 1000

        return RetrievalOutcome(
            retrieved=retrieved,
            context_chunks=context_chunks,
            retrieval_by_id=retrieval_by_id,
            authority_meta=authority_meta,
            preprocess_status=preprocess_status,
            retrieval_latency_ms=retrieval_latency_ms,
            preprocess_latency_ms=preprocess_latency_ms,
            rerank_latency_ms=rerank_latency_ms,
            compression_latency_ms=compression_latency_ms,
            compression_fallback=compression_fallback,
        )
