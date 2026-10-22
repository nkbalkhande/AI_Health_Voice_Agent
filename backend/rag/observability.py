"""Observability helpers for retrieved chunks and pipeline config.

These helpers never alter retrieval, reranking, generation, or scoring. They only
serialize metadata already available on chunks and settings for offline analysis.
"""

from __future__ import annotations

from typing import Any

from backend.core.config import settings
from backend.domain.entities.chunk import Chunk
from backend.rag.retrieval.authority import get_chunk_authority


def snapshot_retrieval(chunks: list[Chunk]) -> dict[str, dict[str, Any]]:
    """Record original retriever rank/score keyed by chunk id (before rerank)."""
    return {
        chunk.id: {"rank": index, "score": chunk.score}
        for index, chunk in enumerate(chunks, start=1)
    }


def approximate_token_count(text: str) -> int:
    """Rough token estimate (~4 chars/token) for debugging, not billing."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def build_context_details(
    chunks: list[Chunk],
    *,
    retrieval_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build per-chunk metadata aligned with final contexts used for generation."""
    retrieval_by_id = retrieval_by_id or {}
    details: list[dict[str, Any]] = []

    for rank, chunk in enumerate(chunks, start=1):
        metadata = dict(chunk.metadata or {})
        retrieval = retrieval_by_id.get(chunk.id, {})
        authority = get_chunk_authority(chunk)
        filename = metadata.get("filename") or metadata.get("source")
        page = metadata.get("page_number", metadata.get("page"))

        retrieval_score = metadata.get("retrieval_score", retrieval.get("score"))
        rerank_score = metadata.get("rerank_score", chunk.score)
        details.append(
            {
                "rank": rank,
                "content": chunk.content,
                "scores": {
                    "rank": rank,
                    "retrieval_rank": retrieval.get("rank"),
                    "retrieval_score": retrieval_score,
                    "rerank_score": rerank_score,
                    "final_score": chunk.score,
                },
                "filename": filename,
                "authority": authority,
                "page": page,
                "document_id": chunk.document_id,
                "chunk_id": chunk.id,
                "chunk_index": metadata.get("chunk_index"),
                "token_count": approximate_token_count(chunk.content),
                "metadata": metadata,
            }
        )

    return details


def build_pipeline_debug(
    *,
    query_latency_ms: float | None = None,
    retrieval_latency_ms: float | None = None,
    preprocess_latency_ms: float | None = None,
    rerank_latency_ms: float | None = None,
    compression_latency_ms: float | None = None,
    generation_latency_ms: float | None = None,
    embedding_model: str | None = None,
    reranker_model: str | None = None,
    generator_model: str | None = None,
    authority_meta: dict[str, Any] | None = None,
    preprocess_status: str | None = None,
) -> dict[str, Any]:
    """Snapshot pipeline configuration and stage latencies for a single query."""
    debug: dict[str, Any] = {
        "retrieval_top_k": settings.retrieval_top_k,
        "rerank_top_k": settings.rerank_top_k,
        "reranker_threshold": settings.reranker_threshold,
        "score_fusion_enabled": settings.score_fusion_enabled,
        "compression_enabled": settings.compression_enabled,
        "preprocessing_enabled": settings.preprocessing_enabled,
        "authority_reordering_enabled": settings.authority_reordering_enabled,
        "embedding_model": embedding_model or settings.embedding_model,
        "reranker_model": reranker_model or settings.reranker_model,
        "generator_model": generator_model or settings.llm_model,
        "collection": settings.vectordb_collection,
        "query_latency_ms": _round_ms(query_latency_ms),
        "retrieval_latency_ms": _round_ms(retrieval_latency_ms),
        "preprocess_latency_ms": _round_ms(preprocess_latency_ms),
        "rerank_latency_ms": _round_ms(rerank_latency_ms),
        "compression_latency_ms": _round_ms(compression_latency_ms),
        "generation_latency_ms": _round_ms(generation_latency_ms),
        # R-63: dedup degrades silently to "no dedup" on an embedding error;
        # record it so the drift is visible per query instead of log-only.
        "preprocess_status": preprocess_status,
    }
    if authority_meta:
        debug.update(authority_meta)
    return debug


def _round_ms(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 2)
