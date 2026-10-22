"""Weighted fusion of retrieval and rerank scores (min-max normalized).

Admission and ranking are separate concerns. Chunks are admitted on the
reranker's absolute relevance score, and only the admitted ones are fused into
a ranking signal -- see ``fuse_retrieval_and_rerank``.
"""

from __future__ import annotations

import math
from typing import Any

from backend.domain.entities.chunk import Chunk

_DEFAULT_RETRIEVAL_WEIGHT = 0.5
_DEFAULT_RERANK_WEIGHT = 0.5


def _as_float(raw: Any, default: float = 0.0) -> float:
    try:
        if raw is None:
            return default
        value = float(raw)
        if not math.isfinite(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [1.0] * len(values)
    span = hi - lo
    return [(v - lo) / span for v in values]


def _retrieval_score(
    chunk: Chunk,
    retrieval_by_id: dict[str, dict[str, Any]],
) -> float:
    snap = retrieval_by_id.get(chunk.id) or {}
    return _as_float(snap.get("score"), 0.0)


def _rerank_scores(chunks: list[Chunk]) -> list[tuple[float, bool]]:
    """Return ``(score, is_absolute)`` per chunk.

    ``is_absolute`` is False when the reranker left no usable score and the
    value is a positional stand-in (rank 1 = n, last = 1). A positional value
    encodes order only, so it must never be compared against a relevance
    threshold.
    """
    n = len(chunks)
    scores: list[tuple[float, bool]] = []
    for index, chunk in enumerate(chunks):
        value = _as_float(chunk.score, math.nan)
        if math.isnan(value):
            scores.append((float(n - index), False))
        else:
            scores.append((value, True))
    return scores


def _normalized_weights(
    retrieval_weight: float, rerank_weight: float
) -> tuple[float, float]:
    """Scale weights to sum to 1 so the fused score stays inside [0, 1].

    A negative weight would invert its channel and push the blend outside the
    documented range, so it is rejected rather than normalized.
    """
    if retrieval_weight < 0 or rerank_weight < 0:
        return 0.0, 0.0
    total = retrieval_weight + rerank_weight
    if not math.isfinite(total) or total <= 0:
        # No usable ranking signal; the stable sort preserves reranker order.
        return 0.0, 0.0
    return retrieval_weight / total, rerank_weight / total


def fuse_retrieval_and_rerank(
    chunks: list[Chunk],
    retrieval_by_id: dict[str, dict[str, Any]] | None = None,
    *,
    threshold: float | None = None,
    retrieval_weight: float = _DEFAULT_RETRIEVAL_WEIGHT,
    rerank_weight: float = _DEFAULT_RERANK_WEIGHT,
) -> list[Chunk]:
    """Filter on absolute relevance, then blend the survivors into a ranking.

    ``threshold`` gates on the raw reranker score, never on the fused value.
    Min-max output is relative to the set being normalized -- the weakest chunk
    is always 0.0 and the strongest always 1.0 -- so it expresses order but not
    quality, and thresholding it would drop strong chunks from a strong set
    while admitting weak chunks from a weak one.

    For the same reason the gate runs *before* fusion: normalizing over the
    full candidate set would let a rejected chunk stretch the range and reorder
    the survivors. Chunks whose rerank score is a positional stand-in are never
    gated.

    Weights are normalized to sum to 1, so the fused ``Chunk.score`` spans
    [0, 1]. Raw channel scores are recorded on ``chunk.metadata`` as
    ``retrieval_score`` and ``rerank_score`` for every chunk, rejected ones
    included; rejected chunks keep their original ``score``.
    """
    if not chunks:
        return []

    retrieval_by_id = retrieval_by_id or {}
    admitted: list[tuple[Chunk, float, float]] = []

    for chunk, (rerank_raw, rerank_is_absolute) in zip(
        chunks, _rerank_scores(chunks), strict=True
    ):
        retrieval_raw = _retrieval_score(chunk, retrieval_by_id)
        meta = dict(chunk.metadata or {})
        meta["retrieval_score"] = retrieval_raw
        meta["rerank_score"] = rerank_raw
        chunk.metadata = meta
        if threshold is not None and rerank_is_absolute and rerank_raw < threshold:
            continue
        admitted.append((chunk, retrieval_raw, rerank_raw))

    if not admitted:
        return []

    retrieval_norm = _minmax([raw for _, raw, _ in admitted])
    rerank_norm = _minmax([raw for _, _, raw in admitted])
    retrieval_weight, rerank_weight = _normalized_weights(
        retrieval_weight, rerank_weight
    )

    fused: list[tuple[float, Chunk]] = []
    for (chunk, _, _), r_n, k_n in zip(
        admitted, retrieval_norm, rerank_norm, strict=True
    ):
        chunk.score = retrieval_weight * r_n + rerank_weight * k_n
        fused.append((chunk.score, chunk))

    fused.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in fused]
