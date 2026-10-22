"""Authority-aware soft reordering of reranked chunks.

If the query names a standards body, chunks attributed to that body float
above chunks from sibling bodies. Nothing is dropped — recall-safe. This
targets the dominant precision-junk class (wrong-authority siblings).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from backend.core.config import settings
from backend.domain.entities.chunk import Chunk

if TYPE_CHECKING:
    from collections.abc import Iterable

# Keys are uppercase to match every lookup path (_normalize_authority,
# supported_authorities, detect_authority). Values are lists: explicit name
# pattern first, then standard-number / alias heuristics — so
# "PCAOB AS 2405" resolves to PCAOB rather than a sibling heuristic.

_AUTHORITY_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "PCAOB": [
        re.compile(r"\bpcaob\b", re.I),
        re.compile(r"\bas\s?\d{3,4}\b", re.I),
    ],
    "AICPA": [
        re.compile(r"\baicpa\b", re.I),
        re.compile(r"\bau-c\b", re.I),
        re.compile(r"\bsas\s?\d\b", re.I),
        re.compile(r"\bssars\b", re.I),
    ],
    "GAO": [
        re.compile(r"\bgao\b", re.I),
        re.compile(r"yellow\s?book", re.I),
        re.compile(r"government auditing standards", re.I),
    ],
    "FASAB": [
        re.compile(r"\bfasab\b", re.I),
        re.compile(r"\bsffas\b", re.I),
    ],
    "GASB": [re.compile(r"\bgasb\b", re.I)],
    "FASB": [
        re.compile(r"\bfasb\b", re.I),
        re.compile(r"\basc\s?\d", re.I),
    ],
    "IRS": [
        re.compile(r"\birs\b", re.I),
        re.compile(r"\birc\b", re.I),
        re.compile(r"publication\s?\d+", re.I),
        # R-91: a bare "Form N" used to claim SEC forms (Form 10-K, Form S-1)
        # for the IRS. Require an IRS-shaped form number (3-4 digits, e.g.
        # Form 1040 / Form 990) instead of any digit run.
        re.compile(r"\bform\s?\d{3,4}\b", re.I),
    ],
    "COSO": [re.compile(r"\bcoso\b", re.I)],
    "SEC": [
        re.compile(r"\bsec\b", re.I),
        # R-91: only 10-K was recognised, so 10-Q / 8-K / S-1 queries fell
        # through to whatever else matched.
        re.compile(r"\b10-?k\b", re.I),
        re.compile(r"\b10-?q\b", re.I),
        re.compile(r"\b8-?k\b", re.I),
        re.compile(r"\bs-?1\b", re.I),
    ],
}

_DEFAULT_SUPPORTED_AUTHORITIES = tuple(_AUTHORITY_PATTERNS.keys())


def supported_authorities() -> tuple[str, ...]:
    configured = settings.authority_reordering_supported_authorities
    if not configured:
        return _DEFAULT_SUPPORTED_AUTHORITIES
    normalized = tuple(
        authority.upper()
        for authority in configured
        if authority and authority.upper() in _AUTHORITY_PATTERNS
    )
    return normalized or _DEFAULT_SUPPORTED_AUTHORITIES


def _normalize_authority(name: str | None) -> str | None:
    if not name:
        return None
    upper = name.strip().upper()
    if upper in _AUTHORITY_PATTERNS:
        return upper
    return None


def detect_authority(
    text: str,
    *,
    authorities: Iterable[str] | None = None,
) -> str | None:
    """Return the first supported authority referenced in *text*, or None."""
    active = tuple(
        authority.upper()
        for authority in (authorities or supported_authorities())
        if authority
    )
    haystack = text or ""

    for authority in active:
        patterns = _AUTHORITY_PATTERNS.get(authority, [])
        if patterns and patterns[0].search(haystack):
            return authority

    for authority in active:
        patterns = _AUTHORITY_PATTERNS.get(authority, [])
        for pattern in patterns[1:]:
            if pattern.search(haystack):
                return authority

    return None


def get_chunk_authority(chunk: Chunk) -> str | None:
    """Return the authority associated with a retrieved chunk, if known."""
    metadata = chunk.metadata or {}

    for key in ("authority", "standards_body", "source_authority"):
        normalized = _normalize_authority(metadata.get(key))
        if normalized is not None:
            return normalized

    haystack = " ".join(
        str(part)
        for part in (
            metadata.get("source"),
            metadata.get("filename"),
            chunk.document_id,
            chunk.id,
            chunk.content[:300],
        )
        if part
    )
    return detect_authority(haystack)


def chunk_authority(chunk: Chunk) -> str | None:
    """Backward-compatible alias for :func:`get_chunk_authority`."""
    return get_chunk_authority(chunk)


def reorder_by_authority(
    query_authority: str | None,
    chunks: list[Chunk],
) -> list[Chunk]:
    """Promote chunks matching *query_authority* while preserving rerank order."""
    wanted = _normalize_authority(query_authority)
    if wanted is None or not chunks:
        return chunks

    # Compute each chunk's authority once; get_chunk_authority runs regex over
    # content, so partitioning without caching would re-derive it 2x per chunk.
    matching: list[Chunk] = []
    rest: list[Chunk] = []
    for chunk in chunks:
        (matching if get_chunk_authority(chunk) == wanted else rest).append(chunk)

    if not matching:
        return chunks
    return matching + rest


def authority_reorder_metadata(
    query_authority: str | None,
    chunks: list[Chunk],
) -> dict[str, str | int | bool | None]:
    """Summarize authority reordering for evaluation and debug logs."""
    normalized = _normalize_authority(query_authority)
    # R-80: with normalized None, `get_chunk_authority(chunk) == normalized`
    # counted every chunk that had NO detectable authority and reported it as
    # "authority_matches" — the opposite of the intended meaning. Skip the
    # regex sweep entirely when there is nothing to match (or the feature is
    # off), which is also the common case.
    if normalized is None or not settings.authority_reordering_enabled:
        matches = 0
    else:
        matches = sum(
            1 for chunk in chunks if get_chunk_authority(chunk) == normalized
        )
    return {
        "query_authority": normalized,
        "authority_reordering": bool(
            settings.authority_reordering_enabled
            and normalized is not None
            and matches > 0
        ),
        "authority_matches": matches,
    }
