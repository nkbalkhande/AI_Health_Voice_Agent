"""Deterministic chunk identity for idempotent vector upserts."""

from __future__ import annotations

import hashlib
import uuid

# Fixed namespace so UUID5 outputs are stable across processes and deploys.
_CHUNK_ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # NAMESPACE_URL


def deterministic_chunk_id(
    source: str,
    page: int | None,
    chunk_index: int,
    content: str,
) -> str:
    """Return a stable UUID for a chunk based on immutable chunk properties.

    Same source + page + chunk_index + content always yields the same ID, so
    re-ingesting a document upserts existing Qdrant points instead of duplicating.
    """
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    page_part = "" if page is None else str(page)
    name = f"{source}|{page_part}|{chunk_index}|{content_hash}"
    return str(uuid.uuid5(_CHUNK_ID_NAMESPACE, name))
