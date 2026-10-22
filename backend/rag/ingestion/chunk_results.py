from dataclasses import dataclass, field
from typing import Any


@dataclass
class PageChunkResult:
    """Intermediate page-level chunking output.

    ``page_number`` is the logical page identity used for chunk ids and
    overlap. Format-specific provenance (``sheet_name``, ``slide_title``, …)
    is carried in ``metadata`` and merged onto each stored chunk.
    """

    page_number: int
    page_content: str
    chunks: list[str] = field(default_factory=list)
    enriched_chunks: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
