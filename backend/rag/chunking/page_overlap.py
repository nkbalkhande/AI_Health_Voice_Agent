import re
from dataclasses import dataclass, field
from typing import Any

from backend.rag.chunking.constants import PAGE_BREAK_PLACEHOLDER
from backend.rag.ingestion.models import ParsedDocument

HEADING_PATTERN = re.compile(r"^#{1,3}\s+", re.MULTILINE)


@dataclass(frozen=True)
class PageSpec:
    """One logical page ready for chunking.

    Format extras (``sheet_name``, later ``slide_title``, …) travel in
    ``metadata`` so the chunker does not grow per-format fields.
    """

    page_number: int
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def split_markdown_into_pages(text: str) -> list[str]:
    """Split exported Docling markdown into one string per page."""
    if not text.strip():
        return []
    return [
        page.strip()
        for page in text.split(PAGE_BREAK_PLACEHOLDER)
        if page.strip()
    ]


def resolve_page_contents(*, pages: list[str] | None, text: str) -> list[str]:
    """Prefer parser-provided pages; fall back to splitting combined markdown."""
    if pages:
        return [page.strip() for page in pages if page.strip()]
    return split_markdown_into_pages(text)


def resolve_page_specs(parsed: ParsedDocument) -> list[PageSpec]:
    """Build chunkable pages, preserving XLSX sheet indices when present (R-52).

    When ``page_metadata`` is aligned with ``pages``, empty sheets are skipped
    but ``page_number`` stays the original 1-based workbook sheet index.
    Formats without page metadata keep the historical renumbered behavior.
    """
    pages = list(parsed.pages or [])
    metas = list(parsed.page_metadata or [])
    if pages and metas and len(metas) == len(pages):
        specs = [
            PageSpec(
                page_number=meta.page_number,
                content=page.strip(),
                metadata={
                    key: value
                    for key, value in dict(meta.metadata).items()
                    if value not in (None, "")
                },
            )
            for page, meta in zip(pages, metas)
            if page.strip()
        ]
        if specs:
            return specs

    contents = resolve_page_contents(pages=parsed.pages, text=parsed.text)
    return [
        PageSpec(page_number=index, content=content)
        for index, content in enumerate(contents, start=1)
    ]


def get_overlap_text(next_page: str) -> str:
    """Return text from the beginning of the next page until the first heading."""
    match = HEADING_PATTERN.search(next_page)

    if match:
        return next_page[: match.start()].rstrip()

    return next_page.rstrip()


def overlap_pages(pages: list[str]) -> list[str]:
    result: list[str] = []

    for i, page in enumerate(pages):
        merged = page.rstrip()

        if i < len(pages) - 1:
            overlap = get_overlap_text(pages[i + 1])

            if overlap:
                merged += "\n\n" + overlap

        result.append(merged)

    return result
