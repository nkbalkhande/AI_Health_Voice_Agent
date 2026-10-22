from __future__ import annotations

from backend.rag.ingestion.parsers.page_counters.base import BasePageCounter


class UnsupportedPageCounter(BasePageCounter):
    """Formats without a reliable pre-conversion page count (DOCX/PPTX/XLSX/etc.)."""

    def total_pages(self, document_bytes: bytes) -> int | None:
        return None
