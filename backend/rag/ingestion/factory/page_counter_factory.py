from __future__ import annotations

from pathlib import Path

from backend.core.logger import logger
from backend.rag.ingestion.parsers.page_counters.base import BasePageCounter
from backend.rag.ingestion.parsers.page_counters.pdf_page_counter import PdfPageCounter
from backend.rag.ingestion.parsers.page_counters.unsupported_page_counter import (
    UnsupportedPageCounter,
)

log = logger.getChild("rag.ingestion.factory.page_counter_factory")

_PDF_EXTENSIONS = frozenset({".pdf"})


class PageCounterFactory:
    """Map file extensions to page-counter implementations."""

    @staticmethod
    def create(filename: str) -> BasePageCounter:
        extension = Path(filename).suffix.lower()
        if extension in _PDF_EXTENSIONS:
            counter: BasePageCounter = PdfPageCounter()
        else:
            # DOCX/PPTX/XLSX (and unknown formats) have no stable page count
            # before layout/conversion, so batching is disabled.
            counter = UnsupportedPageCounter()

        log.info(
            "page_counter.selected filename=%s file_type=%s counter_type=%s",
            filename,
            extension or "(none)",
            type(counter).__name__,
        )
        return counter
