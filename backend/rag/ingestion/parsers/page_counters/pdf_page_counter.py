from __future__ import annotations

import io

from PyPDF2 import PdfReader

from backend.core.logger import logger
from backend.rag.ingestion.parsers.page_counters.base import BasePageCounter

log = logger.getChild("rag.ingestion.parsers.page_counters.pdf")


class PdfPageCounter(BasePageCounter):
    """Count physical pages for PDF documents."""

    def total_pages(self, document_bytes: bytes) -> int | None:
        try:
            page_count = len(PdfReader(io.BytesIO(document_bytes)).pages)
        except Exception as exc:  # noqa: BLE001 - fall back to whole conversion
            log.warning(
                "Unable to determine PDF page count; disabling batching: error=%s",
                exc,
            )
            return None
        return page_count if page_count > 0 else None
