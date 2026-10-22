from __future__ import annotations

from backend.core.logger import logger
from backend.rag.ingestion.models import ParsedDocument
from backend.rag.ingestion.parsers.base_parser import BaseDocumentParser

log = logger.getChild("rag.ingestion.parsers.text_parser")


class TextParser(BaseDocumentParser):
    """Decode plain-text documents into a single-page ParsedDocument."""

    def parse(
        self,
        document_bytes: bytes,
        *,
        filename: str,
    ) -> ParsedDocument:
        text = document_bytes.decode("utf-8", errors="replace")
        log.info(
            "parsing.completed filename=%s parser_type=%s status=ok",
            filename,
            type(self).__name__,
        )
        return ParsedDocument(
            filename=filename,
            text=text,
            pages=[text],
            images=[],
            docling_document=[],
        )
