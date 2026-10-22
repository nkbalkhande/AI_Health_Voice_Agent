from __future__ import annotations

from bs4 import BeautifulSoup

from backend.core.logger import logger
from backend.rag.ingestion.models import ParsedDocument
from backend.rag.ingestion.parsers.base_parser import BaseDocumentParser

log = logger.getChild("rag.ingestion.parsers.html_parser")

_NON_CONTENT_TAGS = ("script", "style", "noscript")


class HtmlParser(BaseDocumentParser):
    """Extract readable text from HTML into a single-page ParsedDocument."""

    def parse(
        self,
        document_bytes: bytes,
        *,
        filename: str,
    ) -> ParsedDocument:
        html = document_bytes.decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(_NON_CONTENT_TAGS):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
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
