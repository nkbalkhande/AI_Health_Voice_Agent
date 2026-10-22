from __future__ import annotations

from pathlib import Path

from backend.core.exceptions import UnsupportedFileType
from backend.core.logger import logger
from backend.rag.ingestion.parsers.base_parser import BaseDocumentParser
from backend.rag.ingestion.parsers.html_parser import HtmlParser
from backend.rag.ingestion.parsers.text_parser import TextParser

log = logger.getChild("rag.ingestion.factory.parser_factory")

_DOCLING_EXTENSIONS = frozenset({".pdf", ".docx", ".pptx"})
_XLSX_EXTENSIONS = frozenset({".xlsx"})
_TEXT_EXTENSIONS = frozenset({".txt"})
_HTML_EXTENSIONS = frozenset({".html", ".htm"})


class ParserFactory:
    """Map file extensions to parser implementations."""

    @staticmethod
    def create(
        filename: str,
        *,
        docling_parser: BaseDocumentParser | None = None,
    ) -> BaseDocumentParser:
        extension = Path(filename).suffix.lower()
        if extension in _DOCLING_EXTENSIONS:
            if docling_parser is None:
                from backend.rag.ingestion.parsers.docling_parser import DoclingParser

                parser: BaseDocumentParser = DoclingParser()
            else:
                parser = docling_parser
            log.info(
                "parser.selected filename=%s file_type=%s parser_type=%s",
                filename,
                extension,
                type(parser).__name__,
            )
            return parser

        if extension in _XLSX_EXTENSIONS:
            from backend.rag.ingestion.parsers.xlsx_parser import XlsxParser

            parser = XlsxParser()
            log.info(
                "parser.selected filename=%s file_type=%s parser_type=%s",
                filename,
                extension,
                type(parser).__name__,
            )
            return parser

        if extension in _TEXT_EXTENSIONS:
            parser = TextParser()
            log.info(
                "parser.selected filename=%s file_type=%s parser_type=%s",
                filename,
                extension,
                type(parser).__name__,
            )
            return parser

        if extension in _HTML_EXTENSIONS:
            parser = HtmlParser()
            log.info(
                "parser.selected filename=%s file_type=%s parser_type=%s",
                filename,
                extension,
                type(parser).__name__,
            )
            return parser

        raise UnsupportedFileType(
            f"Unsupported file type {extension or '(none)'} for filename={filename!r}"
        )
