"""Format-specific document parsers.

Prefer importing concrete parsers from their modules to avoid loading Docling
when only Text/HTML parsers are needed.
"""

from backend.rag.ingestion.parsers.base_parser import BaseDocumentParser

__all__ = [
    "BaseDocumentParser",
    "DoclingParser",
    "HtmlParser",
    "TextParser",
    "XlsxParser",
]


def __getattr__(name: str):
    if name == "DoclingParser":
        from backend.rag.ingestion.parsers.docling_parser import DoclingParser

        return DoclingParser
    if name == "HtmlParser":
        from backend.rag.ingestion.parsers.html_parser import HtmlParser

        return HtmlParser
    if name == "TextParser":
        from backend.rag.ingestion.parsers.text_parser import TextParser

        return TextParser
    if name == "XlsxParser":
        from backend.rag.ingestion.parsers.xlsx_parser import XlsxParser

        return XlsxParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
