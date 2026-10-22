"""Backward-compatible re-exports for the Docling document parser.

Prefer ``backend.rag.ingestion.parsers.docling_parser.DoclingParser`` for new code.
"""

from backend.rag.ingestion.parsers.docling_parser import DoclingParser
from backend.rag.ingestion.parsers.docling_parser import DoclingParser as DoclingDocumentParser

__all__ = ["DoclingDocumentParser", "DoclingParser"]
