from __future__ import annotations

from abc import ABC, abstractmethod

from backend.rag.ingestion.models import ParsedDocument


class BaseDocumentParser(ABC):
    """Common interface for format-specific document parsers."""

    @abstractmethod
    def parse(
        self,
        document_bytes: bytes,
        *,
        filename: str,
    ) -> ParsedDocument:
        """Parse document bytes into ParsedDocument."""
        raise NotImplementedError
