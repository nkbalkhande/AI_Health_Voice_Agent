from abc import ABC, abstractmethod

from backend.domain.entities.document import Document
from backend.rag.ingestion.models import ParsedDocument


class PdfChunker(ABC):
    @abstractmethod
    async def chunk(
        self,
        parsed: ParsedDocument,
        *,
        source: str | None = None,
        metadata: dict | None = None,
    ) -> list[Document]:
        """Convert a parsed PDF into domain documents ready for embedding."""
