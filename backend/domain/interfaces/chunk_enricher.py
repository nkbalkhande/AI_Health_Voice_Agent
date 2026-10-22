from abc import ABC, abstractmethod

from backend.rag.ingestion.chunk_results import PageChunkResult


class ChunkEnricher(ABC):
    @abstractmethod
    async def enrich(
        self,
        page_results: list[PageChunkResult],
        *,
        all_pages: list[str],
    ) -> list[PageChunkResult]:
        """Augment chunk text with missing page context where required."""
