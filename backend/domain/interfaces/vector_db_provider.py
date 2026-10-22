from abc import ABC, abstractmethod

from backend.domain.entities.chunk import Chunk
from backend.domain.entities.document import Document


class VectorDBProvider(ABC):
    @abstractmethod
    async def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        pass

    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        filters: dict | None = None,
        query_text: str | None = None,
    ) -> list[Chunk]:
        pass

    @abstractmethod
    async def delete(self, document_ids: list[str]) -> None:
        pass

    @abstractmethod
    async def update(
        self,
        document_id: str,
        content: str,
        embedding: list[float],
    ) -> None:
        pass
