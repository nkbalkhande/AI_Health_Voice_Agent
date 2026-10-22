from abc import ABC, abstractmethod

from backend.domain.entities.chunk import Chunk


class Retriever(ABC):
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[Chunk]:
        pass
