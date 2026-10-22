from abc import ABC, abstractmethod

from backend.domain.entities.chunk import Chunk


class RerankerProvider(ABC):
    @abstractmethod
    async def rank(
        self,
        query: str,
        chunks: list[Chunk],
        *,
        top_k: int | None = None,
    ) -> list[Chunk]:
        pass
