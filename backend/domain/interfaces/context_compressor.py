from abc import ABC, abstractmethod

from backend.domain.entities.chunk import Chunk


class ContextCompressor(ABC):
    @abstractmethod
    async def process(
        self,
        query: str,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        pass
