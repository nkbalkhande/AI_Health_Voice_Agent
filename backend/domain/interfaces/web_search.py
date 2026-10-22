from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class WebDocument:
    title: str
    url: str
    snippet: str


class WebSearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, *, limit: int = 5) -> list[WebDocument]:
        """Return external pages for a standalone query. Empty on failure."""
