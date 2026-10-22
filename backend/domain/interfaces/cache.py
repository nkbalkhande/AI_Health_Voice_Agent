from abc import ABC, abstractmethod
from typing import Any


class CacheStore(ABC):
    """Short-lived cache. A miss or an outage must never break a reply."""

    @abstractmethod
    async def get_json(self, key: str) -> Any | None:
        pass

    @abstractmethod
    async def set_json(self, key: str, value: Any) -> None:
        pass
