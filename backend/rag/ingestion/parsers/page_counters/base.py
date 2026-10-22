from __future__ import annotations

from abc import ABC, abstractmethod


class BasePageCounter(ABC):
    """Strategy interface for format-specific page counting."""

    @abstractmethod
    def total_pages(self, document_bytes: bytes) -> int | None:
        """Return a reliable page count, or ``None`` when batching is not possible."""
        raise NotImplementedError
