from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager

from backend.domain.entities.chat import (
    ChatMessage,
    ChatThread,
    SessionMemory,
    ThreadSummary,
)


class ChatThreadRepository(ABC):
    """Persist and load conversation threads. Implementations must be
    concurrency-safe."""

    @abstractmethod
    async def create(self, *, metadata: dict | None = None) -> ChatThread:
        pass

    @abstractmethod
    async def get(self, thread_id: str) -> ChatThread | None:
        pass

    @abstractmethod
    async def list_messages(self, thread_id: str) -> list[ChatMessage]:
        pass

    @abstractmethod
    async def append_messages(
        self, thread_id: str, messages: list[ChatMessage]
    ) -> None:
        pass

    @abstractmethod
    async def get_memory(self, thread_id: str) -> SessionMemory:
        """Session summary and facts. Empty when the thread has none yet."""

    @abstractmethod
    async def save_memory(self, thread_id: str, memory: SessionMemory) -> None:
        pass

    @abstractmethod
    async def list_threads(self, *, limit: int = 100) -> list[ThreadSummary]:
        """Threads that have at least one message, most recently active first.
        Title is metadata["title"] when set, else the first user message."""

    @abstractmethod
    async def rename(self, thread_id: str, title: str) -> None:
        pass

    @abstractmethod
    async def delete(self, thread_id: str) -> None:
        pass

    @abstractmethod
    def run_in_thread(self, thread_id: str) -> AbstractAsyncContextManager[None]:
        """
        Async context manager: serializes read/modify/write for one thread.
        Use around load → LLM → append so concurrent requests stay ordered.
        """
