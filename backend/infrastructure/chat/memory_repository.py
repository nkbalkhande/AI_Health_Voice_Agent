import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from backend.core.exceptions import ChatThreadNotFoundError
from backend.domain.entities.chat import (
    ChatMessage,
    ChatThread,
    SessionMemory,
    ThreadSummary,
    thread_title,
)
from backend.domain.interfaces.chat_thread_repository import ChatThreadRepository


class InMemoryChatThreadRepository(ChatThreadRepository):
    """
    Thread-safe in-memory store. Per-thread asyncio.Lock prevents lost updates
    when multiple requests hit the same conversation concurrently.
    """

    def __init__(self) -> None:
        self._threads: dict[str, ChatThread] = {}
        self._messages: dict[str, list[ChatMessage]] = {}
        self._memory: dict[str, SessionMemory] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def _get_lock(self, thread_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            if thread_id not in self._locks:
                self._locks[thread_id] = asyncio.Lock()
            return self._locks[thread_id]

    @asynccontextmanager
    async def run_in_thread(self, thread_id: str) -> AsyncIterator[None]:
        if thread_id not in self._threads:
            raise ChatThreadNotFoundError(thread_id)
        lock = await self._get_lock(thread_id)
        async with lock:
            yield

    async def create(self, *, metadata: dict | None = None) -> ChatThread:
        thread = ChatThread(id=str(uuid4()), metadata=metadata or {})
        async with self._locks_guard:
            self._threads[thread.id] = thread
            self._messages[thread.id] = []
            self._memory[thread.id] = SessionMemory()
            self._locks[thread.id] = asyncio.Lock()
        return thread

    async def get(self, thread_id: str) -> ChatThread | None:
        return self._threads.get(thread_id)

    async def list_messages(self, thread_id: str) -> list[ChatMessage]:
        if thread_id not in self._threads:
            raise ChatThreadNotFoundError(thread_id)
        return list(self._messages.get(thread_id, []))

    async def append_messages(
        self, thread_id: str, messages: list[ChatMessage]
    ) -> None:
        if thread_id not in self._threads:
            raise ChatThreadNotFoundError(thread_id)
        self._messages[thread_id].extend(messages)

    async def get_memory(self, thread_id: str) -> SessionMemory:
        if thread_id not in self._threads:
            raise ChatThreadNotFoundError(thread_id)
        memory = self._memory.get(thread_id) or SessionMemory()
        return SessionMemory(summary=memory.summary, facts=list(memory.facts))

    async def save_memory(self, thread_id: str, memory: SessionMemory) -> None:
        if thread_id not in self._threads:
            raise ChatThreadNotFoundError(thread_id)
        self._memory[thread_id] = SessionMemory(
            summary=memory.summary,
            facts=list(memory.facts),
        )

    async def list_threads(self, *, limit: int = 100) -> list[ThreadSummary]:
        summaries = []
        for thread in self._threads.values():
            messages = self._messages.get(thread.id) or []
            if not messages:
                continue
            first_user = next((m.content for m in messages if m.role == "user"), None)
            summaries.append(
                ThreadSummary(
                    id=thread.id,
                    title=thread_title(thread.metadata, first_user),
                    created_at=thread.created_at,
                    updated_at=messages[-1].created_at,
                    message_count=len(messages),
                )
            )
        summaries.sort(key=lambda s: s.updated_at, reverse=True)
        return summaries[:limit]

    async def rename(self, thread_id: str, title: str) -> None:
        thread = self._threads.get(thread_id)
        if thread is None:
            raise ChatThreadNotFoundError(thread_id)
        thread.metadata = {**thread.metadata, "title": title}

    async def delete(self, thread_id: str) -> None:
        async with self._locks_guard:
            self._threads.pop(thread_id, None)
            self._messages.pop(thread_id, None)
            self._memory.pop(thread_id, None)
            self._locks.pop(thread_id, None)
