from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from backend.core.exceptions import ChatThreadNotFoundError
from backend.domain.entities.chat import ChatMessage, SessionMemory
from backend.domain.interfaces.chat_thread_repository import ChatThreadRepository


@dataclass
class ChatTurnSession:
    """Loaded thread state for one chat turn; call record_exchange after the
    LLM reply."""

    _repository: ChatThreadRepository
    _thread_id: str
    messages: list[ChatMessage]

    async def record_exchange(self, user_message: str, assistant_message: str) -> int:
        await self._repository.append_messages(
            self._thread_id,
            [
                ChatMessage(role="user", content=user_message),
                ChatMessage(role="assistant", content=assistant_message),
            ],
        )
        return len(self.messages) + 2


class ChatHistoryHandler:
    """
    Single place for thread history I/O (load, persist, API serialization).
    Change history behavior here without touching RAG/LLM flow in ChatUseCase.
    """

    def __init__(self, repository: ChatThreadRepository) -> None:
        self._repository = repository

    async def create_thread(self, *, metadata: dict | None = None) -> str:
        thread = await self._repository.create(metadata=metadata)
        return thread.id

    async def resolve_thread(self, thread_id: str | None) -> str:
        """Create a thread when omitted, or verify an existing thread id."""
        if thread_id is None:
            return await self.create_thread()
        if await self._repository.get(thread_id) is None:
            raise ChatThreadNotFoundError(thread_id)
        return thread_id

    async def get_memory(self, thread_id: str) -> SessionMemory:
        return await self._repository.get_memory(thread_id)

    async def save_memory(self, thread_id: str, memory: SessionMemory) -> None:
        await self._repository.save_memory(thread_id, memory)

    async def list_threads(self, *, limit: int = 100) -> list[dict]:
        threads = await self._repository.list_threads(limit=limit)
        return [
            {
                "thread_id": t.id,
                "title": t.title,
                "created_at": t.created_at.isoformat(),
                "updated_at": t.updated_at.isoformat(),
                "message_count": t.message_count,
            }
            for t in threads
        ]

    async def rename(self, thread_id: str, title: str) -> None:
        await self._repository.rename(thread_id, title)

    async def delete(self, thread_id: str) -> None:
        if await self._repository.get(thread_id) is None:
            raise ChatThreadNotFoundError(thread_id)
        await self._repository.delete(thread_id)

    async def read(self, thread_id: str) -> list[dict]:
        async with self._repository.run_in_thread(thread_id):
            messages = await self._repository.list_messages(thread_id)
        return self._to_api_dicts(messages)

    @asynccontextmanager
    async def turn(self, thread_id: str) -> AsyncIterator[ChatTurnSession]:
        async with self._repository.run_in_thread(thread_id):
            messages = await self._repository.list_messages(thread_id)
            session = ChatTurnSession(
                _repository=self._repository,
                _thread_id=thread_id,
                messages=messages,
            )
            yield session

    @staticmethod
    def _to_api_dicts(messages: list[ChatMessage]) -> list[dict]:
        return [
            {
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ]
