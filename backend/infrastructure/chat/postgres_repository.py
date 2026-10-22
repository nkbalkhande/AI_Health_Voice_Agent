import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row

from backend.core.exceptions import ChatThreadNotFoundError
from backend.core.logger import logger
from backend.domain.entities.chat import (
    ChatMessage,
    ChatThread,
    SessionMemory,
    ThreadSummary,
    thread_title,
)
from backend.domain.interfaces.chat_thread_repository import ChatThreadRepository

log = logger.getChild("chat.postgres")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_sessions (
    id UUID PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'active',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES conversation_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS conversation_messages_session_idx
    ON conversation_messages (session_id, id);

CREATE INDEX IF NOT EXISTS conversation_sessions_updated_idx
    ON conversation_sessions (updated_at DESC);

CREATE TABLE IF NOT EXISTS conversation_memory (
    session_id UUID PRIMARY KEY REFERENCES conversation_sessions(id) ON DELETE CASCADE,
    summary TEXT NOT NULL DEFAULT '',
    facts JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class PostgresChatThreadRepository(ChatThreadRepository):
    """Durable sessions, transcripts, and extracted facts in PostgreSQL.

    Queries run on a normal connection inside a worker thread. Psycopg's
    async mode cannot use the Windows default event loop, and a thread
    keeps the FastAPI handlers async either way.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()
        self._schema_ready = False
        self._schema_lock = asyncio.Lock()

    @classmethod
    def from_settings(cls) -> "PostgresChatThreadRepository":
        from backend.core.config import settings

        return cls(settings.postgres_dsn)

    def _connect(self) -> psycopg.Connection:
        self._ensure_schema_sync()
        return psycopg.connect(self._dsn, row_factory=dict_row)

    def _ensure_schema_sync(self) -> None:
        if self._schema_ready:
            return
        with psycopg.connect(self._dsn) as conn:
            for statement in _SCHEMA.split(";"):
                sql = statement.strip()
                if sql:
                    conn.execute(sql)
            conn.commit()
        self._schema_ready = True
        log.info("Conversation tables are ready")

    async def _get_lock(self, thread_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            if thread_id not in self._locks:
                self._locks[thread_id] = asyncio.Lock()
            return self._locks[thread_id]

    @asynccontextmanager
    async def run_in_thread(self, thread_id: str) -> AsyncIterator[None]:
        if await self.get(thread_id) is None:
            raise ChatThreadNotFoundError(thread_id)
        lock = await self._get_lock(thread_id)
        async with lock:
            yield

    async def create(self, *, metadata: dict | None = None) -> ChatThread:
        thread = ChatThread(id=str(uuid4()), metadata=metadata or {})
        await asyncio.to_thread(self._create_sync, thread)
        return thread

    def _create_sync(self, thread: ChatThread) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_sessions (id, metadata)
                VALUES (%s, %s::jsonb)
                """,
                (thread.id, json.dumps(thread.metadata)),
            )
            conn.execute(
                """
                INSERT INTO conversation_memory (session_id, summary, facts)
                VALUES (%s, '', '[]'::jsonb)
                """,
                (thread.id,),
            )
            conn.commit()

    async def get(self, thread_id: str) -> ChatThread | None:
        row = await asyncio.to_thread(self._get_sync, thread_id)
        if row is None:
            return None
        return ChatThread(
            id=str(row["id"]),
            created_at=row["created_at"],
            metadata=row["metadata"] or {},
        )

    def _get_sync(self, thread_id: str):
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT id, metadata, created_at
                FROM conversation_sessions
                WHERE id = %s
                """,
                (thread_id,),
            )
            return cursor.fetchone()

    async def list_messages(self, thread_id: str) -> list[ChatMessage]:
        rows = await asyncio.to_thread(self._list_messages_sync, thread_id)
        return [
            ChatMessage(role=row["role"], content=row["content"], created_at=row["created_at"])
            for row in rows
        ]

    def _list_messages_sync(self, thread_id: str):
        with self._connect() as conn:
            found = conn.execute(
                "SELECT 1 FROM conversation_sessions WHERE id = %s",
                (thread_id,),
            ).fetchone()
            if found is None:
                raise ChatThreadNotFoundError(thread_id)
            return conn.execute(
                """
                SELECT role, content, created_at
                FROM conversation_messages
                WHERE session_id = %s
                ORDER BY id
                """,
                (thread_id,),
            ).fetchall()

    async def append_messages(
        self, thread_id: str, messages: list[ChatMessage]
    ) -> None:
        await asyncio.to_thread(self._append_sync, thread_id, messages)

    def _append_sync(self, thread_id: str, messages: list[ChatMessage]) -> None:
        with self._connect() as conn:
            found = conn.execute(
                "SELECT 1 FROM conversation_sessions WHERE id = %s",
                (thread_id,),
            ).fetchone()
            if found is None:
                raise ChatThreadNotFoundError(thread_id)
            for message in messages:
                conn.execute(
                    """
                    INSERT INTO conversation_messages (session_id, role, content, created_at)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (thread_id, message.role, message.content, message.created_at),
                )
            conn.execute(
                "UPDATE conversation_sessions SET updated_at = now() WHERE id = %s",
                (thread_id,),
            )
            conn.commit()

    async def get_memory(self, thread_id: str) -> SessionMemory:
        row = await asyncio.to_thread(self._get_memory_sync, thread_id)
        if row is None:
            return SessionMemory()
        facts = row["facts"] if isinstance(row["facts"], list) else []
        return SessionMemory(summary=row["summary"] or "", facts=list(facts))

    def _get_memory_sync(self, thread_id: str):
        with self._connect() as conn:
            found = conn.execute(
                "SELECT 1 FROM conversation_sessions WHERE id = %s",
                (thread_id,),
            ).fetchone()
            if found is None:
                raise ChatThreadNotFoundError(thread_id)
            return conn.execute(
                """
                SELECT summary, facts
                FROM conversation_memory
                WHERE session_id = %s
                """,
                (thread_id,),
            ).fetchone()

    async def save_memory(self, thread_id: str, memory: SessionMemory) -> None:
        await asyncio.to_thread(self._save_memory_sync, thread_id, memory)

    def _save_memory_sync(self, thread_id: str, memory: SessionMemory) -> None:
        with self._connect() as conn:
            found = conn.execute(
                "SELECT 1 FROM conversation_sessions WHERE id = %s",
                (thread_id,),
            ).fetchone()
            if found is None:
                raise ChatThreadNotFoundError(thread_id)
            conn.execute(
                """
                INSERT INTO conversation_memory (session_id, summary, facts)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (session_id) DO UPDATE
                SET summary = EXCLUDED.summary,
                    facts = EXCLUDED.facts,
                    updated_at = now()
                """,
                (thread_id, memory.summary, json.dumps(memory.facts)),
            )
            conn.commit()

    async def list_threads(self, *, limit: int = 100) -> list[ThreadSummary]:
        rows = await asyncio.to_thread(self._list_threads_sync, limit)
        return [
            ThreadSummary(
                id=str(row["id"]),
                title=thread_title(row["metadata"], row["first_user_message"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                message_count=row["message_count"],
            )
            for row in rows
        ]

    def _list_threads_sync(self, limit: int):
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT s.id, s.metadata, s.created_at, s.updated_at,
                       stats.message_count,
                       (
                           SELECT m.content
                           FROM conversation_messages m
                           WHERE m.session_id = s.id AND m.role = 'user'
                           ORDER BY m.id
                           LIMIT 1
                       ) AS first_user_message
                FROM conversation_sessions s
                JOIN (
                    SELECT session_id, count(*) AS message_count
                    FROM conversation_messages
                    GROUP BY session_id
                ) stats ON stats.session_id = s.id
                ORDER BY s.updated_at DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()

    async def rename(self, thread_id: str, title: str) -> None:
        await asyncio.to_thread(self._rename_sync, thread_id, title)

    def _rename_sync(self, thread_id: str, title: str) -> None:
        with self._connect() as conn:
            updated = conn.execute(
                """
                UPDATE conversation_sessions
                SET metadata = metadata || jsonb_build_object('title', %s::text)
                WHERE id = %s
                """,
                (title, thread_id),
            ).rowcount
            if not updated:
                raise ChatThreadNotFoundError(thread_id)
            conn.commit()

    async def delete(self, thread_id: str) -> None:
        await asyncio.to_thread(self._delete_sync, thread_id)

    def _delete_sync(self, thread_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM conversation_sessions WHERE id = %s",
                (thread_id,),
            )
            conn.commit()
