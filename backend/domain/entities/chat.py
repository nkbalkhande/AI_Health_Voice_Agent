from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

ChatRole = Literal["user", "assistant", "system"]


@dataclass
class ChatMessage:
    role: ChatRole
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ChatThread:
    id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreadSummary:
    """One row of the conversation list shown in the sidebar."""

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


def thread_title(metadata: dict[str, Any] | None, first_user_message: str | None) -> str:
    custom = str((metadata or {}).get("title") or "").strip()
    if custom:
        return custom
    text = " ".join((first_user_message or "").split())
    if not text:
        return "New conversation"
    return text if len(text) <= 60 else text[:57].rstrip() + "…"


@dataclass
class SessionMemory:
    """Facts and summary kept for one conversation, separate from the transcript."""

    summary: str = ""
    facts: list[dict[str, Any]] = field(default_factory=list)
