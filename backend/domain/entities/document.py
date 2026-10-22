from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Document:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str | None = None
    created_at: datetime | None = None
