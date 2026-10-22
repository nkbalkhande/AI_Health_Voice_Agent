import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from backend.core.config import settings

_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def _provenance() -> dict[str, str]:
    """git SHA + settings hash, resolved once per process.

    R-38: audit rows had nothing to join them to an eval run manifest, so a
    logged answer could not be tied to the code/config that produced it.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        sha = (
            completed.stdout.strip() if completed.returncode == 0 else ""
        ) or "unknown"
    except Exception:
        sha = "unknown"
    try:
        with open(_SETTINGS_PATH, "rb") as f:
            settings_hash = hashlib.sha256(f.read()).hexdigest()[:12]
    except OSError:
        settings_hash = "unknown"
    return {"git_sha": sha, "settings_sha256_12": settings_hash}


@dataclass
class AuditRecord:
    query: str
    retrieved_chunks: list[str]
    reranked_chunks: list[str]
    citations: list[dict]
    model: str
    latency_ms: float
    cost: float | None
    answer: str
    thread_id: str | None = None
    timestamp: str | None = None
    # R-37: without these an audit row cannot distinguish a refusal from an
    # error from a truncated answer - all three look like a short answer.
    generation_status: str | None = None
    finish_reason: str | None = None
    provenance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC).isoformat()
        if not self.provenance:
            self.provenance = dict(_provenance())


class AuditLogger:
    async def log(self, record: AuditRecord) -> None:
        # R-38: the old switch silently wrote JSONL for "postgres" and for any
        # unknown backend, so an operator who configured postgres got JSONL and
        # no warning. Settings now rejects unimplemented backends at boot; this
        # stays defensive for programmatic construction.
        backend = settings.audit_backend.lower()
        if backend != "jsonl":
            raise NotImplementedError(
                f"audit.backend {backend!r} is not implemented; only 'jsonl' "
                "is supported."
            )
        await self._log_jsonl(record)

    async def _log_jsonl(self, record: AuditRecord) -> None:
        path = Path(settings.audit_jsonl_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(record), ensure_ascii=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

