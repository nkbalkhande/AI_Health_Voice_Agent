"""Filesystem checkpoints for VLM page parse + index-batch resume."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from backend.core.config import settings
from backend.core.exceptions import IngestionError
from backend.core.logger import logger

log = logger.getChild("rag.ingestion.checkpoint")

_SAFE_ID = re.compile(r"[^a-zA-Z0-9_-]+")


def content_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_doc_id(
    *,
    checkpoint_id: str | None,
    filename: str,
    content_hash: str,
) -> str:
    """Client checkpoint_id wins; else stable key from filename + content hash."""
    if checkpoint_id and checkpoint_id.strip():
        cleaned = _SAFE_ID.sub("_", checkpoint_id.strip()).strip("._-")
        if not cleaned or cleaned in {".", ".."}:
            raise IngestionError("Invalid checkpoint_id")
        return cleaned[:200]
    raw = f"{filename}:{content_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


_SETTINGS_PATH = Path(__file__).resolve().parents[3] / "config" / "settings.yaml"


def ingest_provenance() -> dict[str, Any]:
    """What produced this checkpoint: parser path, code SHA, settings sha.

    Sep 2 2026. The manifest used to record doc_id / content sha / pages only,
    so nobody could tell after the fact whether a collection was parsed with
    the Azure VLM or with Tesseract (the Aug 27 collection of record was
    Tesseract while main said ``use_vlm: true``). Values are read from the
    live settings singleton, i.e. what was actually in force, and are set
    once at checkpoint creation — a resumed checkpoint keeps the provenance
    of the run that started it.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        sha = completed.stdout.strip() if completed.returncode == 0 else ""
    except Exception:  # noqa: BLE001 - provenance must never fail an ingest
        sha = ""
    try:
        settings_sha = hashlib.sha256(_SETTINGS_PATH.read_bytes()).hexdigest()[:12]
    except OSError:
        settings_sha = "unknown"
    use_vlm = bool(getattr(settings, "ingestion_use_vlm", False))
    return {
        "parser_mode": "azure_vlm" if use_vlm else "docling_tesseract",
        "use_vlm": use_vlm,
        "xlsx_analyze_images": bool(getattr(settings, "ingestion_xlsx_analyze_images", False)),
        "analyze_images": bool(getattr(settings, "ingestion_analyze_images", False)),
        "git_sha": sha or "unknown",
        "settings_sha256_12": settings_sha,
        "embedding_model": getattr(settings, "embedding_model", None),
        "collection": getattr(settings, "vectordb_collection", None),
    }


class IngestionCheckpoint:
    """Persist page markdown and index-batch completion under checkpoint_dir."""

    def __init__(
        self,
        doc_id: str,
        *,
        filename: str,
        content_hash: str,
        root: str | Path | None = None,
    ) -> None:
        self.doc_id = doc_id
        self.filename = filename
        self.content_hash = content_hash
        base = Path(root or settings.ingestion_checkpoint_dir).resolve()
        self.dir = (base / doc_id).resolve()
        if not str(self.dir).startswith(str(base)):
            raise IngestionError("checkpoint path escapes checkpoint_dir")
        self.pages_dir = self.dir / "pages"
        self.manifest_path = self.dir / "manifest.json"
        self._manifest: dict[str, Any] = {
            "doc_id": doc_id,
            "filename": filename,
            "content_sha256": content_hash,
            "page_count": None,
            "parse_completed_pages": [],
            "index_batches": {},
            "index_batch_size": None,
            "provenance": ingest_provenance(),
        }
        self._load_or_init()

    def _load_or_init(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            try:
                loaded = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    prior_hash = loaded.get("content_sha256")
                    if (
                        prior_hash
                        and prior_hash != self.content_hash
                    ):
                        raise IngestionError(
                            "Checkpoint content hash mismatch for "
                            f"{self.doc_id}: stored={prior_hash} "
                            f"upload={self.content_hash}"
                        )
                    self._manifest.update(loaded)
            except IngestionError:
                raise
            except (OSError, json.JSONDecodeError) as exc:
                log.warning(
                    "Corrupt checkpoint manifest %s: %s", self.manifest_path, exc
                )
        self._manifest["doc_id"] = self.doc_id
        self._manifest["filename"] = self.filename
        self._manifest["content_sha256"] = self.content_hash
        self._persist()

    def _persist(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self._manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def set_page_count(self, page_count: int) -> None:
        self._manifest["page_count"] = page_count
        self._persist()

    @property
    def page_count(self) -> int | None:
        count = self._manifest.get("page_count")
        return int(count) if count is not None else None

    def page_path(self, page_number: int) -> Path:
        return self.pages_dir / f"{page_number:04d}.md"

    def has_page(self, page_number: int) -> bool:
        completed = self._manifest.get("parse_completed_pages") or []
        return page_number in completed and self.page_path(page_number).exists()

    def load_page(self, page_number: int) -> str:
        return self.page_path(page_number).read_text(encoding="utf-8")

    def save_page(self, page_number: int, markdown: str) -> None:
        self.page_path(page_number).write_text(markdown, encoding="utf-8")
        completed = list(self._manifest.get("parse_completed_pages") or [])
        if page_number not in completed:
            completed.append(page_number)
            completed.sort()
            self._manifest["parse_completed_pages"] = completed
        self._persist()

    def range_complete(self, start_page: int, end_page: int) -> bool:
        return all(self.has_page(p) for p in range(start_page, end_page + 1))

    def load_pages(self, page_count: int) -> list[str]:
        pages: list[str] = []
        for page_number in range(1, page_count + 1):
            if self.has_page(page_number):
                pages.append(self.load_page(page_number))
            else:
                pages.append("")
        return pages

    def ensure_index_batch_size(self, batch_size: int) -> None:
        prior = self._manifest.get("index_batch_size")
        if prior is not None and int(prior) != int(batch_size):
            log.warning(
                "index_batch_size changed (%s -> %s); clearing index_batches",
                prior,
                batch_size,
            )
            self._manifest["index_batches"] = {}
        self._manifest["index_batch_size"] = int(batch_size)
        self._persist()

    def is_index_batch_done(self, batch_key: str) -> bool:
        batches = self._manifest.get("index_batches") or {}
        entry = batches.get(batch_key)
        return isinstance(entry, dict) and entry.get("status") == "done"

    def mark_index_batch_done(self, batch_key: str, document_ids: list[str]) -> None:
        batches = dict(self._manifest.get("index_batches") or {})
        batches[batch_key] = {"status": "done", "document_ids": list(document_ids)}
        self._manifest["index_batches"] = batches
        self._persist()

    def clear_index_batch(self, batch_key: str) -> None:
        batches = dict(self._manifest.get("index_batches") or {})
        if batch_key in batches:
            del batches[batch_key]
            self._manifest["index_batches"] = batches
            self._persist()
