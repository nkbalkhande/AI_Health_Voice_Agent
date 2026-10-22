from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

from starlette.concurrency import run_in_threadpool

from backend.core.exceptions import IngestionError
from backend.core.logger import logger
from backend.domain.interfaces.embedding_provider import EmbeddingProvider
from backend.domain.interfaces.pdf_chunker import PdfChunker
from backend.domain.interfaces.vector_db_provider import VectorDBProvider
from backend.rag.ingestion.factory.parser_factory import ParserFactory
from backend.rag.ingestion.models import ParsedDocument
from backend.rag.ingestion.parsers.base_parser import BaseDocumentParser

log = logger.getChild("rag.ingestion.facade.ingestion_facade")

_CHUNK_RETRY_ATTEMPTS = 4
_CHUNK_RETRY_BASE_SECONDS = 2.0


@dataclass
class IngestionFacadeResult:
    document_count: int
    chunk_count: int
    filename: str
    page_count: int
    image_count: int
    has_content: bool


class IngestionFacade:
    """Unified entry point for multi-format document ingestion."""

    def __init__(
        self,
        *,
        chunker: PdfChunker,
        embedding: EmbeddingProvider,
        vector_db: VectorDBProvider,
        docling_parser: BaseDocumentParser | None = None,
        parser_factory: type[ParserFactory] = ParserFactory,
    ) -> None:
        self._chunker = chunker
        self._embedding = embedding
        self._vector_db = vector_db
        self._docling_parser = docling_parser
        self._parser_factory = parser_factory

    async def ingest(
        self,
        document_bytes: bytes,
        *,
        filename: str,
        source: str | None = None,
        metadata: dict | None = None,
    ) -> IngestionFacadeResult:
        started = time.perf_counter()
        file_type = Path(filename).suffix.lower()
        log.info(
            "ingestion.started filename=%s file_type=%s status=started",
            filename,
            file_type,
        )
        try:
            self._validate_input(document_bytes, filename=filename)

            parser = self._parser_factory.create(
                filename,
                docling_parser=self._docling_parser,
            )
            parser_type = type(parser).__name__

            log.info(
                "parsing.started filename=%s file_type=%s parser_type=%s",
                filename,
                file_type,
                parser_type,
            )
            parse_started = time.perf_counter()
            parsed = await run_in_threadpool(
                parser.parse,
                document_bytes,
                filename=filename,
            )
            log.info(
                "parsing.completed filename=%s parser_type=%s duration=%.3fs status=ok",
                filename,
                parser_type,
                time.perf_counter() - parse_started,
            )

            self._validate_parsed(parsed)

            if not parsed.has_content:
                raise IngestionError(
                    "Document contains no extractable text, tables, or captions"
                )

            resolved_source = source or filename

            log.info(
                "chunking.started filename=%s parser_type=%s",
                filename,
                parser_type,
            )
            chunk_started = time.perf_counter()
            documents = await self._chunker.chunk(
                parsed,
                source=resolved_source,
                metadata=metadata,
            )
            for doc in documents:
                doc.content = f"{doc.content}\n\nFile name : {parsed.filename}"
            log.info(
                "chunking.completed filename=%s duration=%.3fs chunk_count=%d status=ok",
                filename,
                time.perf_counter() - chunk_started,
                len(documents),
            )

            log.info("embedding.started filename=%s", filename)
            embed_started = time.perf_counter()
            texts = [doc.content for doc in documents]
            embeddings = await self._embed_with_chunk_retry(texts, filename=filename)
            log.info(
                "embedding.completed filename=%s duration=%.3fs status=ok",
                filename,
                time.perf_counter() - embed_started,
            )

            log.info("vector_upsert.started filename=%s", filename)
            upsert_started = time.perf_counter()
            await self._vector_db.add_documents(documents, embeddings)
            log.info(
                "vector_upsert.completed filename=%s duration=%.3fs status=ok",
                filename,
                time.perf_counter() - upsert_started,
            )

            result = IngestionFacadeResult(
                document_count=len(documents),
                chunk_count=len(documents),
                filename=parsed.filename,
                page_count=len(parsed.pages or []),
                image_count=len(parsed.images or []),
                has_content=parsed.has_content,
            )
            log.info(
                "ingestion.completed filename=%s file_type=%s parser_type=%s "
                "duration=%.3fs status=ok",
                filename,
                file_type,
                parser_type,
                time.perf_counter() - started,
            )
            return result
        except Exception:
            log.exception(
                "ingestion.completed filename=%s file_type=%s duration=%.3fs status=error",
                filename,
                file_type,
                time.perf_counter() - started,
            )
            raise

    async def _embed_with_chunk_retry(
        self, texts: list[str], *, filename: str
    ) -> list[list[float]]:
        """Embed all chunks; if that fails, embed and retry each chunk on its own."""
        try:
            return await self._embedding.embed_documents(texts)
        except Exception as exc:
            log.warning(
                "embedding.batch_failed filename=%s chunks=%d error=%s; "
                "retrying chunk by chunk",
                filename,
                len(texts),
                exc,
            )

        embeddings: list[list[float] | None] = [None] * len(texts)
        for attempt in range(1, _CHUNK_RETRY_ATTEMPTS + 1):
            pending = [i for i, vector in enumerate(embeddings) if vector is None]
            if not pending:
                break
            if attempt > 1:
                log.warning(
                    "embedding.retry filename=%s attempt=%d/%d failed_chunks=%d",
                    filename,
                    attempt,
                    _CHUNK_RETRY_ATTEMPTS,
                    len(pending),
                )
                await asyncio.sleep(_CHUNK_RETRY_BASE_SECONDS * 2 ** (attempt - 2))
            for index in pending:
                try:
                    embeddings[index] = (
                        await self._embedding.embed_documents([texts[index]])
                    )[0]
                except Exception as exc:
                    log.debug("Chunk %d embedding failed: %s", index, exc)

        missing = sum(vector is None for vector in embeddings)
        if missing:
            raise IngestionError(
                f"{missing} of {len(texts)} chunks could not be embedded "
                f"after {_CHUNK_RETRY_ATTEMPTS} attempts per chunk"
            )
        return embeddings  # type: ignore[return-value]

    @staticmethod
    def _validate_input(document_bytes: bytes, *, filename: str) -> None:
        if not filename or not str(filename).strip():
            raise IngestionError("filename is required for ingestion")
        if document_bytes is None:
            raise IngestionError("document_bytes is required for ingestion")
        if len(document_bytes) == 0:
            raise IngestionError("document_bytes must not be empty")

    @staticmethod
    def _validate_parsed(parsed: ParsedDocument) -> None:
        if not isinstance(parsed, ParsedDocument):
            raise IngestionError(
                f"Parser returned unexpected type: {type(parsed).__name__}"
            )
