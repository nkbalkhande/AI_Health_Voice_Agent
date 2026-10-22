import asyncio
from functools import lru_cache

from langchain_experimental.text_splitter import SemanticChunker

from backend.core.config import settings
from backend.core.exceptions import IngestionError
from backend.core.logger import logger
from backend.domain.entities.document import Document
from backend.domain.interfaces.chunk_enricher import ChunkEnricher
from backend.domain.interfaces.pdf_chunker import PdfChunker
from backend.infrastructure.embeddings.langchain_adapter import EmbeddingProviderAdapter
from backend.infrastructure.embeddings.litellm_provider import LiteLLMEmbedding
from backend.rag.chunking.ids import deterministic_chunk_id
from backend.rag.chunking.page_overlap import PageSpec, overlap_pages, resolve_page_specs
from backend.rag.ingestion.chunk_results import PageChunkResult
from backend.rag.ingestion.document_mapping import append_image_documents, base_metadata
from backend.rag.ingestion.models import ParsedDocument

log = logger.getChild("rag.chunking.semantic")


_LITELLM_EMBEDDING_PROVIDERS = {"litellm", "openai", "azure", "azure_openai", "cohere"}


@lru_cache
def _build_text_splitter(
    breakpoint_threshold_type: str,
    breakpoint_threshold_amount: float,
) -> SemanticChunker:
    # R-58: this hardcodes LiteLLMEmbedding, so with the documented offline
    # `embedding.provider: bge` config the chunker silently kept calling the
    # LiteLLM API - chunk boundaries came from a different embedding space
    # than the vectors that get stored, and an offline run failed on a network
    # call it was configured not to make. Fail fast with a clear message until
    # the splitter is routed through ProviderFactory behind a sync adapter.
    provider = settings.embedding_provider.lower()
    if provider not in _LITELLM_EMBEDDING_PROVIDERS:
        raise IngestionError(
            f"chunking.strategy 'semantic' requires a LiteLLM-routed embedding "
            f"provider; embedding.provider is {provider!r}. Use "
            f"chunking.strategy 'hybrid', or set a LiteLLM embedding provider."
        )
    embeddings = EmbeddingProviderAdapter(LiteLLMEmbedding())
    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )


class SemanticPageChunker(PdfChunker):
    """Apply cross-page overlap, semantic chunking, and optional LLM enrichment."""
    def __init__(
        self,
        *,
        breakpoint_threshold_type: str | None = None,
        breakpoint_threshold_amount: float | None = None,
        batch_size: int | None = None,
        enricher: ChunkEnricher | None = None,

    ) -> None:

        self._breakpoint_threshold_type = (
            breakpoint_threshold_type or settings.semantic_chunk_breakpoint_type
        )
        self._breakpoint_threshold_amount = (
            breakpoint_threshold_amount or settings.semantic_chunk_breakpoint_amount
        )
        self._batch_size = batch_size or settings.semantic_chunk_batch_size
        self._splitter = _build_text_splitter(
            self._breakpoint_threshold_type,
            self._breakpoint_threshold_amount,
        )
        self._enricher = enricher

    async def chunk(
        self,
        parsed: ParsedDocument,
        *,
        source: str | None = None,
        metadata: dict | None = None,

    ) -> list[Document]:

        base_meta = base_metadata(parsed, metadata=metadata)
        resolved_source = source or parsed.filename
        page_specs = resolve_page_specs(parsed)
        if not page_specs:
            raise IngestionError(
                "Parsed PDF text contains no page content for semantic chunking"
            )

        page_contents = [spec.content for spec in page_specs]
        chunk_results = await self._chunk_pages(page_specs)
        if self._enricher is not None:
            chunk_results = await self._enricher.enrich(
                chunk_results,
                all_pages=page_contents,
            )

        documents: list[Document] = []

        for result in chunk_results:

            for index, chunk_text in enumerate(result.chunks):
                if not chunk_text.strip():
                    continue

                chunk_meta = {
                    **base_meta,
                    **result.metadata,
                    "content_type": "text",
                    "page_number": result.page_number,
                    "chunk_index": index,
                }

                documents.append(
                    Document(
                        id=deterministic_chunk_id(
                            resolved_source,
                            page=result.page_number,
                            chunk_index=index,
                            content=chunk_text,
                        ),
                        content=chunk_text,
                        metadata=chunk_meta,
                        source=resolved_source,
                    )
                )

        append_image_documents(
            documents,
            parsed,
            base_meta=base_meta,
            resolved_source=resolved_source,
        )
        return documents

    async def _chunk_pages(self, pages: list[PageSpec]) -> list[PageChunkResult]:
        if not pages:
            return []
        contents = [page.content for page in pages]
        # Page overlap carries a PDF page's run-on paragraph back onto the
        # page it started on. Worksheets are independent documents, so the
        # next sheet's opening rows must not be appended to this one.
        if any("sheet_name" in page.metadata for page in pages):
            overlapped_pages = [content.rstrip() for content in contents]
        else:
            overlapped_pages = overlap_pages(contents)
        overlap_count = sum(
            1
            for original, merged in zip(contents, overlapped_pages)
            if merged != original.rstrip()
        )
        if overlap_count:
            log.info(
                "Applied cross-page overlap to %d/%d pages before chunking",
                overlap_count,
                len(pages),
            )

        total_batches = (
            len(overlapped_pages) + self._batch_size - 1
        ) // self._batch_size
        results: list[PageChunkResult] = []
        for start in range(0, len(overlapped_pages), self._batch_size):
            batch_specs = pages[start : start + self._batch_size]
            batch_texts = overlapped_pages[start : start + self._batch_size]
            batch_results = await asyncio.gather(
                *[
                    self._chunk_page(
                        spec.page_number,
                        page_text,
                        metadata=spec.metadata,
                    )
                    for spec, page_text in zip(batch_specs, batch_texts)
                ]
            )
            results.extend(batch_results)
            log.info(
                "Completed semantic chunk batch %d/%d",
                start // self._batch_size + 1,
                total_batches,
            )
        return results

    async def _chunk_page(
        self,
        page_number: int,
        page_text: str,
        *,
        metadata: dict | None = None,
    ) -> PageChunkResult:
        page_meta = dict(metadata or {})
        if not page_text.strip():
            return PageChunkResult(
                page_number=page_number,
                page_content=page_text,
                chunks=[],
                metadata=page_meta,
            )

        chunks = await asyncio.to_thread(self._splitter.split_text, page_text)
        return PageChunkResult(
            page_number=page_number,
            page_content=page_text,
            chunks=chunks,
            metadata=page_meta,
        )

