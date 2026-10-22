import asyncio
from typing import Any

from backend.core.config import settings
from backend.core.logger import logger
from backend.domain.interfaces.chunk_enricher import ChunkEnricher
from backend.domain.interfaces.llm_provider import LLMProvider
from backend.rag.chunking.enrichment_prompt import DEFAULT_ENRICHMENT_SYSTEM_PROMPT
from backend.rag.chunking.enrichment_utils import (
    apply_enrichment,
    build_page_context,
    response_to_json_safe,
)
from backend.rag.ingestion.chunk_results import PageChunkResult

log = logger.getChild("infrastructure.chunking.llm_chunk_enricher")


class LlmChunkEnricher(ChunkEnricher):
    """Use the shared LLM provider to add missing page context before embedding."""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        max_completion_tokens: int | None = None,
        max_page_context_chars: int | None = None,
        include_neighbor_pages: bool | None = None,
        concurrent_chunks: int | None = None,
        system_prompt_template: str | None = None,
    ) -> None:
        self._llm = llm
        self._max_completion_tokens = (
            max_completion_tokens or settings.chunk_enrichment_max_completion_tokens
        )
        self._max_page_context_chars = (
            max_page_context_chars or settings.chunk_enrichment_max_page_context_chars
        )
        self._include_neighbor_pages = (
            include_neighbor_pages
            if include_neighbor_pages is not None
            else settings.chunk_enrichment_include_neighbor_pages
        )
        self._concurrent_chunks = (
            concurrent_chunks or settings.chunk_enrichment_concurrent_chunks
        )
        # R-29: the semaphore used to be created per page inside
        # _enrich_chunks_for_page, and pages were awaited one at a time, so
        # concurrency was capped per page and every page paid a full serial
        # round trip. One instance-scoped semaphore bounds TOTAL in-flight
        # calls while pages run concurrently.
        self._enrich_semaphore: asyncio.Semaphore | None = None
        self._system_prompt_template = (
            system_prompt_template
            or settings.chunk_enrichment_system_prompt
            or DEFAULT_ENRICHMENT_SYSTEM_PROMPT
        )
        # R-56: a template missing either placeholder used to silently drop the
        # page or the chunk from the enrichment prompt. Fail at construction.
        missing = [
            token
            for token in ("{PAGE_MARKDOWN}", "{CHUNK}")
            if token not in self._system_prompt_template
        ]
        if missing:
            raise ValueError(
                "chunking.semantic.enrichment.system_prompt is missing required "
                f"placeholder(s): {', '.join(missing)}"
            )

    async def enrich(
        self,
        page_results: list[PageChunkResult],
        *,
        all_pages: list[str],
    ) -> list[PageChunkResult]:
        if not page_results:
            return []

        total_pages = len(page_results)
        log.info(
            "Enriching semantic chunks across %d page(s), max %d concurrent "
            "LLM calls in flight.",
            total_pages,
            self._concurrent_chunks,
        )

        async def _enrich_page(
            page_index: int, page_result: PageChunkResult
        ) -> PageChunkResult:
            page_context = build_page_context(
                all_pages,
                page_index,
                current_page_content=page_result.page_content,
                max_chars=self._max_page_context_chars,
                include_neighbors=self._include_neighbor_pages,
            )
            enrichments = await self._enrich_chunks_for_page(
                page_context=page_context,
                chunks=page_result.chunks,
            )
            final_chunks = [
                apply_enrichment(chunk, enrich)
                for chunk, enrich in zip(
                    page_result.chunks, enrichments, strict=False
                )
            ]
            log.info(
                "Enriched page_number=%d (%d chunks)",
                page_result.page_number,
                len(final_chunks),
            )
            return PageChunkResult(
                page_number=page_result.page_number,
                page_content=page_result.page_content,
                chunks=final_chunks,
                enriched_chunks=enrichments,
                metadata=dict(page_result.metadata),
            )

        # Pages run concurrently; the instance semaphore inside
        # _enrich_chunks_for_page keeps total in-flight calls bounded.
        return list(
            await asyncio.gather(
                *(
                    _enrich_page(page_index, page_result)
                    for page_index, page_result in enumerate(page_results)
                )
            )
        )

    async def _enrich_chunks_for_page(
        self,
        *,
        page_context: str,
        chunks: list[str],
    ) -> list[dict[str, Any]]:
        if not chunks:
            return []

        if self._enrich_semaphore is None:
            # Created lazily so it binds to the running event loop.
            self._enrich_semaphore = asyncio.Semaphore(self._concurrent_chunks)
        semaphore = self._enrich_semaphore

        async def _invoke(chunk: str) -> dict[str, Any]:
            async with semaphore:
                return await self._llm_invoke(page_context, chunk)

        return list(await asyncio.gather(*(_invoke(chunk) for chunk in chunks)))

    async def _llm_invoke(self, page_content: str, chunk: str) -> dict[str, Any]:
        # R-56: str.format made any literal brace in an operator-supplied
        # template (e.g. a JSON example) raise KeyError per chunk, escaping as
        # an unhandled 500. .replace is inert to braces; __init__ validates
        # that both placeholders are present.
        prompt = self._system_prompt_template.replace(
            "{PAGE_MARKDOWN}", page_content
        ).replace("{CHUNK}", chunk)
        result = await self._llm.generate(
            prompt,
            max_tokens=self._max_completion_tokens,
        )
        if result.status != "answered":
            log.warning(
                "Chunk enrichment LLM ended with status=%s; keeping original chunk",
                result.status,
            )
            return {}
        return response_to_json_safe(result.text)
