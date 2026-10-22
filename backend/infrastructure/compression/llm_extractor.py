from dataclasses import replace

from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from backend.core.config import settings
from backend.core.logger import logger
from backend.domain.entities.chunk import Chunk
from backend.domain.interfaces.context_compressor import ContextCompressor

log = logger.getChild("infrastructure.compression.llm_extractor")

_CHUNK_ID_KEY = "_chunk_id"


class LLMChainExtractorCompressor(ContextCompressor):
    """Contextual compression via LangChain's ``LLMChainExtractor``.

    Each reranked chunk is passed through an LLM that extracts only the spans
    relevant to the query, dropping chunks that contribute nothing. Original
    chunk identity and metadata (filename, page number, ...) are preserved by
    keying compressed output back to the source chunk.
    """

    def __init__(self) -> None:
        model = settings.compression_model
        if model.lower().startswith("openai/"):
            llm = ChatOpenAI(
                model=model.split("/", 1)[1],
                api_key=settings.openai_api_key,
                temperature=settings.compression_temperature,
                max_completion_tokens=1024,
            )
        else:
            deployment = (
                model.split("/", 1)[1]
                if model.lower().startswith("azure/")
                else model
            )
            llm = AzureChatOpenAI(
                azure_deployment=deployment,
                model=deployment,
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                temperature=settings.compression_temperature,
                max_completion_tokens=1024,
            )
        self._compressor = LLMChainExtractor.from_llm(llm)

    async def process(
        self,
        query: str,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        if not chunks:
            return []

        by_id = {chunk.id: chunk for chunk in chunks}
        documents =   [
    Document(
        page_content=chunk.content,
        metadata={**chunk.metadata, _CHUNK_ID_KEY: chunk.id},
    )
    for chunk in chunks
    ]

        try:
            compressed = await self._compressor.acompress_documents(
                documents, query
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully to reranked chunks
            log.warning(
                "Context compression failed (%s); falling back to reranked chunks.",
                exc,
            )
            return chunks

        result: list[Chunk] = []
        for doc in compressed:
            chunk_id = doc.metadata.get(_CHUNK_ID_KEY)
            if not isinstance(chunk_id, str):
                continue
            source = by_id.get(chunk_id)
            if source is None:
                continue
            result.append(replace(source, content=doc.page_content))

        # If the extractor filtered everything out, keep the reranked chunks so
        # the pipeline never loses all context.
        if not result:
            log.warning(
                "Context compression dropped all %d chunks; falling back to "
                "reranked chunks.",
                len(chunks),
            )
            return chunks
        return result
