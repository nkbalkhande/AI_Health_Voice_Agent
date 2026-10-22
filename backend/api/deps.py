from functools import lru_cache

from backend.application.use_cases.chat import ChatUseCase
from backend.application.use_cases.evaluate_rag_pipeline import EvaluateRagPipelineUseCase
from backend.application.use_cases.ingest_documents import IngestDocumentsUseCase
from backend.application.use_cases.query_documents import QueryUseCase
from backend.core.factory import ProviderFactory
from backend.rag.ingestion.facade.ingestion_facade import IngestionFacade
from backend.rag.ingestion.parsers.docling_parser import DoclingParser


@lru_cache
def get_query_use_case() -> QueryUseCase:
    return QueryUseCase(
        llm=ProviderFactory.get_llm(),
        retriever=ProviderFactory.get_retriever(),
        reranker=ProviderFactory.get_reranker(),
        preprocessor=ProviderFactory.get_chunk_preprocessor(),
        compressor=ProviderFactory.get_context_compressor(),
    )


@lru_cache
def get_document_parser() -> DoclingParser:
    from backend.core.config import settings

    return DoclingParser(
        use_vlm=settings.ingestion_use_vlm,
        analyze_images=settings.ingestion_analyze_images,
    )


# R-28: these two rebuilt every provider per request — a fresh Qdrant client
# and a fresh BM25 sparse model load on every call, and a store separate from
# the query path (fatal for :memory: / on-disk Qdrant modes).
@lru_cache
def get_ingest_use_case() -> IngestDocumentsUseCase:
    return IngestDocumentsUseCase(
        embedding=ProviderFactory.get_embedding(),
        vector_db=ProviderFactory.get_vector_db(),
    )


def get_ingestion_facade() -> IngestionFacade:
    """Composition root for multi-format ingestion."""
    return IngestionFacade(
        chunker=ProviderFactory.get_pdf_chunker(),
        embedding=ProviderFactory.get_embedding(),
        vector_db=ProviderFactory.get_vector_db(),
        docling_parser=get_document_parser(),
    )


@lru_cache
def get_evaluate_pipeline_use_case() -> EvaluateRagPipelineUseCase:
    return EvaluateRagPipelineUseCase(
        query_use_case=get_query_use_case(),
        runner=ProviderFactory.get_evaluation_runner(),
    )


@lru_cache
def get_chat_use_case() -> ChatUseCase:
    from backend.core.config import settings

    checkpointer_dsn = (
        settings.postgres_dsn
        if settings.agent_checkpointer.lower() == "postgres"
        else None
    )
    return ChatUseCase(
        llm=ProviderFactory.get_llm(),
        retriever=ProviderFactory.get_retriever(),
        reranker=ProviderFactory.get_reranker(),
        preprocessor=ProviderFactory.get_chunk_preprocessor(),
        # R-65: chat used to skip context compression entirely, so /chat and
        # /query answered the same question from different context.
        compressor=ProviderFactory.get_context_compressor(),
        history=ProviderFactory.get_chat_history_handler(),
        web_search=ProviderFactory.get_web_search(),
        cache=ProviderFactory.get_cache(),
        checkpointer_dsn=checkpointer_dsn,
        voice_understand_llm=ProviderFactory.get_voice_understand_llm(),
    )


def get_speech_client():
    return ProviderFactory.get_speech_client()


def get_live_transcriber():
    return ProviderFactory.get_live_transcriber()


def get_voice_use_case():
    from backend.application.use_cases.voice import VoiceUseCase

    return VoiceUseCase(speech=get_speech_client(), chat=get_chat_use_case())
