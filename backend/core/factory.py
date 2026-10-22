from functools import lru_cache

from backend.application.chat.history import ChatHistoryHandler
from backend.core.config import settings
from backend.core.exceptions import ProviderNotFoundError
from backend.domain.interfaces.chat_thread_repository import ChatThreadRepository
from backend.domain.interfaces.chunk_enricher import ChunkEnricher
from backend.domain.interfaces.chunk_preprocessor import ChunkPreprocessor
from backend.domain.interfaces.context_compressor import ContextCompressor
from backend.domain.interfaces.embedding_provider import EmbeddingProvider
from backend.domain.interfaces.evaluation_runner import EvaluationRunner
from backend.domain.interfaces.llm_provider import LLMProvider
from backend.domain.interfaces.pdf_chunker import PdfChunker
from backend.domain.interfaces.reranker_provider import RerankerProvider
from backend.domain.interfaces.retriever import Retriever
from backend.domain.interfaces.vector_db_provider import VectorDBProvider

# Legacy provider names still resolve model prefixes via model_resolver
_LITELLM_LLM_PROVIDERS = frozenset(
    {
        "litellm",
        "anthropic",
        "openai",
        "azure",
        "azure_openai",
        "gemini",
        "google",
        "bedrock",
        "cohere",
        "mistral",
        "ollama",
        "groq",
        "deepseek",
    }
)

_LITELLM_EMBEDDING_PROVIDERS = frozenset(
    {
        "litellm",
        "openai",
        "azure",
        "azure_openai",
        "cohere",
        "voyage",
        "bedrock",
        "gemini",
        "huggingface",
    }
)


class ProviderFactory:
    """Config-driven factory. Swap providers via settings.yaml / env."""

    @staticmethod
    def get_llm() -> LLMProvider:
        provider = settings.llm_provider.lower()
        if provider in _LITELLM_LLM_PROVIDERS:
            from backend.infrastructure.llm.litellm_provider import LiteLLMProvider

            return LiteLLMProvider()
        raise ProviderNotFoundError(
            f"Unknown LLM provider: {provider}. "
            f"Use litellm with a full model id "
            f"(e.g. anthropic/claude-sonnet-4-20250514)."
        )

    @staticmethod
    @lru_cache
    def get_embedding() -> EmbeddingProvider:
        provider = settings.embedding_provider.lower()
        if provider == "bge":
            from backend.infrastructure.embeddings.bge import BGEEmbedding

            return BGEEmbedding()
        if provider in _LITELLM_EMBEDDING_PROVIDERS:
            from backend.infrastructure.embeddings.litellm_provider import (
                LiteLLMEmbedding,
            )

            return LiteLLMEmbedding()
        raise ProviderNotFoundError(
            f"Unknown embedding provider: {provider}. "
            "Use litellm + model (e.g. openai/text-embedding-3-small) or bge for local."
        )

    # R-28: one shared instance across query and ingest. Separate instances
    # meant a separate Qdrant client (and, in :memory:/path modes, a separate
    # store) plus a repeated BM25 sparse-model load.
    @staticmethod
    @lru_cache
    def get_vector_db() -> VectorDBProvider:
        provider = settings.vectordb_provider.lower()
        if provider == "qdrant":
            from backend.infrastructure.vectordb.qdrant import QdrantVectorDB

            return QdrantVectorDB()
        if provider == "chroma":
            from backend.infrastructure.vectordb.chroma import ChromaVectorDB

            return ChromaVectorDB()
        if provider == "pinecone":
            from backend.infrastructure.vectordb.pinecone import PineconeVectorDB

            return PineconeVectorDB()
        if provider == "azure_ai_search":
            from backend.infrastructure.vectordb.azure_ai_search import (
                AzureAISearchVectorDB,
            )

            return AzureAISearchVectorDB()
        raise ProviderNotFoundError(f"Unknown vector DB provider: {provider}")

    @staticmethod
    def get_reranker() -> RerankerProvider:
        provider = settings.reranker_provider.lower()
        if provider in ("bge", "lexical"):
            from backend.infrastructure.rerankers.bge_reranker import (
                LexicalOverlapReranker,
            )

            return LexicalOverlapReranker()
        if provider == "cohere":
            from backend.infrastructure.rerankers.cohere_reranker import CohereReranker

            return CohereReranker()
        raise ProviderNotFoundError(f"Unknown reranker provider: {provider}")

    @staticmethod
    def get_context_compressor() -> ContextCompressor | None:
        if not settings.compression_enabled:
            return None
        from backend.infrastructure.compression.llm_extractor import (
            LLMChainExtractorCompressor,
        )

        return LLMChainExtractorCompressor()

    @staticmethod
    def get_chunk_preprocessor() -> ChunkPreprocessor | None:
        if not settings.preprocessing_enabled:
            return None
        provider = settings.preprocessing_provider.lower()
        if provider == "embeddings":
            from backend.infrastructure.preprocessing.embeddings_filter import (
                EmbeddingsChunkPreprocessor,
            )

            return EmbeddingsChunkPreprocessor()
        raise ProviderNotFoundError(f"Unknown chunk preprocessor provider: {provider}")

    @staticmethod
    def get_enrichment_llm() -> LLMProvider:
        provider = settings.llm_provider.lower()
        if provider in _LITELLM_LLM_PROVIDERS:
            from backend.infrastructure.llm.litellm_provider import LiteLLMProvider

            return LiteLLMProvider(
                provider=provider,
                model=settings.chunk_enrichment_model or settings.llm_model,
                max_completion_tokens=settings.chunk_enrichment_max_completion_tokens,
            )
        raise ProviderNotFoundError(
            f"Unknown LLM provider: {provider}. "
            f"Use litellm with a full model id "
            f"(e.g. anthropic/claude-sonnet-4-20250514)."
        )

    @staticmethod
    @lru_cache
    def get_chunk_enricher() -> ChunkEnricher | None:
        if not settings.chunk_enrichment_enabled:
            return None
        from backend.infrastructure.chunking.llm_chunk_enricher import LlmChunkEnricher

        return LlmChunkEnricher(llm=ProviderFactory.get_enrichment_llm())

    @staticmethod
    @lru_cache
    def get_pdf_chunker() -> PdfChunker:
        strategy = settings.chunking_strategy.lower()
        if strategy == "semantic":
            from backend.rag.chunking.semantic import SemanticPageChunker

            return SemanticPageChunker(enricher=ProviderFactory.get_chunk_enricher())
        if strategy == "hybrid":
            from backend.rag.chunking.hybrid import DoclingHybridChunker

            return DoclingHybridChunker()
        raise ProviderNotFoundError(f"Unknown chunking strategy: {strategy}")

    @staticmethod
    def get_retriever(
        *,
        embedding: EmbeddingProvider | None = None,
        vector_db: VectorDBProvider | None = None,
    ) -> Retriever:
        strategy = settings.retrieval_strategy.lower()
        embedding = embedding or ProviderFactory.get_embedding()
        vector_db = vector_db or ProviderFactory.get_vector_db()

        if strategy == "vector":
            from backend.rag.retrieval.vector import VectorRetriever

            return VectorRetriever(embedding=embedding, vector_db=vector_db)
        if strategy == "hybrid":
            from backend.rag.retrieval.hybrid import HybridRetriever

            return HybridRetriever(embedding=embedding, vector_db=vector_db)
        raise ProviderNotFoundError(f"Unknown retrieval strategy: {strategy}")

    @staticmethod
    @lru_cache
    def get_chat_thread_repository() -> ChatThreadRepository:
        store = settings.chat_store.lower()
        if store == "memory":
            from backend.infrastructure.chat.memory_repository import (
                InMemoryChatThreadRepository,
            )

            return InMemoryChatThreadRepository()
        if store == "postgres":
            from backend.infrastructure.chat.postgres_repository import (
                PostgresChatThreadRepository,
            )

            return PostgresChatThreadRepository.from_settings()
        raise ProviderNotFoundError(f"Unknown chat store: {store}")

    @staticmethod
    @lru_cache
    def get_web_search():
        if not settings.web_search_enabled:
            return None
        provider = settings.web_search_provider.lower()
        if provider == "serpapi":
            from backend.core.exceptions import ProviderNotConfiguredError
            from backend.infrastructure.web.serpapi import SerpApiSearch

            if not settings.serpapi_api_key:
                raise ProviderNotConfiguredError("SerpApi API key is not configured")
            return SerpApiSearch(
                settings.serpapi_api_key,
                max_results=settings.web_search_max_results,
                timeout_seconds=settings.web_search_timeout_seconds,
                domains=settings.web_search_domains,
                engine=settings.web_search_engine,
                language=settings.web_search_language,
                country=settings.web_search_country,
            )
        if provider == "duckduckgo":
            from backend.infrastructure.web.duckduckgo import DuckDuckGoSearch

            return DuckDuckGoSearch(
                max_results=settings.web_search_max_results,
                timeout_seconds=settings.web_search_timeout_seconds,
                domains=settings.web_search_domains,
            )
        raise ProviderNotFoundError(f"Unknown web search provider: {provider}")

    @staticmethod
    @lru_cache
    def get_speech_client():
        from backend.core.exceptions import ProviderNotConfiguredError
        from backend.infrastructure.voice.composite import CompositeSpeechClient
        from backend.infrastructure.voice.deepgram import (
            DeepgramSpeechClient,
            DeepgramTTSClient,
        )
        from backend.infrastructure.voice.elevenlabs import ElevenLabsSpeechClient

        if not settings.deepgram_api_key:
            raise ProviderNotConfiguredError("Deepgram API key is not configured")
        stt = DeepgramSpeechClient(
            settings.deepgram_api_key,
            stt_model=settings.voice_stt_model,
            timeout_seconds=settings.voice_timeout_seconds,
        )
        provider = settings.voice_tts_provider.strip().lower()
        if provider == "deepgram":
            tts = DeepgramTTSClient(
                settings.deepgram_api_key,
                model=settings.deepgram_tts_model,
                voices=tuple(str(v.get("id")) for v in settings.deepgram_tts_voices),
                timeout_seconds=settings.voice_timeout_seconds,
            )
        elif provider == "elevenlabs":
            if not settings.elevenlabs_api_key:
                raise ProviderNotConfiguredError("ElevenLabs API key is not configured")
            tts = ElevenLabsSpeechClient(
                settings.elevenlabs_api_key,
                voice_id=settings.elevenlabs_voice_id,
                model=settings.elevenlabs_model,
                timeout_seconds=settings.voice_timeout_seconds,
            )
        else:
            raise ProviderNotFoundError(f"Unknown TTS provider: {provider}")
        return CompositeSpeechClient(stt=stt, tts=tts)

    @staticmethod
    def get_voice_understand_llm() -> LLMProvider | None:
        if not settings.voice_understand_model:
            return None
        from backend.infrastructure.llm.litellm_provider import LiteLLMProvider

        return LiteLLMProvider(model=settings.voice_understand_model)

    @staticmethod
    def get_live_transcriber():
        from backend.core.exceptions import ProviderNotConfiguredError
        from backend.infrastructure.voice.deepgram import DeepgramLiveTranscriber

        if not settings.deepgram_api_key:
            raise ProviderNotConfiguredError("Deepgram API key is not configured")
        return DeepgramLiveTranscriber(
            settings.deepgram_api_key,
            model=settings.voice_stt_model,
            endpointing_ms=settings.voice_endpointing_ms,
            utterance_end_ms=settings.voice_utterance_end_ms,
        )

    @staticmethod
    @lru_cache
    def get_cache():
        if not settings.redis_enabled:
            return None
        from backend.infrastructure.cache.redis_cache import RedisCache

        return RedisCache(
            settings.redis_url,
            ttl_seconds=settings.redis_web_cache_ttl_seconds,
        )

    @staticmethod
    @lru_cache
    def get_chat_history_handler() -> ChatHistoryHandler:
        return ChatHistoryHandler(ProviderFactory.get_chat_thread_repository())

    @staticmethod
    def get_evaluation_runner() -> EvaluationRunner:
        provider = settings.evaluation_provider.lower()
        if provider == "ragas":
            from backend.evaluation.ragas_runner import RagasRunner

            return RagasRunner()
        if provider == "deepeval":
            from backend.evaluation.deepeval_runner import DeepEvalRunner

            return DeepEvalRunner()
        raise ProviderNotFoundError(f"Unknown evaluation provider: {provider}")
