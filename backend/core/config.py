import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# R-69: the ONE load_dotenv for the process. It used to be repeated in four
# modules, so which .env won depended on import order, and a test or script
# that imported a provider module first got a different environment than one
# that imported config first. Everything reads settings from here.
load_dotenv(find_dotenv() or ".env")

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def _load_yaml_config() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        return {}
    with _CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_yaml = _load_yaml_config()
_vectordb_yaml = _yaml.get("vectordb", {})
_qdrant_yaml = _vectordb_yaml.get("qdrant", {})
_chroma_yaml = _vectordb_yaml.get("chroma", {})
_chat_yaml = _yaml.get("chat", {})
_web_search_yaml = _yaml.get("web_search", {})
_voice_yaml = _yaml.get("voice", {})
_agent_yaml = _yaml.get("agent", {})
_redis_yaml = _yaml.get("redis", {})
# R-86/R-73: chat.mysql parsing removed - the MySQL chat store was never
# implemented (app/infrastructure/chat/mysql/ was an empty package).
_preprocessing_yaml = _yaml.get("preprocessing", {})
_evaluation_yaml = _yaml.get("evaluation", {})
_ragas_yaml = _evaluation_yaml.get("ragas", {})
_deepeval_yaml = _evaluation_yaml.get("deepeval", {})
_authority_reordering_yaml = _yaml.get("authority_reordering", {})
_chunking_yaml = _yaml.get("chunking", {})
_semantic_chunking_yaml = _chunking_yaml.get("semantic", {})
_chunk_enrichment_yaml = _semantic_chunking_yaml.get("enrichment", {})
_logging_yaml = _yaml.get("logging", {})
_ingestion_yaml = _yaml.get("ingestion", {})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "GenAI Platform Framework"
    debug: bool = False
    log_level: str = Field(default=_logging_yaml.get("level", "INFO"))
    log_file_path: str | None = Field(
        default=_logging_yaml.get("file_path", "./data/logs/backend.log")
    )

    llm_provider: str = Field(
        default=_yaml.get("llm", {}).get("provider", "litellm")
    )
    llm_model: str = Field(
        default=_yaml.get("llm", {}).get(
            "model", "anthropic/claude-sonnet-4-20250514"
        )
    )
    llm_max_completion_tokens: int = Field(
        default=_yaml.get("llm", {}).get("max_completion_tokens", 8192)
    )
    llm_reasoning_effort: str | None = Field(
        default=_yaml.get("llm", {}).get("reasoning_effort", None)
    )
    llm_timeout: float = Field(
        default=float(_yaml.get("llm", {}).get("timeout", 300))
    )
    litellm_llm_kwargs: dict[str, Any] = Field(
        default_factory=lambda: _yaml.get("llm", {}).get("kwargs", {})
    )
    openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    # R-50: one api-version field resolved by pydantic-settings, not two split
    # import-time os.getenv reads. AZURE_OPENAI_API_VERSION is the documented
    # variable and wins; OPENAI_API_VERSION stays accepted for back-compat.
    # openai_api_version is kept as an alias property so existing call sites
    # keep working.
    azure_openai_api_version: str | None = Field(
        default="2025-03-01-preview",
        validation_alias=AliasChoices(
            "AZURE_OPENAI_API_VERSION", "OPENAI_API_VERSION"
        ),
    )
    # R-86: azure_openai_deployment and anthropic_api_key were removed - both
    # were read by nothing. Azure deployments are derived from the model id
    # (see judge_config._openai_model_id) and Anthropic models route through
    # litellm, which reads ANTHROPIC_API_KEY from the environment directly.

    @property
    def openai_api_version(self) -> str | None:
        """Back-compat alias for :attr:`azure_openai_api_version`."""
        return self.azure_openai_api_version

    chunking_strategy: str = Field(
        default=_chunking_yaml.get("strategy", "semantic")
    )
    semantic_chunk_breakpoint_type: str = Field(
        default=_semantic_chunking_yaml.get("breakpoint_threshold_type", "percentile")
    )
    semantic_chunk_breakpoint_amount: float = Field(
        default=_semantic_chunking_yaml.get("breakpoint_threshold_amount", 90)
    )
    semantic_chunk_batch_size: int = Field(
        default=_semantic_chunking_yaml.get("batch_size", 5)
    )
    chunk_enrichment_enabled: bool = Field(
        default=_chunk_enrichment_yaml.get("enabled", False)
    )
    chunk_enrichment_model: str | None = Field(
        default=_chunk_enrichment_yaml.get("model")
    )
    chunk_enrichment_max_completion_tokens: int = Field(
        default=_chunk_enrichment_yaml.get("max_completion_tokens", 1024)
    )
    chunk_enrichment_max_page_context_chars: int = Field(
        default=_chunk_enrichment_yaml.get("max_page_context_chars", 100_000)
    )
    chunk_enrichment_include_neighbor_pages: bool = Field(
        default=_chunk_enrichment_yaml.get("include_neighbor_pages", True)
    )
    chunk_enrichment_concurrent_chunks: int = Field(
        default=_chunk_enrichment_yaml.get("concurrent_chunks", 5)
    )
    chunk_enrichment_system_prompt: str | None = Field(
        default=_chunk_enrichment_yaml.get("system_prompt")
    )

    embedding_provider: str = Field(
        default=_yaml.get("embedding", {}).get("provider", "litellm")
    )
    embedding_model: str = Field(
        default=_yaml.get("embedding", {}).get(
            "model", "openai/text-embedding-3-small"
        )
    )
    litellm_embedding_kwargs: dict[str, Any] = Field(
        default_factory=lambda: _yaml.get("embedding", {}).get("kwargs", {})
    )
    embedding_max_tokens_per_request: int = Field(
        default=_yaml.get("embedding", {}).get("max_tokens_per_request", 250_000)
    )
    embedding_max_batch_size: int = Field(
        default=_yaml.get("embedding", {}).get("max_batch_size", 2048)
    )
    embedding_timeout: float = Field(
        default=float(_yaml.get("embedding", {}).get("timeout", 60))
    )

    vectordb_provider: str = Field(
        default=_vectordb_yaml.get("provider", "qdrant")
    )
    vectordb_collection: str = Field(
        default=_vectordb_yaml.get("collection", "documents-normal")
    )
    qdrant_url: str = Field(default=_qdrant_yaml.get("url", ":memory:"))
    qdrant_path: str | None = Field(default=_qdrant_yaml.get("path"))
    qdrant_dense_dim: int = Field(default=_qdrant_yaml.get("dense_dim", 0))
    qdrant_api_key: str | None = Field(default=_qdrant_yaml.get("api_key"))
    qdrant_timeout: int = Field(default=_qdrant_yaml.get("timeout", 120))
    qdrant_upsert_batch_size: int = Field(
        default=_qdrant_yaml.get("upsert_batch_size", 32)
    )
    # Hybrid fusion knobs. Defaults reproduce the historical hard-coded behavior
    # exactly (RRF, 2x candidate pool per arm, exact/brute-force search) so the
    # pinned baseline is unaffected until these are deliberately tuned under eval.
    qdrant_hybrid_fusion: str = Field(
        default=_qdrant_yaml.get("hybrid_fusion", "rrf")
    )
    qdrant_prefetch_multiplier: int = Field(
        default=_qdrant_yaml.get("prefetch_multiplier", 2)
    )
    qdrant_search_exact: bool = Field(
        default=_qdrant_yaml.get("search_exact", True)
    )
    chroma_persist_dir: str = Field(
        default=_chroma_yaml.get(
            "persist_dir",
            _vectordb_yaml.get("persist_dir", "./data/chroma"),
        )
    )

    # R-40: the default used to be "bge" - a lexical-overlap stub - so a
    # truncated or renamed settings.yaml silently swapped Cohere semantic
    # reranking for bag-of-words with no error.
    reranker_provider: str = Field(
        default=_yaml.get("reranker", {}).get("provider", "cohere")
    )
    reranker_model: str = Field(
        default=_yaml.get("reranker", {}).get("model", "cohere/rerank-english-v3.0")
    )
    reranker_api_key: str | None = os.getenv("COHERE_API_KEY")
    reranker_rate_limit_per_min: int = Field(
        default=_yaml.get("reranker", {}).get("rate_limit_per_min", 10)
    )
    retrieval_strategy: str = Field(
        default=_yaml.get("retrieval", {}).get("strategy", "hybrid")
    )
    retrieval_top_k: int = Field(
        default=_yaml.get("retrieval", {}).get("top_k", 10)
    )
    rerank_top_k: int = Field(
        default=_yaml.get("retrieval", {}).get("rerank_top_k", 5)
    )

    compression_enabled: bool = Field(
        default=_yaml.get("compression", {}).get("enabled", False)
    )
    compression_model: str = Field(
        default=_yaml.get("compression", {}).get("model", "gpt-4o")
    )
    compression_temperature: float = Field(
        default=_yaml.get("compression", {}).get("temperature", 0.0)
    )
    # R-51: compression.chunk_threshold was removed. It was read into
    # settings and never used anywhere - the "keep only reranked chunks above
    # this relevance" filter it documented does not exist. Thresholding is
    # reranker.reranker_threshold, applied to the fused score.
    preprocessing_enabled: bool = Field(
        default=_preprocessing_yaml.get("enabled", False)
    )
    preprocessing_provider: str = Field(
        default=_preprocessing_yaml.get("provider", "embeddings")
    )
    preprocessing_embedding_provider: str = Field(
        default=_preprocessing_yaml.get("embedding", {}).get(
            "provider", _yaml.get("embedding", {}).get("provider", "litellm")
        )
    )
    preprocessing_embedding_model: str = Field(
        default=_preprocessing_yaml.get("embedding", {}).get(
            "model",
            _yaml.get("embedding", {}).get("model", "openai/text-embedding-3-small"),
        )
    )
    preprocessing_redundant_threshold: float = Field(
        default=_preprocessing_yaml.get("redundant_similarity_threshold", 0.85)
    )
    reranker_threshold: float | None = Field(
        default=_yaml.get("reranker", {}).get("reranker_threshold",0.5)
    )
    min_context_chunks: int = Field(
        default=_yaml.get("reranker", {}).get("min_context_chunks", 0)
    )
    score_fusion_enabled: bool = Field(
        default=_yaml.get("reranker", {}).get("score_fusion", {}).get("enabled", True)
    )
    # Fusion channel weights — RANKING only; R-19 moved thresholding to raw
    # rerank relevance, pre-fusion. Keep the sum at 1.0 so the fused ranking
    # score keeps its [0, 1] scale (R-52). Tune only via settings.yaml so the
    # run manifest captures changes.
    score_fusion_retrieval_weight: float = Field(
        default=_yaml.get("reranker", {})
        .get("score_fusion", {})
        .get("retrieval_weight", 0.5)
    )
    score_fusion_rerank_weight: float = Field(
        default=_yaml.get("reranker", {})
        .get("score_fusion", {})
        .get("rerank_weight", 0.5)
    )

    authority_reordering_enabled: bool = Field(
        default=_authority_reordering_yaml.get("enabled", False)
    )
    authority_reordering_supported_authorities: list[str] = Field(
        default_factory=lambda: _authority_reordering_yaml.get(
            "supported_authorities",
            [
                "PCAOB",
                "AICPA",
                "SEC",
                "IRS",
                "COSO",
                "FASB",
                "GASB",
                "FASAB",
            ],
        )
    )

    evaluation_provider: str = Field(
        default=_evaluation_yaml.get("provider", "ragas")
    )
    # Shared judge LLM for ragas + deepeval (prefer LiteLLM-style ids).
    evaluation_model: str = Field(
        default=_evaluation_yaml.get("model", "openai/gpt-4.1")
    )
    evaluation_temperature: float = Field(
        default=float(_evaluation_yaml.get("temperature", 0))
    )
    evaluation_embedding_model: str = Field(
        default=_evaluation_yaml.get("embedding_model", "text-embedding-3-large")
    )
    evaluation_judge_max_tokens: int = Field(
        default=int(_deepeval_yaml.get("evaluation_judge_max_tokens", 8192))
    )
    # Provider-specific knobs (nested under evaluation.ragas / evaluation.deepeval).
    ragas_max_retries: int = Field(default=int(_ragas_yaml.get("max_retries", 30)))
    ragas_max_wait: int = Field(default=int(_ragas_yaml.get("max_wait", 180)))
    ragas_timeout: int = Field(default=int(_ragas_yaml.get("timeout", 3600)))
    ragas_max_workers: int = Field(default=int(_ragas_yaml.get("max_workers", 8)))
    # R-32 / R-49: pin AnswerRelevancy strictness (ragas library default is 3).
    ragas_answer_relevancy_strictness: int = Field(
        default=int(
            1
            if _ragas_yaml.get("answer_relevancy_strictness") is None
            else _ragas_yaml.get("answer_relevancy_strictness", 1)
        )
    )
    deepeval_max_concurrent: int = Field(
        default=int(_deepeval_yaml.get("max_concurrent", 5))
    )
    deepeval_per_task_timeout_seconds: int = Field(
        default=int(_deepeval_yaml.get("per_task_timeout_seconds", 1800))
    )
    # Seconds allowed for a single DeepEval provider attempt. ``null`` disables
    # DeepEval's timeout wrappers; LiteLLM's provider timeout remains in force.
    deepeval_per_attempt_timeout_seconds: int | None = Field(
        default=(
            None
            if _deepeval_yaml.get("per_attempt_timeout_seconds") is None
            else int(_deepeval_yaml["per_attempt_timeout_seconds"])
        )
    )
    deepeval_retry_max_attempts: int = Field(
        default=int(_deepeval_yaml.get("retry_max_attempts", 5))
    )

    # R-53/R-55: evaluation.artifacts_dir was declared in settings.yaml and
    # read by nothing, so no result artifact was ever written server-side.
    evaluation_artifacts_dir: str = Field(
        default=_evaluation_yaml.get("artifacts_dir", "./data/results/artifacts")
    )

    audit_backend: str = Field(
        default=_yaml.get("audit", {}).get("backend", "jsonl")
    )
    audit_jsonl_path: str = Field(
        default=_yaml.get("audit", {}).get("path", "./data/audit/queries.jsonl")
    )

    chat_store: str = Field(default=_chat_yaml.get("store", "memory"))
    chat_max_history_messages: int = Field(
        default=_chat_yaml.get("max_history_messages", 20)
    )

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = ""
    postgres_db: str = "ai_voice_agent"

    web_search_enabled: bool = Field(
        default=bool(_web_search_yaml.get("enabled", False))
    )
    web_search_provider: str = Field(
        default=_web_search_yaml.get("provider", "serpapi")
    )
    web_search_engine: str = Field(
        default=_web_search_yaml.get("engine", "google")
    )
    web_search_language: str = Field(
        default=_web_search_yaml.get("language", "en")
    )
    web_search_country: str = Field(
        default=_web_search_yaml.get("country", "in")
    )
    serpapi_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SERPAPI_API_KEY", "serpapi"),
    )
    web_search_max_results: int = Field(
        default=int(_web_search_yaml.get("max_results", 5))
    )
    web_search_timeout_seconds: float = Field(
        default=float(_web_search_yaml.get("timeout_seconds", 12))
    )
    web_search_domains: list[str] = Field(
        default_factory=lambda: list(_web_search_yaml.get("domains") or [])
    )

    deepgram_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DEEPGRAM_API_KEY", "deepgram"),
    )
    elevenlabs_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "ELEVENLABS_API_KEY", "elevenlabs", "elevenlab"
        ),
    )
    voice_stt_provider: str = Field(
        default=_voice_yaml.get("stt_provider", "deepgram")
    )
    voice_stt_model: str = Field(default=_voice_yaml.get("stt_model", "nova-3"))
    voice_tts_provider: str = Field(
        default=_voice_yaml.get("tts_provider", "elevenlabs")
    )
    elevenlabs_voice_id: str = Field(
        default=_voice_yaml.get("elevenlabs_voice_id", "21m00Tcm4TlvDq8ikWAM")
    )
    elevenlabs_model: str = Field(
        default=_voice_yaml.get("elevenlabs_model", "eleven_flash_v2_5")
    )
    deepgram_tts_model: str = Field(
        default=_voice_yaml.get("deepgram_tts_model", "aura-2-thalia-en")
    )
    deepgram_tts_voices: list[dict] = Field(
        default=_voice_yaml.get("deepgram_tts_voices") or []
    )
    # Live voice: silence (ms) before Deepgram ends a segment, and the gap
    # that ends the whole utterance even if endpointing missed it.
    voice_endpointing_ms: int = Field(
        default=int(_voice_yaml.get("endpointing_ms", 400))
    )
    voice_utterance_end_ms: int = Field(
        default=int(_voice_yaml.get("utterance_end_ms", 1000))
    )
    # Faster model for the understand step of live voice turns only (it is
    # on the critical path to first audio). Empty uses llm.model.
    voice_understand_model: str = Field(
        default=_voice_yaml.get("understand_model") or ""
    )
    voice_tts_sample_rate: int = Field(
        default=int(_voice_yaml.get("tts_sample_rate", 24000))
    )
    voice_timeout_seconds: float = Field(
        default=float(_voice_yaml.get("timeout_seconds", 60))
    )

    agent_checkpointer: str = Field(
        default=_agent_yaml.get("checkpointer", "postgres")
    )
    redis_enabled: bool = Field(default=bool(_redis_yaml.get("enabled", False)))
    redis_url: str = Field(
        default=_redis_yaml.get("url", "redis://localhost:6379/0")
    )
    redis_web_cache_ttl_seconds: int = Field(
        default=int(_redis_yaml.get("web_cache_ttl_seconds", 300))
    )

    @property
    def postgres_dsn(self) -> str:
        user = quote_plus(self.postgres_user)
        password = quote_plus(self.postgres_password)
        database = quote_plus(self.postgres_db)
        return (
            f"postgresql://{user}:{password}@{self.postgres_host}:"
            f"{self.postgres_port}/{database}"
        )

    ingestion_use_vlm: bool = Field(
        default=_ingestion_yaml.get("use_vlm", False)
    )
    ingestion_vlm_prompt: str | None = Field(
        default=_ingestion_yaml.get("vlm_prompt")
    )
    ingestion_vlm_concurrency: int = Field(
        default=_ingestion_yaml.get("vlm_concurrency", 8), ge=1
    )
    ingestion_analyze_images: bool = Field(
        default=_ingestion_yaml.get("analyze_images", False)
    )
    # XLSX pictures: False = anchored placeholder only (refer-to-file);
    # True = also transcribe each picture via the vision LLM into an image chunk.
    ingestion_xlsx_analyze_images: bool = Field(
        default=_ingestion_yaml.get("xlsx_analyze_images", False)
    )
    ingestion_checkpoint_dir: str = Field(
        default=_ingestion_yaml.get("checkpoint_dir", "./data/ingestion_checkpoints")
    )

    @model_validator(mode="after")
    def _check_invariants(self) -> "Settings":
        """Fail at boot on configs that would silently distort a run.

        R-52: fusion weights must sum to 1.0 so the fused RANKING score keeps
        its [0, 1] scale. (Pre-R-19, reranker_threshold was applied to the
        fused score and calibration depended on this; the threshold now gates
        raw rerank relevance pre-fusion, so only ordering stability depends on
        the sum.) R-38: an unimplemented audit backend used to fall through to
        JSONL silently.
        """
        if self.score_fusion_enabled:
            total = (
                self.score_fusion_retrieval_weight + self.score_fusion_rerank_weight
            )
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    "reranker.score_fusion weights must sum to 1.0 (got "
                    f"{self.score_fusion_retrieval_weight} + "
                    f"{self.score_fusion_rerank_weight} = {total}); "
                    "the fused ranking score is calibrated to [0, 1]."
                )
        if self.reranker_threshold is not None and not (
            0.0 <= self.reranker_threshold <= 1.0
        ):
            raise ValueError(
                f"reranker.reranker_threshold must be in [0, 1] "
                f"(got {self.reranker_threshold})."
            )
        if self.rerank_top_k > self.retrieval_top_k:
            raise ValueError(
                f"retrieval.rerank_top_k ({self.rerank_top_k}) cannot exceed "
                f"retrieval.top_k ({self.retrieval_top_k}); the reranker would "
                "be asked for more chunks than retrieval returns."
            )
        if self.ragas_answer_relevancy_strictness < 1:
            raise ValueError(
                "evaluation.ragas.answer_relevancy_strictness must be >= 1 "
                f"(got {self.ragas_answer_relevancy_strictness})."
            )
        supported_audit_backends = {"jsonl"}
        if self.audit_backend not in supported_audit_backends:
            raise ValueError(
                f"audit.backend {self.audit_backend!r} is not implemented "
                f"(supported: {sorted(supported_audit_backends)}). It used to "
                "fall through to JSONL silently."
            )
        return self




@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
