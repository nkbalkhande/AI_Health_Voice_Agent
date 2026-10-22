import hashlib
import math
from collections import OrderedDict

from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

from backend.core.config import settings
from backend.core.logger import logger
from backend.domain.entities.chunk import Chunk
from backend.domain.interfaces.context_compressor import ContextCompressor
from backend.infrastructure.litellm.model_resolver import resolve_embedding_model

log = logger.getChild("infrastructure.preprocessing.embeddings_filter")

_CHUNK_ID_KEY = "_chunk_id"


# R-89: a second, identical _openai_embedding_model_name() was defined
# below and silently shadowed the first. Only one remains.
def _resolved_embedding_model() -> str:
    return resolve_embedding_model(
        settings.preprocessing_embedding_provider,
        settings.preprocessing_embedding_model,
    )


def _openai_embedding_model_name() -> str:
    """LangChain OpenAIEmbeddings expects a bare model id, not provider/model."""
    model = _resolved_embedding_model()
    if "/" in model:
        return model.split("/", 1)[1]
    return model


def _uses_azure_embeddings() -> bool:
    provider = settings.preprocessing_embedding_provider.lower()
    model = _resolved_embedding_model().lower()
    return provider in {"azure", "azure_openai"} or model.startswith("azure/")


def _build_embeddings() -> AzureOpenAIEmbeddings | OpenAIEmbeddings:
    model = _openai_embedding_model_name()
    if _uses_azure_embeddings():
        log.info("Using Azure embeddings for chunk preprocessing (model=%s)", model)
        return AzureOpenAIEmbeddings(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.openai_api_version,
            azure_deployment=model,
            model=model,
            max_retries=5,
        )
    return OpenAIEmbeddings(model=model)
class EmbeddingsChunkPreprocessor(ContextCompressor):
    """Pre-retrieval-rerank chunk preprocessing: drop near-duplicate chunks.

    Deduplicates near-duplicate chunks before reranking so the reranker sees
    the full retrieved candidate set (up to top_k) rather than a clustered
    subset.

    NOTE: EmbeddingsClusteringFilter was removed deliberately (earlier fix).
    Its similarity_threshold/sort_by_similarity kwargs were silently ignored
    and it ran at defaults num_clusters=5 x num_closest=1, capping candidates
    at ~5/30 before reranking — a hard ceiling on context recall.

    Two further behaviours were corrected against the LangChain-based
    implementation this replaces:

    * R-77 — ``EmbeddingsRedundantFilter`` delegates to
      ``_filter_similar_embeddings``, which takes ``np.tril(sim, k=-1)`` and
      removes the *column* index of each above-threshold pair, i.e. the
      EARLIER document. Chunks arrive in retrieval-ranked order, so it
      discarded the higher-ranked twin and kept its lower-ranked duplicate.
      The dedup below is keep-first: it walks in rank order and drops only
      later near-duplicates.
    * R-63 — the filter re-embedded every retrieved chunk on every query
      (~25 embedding calls per question, the dominant share of median query
      latency). Embeddings are now cached by content hash across queries.

    Follow-up (deliberately not done here): Qdrant already stores these dense
    vectors, so ``similarity_search(..., with_vectors=True)`` could remove the
    embedding call entirely. That change spans the retriever interface and is
    out of scope for this batch.
    """

    #: Bounded so a long-lived process cannot accumulate vectors without limit.
    _CACHE_MAX_ENTRIES = 4096

    def __init__(self) -> None:
        self._embeddings = _build_embeddings()
        self._threshold = settings.preprocessing_redundant_threshold
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        #: "ok" or "skipped: <reason>" for the last process() call, so a
        #: silently-disabled dedup is visible to callers (R-63).
        self.last_status: str = "not_run"

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    def _cached(self, key: str) -> list[float] | None:
        vector = self._cache.get(key)
        if vector is not None:
            self._cache.move_to_end(key)
        return vector

    def _store(self, key: str, vector: list[float]) -> None:
        self._cache[key] = vector
        self._cache.move_to_end(key)
        while len(self._cache) > self._CACHE_MAX_ENTRIES:
            self._cache.popitem(last=False)

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(text) for text in texts]
        # dict() de-duplicates identical texts within this batch too.
        missing = [
            (key, text)
            for key, text in dict(zip(keys, texts)).items()
            if self._cached(key) is None
        ]
        if missing:
            vectors = await self._embeddings.aembed_documents(
                [text for _key, text in missing]
            )
            for (key, _text), vector in zip(missing, vectors, strict=True):
                self._store(key, vector)
        return [self._cache[key] for key in keys]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        return dot / norm if norm else 0.0

    async def process(
        self,
        query: str,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        if not chunks:
            self.last_status = "ok"
            return []

        try:
            vectors = await self._embed([chunk.content for chunk in chunks])
        except Exception as exc:  # noqa: BLE001 - degrade gracefully to input chunks
            self.last_status = f"skipped: {type(exc).__name__}"
            log.warning(
                "Chunk preprocessing failed (%s); falling back to retrieved "
                "chunks WITHOUT dedup.",
                exc,
            )
            return chunks

        kept: list[Chunk] = []
        kept_vectors: list[list[float]] = []
        dropped = 0
        for chunk, vector in zip(chunks, vectors, strict=True):
            if any(
                self._cosine(vector, other) > self._threshold
                for other in kept_vectors
            ):
                dropped += 1
                continue
            kept.append(chunk)
            kept_vectors.append(vector)

        self.last_status = "ok"
        if dropped:
            log.info(
                "Chunk preprocessing dropped %d/%d near-duplicate chunks "
                "(threshold=%.2f, keep-first).",
                dropped,
                len(chunks),
                self._threshold,
            )
        return kept
