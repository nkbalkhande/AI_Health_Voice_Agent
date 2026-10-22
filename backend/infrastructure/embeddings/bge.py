import hashlib

from backend.core.config import settings
from backend.core.logger import logger
from backend.domain.interfaces.embedding_provider import EmbeddingProvider

log = logger.getChild("infrastructure.embeddings.bge")

_EMBED_DIM = 384


def _deterministic_embed(text: str, dim: int = _EMBED_DIM) -> list[float]:
    """Placeholder until sentence-transformers is wired. Stable per text."""
    digest = hashlib.sha256(text.encode()).digest()
    values = []
    for i in range(dim):
        byte = digest[i % len(digest)]
        values.append((byte / 255.0) * 2 - 1)
    return values


class BGEEmbedding(EmbeddingProvider):
    """PLACEHOLDER embeddings — sha256 hash noise, NOT a BGE model.

    R-39: nothing here loads BAAI/bge-small-en-v1.5 or any other model.
    ``_deterministic_embed`` tiles sha256 digest bytes into a 384-dim vector,
    so the vectors carry ZERO semantics: retrieval against them is effectively
    random. settings.yaml advertised this as "Offline/local deterministic
    embeddings" next to a real model name, which reads like a working local
    option. Never use it for an eval run. Replace ``_deterministic_embed``
    with a real SentenceTransformer before this provider means anything.
    """

    def __init__(self) -> None:
        self._model = settings.embedding_model
        log.warning(
            "embedding.provider='bge' is a PLACEHOLDER (sha256 hash vectors, "
            "%d dims, no semantics). Retrieval quality will be random. Do not "
            "use for evaluation runs.",
            _EMBED_DIM,
        )
        # dense_dim 0 means "size the collection from the first batch".
        if settings.qdrant_dense_dim and settings.qdrant_dense_dim != _EMBED_DIM:
            log.error(
                "Embedding dim %d does not match vectordb.qdrant.dense_dim %d; "
                "ingest/query against the configured collection will fail.",
                _EMBED_DIM,
                settings.qdrant_dense_dim,
            )

    @property
    def model_name(self) -> str:
        return self._model

    async def embed_query(self, text: str) -> list[float]:
        return _deterministic_embed(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_deterministic_embed(t) for t in texts]
