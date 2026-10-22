from backend.domain.entities.chunk import Chunk
from backend.domain.interfaces.embedding_provider import EmbeddingProvider
from backend.domain.interfaces.retriever import Retriever
from backend.domain.interfaces.vector_db_provider import VectorDBProvider


class HybridRetriever(Retriever):
    """Dense + sparse retrieval, fused inside the vector store.

    R-47: this used to delegate straight to ``VectorRetriever`` — which itself
    passed ``query_text``, i.e. did the hybrid search — so both strategies ran
    the same code and neither name meant anything. The hybrid retriever now
    owns the embed + ``query_text`` call, and VectorRetriever is dense-only.
    """

    def __init__(
        self,
        *,
        embedding: EmbeddingProvider,
        vector_db: VectorDBProvider,
    ) -> None:
        self._embedding = embedding
        self._vector_db = vector_db

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[Chunk]:
        query_embedding = await self._embedding.embed_query(query)
        return await self._vector_db.similarity_search(
            query_embedding,
            top_k=top_k,
            filters=filters,
            query_text=query,
        )
