from backend.domain.entities.chunk import Chunk
from backend.domain.interfaces.embedding_provider import EmbeddingProvider
from backend.domain.interfaces.retriever import Retriever
from backend.domain.interfaces.vector_db_provider import VectorDBProvider


class VectorRetriever(Retriever):
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
        """Dense-only retrieval.

        R-59: this used to pass ``query_text=query``, which makes the Qdrant
        provider run the dense+sparse fusion path — so ``retrieval.strategy:
        vector`` and ``retrieval.strategy: hybrid`` executed byte-identical
        code and the knob was a no-op. Omitting query_text is what makes this
        retriever actually dense-only.
        """
        query_embedding = await self._embedding.embed_query(query)
        return await self._vector_db.similarity_search(
            query_embedding,
            top_k=top_k,
            filters=filters,
        )
