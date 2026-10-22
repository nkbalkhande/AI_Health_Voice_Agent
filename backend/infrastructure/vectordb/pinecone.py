from backend.core.exceptions import ProviderNotConfiguredError
from backend.domain.entities.chunk import Chunk
from backend.domain.entities.document import Document
from backend.domain.interfaces.vector_db_provider import VectorDBProvider


class PineconeVectorDB(VectorDBProvider):
    """Pinecone implementation stub — wire pinecone SDK here."""

    async def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        raise ProviderNotConfiguredError(
            "Pinecone provider not yet implemented. Install pinecone-client."
        )

    async def similarity_search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        filters: dict | None = None,
        query_text: str | None = None,
    ) -> list[Chunk]:
        raise ProviderNotConfiguredError("Pinecone provider not yet implemented.")

    async def delete(self, document_ids: list[str]) -> None:
        raise ProviderNotConfiguredError("Pinecone provider not yet implemented.")

    async def update(
        self,
        document_id: str,
        content: str,
        embedding: list[float],
    ) -> None:
        raise ProviderNotConfiguredError("Pinecone provider not yet implemented.")
