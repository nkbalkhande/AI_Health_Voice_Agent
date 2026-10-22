from backend.core.exceptions import ProviderNotConfiguredError
from backend.domain.entities.chunk import Chunk
from backend.domain.entities.document import Document
from backend.domain.interfaces.vector_db_provider import VectorDBProvider


class AzureAISearchVectorDB(VectorDBProvider):
    """Azure AI Search implementation stub."""

    async def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        raise ProviderNotConfiguredError(
            "Azure AI Search provider not yet implemented."
        )

    async def similarity_search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        filters: dict | None = None,
        query_text: str | None = None,
    ) -> list[Chunk]:
        raise ProviderNotConfiguredError(
            "Azure AI Search provider not yet implemented."
        )

    async def delete(self, document_ids: list[str]) -> None:
        raise ProviderNotConfiguredError(
            "Azure AI Search provider not yet implemented."
        )

    async def update(
        self,
        document_id: str,
        content: str,
        embedding: list[float],
    ) -> None:
        raise ProviderNotConfiguredError(
            "Azure AI Search provider not yet implemented."
        )
