"""Chroma vector store — NOT IMPLEMENTED.

R-42: this used to be a no-op stub whose methods took no arguments and
returned ``None``. Because ``VectorDBProvider`` is an ABC with matching method
NAMES, it satisfied the abstract check and constructed happily, so selecting
``vectordb.provider: chroma`` produced a pipeline where ingest silently stored
nothing and every search raised TypeError on the first call. The pinecone and
azure_ai_search stubs at least fail cleanly; this one now does too.
"""

from backend.core.exceptions import ProviderNotFoundError
from backend.domain.entities.chunk import Chunk
from backend.domain.entities.document import Document
from backend.domain.interfaces.vector_db_provider import VectorDBProvider

_NOT_IMPLEMENTED = (
    "vectordb.provider 'chroma' is not implemented. Use 'qdrant'."
)


class ChromaVectorDB(VectorDBProvider):
    def __init__(self) -> None:
        raise ProviderNotFoundError(_NOT_IMPLEMENTED)

    async def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        raise ProviderNotFoundError(_NOT_IMPLEMENTED)

    async def similarity_search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        filters: dict | None = None,
        query_text: str | None = None,
    ) -> list[Chunk]:
        raise ProviderNotFoundError(_NOT_IMPLEMENTED)

    async def delete(self, document_ids: list[str]) -> None:
        raise ProviderNotFoundError(_NOT_IMPLEMENTED)

    async def update(
        self,
        document_id: str,
        content: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> None:
        raise ProviderNotFoundError(_NOT_IMPLEMENTED)
