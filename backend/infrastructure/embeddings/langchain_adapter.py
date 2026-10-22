from langchain_core.embeddings import Embeddings

from backend.infrastructure.embeddings.litellm_provider import LiteLLMEmbedding


class EmbeddingProviderAdapter(Embeddings):
    """Bridge LiteLLMEmbedding to LangChain's synchronous Embeddings interface."""

    def __init__(self, embedding: LiteLLMEmbedding) -> None:
        self._embedding = embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedding.embed_documents_sync(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embedding.embed_query_sync(text)
