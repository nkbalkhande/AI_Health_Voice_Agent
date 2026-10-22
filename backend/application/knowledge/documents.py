from backend.core.logger import logger
from backend.domain.entities.chunk import Chunk
from backend.rag.retrieval.pipeline import RetrievalPipeline

log = logger.getChild("knowledge.documents")


class DocumentKnowledgeBranch:
    """Branch A — trusted knowledge base only.

    Runs the existing RAG pipeline (Qdrant hybrid retrieval, reranking,
    compression) and returns document chunks. It never sees web results,
    so document behaviour can be tuned and tested on its own.
    """

    def __init__(self, pipeline: RetrievalPipeline) -> None:
        self._pipeline = pipeline

    async def search(
        self, query: str, *, filters: dict | None = None, compress: bool = True
    ) -> list[Chunk]:
        try:
            outcome = await self._pipeline.run(query, filters=filters, compress=compress)
        except Exception:
            log.warning("Document retrieval failed for a turn", exc_info=True)
            return []
        return list(outcome.context_chunks)
