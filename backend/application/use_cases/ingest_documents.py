from dataclasses import dataclass

from backend.core.exceptions import IngestionError
from backend.core.factory import ProviderFactory
from backend.core.logger import logger
from backend.domain.entities.document import Document
from backend.domain.interfaces.embedding_provider import EmbeddingProvider
from backend.domain.interfaces.pdf_chunker import PdfChunker
from backend.domain.interfaces.vector_db_provider import VectorDBProvider
from backend.rag.ingestion.models import ParsedDocument

log = logger.getChild("application.use_cases.ingest_documents")


@dataclass
class IngestResult:
    document_count: int
    chunk_count: int


class IngestDocumentsUseCase:
    """Index a parsed PDF: chunk → annotate filename → embed → store."""

    def __init__(
        self,
        *,
        embedding: EmbeddingProvider,
        vector_db: VectorDBProvider,
        chunker: PdfChunker | None = None,
    ) -> None:
        self._embedding = embedding
        self._vector_db = vector_db
        self._chunker = chunker or ProviderFactory.get_pdf_chunker()

    async def execute(
        self,
        parsed: ParsedDocument,
        *,
        source: str | None = None,
        metadata: dict | None = None,
    ) -> IngestResult:
        """Chunk the parsed PDF and index every chunk in the vector store.

        Each chunk content is annotated with ``File name : <filename>`` before
        embedding so retrieval can surface the source document name in context.

        Raises:
            IngestionError: If the parsed PDF has no extractable content.
        """
        if not parsed.has_content:
            raise IngestionError(
                "PDF contains no extractable text, tables, or captions"
            )

        log.info("Building documents from parsed PDF: filename=%s", parsed.filename)
        documents = await self._chunker.chunk(
            parsed,
            source=source,
            metadata=metadata,
        )
        filename = parsed.filename
        for doc in documents:
            doc.content = f"{doc.content}\n\nFile name : {filename}"
        log.info("Total chunks generated: count=%d", len(documents))
        return await self._embed_and_store(documents)

    async def _embed_and_store(
        self,
        documents: list[Document],
    ) -> IngestResult:
        """Embed document texts and upsert them into the vector database."""
        texts = [doc.content for doc in documents]
        embeddings = await self._embedding.embed_documents(texts)
        await self._vector_db.add_documents(documents, embeddings)

        return IngestResult(
            document_count=len(documents),
            chunk_count=len(documents),
        )
