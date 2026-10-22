import asyncio

from docling.chunking import HybridChunker

from backend.core.exceptions import IngestionError
from backend.domain.entities.document import Document
from backend.domain.interfaces.pdf_chunker import PdfChunker
from backend.rag.chunking.ids import deterministic_chunk_id
from backend.rag.ingestion.document_mapping import append_image_documents, base_metadata
from backend.rag.ingestion.models import ParsedDocument


class DoclingHybridChunker(PdfChunker):
    """Docling hybrid chunker used during PDF ingestion."""

    def __init__(self, chunker: HybridChunker | None = None) -> None:
        self._chunker = chunker or HybridChunker()

    async def chunk(
        self,
        parsed: ParsedDocument,
        *,
        source: str | None = None,
        metadata: dict | None = None,
    ) -> list[Document]:
        return await asyncio.to_thread(
            self._chunk_sync,
            parsed,
            source=source,
            metadata=metadata,
        )

    def _chunk_sync(
        self,
        parsed: ParsedDocument,
        *,
        source: str | None,
        metadata: dict | None,
    ) -> list[Document]:
        if not parsed.docling_document:
            raise IngestionError(
                "Parsed PDF is missing docling_document for hybrid chunking"
            )

        base_meta = base_metadata(parsed, metadata=metadata)
        resolved_source = source or parsed.filename
        documents: list[Document] = []

        for doc in parsed.docling_document:
            for chunk_index, chunk in enumerate(self._chunker.chunk(doc)):
                content = self._chunker.contextualize(chunk)
                page_number = _page_number_from_chunk(chunk)
                documents.append(
                    Document(
                        id=deterministic_chunk_id(
                            resolved_source,
                            page=page_number,
                            chunk_index=chunk_index,
                            content=content,
                        ),
                        content=content,
                        metadata={
                            **base_meta,
                            "content_type": "text",
                            "page_number": page_number,
                            "chunk_index": chunk_index,
                        },
                        source=resolved_source,
                    )
                )

        append_image_documents(
            documents,
            parsed,
            base_meta=base_meta,
            resolved_source=resolved_source,
        )
        return documents


def _page_number_from_chunk(chunk) -> int | None:
    if not chunk.meta.doc_items or not chunk.meta.doc_items[0].prov:
        return None
    return chunk.meta.doc_items[0].prov[0].page_no
