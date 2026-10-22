import uuid
from docling.chunking import HybridChunker

from backend.domain.entities.document import Document
from backend.rag.ingestion.models import ParsedPdfDocument


def parsed_pdf_to_documents(
    parsed: ParsedPdfDocument,
    *,
    chunker,
    source: str | None = None,
    metadata: dict | None = None,
) -> list[Document]:
    """Map parsed PDF structure to domain documents before chunking."""
    base_meta = {**(metadata or {}), "filename": parsed.filename}
    resolved_source = source or parsed.filename
    documents: list[Document] = [] 
    for doc in parsed.docling_document:
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            documents.append(
                Document(
                    id=str(uuid.uuid4()),
                    content=chunker.contextualize(chunk),
                    metadata={**base_meta, "content_type": "text","page_number": (chunk.meta.doc_items[0].prov[0].page_no 
                if chunk.meta.doc_items and chunk.meta.doc_items[0].prov 
                else None)},
                    source=resolved_source,
                )
            )

    for index, image in enumerate(parsed.images):
        if not image.caption or not image.caption.strip():
            continue
        documents.append(
            Document(
                id=str(uuid.uuid4()),
                content=image.caption.strip(),
                metadata={**base_meta, "content_type": "image", "image_index": index},
                source=resolved_source,
            )
        )
    

    return documents
