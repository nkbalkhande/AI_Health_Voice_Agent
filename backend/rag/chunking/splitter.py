from backend.domain.entities.chunk import Chunk
from backend.domain.entities.document import Document
from backend.rag.chunking.ids import deterministic_chunk_id


class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def split(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text:
            return []
        chunks: list[Chunk] = []
        start = 0
        index = 0
        source = document.source or document.id
        page = document.metadata.get("page_number", document.metadata.get("page"))
        while start < len(text):
            end = min(start + self._chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append(
                Chunk(
                    id=deterministic_chunk_id(
                        source,
                        page=page if isinstance(page, int) else None,
                        chunk_index=index,
                        content=chunk_text,
                    ),
                    document_id=document.id,
                    content=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_index": index,
                        "source": document.source,
                    },
                )
            )
            index += 1
            if end >= len(text):
                break
            start = end - self._chunk_overlap
        return chunks


def document_from_text(
    content: str,
    *,
    source: str | None = None,
    metadata: dict | None = None,
) -> Document:
    resolved_source = source or ""
    return Document(
        id=deterministic_chunk_id(
            resolved_source,
            page=None,
            chunk_index=0,
            content=content,
        ),
        content=content,
        metadata=metadata or {},
        source=source,
    )
