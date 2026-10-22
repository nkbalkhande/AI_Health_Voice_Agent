from backend.domain.entities.document import Document
from backend.rag.chunking.splitter import TextChunker


class IngestionProcessor:
    def __init__(self, chunker: TextChunker | None = None) -> None:
        self._chunker = chunker or TextChunker()

    def process(self, documents: list[Document]) -> list:
        from backend.domain.entities.chunk import Chunk

        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self._chunker.split(doc))
        return all_chunks
