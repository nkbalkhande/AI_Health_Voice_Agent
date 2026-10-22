from backend.domain.entities.chunk import Chunk


class CitationBuilder:
    def build_web(self, pages: list) -> list[dict]:
        """Citations for Branch B (web). Kept apart from document citations."""
        return [
            {
                "index": i,
                "source_type": "web",
                "document_id": page.url,
                "source": page.title or page.url,
                "url": page.url,
                "snippet": page.snippet[:200],
                "content": page.snippet,
                "score": None,
                "page_number": page.url,
            }
            for i, page in enumerate(pages, start=1)
        ]

    def build(self, chunks: list[Chunk]) -> list[dict]:
        citations = []
        for i, chunk in enumerate(chunks, start=1):
            citation = {
                "index": i,
                "source_type": chunk.metadata.get("source_type", "document"),
                "document_id": chunk.document_id,
                "source": chunk.metadata.get("source"),
                "snippet": chunk.content[:200],
                "content": chunk.content,
                "score": chunk.score,
                "page_number": chunk.metadata.get("page_number", "N/A"),
            }
            # R-52: Excel chunks carry worksheet title alongside sheet-as-page index.
            # Drawing-layer provenance rides along so the audit trail shows when
            # a cited sheet holds un-indexed screenshots, and exactly which
            # cells an image transcription came from.
            for key in _OPTIONAL_CITATION_KEYS:
                value = chunk.metadata.get(key)
                if value not in (None, ""):
                    citation[key] = value
            if chunk.metadata.get("content_type") == "image":
                citation["content_type"] = "image"
            citations.append(citation)
        return citations


_OPTIONAL_CITATION_KEYS = (
    "sheet_name",
    "embedded_images",
    "anchor_range",
    "anchor_label",
    # The chunk failed the relevance gate and was added to meet the minimum.
    "backfilled",
)
