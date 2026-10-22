from backend.domain.entities.chunk import Chunk


def reciprocal_rank_fusion(
    result_lists: list[list[Chunk]],
    *,
    k: int = 60,
    top_k: int = 10,
) -> list[Chunk]:
    """Merge ranked lists using RRF. Same chunk id accumulates score."""
    scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results):
            # Fuse at chunk granularity: keying on document_id would collapse
            # RRF to one chunk per document once document_id stops mirroring
            # chunk.id (today they are equal on every retrieved chunk).
            key = chunk.id
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in chunk_map or (chunk.score or 0) > (chunk_map[key].score or 0):
                chunk_map[key] = chunk

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    output: list[Chunk] = []
    for key, score in merged[:top_k]:
        c = chunk_map[key]
        output.append(
            Chunk(
                id=c.id,
                document_id=c.document_id,
                content=c.content,
                metadata=c.metadata,
                score=score,
            )
        )
    return output
