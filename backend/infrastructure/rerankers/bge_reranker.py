import re

from backend.core.config import settings
from backend.core.logger import logger
from backend.domain.entities.chunk import Chunk
from backend.domain.interfaces.reranker_provider import RerankerProvider

log = logger.getChild("infrastructure.rerankers.lexical")

_WORD = re.compile(r"\w+")


class LexicalOverlapReranker(RerankerProvider):
    """PLACEHOLDER reranker — bag-of-words overlap, NOT a BGE cross-encoder.

    R-40: the class was called BGEReranker and loaded no model. It counts how
    many query words appear in the chunk and sorts by that count. The old
    implementation matched on raw substrings (``"as" in "class"`` scored a
    hit); tokens are now matched on word boundaries, which is the least this
    stub should do. It remains a placeholder: use reranker.provider "cohere"
    for anything whose numbers get reported.
    """

    def __init__(self) -> None:
        log.warning(
            "reranker.provider='bge' is a PLACEHOLDER lexical-overlap stub, "
            "not a BGE cross-encoder. Use 'cohere' for evaluation runs."
        )

    async def rank(
        self,
        query: str,
        chunks: list[Chunk],
        *,
        top_k: int | None = None,
    ) -> list[Chunk]:
        query_terms = set(_WORD.findall(query.lower()))
        scored = []
        for chunk in chunks:
            chunk_terms = set(_WORD.findall((chunk.content or "").lower()))
            overlap = len(query_terms & chunk_terms)
            scored.append((overlap, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        ranked = [c for _, c in scored]
        limit = top_k or settings.rerank_top_k
        ranked = ranked[:limit]
        # Overwrite retrieval scores with position-based rerank scores so fusion
        # has a distinct rerank channel (does not double-count retrieval).
        n = len(ranked)
        for index, chunk in enumerate(ranked):
            chunk.score = float(n - index)
        return ranked


#: Back-compat alias — the class never was a BGE model (R-40).
BGEReranker = LexicalOverlapReranker
