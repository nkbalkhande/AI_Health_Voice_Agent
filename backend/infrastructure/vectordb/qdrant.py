import asyncio
import uuid
from typing import Any

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient

# from qdrant_client.http import models
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    SearchParams,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from backend.core.config import settings
from backend.core.exceptions import IngestionError
from backend.core.logger import logger
from backend.domain.entities.chunk import Chunk
from backend.domain.entities.document import Document
from backend.domain.interfaces.vector_db_provider import VectorDBProvider

logger = logger.getChild("infrastructure.vectordb.qdrant")

_CHUNK_RETRY_ATTEMPTS = 5
_CHUNK_RETRY_PAUSE_SECONDS = 5.0
class QdrantVectorDB(VectorDBProvider):
    """Qdrant vector store with dense + sparse vectors for hybrid retrieval."""
    DENSE_VECTOR_NAME: str = "dense"
    SPARSE_VECTOR_NAME: str = "sparse"
    sparse_embedding_model:str = "Qdrant/bm25"
    def __init__(self) -> None:
        self._collection = settings.vectordb_collection
        self._dense_dim = settings.qdrant_dense_dim
        self.client = _create_qdrant_client()
        self._verify_connection()
        # R-43: do NOT auto-create here. A typo'd collection name used to be
        # silently materialized EMPTY, so every query returned zero chunks and
        # the pipeline refused every question while looking healthy.
        # add_documents() still creates the collection on first ingest.
        if not self._collection_ready():
            logger.warning(
                "Qdrant collection %r not found; queries will return no chunks "
                "until it is ingested. Check vectordb.collection.",
                self._collection,
            )
        self._sparse_embedding = self._create_sparse_embedding()

    @property
    def sparse_embedding(self) -> SparseTextEmbedding:
        return self._sparse_embedding

    def _create_sparse_embedding(self) -> SparseTextEmbedding:
        return SparseTextEmbedding(model_name=self.sparse_embedding_model)

    def _ensure_collection(self, dense_dim: int | None = None) -> None:
        dim = dense_dim or self._dense_dim
        if dim <= 0:
            return
        existing = {c.name for c in self.client.get_collections().collections}
        if self._collection in existing:
            return
        self.client.create_collection(
            collection_name=self._collection,
            vectors_config={
                self.DENSE_VECTOR_NAME: VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self.SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )

    def _collection_ready(self) -> bool:
        return self._collection in {
            c.name for c in self.client.get_collections().collections
        }
    def _embed_sparse_sync(self, text: str) -> SparseVector:
        embedding = next(self.sparse_embedding.embed(text))
        return SparseVector(
            indices=embedding.indices.tolist(),
            values=embedding.values.tolist(),
        )

    def _embed_sparse_batch_sync(self, texts: list[str]) -> list[SparseVector]:
        return [
            SparseVector(
                indices=embedding.indices.tolist(),
                values=embedding.values.tolist(),
            )
            for embedding in self.sparse_embedding.embed(texts)
        ]

    async def text_to_sparse_vector(self, text: str) -> SparseVector:
        return await asyncio.to_thread(self._embed_sparse_sync, text)

    def _verify_connection(self) -> None:
        try:
            self.client.get_collections()
        except Exception as exc:
            target = settings.qdrant_path or settings.qdrant_url
            raise IngestionError(
                f"Cannot connect to Qdrant at {target}. "
                "Ensure the server is running and the URL/port are correct."
            ) from exc

    async def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        if not documents:
            return
        dim = len(embeddings[0])
        if not self._collection_ready():
            self._ensure_collection(dense_dim=dim)
        texts = [doc.content for doc in documents]
        sparse_vectors = await asyncio.to_thread(
            self._embed_sparse_batch_sync, texts
        )
        points = [
            PointStruct(
                id=_point_id(doc.id),
                vector={
                    self.DENSE_VECTOR_NAME: emb,
                    self.SPARSE_VECTOR_NAME: sparse_vector,
                },
                payload=_payload(doc),
            )
            for doc, emb, sparse_vector in zip(
                documents, embeddings, sparse_vectors, strict=True
            )
        ]
        await self._upsert_points(points)

    async def similarity_search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        filters: dict | None = None,
        query_text: str | None = None,
    ) -> list[Chunk]:
        if not self._collection_ready():
            # R-44: silent [] turned config drift into 100% refusals with no
            # signal anywhere in the logs.
            logger.error(
                "Qdrant collection %r not found - returning 0 chunks, so every "
                "answer will be a refusal. Check vectordb.collection.",
                self._collection,
            )
            return []
        logger.info(
            f"Querying Qdrant with top_k={top_k}, filters={filters}, "
            f"query_text={query_text}"
        )
        q_filter = _build_filter(filters) if filters else None

        if query_text:
            sparse = await self.text_to_sparse_vector(query_text)
            prefetch_limit = top_k * max(1, settings.qdrant_prefetch_multiplier)
            response = await asyncio.to_thread(
                self.client.query_points,
                collection_name=self._collection,
                prefetch=[
                    # R-23: with a FusionQuery the actual searches happen
                    # inside the prefetch arms, and top-level search_params do
                    # not reach them - so qdrant.search_exact was a no-op and
                    # the "brute-force baseline" claim was false. Each arm
                    # carries its own params now.
                    Prefetch(
                        query=query_embedding,
                        using=self.DENSE_VECTOR_NAME,
                        limit=prefetch_limit,
                        filter=q_filter,
                        params=SearchParams(exact=settings.qdrant_search_exact),
                    ),
                    Prefetch(
                        query=sparse,
                        using=self.SPARSE_VECTOR_NAME,
                        limit=prefetch_limit,
                        filter=q_filter,
                        params=SearchParams(exact=settings.qdrant_search_exact),
                    ),
                ],
                query=FusionQuery(fusion=_resolve_fusion(settings.qdrant_hybrid_fusion)),
                limit=top_k,
                with_payload=True,
            )
            if not response.points:
                logger.warning(
                    "Hybrid search on %r returned 0 points for top_k=%d.",
                    self._collection,
                    top_k,
                )
            return _points_to_chunks(response.points)

        response = await asyncio.to_thread(
            self.client.query_points,
            collection_name=self._collection,
            query=query_embedding,
            using=self.DENSE_VECTOR_NAME,
            limit=top_k,
            query_filter=q_filter,
            search_params=SearchParams(exact=settings.qdrant_search_exact),
            with_payload=True,
        )
        if not response.points:
            logger.warning(
                "Dense search on %r returned 0 points for top_k=%d.",
                self._collection,
                top_k,
            )
        return _points_to_chunks(response.points)

    async def delete(self, document_ids: list[str]) -> None:
        if not document_ids or not self._collection_ready():
            return
        await asyncio.to_thread(
            self.client.delete,
            collection_name=self._collection,
            points_selector=[_point_id(doc_id) for doc_id in document_ids],
        )

    async def update(
        self,
        document_id: str,
        content: str,
        embedding: list[float],
    ) -> None:
        point = PointStruct(
            id=_point_id(document_id),
            vector={
                self.DENSE_VECTOR_NAME: embedding,
                self.SPARSE_VECTOR_NAME: await self.text_to_sparse_vector(content),
            },
            payload={"content": content, "document_id": document_id},
        )
        if not self._collection_ready():
            self._ensure_collection(dense_dim=len(embedding))
        await self._upsert_points([point])

    async def _upsert_points(self, points: list[PointStruct]) -> None:
        batch_size = max(1, settings.qdrant_upsert_batch_size)
        failed: list[PointStruct] = []
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            try:
                await self._upsert_batch_with_retry(batch)
            except IngestionError:
                logger.warning(
                    "Qdrant batch of %d chunks failed; inserting them one by one",
                    len(batch),
                )
                failed.extend(await self._upsert_each(batch))
        if not failed:
            return
        logger.warning("Retrying %d chunk(s) that failed to insert", len(failed))
        await asyncio.sleep(_CHUNK_RETRY_PAUSE_SECONDS)
        still_failed = await self._upsert_each(failed)
        if still_failed:
            raise IngestionError(
                f"{len(still_failed)} of {len(points)} chunks could not be "
                "stored in Qdrant after retrying each chunk"
            )
        logger.info("All %d previously failed chunk(s) inserted", len(failed))

    async def _upsert_each(self, points: list[PointStruct]) -> list[PointStruct]:
        """Insert chunks one at a time; return the ones that still failed."""
        failed: list[PointStruct] = []
        for point in points:
            try:
                await self._upsert_batch_with_retry(
                    [point], retries=_CHUNK_RETRY_ATTEMPTS
                )
            except IngestionError:
                failed.append(point)
        return failed

    async def _upsert_batch_with_retry(
        self,
        points: list[PointStruct],
        *,
        retries: int = 3,
    ) -> None:
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                await asyncio.to_thread(
                    self.client.upsert,
                    collection_name=self._collection,
                    points=points,
                    wait=True,
                )
                return
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Qdrant upsert attempt %s/%s failed: %s",
                    attempt + 1,
                    retries,
                    exc,
                )
                if attempt < retries - 1:
                    await asyncio.sleep(0.5 * (2**attempt))
        raise IngestionError(
            f"Qdrant upsert failed after {retries} attempts "
            f"({len(points)} points per batch): {last_exc}"
        ) from last_exc


def _resolve_fusion(name: str | None) -> Fusion:
    """Map a config string to a Qdrant Fusion. Unknown values fall back to RRF."""
    mapping = {"rrf": Fusion.RRF, "dbsf": Fusion.DBSF}
    resolved = mapping.get((name or "rrf").strip().lower())
    if resolved is None:
        logger.warning("Unknown hybrid fusion '%s'; using RRF", name)
        return Fusion.RRF
    return resolved


def _create_qdrant_client() -> QdrantClient:
    if settings.qdrant_path:
        return QdrantClient(path=settings.qdrant_path)
    url = settings.qdrant_url.strip()
    if url == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(
        url=url.rstrip("/"),
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout,
    )


def _point_id(document_id: str) -> str:
    try:
        uuid.UUID(document_id)
        return document_id
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, document_id))


def _payload(doc: Document) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "content": doc.content,
        "document_id": doc.id,
    }
    if doc.source:
        payload["source"] = doc.source
    payload.update(doc.metadata)
    return payload


def _build_filter(filters: dict) -> Filter:
    if not filters:
        return None
    conditions = [
        FieldCondition(key=key, match=MatchValue(value=value))
        for key, value in filters.items()
    ]
    return Filter(must=conditions)


def _points_to_chunks(points: list[Any]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for point in points:
        payload = point.payload or {}
        content = payload.get("content", "")
        document_id = payload.get("document_id", str(point.id))
        metadata = {
            k: v
            for k, v in payload.items()
            if k not in ("content", "document_id")
        }
        chunks.append(
            Chunk(
                id=str(point.id),
                document_id=document_id,
                content=content,
                metadata=metadata,
                score=point.score,
            )
        )
    return chunks
