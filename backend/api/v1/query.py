from fastapi import APIRouter, Depends

from backend.api.deps import get_query_use_case
from backend.api.schemas import QueryRequest, QueryResponse
from backend.application.use_cases.query_documents import QueryUseCase
from backend.core.logger import logger

router = APIRouter(prefix="/query", tags=["query"])
log = logger.getChild("api.v1.query")


# R-76: the router already carries the /query prefix, so this published
# /api/v1/query/query. "" publishes /api/v1/query; the doubled path is kept as
# a hidden alias so existing clients and scripts do not break.
@router.post(
    "",
    response_model=QueryResponse,
    summary="Run a single-turn RAG query",
)
@router.post("/query", response_model=QueryResponse, include_in_schema=False)
async def query(
    body: QueryRequest,
    use_case: QueryUseCase = Depends(get_query_use_case),
) -> QueryResponse:
    """Answer a question using retrieval-augmented generation over indexed docs.

    Pipeline:
    1. Retrieve top-k chunks from the vector store (vector or hybrid).
    2. Optionally preprocess / filter retrieved context.
    3. Rerank chunks, then optionally compress context for the LLM.
    4. Build a grounded prompt, generate an answer, and attach citations.

    Optional ``filters`` are passed through to the retriever (provider-specific).
    The response includes the answer, citations, retrieved chunks, model name,
    and end-to-end latency.
    """
    preview = body.query if len(body.query) <= 200 else f"{body.query[:200]}..."
    log.info(
        "Query request received query_len=%d filters=%s preview=%r",
        len(body.query),
        body.filters,
        preview,
    )
    try:
        result = await use_case.execute(body.query, filters=body.filters)
    except Exception:
        log.exception("Query request failed query_len=%d", len(body.query))
        raise

    log.info(
        "Query request completed model=%s latency_ms=%.2f citations=%d chunks=%d",
        result.model,
        result.latency_ms,
        len(result.citations),
        len(result.retrieved_chunks),
    )
    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        retrieved_chunks=result.retrieved_chunks,
        model=result.model,
        latency_ms=result.latency_ms,
        generation_status=result.generation_status,
        finish_reason=result.finish_reason,
    )
