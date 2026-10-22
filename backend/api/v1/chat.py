import json

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse

from backend.api.deps import get_chat_use_case
from backend.api.schemas import (
    ChatMessageResponse,
    ChatRequest,
    ChatResponse,
    CreateThreadResponse,
    RenameThreadRequest,
    ThreadHistoryResponse,
    ThreadListResponse,
    ThreadSummaryResponse,
)
from backend.application.use_cases.chat import ChatUseCase
from backend.core.exceptions import ChatThreadNotFoundError

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/threads", response_model=CreateThreadResponse)
async def create_thread(
    use_case: ChatUseCase = Depends(get_chat_use_case),
) -> CreateThreadResponse:
    thread_id = await use_case.create_thread()
    return CreateThreadResponse(thread_id=thread_id)


@router.get("/threads", response_model=ThreadListResponse)
async def list_threads(
    limit: int = Query(100, ge=1, le=500),
    use_case: ChatUseCase = Depends(get_chat_use_case),
) -> ThreadListResponse:
    threads = await use_case.list_threads(limit=limit)
    return ThreadListResponse(threads=[ThreadSummaryResponse(**t) for t in threads])


@router.patch("/threads/{thread_id}", status_code=204)
async def rename_thread(
    thread_id: str,
    body: RenameThreadRequest,
    use_case: ChatUseCase = Depends(get_chat_use_case),
) -> Response:
    try:
        await use_case.rename_thread(thread_id, body.title.strip())
    except ChatThreadNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return Response(status_code=204)


@router.delete("/threads/{thread_id}", status_code=204)
async def delete_thread(
    thread_id: str,
    use_case: ChatUseCase = Depends(get_chat_use_case),
) -> Response:
    try:
        await use_case.delete_thread(thread_id)
    except ChatThreadNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return Response(status_code=204)


@router.get("/threads/{thread_id}", response_model=ThreadHistoryResponse)
async def get_thread_history(
    thread_id: str,
    use_case: ChatUseCase = Depends(get_chat_use_case),
) -> ThreadHistoryResponse:
    try:
        session = await use_case.get_session(thread_id)
    except ChatThreadNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return ThreadHistoryResponse(
        thread_id=thread_id,
        messages=[ChatMessageResponse(**m) for m in session["messages"]],
        summary=session["summary"],
        facts=session["facts"],
    )


@router.post("", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    use_case: ChatUseCase = Depends(get_chat_use_case),
) -> ChatResponse:
    try:
        result = await use_case.execute(
            body.message,
            thread_id=body.thread_id,
            filters=body.filters,
        )
    except ChatThreadNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return ChatResponse(
        thread_id=result.thread_id,
        answer=result.answer,
        citations=result.citations,
        retrieved_chunks=result.retrieved_chunks,
        model=result.model,
        latency_ms=result.latency_ms,
        message_count=result.message_count,
        standalone_query=result.standalone_query,
        intent=result.intent,
        summary=result.summary,
        facts=result.facts,
        document_results=result.document_results,
        web_results=result.web_results,
    )


@router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    use_case: ChatUseCase = Depends(get_chat_use_case),
) -> StreamingResponse:
    async def events():
        try:
            async for event in use_case.stream(
                body.message,
                thread_id=body.thread_id,
                filters=body.filters,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except ChatThreadNotFoundError as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
