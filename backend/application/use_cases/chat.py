import time
from dataclasses import dataclass, field
from typing import Any

from backend.application.chat.history import ChatHistoryHandler
from backend.application.conversation.graph import ConversationGraph
from backend.application.conversation.understand import merge_facts
from backend.audit.audit_logger import AuditLogger, AuditRecord
from backend.core.config import settings
from backend.domain.entities.chat import SessionMemory
from backend.domain.interfaces.chunk_preprocessor import ChunkPreprocessor
from backend.domain.interfaces.llm_provider import LLMProvider
from backend.domain.interfaces.reranker_provider import RerankerProvider
from backend.domain.interfaces.retriever import Retriever
from backend.domain.interfaces.web_search import WebSearchProvider
from backend.rag.citation.builder import CitationBuilder
from backend.rag.prompting.builder import PromptBuilder
from backend.rag.retrieval.pipeline import RetrievalPipeline

VOICE_STYLE = (
    "This turn is a live spoken conversation and your reply is streamed to speech "
    "as you write it. Answer in two to four short, natural spoken sentences, "
    "under about 80 words. Lead with the direct answer. No lists, headings, "
    "tables, markdown, URLs, or bracketed citations; if a source matters, say it "
    "naturally (for example, 'according to the WHO'). Offer to go deeper only if useful."
)


@dataclass
class ChatResult:
    thread_id: str
    answer: str
    citations: list[dict]
    retrieved_chunks: list[dict]
    model: str
    latency_ms: float
    message_count: int
    standalone_query: str = ""
    intent: str = ""
    summary: str = ""
    facts: list[dict] = field(default_factory=list)
    # Branch A and Branch B, kept apart in the response as in the code.
    document_results: list[dict] = field(default_factory=list)
    web_results: list[dict] = field(default_factory=list)


class ChatUseCase:
    """Multi-turn RAG chat. All thread/history I/O goes through ChatHistoryHandler."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        retriever: Retriever,
        reranker: RerankerProvider,
        history: ChatHistoryHandler,
        preprocessor: ChunkPreprocessor | None = None,
        compressor: ChunkPreprocessor | None = None,
        prompt_builder: PromptBuilder | None = None,
        citation_builder: CitationBuilder | None = None,
        audit_logger: AuditLogger | None = None,
        web_search: WebSearchProvider | None = None,
        cache=None,
        checkpointer_dsn: str | None = None,
        voice_understand_llm: LLMProvider | None = None,
    ) -> None:
        self._llm = llm
        self._pipeline = RetrievalPipeline(
            retriever=retriever,
            reranker=reranker,
            preprocessor=preprocessor,
            compressor=compressor,
        )
        self._history = history
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._citation_builder = citation_builder or CitationBuilder()
        self._audit = audit_logger or AuditLogger()
        self._conversation = ConversationGraph(
            llm=llm,
            pipeline=self._pipeline,
            web_search=web_search,
            cache=cache,
            checkpointer_dsn=checkpointer_dsn,
            voice_understand_llm=voice_understand_llm,
        )

    async def warm_up(self) -> None:
        """Open slow connections before the first live voice turn needs them."""
        await self._conversation.warm_up()

    async def create_thread(self, *, metadata: dict | None = None) -> str:
        return await self._history.create_thread(metadata=metadata)

    async def list_threads(self, *, limit: int = 100) -> list[dict]:
        return await self._history.list_threads(limit=limit)

    async def rename_thread(self, thread_id: str, title: str) -> None:
        await self._history.rename(thread_id, title)

    async def delete_thread(self, thread_id: str) -> None:
        await self._history.delete(thread_id)

    async def get_history(self, thread_id: str) -> list[dict]:
        return await self._history.read(thread_id)

    async def get_session(self, thread_id: str) -> dict:
        messages = await self.get_history(thread_id)
        memory = await self._history.get_memory(thread_id)
        return {
            "messages": messages,
            "summary": memory.summary,
            "facts": memory.facts,
        }

    async def execute(
        self,
        message: str,
        *,
        thread_id: str | None = None,
        filters: dict | None = None,
    ) -> ChatResult:
        thread_id = await self._history.resolve_thread(thread_id)
        start = time.perf_counter()

        async with self._history.turn(thread_id) as session:
            prepared = await self._prepare(thread_id, session.messages, message, filters)
            generation = await self._llm.chat(prepared["messages"])
            answer = generation.text
            memory, message_count = await self._persist_turn(
                thread_id,
                prepared,
                message,
                answer,
                session,
            )

        return await self._finish(
            thread_id=thread_id,
            message=message,
            answer=answer,
            prepared=prepared,
            memory=memory,
            message_count=message_count,
            start=start,
            generation=generation,
        )

    async def stream(
        self,
        message: str,
        *,
        thread_id: str | None = None,
        filters: dict | None = None,
        voice: bool = False,
    ):
        """Yield status, token, and done events for one conversational turn.

        If the consumer stops early (a spoken answer interrupted by the user),
        whatever was generated so far is still saved, marked as cut off.
        """
        thread_id = await self._history.resolve_thread(thread_id)
        start = time.perf_counter()

        async with self._history.turn(thread_id) as session:
            prepared = await self._prepare(
                thread_id, session.messages, message, filters, voice=voice
            )
            yield {
                "type": "status",
                "thread_id": thread_id,
                "intent": prepared["intent"],
                "standalone_query": prepared["standalone_query"],
                "needs_retrieval": prepared["needs_retrieval"],
            }
            parts: list[str] = []
            finished = False
            try:
                async for token in self._llm.stream_chat(prepared["messages"]):
                    parts.append(token)
                    yield {"type": "token", "text": token}
                finished = True
            finally:
                if not finished:
                    partial = "".join(parts).strip()
                    await self._persist_turn(
                        thread_id,
                        prepared,
                        message,
                        f"{partial} …" if partial else "(interrupted)",
                        session,
                    )
            answer = "".join(parts).strip()
            memory, message_count = await self._persist_turn(
                thread_id,
                prepared,
                message,
                answer,
                session,
            )

        citations = self._all_citations(prepared)
        yield {
            "type": "done",
            "thread_id": thread_id,
            "answer": answer,
            "citations": citations,
            "document_results": self._document_results(prepared),
            "web_results": self._web_results(prepared),
            "summary": memory.summary,
            "facts": memory.facts,
            "message_count": message_count,
            "model": self._llm.model_name,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }

    def _all_citations(self, prepared: dict[str, Any]) -> list[dict]:
        return self._citation_builder.build(
            prepared["document_chunks"]
        ) + self._citation_builder.build_web(prepared["web_pages"])

    @staticmethod
    def _document_results(prepared: dict[str, Any]) -> list[dict]:
        return [
            {
                "content": chunk.content,
                "score": chunk.score,
                "source": chunk.metadata.get("source"),
                "page_number": chunk.metadata.get("page_number"),
            }
            for chunk in prepared["document_chunks"]
        ]

    @staticmethod
    def _web_results(prepared: dict[str, Any]) -> list[dict]:
        return [
            {"title": page.title, "url": page.url, "snippet": page.snippet}
            for page in prepared["web_pages"]
        ]

    async def _prepare(
        self,
        thread_id: str,
        history,
        message: str,
        filters: dict | None,
        *,
        voice: bool = False,
    ) -> dict[str, Any]:
        memory = await self._history.get_memory(thread_id)
        state = await self._conversation.run(
            {
                "session_id": thread_id,
                "user_message": message,
                "history": [{"role": item.role, "content": item.content} for item in history],
                "summary": memory.summary,
                "facts": list(memory.facts),
                "filters": filters,
                "voice": voice,
            }
        )
        facts = merge_facts(memory.facts, list(state.get("memory_updates") or []))
        summary = state.get("updated_summary") or memory.summary
        # Branch A and Branch B stay separate from here to the response.
        document_chunks = list(state.get("rag_chunks") or [])
        web_pages = list(state.get("web_documents") or [])
        standalone = state.get("standalone_query") or message
        messages = self._prompt_builder.build_chat_messages(
            history=history,
            query=message,
            document_chunks=document_chunks,
            web_pages=web_pages,
            max_history_messages=settings.chat_max_history_messages,
            summary=summary,
            facts=facts,
            resolved_query=standalone,
        )
        if voice:
            messages.append({"role": "system", "content": VOICE_STYLE})
        return {
            "messages": messages,
            "document_chunks": document_chunks,
            "web_pages": web_pages,
            "rag_chunks": document_chunks,
            "intent": state.get("intent") or "",
            "standalone_query": standalone,
            "needs_retrieval": bool(state.get("needs_retrieval", True)),
            "summary": summary,
            "facts": facts,
            "prior_memory": memory,
        }

    async def _persist_turn(self, thread_id, prepared, message, answer, session):
        memory = SessionMemory(summary=prepared["summary"], facts=prepared["facts"])
        await self._history.save_memory(thread_id, memory)
        message_count = await session.record_exchange(message, answer)
        return memory, message_count

    async def _finish(
        self,
        *,
        thread_id: str,
        message: str,
        answer: str,
        prepared: dict[str, Any],
        memory: SessionMemory,
        message_count: int,
        start: float,
        generation,
    ) -> ChatResult:
        document_chunks = prepared["document_chunks"]
        citations = self._all_citations(prepared)
        latency_ms = (time.perf_counter() - start) * 1000
        record = AuditRecord(
            query=message,
            retrieved_chunks=[c.content for c in document_chunks],
            reranked_chunks=[c.content for c in document_chunks],
            citations=citations,
            model=self._llm.model_name,
            latency_ms=latency_ms,
            cost=getattr(generation, "cost", None),
            answer=answer,
            thread_id=thread_id,
            generation_status=generation.status,
            finish_reason=generation.finish_reason,
        )
        await self._audit.log(record)
        return ChatResult(
            thread_id=thread_id,
            answer=answer,
            citations=citations,
            retrieved_chunks=self._document_results(prepared),
            document_results=self._document_results(prepared),
            web_results=self._web_results(prepared),
            model=self._llm.model_name,
            latency_ms=latency_ms,
            message_count=message_count,
            standalone_query=prepared["standalone_query"],
            intent=prepared["intent"],
            summary=memory.summary,
            facts=list(memory.facts),
        )
