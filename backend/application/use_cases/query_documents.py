import time
from dataclasses import dataclass, field

from backend.audit.audit_logger import AuditLogger, AuditRecord
from backend.domain.interfaces.context_compressor import ContextCompressor
from backend.domain.interfaces.llm_provider import GenerationResult, LLMProvider
from backend.domain.interfaces.reranker_provider import RerankerProvider
from backend.domain.interfaces.retriever import Retriever
from backend.rag.citation.builder import CitationBuilder
from backend.rag.observability import build_context_details, build_pipeline_debug
from backend.rag.prompting.builder import PromptBuilder
from backend.rag.retrieval.pipeline import RetrievalPipeline


# The system prompt's mandated refusal sentence; the evaluation refusal markers
# match it, so these turns are counted as no-evidence refusals.
NO_EVIDENCE_ANSWER = (
    "I could not find sufficient information in the provided context. "
    "Please ask a doctor or pharmacist."
)


@dataclass
class QueryResult:
    answer: str
    citations: list[dict]
    retrieved_chunks: list[dict]
    model: str
    latency_ms: float
    generation_status: str = "answered"
    finish_reason: str | None = None
    # Evaluation/debug only — does not affect generation or citations.
    context_details: list[dict] = field(default_factory=list)
    pipeline_debug: dict = field(default_factory=dict)
    user_prompt: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost: float | None = None


class QueryUseCase:
    def __init__(
        self,
        *,
        llm: LLMProvider,
        retriever: Retriever,
        reranker: RerankerProvider,
        preprocessor: ContextCompressor | None = None,
        compressor: ContextCompressor | None = None,
        prompt_builder: PromptBuilder | None = None,
        citation_builder: CitationBuilder | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self._llm = llm
        # R-65: retrieval lives in one shared pipeline, not duplicated here
        # and in ChatUseCase.
        self._pipeline = RetrievalPipeline(
            retriever=retriever,
            reranker=reranker,
            preprocessor=preprocessor,
            compressor=compressor,
        )
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._citation_builder = citation_builder or CitationBuilder()
        self._audit = audit_logger or AuditLogger()

    async def execute(self, query: str, *, filters: dict | None = None) -> QueryResult:
        start = time.perf_counter()

        outcome = await self._pipeline.run(query, filters=filters)
        retrieved = outcome.retrieved
        context_chunks = outcome.context_chunks
        retrieval_by_id = outcome.retrieval_by_id

        prompt = self._prompt_builder.build(query, context_chunks)
        t0 = time.perf_counter()
        if context_chunks:
            generation = await self._llm.generate(
                prompt, system=PromptBuilder.SYSTEM
            )
        else:
            # Nothing passed the relevance gate. Given an empty context the
            # model answers from memory, so the refusal is enforced here.
            generation = GenerationResult(
                text=NO_EVIDENCE_ANSWER, status="answered", finish_reason="no_evidence"
            )
        generation_latency_ms = (time.perf_counter() - t0) * 1000
        answer = generation.text
        citations = self._citation_builder.build(context_chunks)
        latency_ms = (time.perf_counter() - start) * 1000

        context_details = build_context_details(
            context_chunks, retrieval_by_id=retrieval_by_id
        )
        pipeline_debug = build_pipeline_debug(
            query_latency_ms=latency_ms,
            retrieval_latency_ms=outcome.retrieval_latency_ms,
            preprocess_latency_ms=outcome.preprocess_latency_ms,
            rerank_latency_ms=outcome.rerank_latency_ms,
            compression_latency_ms=outcome.compression_latency_ms,
            generation_latency_ms=generation_latency_ms,
            reranker_model=getattr(self._pipeline._reranker, "model_name", None),
            generator_model=self._llm.model_name,
            authority_meta=outcome.authority_meta,
            preprocess_status=outcome.preprocess_status,
        )
        if outcome.compression_fallback:
            pipeline_debug["compression_fallback"] = True

        record = AuditRecord(
            query=query,
            retrieved_chunks=[c.content for c in retrieved],
            reranked_chunks=[c.content for c in context_chunks],
            citations=citations,
            model=self._llm.model_name,
            latency_ms=latency_ms,
            cost=generation.cost,
            answer=answer,
            generation_status=generation.status,
            finish_reason=generation.finish_reason,
        )
        await self._audit.log(record)

        used_ids = {c.id for c in context_chunks}

        return QueryResult(
            answer=answer,
            citations=citations,
            retrieved_chunks=[
                {
                    "content": c.content,
                    "score": (retrieval_by_id.get(c.id) or {}).get("score", c.score),
                    "rerank_score": (c.metadata or {}).get("rerank_score"),
                    "used_in_answer": c.id in used_ids,
                }
                for c in retrieved
            ],
            model=self._llm.model_name,
            latency_ms=latency_ms,
            generation_status=generation.status,
            finish_reason=generation.finish_reason,
            context_details=context_details,
            pipeline_debug=pipeline_debug,
            user_prompt=prompt,
            prompt_tokens=getattr(generation, "prompt_tokens", None),
            completion_tokens=getattr(generation, "completion_tokens", None),
            cost=getattr(generation, "cost", None),
        )
