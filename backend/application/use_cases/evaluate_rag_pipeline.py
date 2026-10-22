import asyncio
import datetime
import hashlib
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.application.use_cases.query_documents import QueryUseCase
from backend.core.config import settings
from backend.core.logger import logger
from backend.domain.interfaces.evaluation_runner import EvaluationRunner
from backend.rag.agent_prompt.rag import SYSTEM_PROMPT
from backend.rag.prompting.builder import PromptBuilder

_SETTINGS_PATH = Path(__file__).resolve().parents[3] / "config" / "settings.yaml"


@dataclass
class EvalCase:
    question: str
    ground_truth: str = ""


def _run_manifest(
    *,
    run_id: str | None = None,
    runner_name: str | None = None,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # R-49: outside a git work tree rev-parse exits non-zero with empty
        # stdout, which used to be recorded as "" — say "unknown" instead.
        sha = completed.stdout.strip() if completed.returncode == 0 else "unknown"
        sha = sha or "unknown"
    except Exception:
        sha = "unknown"
    try:
        with open(_SETTINGS_PATH, "rb") as f:
            settings_hash = hashlib.sha256(f.read()).hexdigest()[:12]
    except OSError:
        settings_hash = "unknown"

    # Runtime values from the live settings singleton (env overrides included).
    # Never dump Settings.model_dump() — that would leak API keys.
    manifest: dict[str, Any] = {
        "manifest_version": 3,
        "run_id": run_id if run_id else str(uuid.uuid4()),
        "git_sha": sha,
        "settings_sha256_12": settings_hash,
        "judge": settings.evaluation_model,
        "generator": settings.llm_model,
        "collection": settings.vectordb_collection,
        # Judge embeddings (evaluation.*) — kept under historical key name.
        "embedding": settings.evaluation_embedding_model,
        "retrieval_embedding": settings.embedding_model,
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "evaluation_provider": settings.evaluation_provider,
        "evaluation_temperature": settings.evaluation_temperature,
        # The settings hash proves only the yaml file on disk; env vars can
        # override any of these, so record the values actually in force.
        "prompt_sha256_12": hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest()[:12],
        "reranker": {
            "provider": settings.reranker_provider,
            "model": settings.reranker_model,
        },
        "retrieval": {
            "strategy": settings.retrieval_strategy,
            "top_k": settings.retrieval_top_k,
            "rerank_top_k": settings.rerank_top_k,
            "reranker_threshold": settings.reranker_threshold,
            # R-19: admission gates on absolute Cohere relevance (rerank_raw /
            # k_raw), not the min-max fused ranking score.
            "threshold_applies_to": "raw_rerank",
            "score_fusion_enabled": settings.score_fusion_enabled,
            "weights": [
                settings.score_fusion_retrieval_weight,
                settings.score_fusion_rerank_weight,
            ],
        },
        "pipeline": {
            "compression_enabled": settings.compression_enabled,
            "preprocessing_enabled": settings.preprocessing_enabled,
            "authority_reordering_enabled": settings.authority_reordering_enabled,
        },
    }

    # R-32: record the strictness the actual runner uses. Prefer runner_name
    # (DI truth) over settings.evaluation_provider; factory lowercases both.
    effective_runner = (runner_name or settings.evaluation_provider).lower()
    if effective_runner == "ragas":
        manifest["answer_relevancy"] = {
            "strictness": settings.ragas_answer_relevancy_strictness,
        }
        manifest["ragas"] = {
            "max_retries": settings.ragas_max_retries,
            "max_wait": settings.ragas_max_wait,
            "timeout": settings.ragas_timeout,
            "max_workers": settings.ragas_max_workers,
        }
    return manifest


def _dataset_fingerprint(cases: list["EvalCase"]) -> str:
    """Stable hash of the evaluated question/ground-truth pairs.

    The golden set lives outside the repo, so neither the git SHA nor the
    settings hash identifies it. Without this, two result files with identical
    manifests can come from different golden-set revisions (R-49).
    """
    payload = "\n".join(f"{c.question}\t{c.ground_truth}" for c in cases)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


class EvaluateRagPipelineUseCase:
    """Run the live RAG pipeline for each question then score it with the
    configured runner.

    Callers provide only ``question`` (+ optional ``ground_truth``). The pipeline
    retrieves, reranks and generates an answer, building the dataset that the
    evaluation runner (ragas / deepeval) needs.

    Runner knobs come from ``config/settings.yaml`` only.
    """

    def __init__(
        self,
        *,
        query_use_case: QueryUseCase,
        runner: EvaluationRunner,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self._query_use_case = query_use_case
        self._runner = runner
        self._prompt_builder = prompt_builder or PromptBuilder()

    async def execute(
        self,
        cases: list[EvalCase],
        *,
        max_concurrency: int = 8,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded(case: EvalCase) -> dict[str, Any]:
            async with semaphore:
                try:
                    return await self._build_row(case)
                except Exception as exc:  # noqa: BLE001 - isolate one case
                    # R-13: without this, one transient retrieval/rerank/embed
                    # failure propagates out of gather() and marks a whole
                    # 166-question run FAILED after hours of judge spend.
                    # classify_for_evaluation excludes any non-"answered"
                    # status with the status as its reason, so the row is
                    # reported rather than scored.
                    logger.exception(
                        "RAG pipeline failed for eval question: %s", case.question
                    )
                    return {
                        "question": case.question,
                        "answer": "",
                        "contexts": [],
                        "ground_truth": case.ground_truth,
                        "generation_status": "pipeline_error",
                        "finish_reason": None,
                        "context_count": 0,
                        "context_details": [],
                        "pipeline_debug": {"error": str(exc)},
                    }

        rows = await asyncio.gather(*[_bounded(case) for case in cases])
        result = await self._runner.run(rows)
        manifest = _run_manifest(run_id=run_id, runner_name=self._runner.name)
        manifest["runner"] = self._runner.name
        manifest["n_cases"] = len(cases)
        manifest["dataset_sha256_12"] = _dataset_fingerprint(cases)
        result["manifest"] = manifest
        return result

    async def _build_row(self, case: EvalCase) -> dict[str, Any]:
        question = case.question
        result = await self._query_use_case.execute(question)

        # R-01: give the judge the same evidence the generator saw — byte-for-
        # byte. Rebuild Chunks from the citations and reuse PromptBuilder's own
        # chunk markup (source/page attributes included), so Faithfulness and
        # the citation judges score against identical context. Without the
        # attributes, every citation-bearing sentence looks unsupported to the
        # judge and faithfulness hard-zeros.
        chunks = self._prompt_builder.chunks_from_citations(result.citations)
        contexts = self._prompt_builder.format_contexts(chunks)

        return {
            "question": question,
            "answer": result.answer,
            "contexts": contexts,
            "ground_truth": case.ground_truth,
            # Carry the generation outcome through to the evaluator so it can
            # decide eligibility and preserve status per sample.
            "generation_status": result.generation_status,
            "finish_reason": result.finish_reason,
            "context_count": len(contexts),
            # Observability only — ignored by Ragas/DeepEval scoring inputs.
            "context_details": result.context_details,
            "pipeline_debug": result.pipeline_debug,
        }
