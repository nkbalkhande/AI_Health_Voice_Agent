from __future__ import annotations

import sys
import types
from typing import Any

from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from openai import AsyncAzureOpenAI, AsyncOpenAI

from backend.core.config import settings
from backend.core.logger import logger
from backend.evaluation.base import BaseEvaluationRunner
from backend.evaluation.eligibility import (  # noqa: F401 - re-exported for callers/tests
    REFUSAL_MARKERS,
    _clean_float,
    classify_for_evaluation,
    clean_float,
    looks_like_refusal,
)
from backend.evaluation.judge_config import (  # noqa: F401 - re-exported for callers/tests
    _REASONING_MODEL_PREFIXES,
    _is_azure_model,
    _is_reasoning_model,
    _judge_llm_kwargs,
    _missing_judge_credentials,
    _openai_model_id,
    _sanitize_reasoning_judge,
)
from backend.evaluation.provider_credentials import (
    azure_api_version,
    is_azure_model,
    missing_credentials,
)
from backend.evaluation.strip_citations import strip_answer_citations


def _patch_ragas_vertex_import() -> None:
    """ragas imports a langchain_community module that no longer exists."""
    chat_module_name = "langchain_community.chat_models.vertexai"
    if chat_module_name not in sys.modules:
        vertexai_module = types.ModuleType(chat_module_name)

        class ChatVertexAI:
            pass

        vertexai_module.ChatVertexAI = ChatVertexAI
        sys.modules[chat_module_name] = vertexai_module
    llms_module_name = "langchain_community.llms"
    llms_module = sys.modules.get(llms_module_name)
    if llms_module is None:
        llms_module = types.ModuleType(llms_module_name)
        sys.modules[llms_module_name] = llms_module
    if not hasattr(llms_module, "VertexAI"):

        class VertexAI:
            pass

        llms_module.VertexAI = VertexAI


_RAGAS: types.SimpleNamespace | None = None


def _import_ragas() -> types.SimpleNamespace:
    """Import ragas/datasets on first use, not at module import (R-69).

    The vertex shim and the heavy ragas/datasets import chain used to run as
    module-level side effects (behind ``noqa: E402`` suppressions). Deferring
    them means importing this module is side-effect free; the patch still runs
    before the first ragas import, which is the ordering it exists to protect.
    """
    global _RAGAS
    if _RAGAS is None:
        _patch_ragas_vertex_import()
        from datasets import Dataset
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import llm_factory
        from ragas.metrics._answer_correctness import AnswerCorrectness
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.metrics._context_precision import ContextPrecision
        from ragas.metrics._context_recall import ContextRecall
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.run_config import RunConfig

        _RAGAS = types.SimpleNamespace(
            Dataset=Dataset,
            evaluate=evaluate,
            LangchainEmbeddingsWrapper=LangchainEmbeddingsWrapper,
            llm_factory=llm_factory,
            AnswerCorrectness=AnswerCorrectness,
            AnswerRelevancy=AnswerRelevancy,
            ContextPrecision=ContextPrecision,
            ContextRecall=ContextRecall,
            Faithfulness=Faithfulness,
            RunConfig=RunConfig,
        )
    return _RAGAS


_INPUT_COLUMNS = frozenset(
    {
        "question",
        "answer",
        "contexts",
        "ground_truth",
        "user_input",
        "response",
        "retrieved_contexts",
        "reference",
    }
)


def _result_traces(result: Any) -> list[Any]:
    """ragas >= 0.2 exposes ``EvaluationResult.traces``: one dict per sample,
    in dataset order, keyed by metric name -> prompt name -> {input, output}.
    Older/faked results have no such attribute; return an empty list."""
    traces = getattr(result, "traces", None)
    return list(traces) if isinstance(traces, (list, tuple)) else []


def _jsonable(value: Any) -> Any:
    """Recursively reduce a ragas trace to JSON-serialisable primitives.
    Prompt inputs/outputs are pydantic models; anything else falls back to
    ``str`` so a trace can never break the write of a multi-hour run."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _jsonable(dump())
        except Exception:  # noqa: BLE001 - fall through to str
            pass
    return str(value)


def _build_metrics(*, has_ground_truth: bool) -> list[Any]:
    """Build per-run metric instances (avoid mutating ragas singletons).
    Every metric here is a fresh instance: ragas metric objects carry mutable
    per-run state (llm, embeddings), so a module-level singleton would leak the
    previous run's judge into the next one (R-95).

    R-32 / R-49: AnswerRelevancy strictness comes from
    ``settings.ragas_answer_relevancy_strictness`` (shipped default 1 —
    structured adapters return one generation per call). Changing it moves both cost and answer_relevancy scores;
    the run manifest records the value actually used.
    """
    ragas = _import_ragas()
    relevancy = ragas.AnswerRelevancy(
        strictness=settings.ragas_answer_relevancy_strictness
    )
    if has_ground_truth:
        return [
            ragas.Faithfulness(),
            relevancy,
            ragas.ContextPrecision(),
            ragas.ContextRecall(),
            ragas.AnswerCorrectness(),
        ]
    return [ragas.Faithfulness(), relevancy]


class RagasRunner(BaseEvaluationRunner):
    """Evaluate RAG samples with the ragas library.

    Each sample must already contain ``question``, ``answer``, ``contexts`` and
    (optionally) ``ground_truth``. Use :class:`EvaluateRagPipelineUseCase` to
    run the live RAG pipeline first when only questions + ground truths are
    provided. Shared scoring/aggregation lives in
    :class:`~backend.evaluation.base.BaseEvaluationRunner`.
    """

    # Decision metrics reported as headline scores; answer_relevancy is
    # diagnostic only.
    HEADLINE_METRIC_NAMES = frozenset(
        {
            "context_recall",
            "context_precision",
            "faithfulness",
            "answer_correctness",
        }
    )
    DIAGNOSTIC_METRIC_NAMES = frozenset({"answer_relevancy"})
    GT_METRIC_NAMES = frozenset(
        {"context_recall", "context_precision", "answer_correctness"}
    )

    @property
    def name(self) -> str:
        return "ragas"

    def _evaluate(
        self,
        dataset: list[dict[str, Any]],
    ) -> dict[str, Any]:
        evaluator_model = settings.evaluation_model
        embedding_model = settings.evaluation_embedding_model
        # R-21 / HEAD: gate via provider_credentials on the providers these two
        # models actually route to — Azure-only runs need no OpenAI key.
        missing = missing_credentials(
            {"judge": evaluator_model, "embeddings": embedding_model},
            runner=self.name,
        )
        if missing:
            return self._credential_error(missing, dataset)

        api_key = settings.openai_api_key
        evaluator_model_openai = _openai_model_id(evaluator_model)
        azure_api_key = settings.azure_openai_api_key
        azure_endpoint = settings.azure_openai_endpoint
        azure_version = azure_api_version()

        rows = [self._normalize_row(row) for row in dataset]
        eligibility = [classify_for_evaluation(row) for row in rows]
        eligible_indices = [i for i, e in enumerate(eligibility) if e["eligible"]]
        eligible_rows = [rows[i] for i in eligible_indices]
        skipped = len(rows) - len(eligible_rows)
        if skipped:
            logger.info(
                "Excluding %d/%d samples from RAGAS (not answered).",
                skipped,
                len(rows),
            )
        # Nothing eligible: still emit a fully-formed result with statuses.
        if not eligible_rows:
            logger.warning("No eligible samples to score; returning statuses only.")
            return self._build_output(
                rows=rows,
                eligibility=eligibility,
                metric_names=[],
                scores_by_index={},
            )
        # R-05: gate ground-truth metrics PER ROW, not per run. The old
        # run-global all() meant one blank ground_truth cell stripped
        # context_recall/context_precision/answer_correctness from every row.
        # Covered here + BaseEvaluationRunner.GT_METRIC_NAMES (main); HEAD's
        # ground_truth.partition_by_ground_truth is equivalent but redundant.
        gt_indices = [i for i in eligible_indices if rows[i].get("ground_truth")]
        nogt_indices = [i for i in eligible_indices if not rows[i].get("ground_truth")]
        if nogt_indices:
            logger.warning(
                "%d/%d eligible rows lack ground_truth; scoring "
                "context_recall/context_precision/answer_correctness on the %d "
                "rows that have it (per-row gating, not run-global).",
                len(nogt_indices),
                len(eligible_indices),
                len(gt_indices),
            )
        subset_plan = [
            (gt_indices, _build_metrics(has_ground_truth=True)),
            (nogt_indices, _build_metrics(has_ground_truth=False)),
        ]
        subset_plan = [(idx, metrics) for idx, metrics in subset_plan if idx]

        job_count = sum(len(idx) * len(metrics) for idx, metrics in subset_plan)
        logger.info(
            "RAGAS judge workload: %d jobs across %d subset(s). "
            "Each sample needs several LLM judge calls — expect ~1–3 min per sample; "
            "the progress bar advances per completed job.",
            job_count,
            len(subset_plan),
        )

        logger.info(
            "Running ragas evaluation: %d eligible samples, model=%s, metrics=%s",
            len(eligible_rows),
            evaluator_model,
            [[m.name for m in metrics] for _idx, metrics in subset_plan],
        )
        # R-70: these used to be the SYNC OpenAI/AzureOpenAI clients, which
        # sends ragas' InstructorLLM down its blocking generate() path: every
        # judge call ran one at a time and RunConfig's concurrency and timeout
        # settings could not be enforced. Async clients make ragas_max_workers
        # and ragas_timeout mean something.
        if is_azure_model(evaluator_model):
            logger.info("Judge client: Azure (%s)", evaluator_model)
            openai_client = AsyncAzureOpenAI(
                api_key=azure_api_key,
                azure_deployment=evaluator_model_openai,
                azure_endpoint=azure_endpoint,
                api_version=azure_version,
            )
        else:
            logger.info("Judge client: OpenAI (%s)", evaluator_model)
            openai_client = AsyncOpenAI(api_key=api_key)
        ragas = _import_ragas()
        evaluator_llm = ragas.llm_factory(
            evaluator_model_openai,
            client=openai_client,
            **_judge_llm_kwargs(evaluator_model_openai),
        )
        # Keep the guard: an unguarded call would force temperature=1.0 and
        # strip max_tokens on classic judges (e.g. openai/gpt-4.1), silently
        # de-determinizing them.
        if _is_reasoning_model(evaluator_model_openai):
            _sanitize_reasoning_judge(evaluator_llm)
        embedding_deployment = _openai_model_id(embedding_model)
        # Route Azure-hosted embedding deployments (e.g.
        # "azure/embedv3-...") to Azure; the deployment name is not a valid
        # OpenAI model, so sending it to api.openai.com 404s.
        if is_azure_model(embedding_model):
            embeddings_client = AzureOpenAIEmbeddings(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=azure_version,
                azure_deployment=embedding_deployment,
                model=embedding_deployment,
                max_retries=5,
            )
        else:
            embeddings_client = OpenAIEmbeddings(
                api_key=api_key, model=embedding_deployment
            )
        evaluator_embeddings = ragas.LangchainEmbeddingsWrapper(embeddings_client)
        run_config = ragas.RunConfig(
            max_retries=settings.ragas_max_retries,
            max_wait=settings.ragas_max_wait,
            timeout=settings.ragas_timeout,
            max_workers=settings.ragas_max_workers,
        )
        # One evaluate() per ground-truth subset; scores merge back onto the
        # original row indices so the output shape is unchanged.
        metric_names: list[str] = []
        scores_by_index: dict[int, dict[str, Any]] = {}
        # Sep 2 2026: keep the judge's intermediate work (statement lists,
        # TP/FP/FN verdicts, faithfulness claims) per sample. Until now only
        # the final score columns survived, so a 0.0 on a correct answer could
        # not be explained after the fact. Row index -> ragas trace dict.
        traces_by_index: dict[int, dict[str, Any]] = {}
        for subset_indices, metric_objs in subset_plan:
            eval_dataset = ragas.Dataset.from_list(
                [self._to_ragas_input(rows[i]) for i in subset_indices]
            )
            result = ragas.evaluate(
                eval_dataset,
                metrics=metric_objs,
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                run_config=run_config,
                raise_exceptions=False,
            )
            df = result.to_pandas()
            subset_metric_names = [c for c in df.columns if c not in _INPUT_COLUMNS]
            for name in subset_metric_names:
                if name not in metric_names:
                    metric_names.append(name)
            records = df.to_dict(orient="records")
            traces = _result_traces(result)
            for local_pos, record in enumerate(records):
                original_index = subset_indices[local_pos]
                scores_by_index[original_index] = {
                    col: _clean_float(record.get(col)) for col in subset_metric_names
                }
                if local_pos < len(traces):
                    traces_by_index[original_index] = _jsonable(traces[local_pos])
        output = self._build_output(
            rows=rows,
            eligibility=eligibility,
            metric_names=metric_names,
            scores_by_index=scores_by_index,
            judge_model=evaluator_model,
            embedding_model=embedding_model,
        )
        # Side channel, not part of per_sample: the background job writes it
        # to <artifact>_traces.jsonl and drops the key from the stored result
        # so the report shape is unchanged.
        output["judge_traces"] = [
            {"index": i, "question": rows[i]["question"], "trace": traces_by_index[i]}
            for i in sorted(traces_by_index)
        ]
        return output

    @staticmethod
    def _to_ragas_input(row: dict[str, Any]) -> dict[str, Any]:
        """Project a normalized row down to only the columns RAGAS consumes.
        Metadata (generation_status, finish_reason, ...) is deliberately omitted
        so it isn't mistaken for a metric column in the result frame.

        R-04: citation markup is stripped from the SCORING inputs only — both
        the generated answer and the golden reference. Citations are metadata
        for humans, not factual statements the judge should verify (they
        depressed faithfulness and capped answer_correctness). ``row["answer"]``
        used for reporting stays unchanged.
        """
        scored_answer = strip_answer_citations(row["answer"])
        scored_reference = strip_answer_citations(row["ground_truth"])
        return {
            "question": row["question"],
            "answer": scored_answer,
            "contexts": row["contexts"],
            "ground_truth": scored_reference,
            "user_input": row["question"],
            "response": scored_answer,
            "retrieved_contexts": row["contexts"],
            "reference": scored_reference,
        }
