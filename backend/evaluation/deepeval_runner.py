from __future__ import annotations

import os
import sys
from typing import Any

from backend.core.config import settings
from backend.core.logger import logger
from backend.evaluation.base import BaseEvaluationRunner
from backend.evaluation.eligibility import (  # noqa: F401 - re-exported for tests
    _clean_float,
    classify_for_evaluation,
)
from backend.evaluation.provider_credentials import (
    azure_api_version,
    is_azure_model,
    missing_credentials,
    model_id,
)
from backend.evaluation.strip_citations import strip_answer_citations

# Swapping the api_key alone is not enough for DeepEval: LiteLLM also needs the
# azure/* route and api_base, so the Azure error keeps pointing operators there.
_AZURE_ROUTE_HINT = "LiteLLM needs azure/* plus api_base."

_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _missing_judge_credentials(judge_model: str) -> str | None:
    """Return an error message when judge credentials/route are incomplete."""
    return missing_credentials(
        {"judge": judge_model}, runner="deepeval", azure_hint=_AZURE_ROUTE_HINT
    )


def _is_reasoning_judge_model(model: str) -> bool:
    return model_id(model).startswith(_REASONING_MODEL_PREFIXES)


def _resolve_judge_litellm_client_kwargs(judge_model: str) -> dict[str, Any]:
    """Build LiteLLMModel kwargs so the judge actually reaches its provider.

    Swapping only ``api_key`` to an Azure key is not enough: LiteLLM still needs
    the ``azure/`` model route and ``api_base`` (DeepEval's ``base_url``) — and
    usually ``api_version`` — or calls go to the wrong host and fail quietly
    into null/zero scores.

    Always honor ``settings.evaluation_judge_max_tokens`` via generation_kwargs
    (``max_completion_tokens`` for GPT-5 / o-series, ``max_tokens`` otherwise).
    Falls back through :func:`azure_api_version` so deployments that only set
    ``OPENAI_API_VERSION`` keep working.
    """
    generation_kwargs: dict[str, Any] = {
        "timeout": settings.deepeval_per_task_timeout_seconds,
        "num_retries": 0,
    }
    if _is_reasoning_judge_model(judge_model):
        generation_kwargs["max_completion_tokens"] = (
            settings.evaluation_judge_max_tokens
        )
    else:
        generation_kwargs["max_tokens"] = settings.evaluation_judge_max_tokens
    if is_azure_model(judge_model):
        api_version = azure_api_version()
        if api_version:
            generation_kwargs["api_version"] = api_version
        return {
            "api_key": settings.azure_openai_api_key,
            "base_url": settings.azure_openai_endpoint,
            "generation_kwargs": generation_kwargs,
        }
    return {
        "api_key": settings.openai_api_key,
        "base_url": None,
        "generation_kwargs": generation_kwargs,
    }


def _configure_utf8_stdio() -> None:
    """Avoid Windows cp1252 crashes when DeepEval/rich prints emoji.

    R-87: this used to reconfigure stdout/stderr on every platform from inside
    a worker thread. Only Windows consoles need it.
    """
    if not sys.platform.startswith("win"):
        return
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _load_judge_json(content: str) -> dict[str, Any]:
    import json
    import re

    text = content or ""
    start = text.find("{")
    end = text.rfind("}") + 1
    json_str = text[start:end] if start != -1 and end != 0 else text
    json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
    return json.loads(json_str)


def _strict_json_schema(schema: Any) -> dict[str, Any]:
    """Build an OpenAI strict-mode json_schema response_format for *schema*.

    LiteLLMModel._generate() hands litellm the raw pydantic class, which gets
    converted to a *non-strict* json_schema. Non-strict mode only asks the
    model to follow the schema — it does NOT guarantee syntactically valid
    JSON, so long verdict arrays (Faithfulness/ContextualRecall over many
    chunks) occasionally corrupt (a stray unescaped quote/newline) and
    trim_and_load_json raises "invalid JSON". Strict mode is what actually
    guarantees valid, schema-conformant JSON every time, so we build it
    ourselves — every object in the schema graph (including nested $defs)
    needs additionalProperties=False and all properties required.
    """
    json_schema = schema.model_json_schema()

    def _tighten(node: Any) -> None:
        if isinstance(node, dict):
            if "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
            for value in node.values():
                _tighten(value)
        elif isinstance(node, list):
            for item in node:
                _tighten(item)

    _tighten(json_schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "strict": True,
            "schema": json_schema,
        },
    }


def _build_metrics(*, has_ground_truth: bool, judge_model: str) -> list[Any]:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.models import LiteLLMModel

    class _StrictJSONLiteLLMModel(LiteLLMModel):
        """LiteLLMModel with two hardening layers over the stock judge model.

        1. Force OpenAI strict json_schema mode (see :func:`_strict_json_schema`)
           so the model is asked to emit valid, schema-conformant JSON.
        2. Repair the response before parsing. DeepEval's stock parser
           (``trim_and_load_json``) only strips trailing commas and does a naive
           ``{``..``}`` slice, so any residual glitch — an unescaped quote in a
           ``reason`` string, or a length-truncated object — makes it raise
           ``DeepEvalError("invalid JSON")`` and abort the whole eval. We instead
           fall back to ``json_repair``, which closes/repairs the structure, and
           we log ``finish_reason`` so truncation is distinguishable from
           corruption in the logs.
        """

        def _generate(self, prompt, schema=None):
            from litellm import completion

            params = self._completion_params(self._build_content(prompt))
            if schema:
                params["response_format"] = _strict_json_schema(schema)
            response = completion(**params)
            return self._parse_response(response, schema)

        async def _a_generate(self, prompt, schema=None):
            from litellm import acompletion

            params = self._completion_params(self._build_content(prompt))
            if schema:
                params["response_format"] = _strict_json_schema(schema)
            response = await acompletion(**params)
            return self._parse_response(response, schema)

        def _parse_response(self, response, schema):
            choice = response.choices[0]
            content = choice.message.content
            cost = self._response_cost(response)
            if schema is None:
                return content, cost
            try:
                data = _load_judge_json(content)
            except Exception:
                finish_reason = getattr(choice, "finish_reason", None)
                logger.warning(
                    "DeepEval judge returned unparseable JSON "
                    "(finish_reason=%s, chars=%d); repairing with json_repair. "
                    "If finish_reason='length', raise evaluation.deepeval."
                    "evaluation_judge_max_tokens.",
                    finish_reason,
                    len(content or ""),
                )
                from json_repair import repair_json

                data = repair_json(content or "", return_objects=True)
            # repair_json returns "" for empty content and a list for
            # array-shaped output; schema(**data) then raised TypeError
            # OUTSIDE any handler and aborted the whole run (R-14).
            if not isinstance(data, dict):
                raise ValueError(
                    "Judge output is not a JSON object "
                    f"(got {type(data).__name__}, "
                    f"finish_reason={getattr(choice, 'finish_reason', None)})"
                )
            return schema.model_validate(data), cost

    # Large retrieval contexts produce long per-statement verdict JSON;
    # without an explicit ceiling the judge call truncates mid-JSON and
    # trim_and_load_json raises DeepEvalError("invalid JSON"). Azure judges
    # additionally need base_url + api_version wired explicitly, which the
    # api_key-only construction never did (R-02 / R-03).
    client_kwargs = _resolve_judge_litellm_client_kwargs(judge_model)
    logger.info(
        "DeepEval judge LiteLLM route: model=%s, api_base=%s, api_version=%s",
        judge_model,
        client_kwargs.get("base_url"),
        client_kwargs["generation_kwargs"].get("api_version"),
    )
    model = _StrictJSONLiteLLMModel(
        model=judge_model,
        **client_kwargs,
    )
    metrics: list[Any] = [
        AnswerRelevancyMetric(model=model),
        FaithfulnessMetric(model=model),
    ]
    if has_ground_truth:
        metrics.extend(
            [
                ContextualPrecisionMetric(model=model),
                ContextualRecallMetric(model=model),
            ]
        )
    return metrics


def _metric_key(metric: Any) -> str:
    raw = getattr(metric, "name", None) or metric.__class__.__name__
    text = str(raw).lower().replace(" ", "_")
    aliases = {
        "answerrelevancymetric": "answer_relevancy",
        "answer_relevancy": "answer_relevancy",
        "faithfulnessmetric": "faithfulness",
        "faithfulness": "faithfulness",
        "contextualprecisionmetric": "contextual_precision",
        "contextual_precision": "contextual_precision",
        "contextualrecallmetric": "contextual_recall",
        "contextual_recall": "contextual_recall",
    }
    cleaned = text.replace("metric", "").strip("_")
    return aliases.get(text, aliases.get(cleaned, cleaned))


class DeepEvalRunner(BaseEvaluationRunner):
    """Evaluate RAG samples with the deepeval library.

    Each sample must already contain ``question``, ``answer``, ``contexts`` and
    (optionally) ``ground_truth``. Use :class:`EvaluateRagPipelineUseCase` to
    run the live RAG pipeline first when only questions + ground truths are
    provided. Shared scoring/aggregation lives in
    :class:`~backend.evaluation.base.BaseEvaluationRunner`.
    """

    # Aligned with the Ragas headline / diagnostic split where DeepEval has
    # equivalents.
    HEADLINE_METRIC_NAMES = frozenset(
        {
            "contextual_recall",
            "contextual_precision",
            "faithfulness",
        }
    )
    DIAGNOSTIC_METRIC_NAMES = frozenset({"answer_relevancy"})
    GT_METRIC_NAMES = frozenset({"contextual_recall", "contextual_precision"})

    @property
    def name(self) -> str:
        return "deepeval"

    def _evaluate(
        self,
        dataset: list[dict[str, Any]],
    ) -> dict[str, Any]:
        judge_model = settings.evaluation_model
        # R-54: gate on the credentials this judge actually needs. The old gate
        # demanded OPENAI_API_KEY (never used on the shipped azure/* judge) and
        # never checked the Azure credentials it does use, so a box with a
        # stale OpenAI key burned a full run before failing.
        missing = _missing_judge_credentials(judge_model)
        if missing:
            return self._credential_error(missing, dataset)

        max_concurrent = settings.deepeval_max_concurrent
        # DeepEval's timeout wrapper can terminate a valid long structured-output
        # request. A null setting explicitly disables those wrappers; LiteLLM
        # still has the finite provider timeout configured in _build_metrics().
        attempt_timeout = settings.deepeval_per_attempt_timeout_seconds
        # R-87: these env vars are read by deepeval AT IMPORT TIME, so the
        # ordering here is load-bearing and was entirely implicit. Fail loudly
        # if something imported deepeval first, rather than silently running
        # with the wrong timeout policy.
        if "deepeval" in sys.modules:
            logger.warning(
                "deepeval was already imported before the timeout policy was "
                "set; DEEPEVAL_* overrides may not take effect for this run."
            )
        if attempt_timeout is None:
            os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "1"
            os.environ.pop(
                "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", None
            )
            os.environ.pop("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", None)
        else:
            os.environ.pop("DEEPEVAL_DISABLE_TIMEOUTS", None)
            os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = str(
                attempt_timeout
            )
            os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = str(
                settings.deepeval_per_task_timeout_seconds
            )
        os.environ["DEEPEVAL_RETRY_MAX_ATTEMPTS"] = str(
            settings.deepeval_retry_max_attempts
        )
        logger.info(
            "DeepEval timeout policy: enforced=%s, per_attempt=%s, "
            "provider_timeout=%ss, attempts=%d",
            attempt_timeout is not None,
            attempt_timeout,
            settings.deepeval_per_task_timeout_seconds,
            settings.deepeval_retry_max_attempts,
        )

        rows = [self._normalize_row(row) for row in dataset]
        eligibility = [classify_for_evaluation(row) for row in rows]
        eligible_indices = [i for i, e in enumerate(eligibility) if e["eligible"]]
        eligible_rows = [rows[i] for i in eligible_indices]

        skipped = len(rows) - len(eligible_rows)
        if skipped:
            logger.info(
                "Excluding %d/%d samples from DeepEval (not answered).",
                skipped,
                len(rows),
            )

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
        # contextual_recall/contextual_precision from every row.
        # Covered here + BaseEvaluationRunner.GT_METRIC_NAMES (main); HEAD's
        # ground_truth.partition_by_ground_truth is equivalent but redundant.
        gt_indices = [i for i in eligible_indices if rows[i].get("ground_truth")]
        nogt_indices = [i for i in eligible_indices if not rows[i].get("ground_truth")]
        if nogt_indices:
            logger.warning(
                "%d/%d eligible rows lack ground_truth; scoring "
                "contextual_recall/contextual_precision on the %d rows that "
                "have it (per-row gating, not run-global).",
                len(nogt_indices),
                len(eligible_indices),
                len(gt_indices),
            )
        subset_plan = [
            (gt_indices, True),
            (nogt_indices, False),
        ]
        subset_plan = [(idx, has_gt) for idx, has_gt in subset_plan if idx]

        logger.info(
            "Running deepeval evaluation: %d eligible samples across %d "
            "subset(s), model=%s",
            len(eligible_rows),
            len(subset_plan),
            judge_model,
        )

        from deepeval import evaluate
        from deepeval.evaluate import AsyncConfig
        from deepeval.evaluate.configs import DisplayConfig, ErrorConfig
        from deepeval.test_case import LLMTestCase

        # DeepEval prints emoji via rich; Windows cp1252 consoles raise
        # UnicodeEncodeError.
        _configure_utf8_stdio()

        metric_names: list[str] = []
        scores_by_index: dict[int, dict[str, Any]] = {}
        for subset_indices, subset_has_gt in subset_plan:
            metric_objs = _build_metrics(
                has_ground_truth=subset_has_gt, judge_model=judge_model
            )
            subset_metric_names = [_metric_key(m) for m in metric_objs]
            for name in subset_metric_names:
                if name not in metric_names:
                    metric_names.append(name)
            logger.info(
                "DeepEval subset: %d samples × %d metrics = %d judge jobs (%s).",
                len(subset_indices),
                len(metric_objs),
                len(subset_indices) * len(metric_objs),
                subset_metric_names,
            )
            # R-04: strip citation markup from the SCORING inputs only (both
            # the generated answer and the golden expected_output); the
            # reported per_sample answer stays unchanged.
            test_cases = [
                LLMTestCase(
                    input=rows[i]["question"],
                    actual_output=strip_answer_citations(rows[i]["answer"]),
                    expected_output=strip_answer_citations(
                        rows[i]["ground_truth"]
                    ),
                    retrieval_context=rows[i]["contexts"],
                )
                for i in subset_indices
            ]
            # Synchronous evaluation (run_async=False) — same blocking style as Ragas.
            raw = evaluate(
                test_cases=test_cases,
                metrics=metric_objs,
                async_config=AsyncConfig(
                    run_async=False,
                    max_concurrent=max_concurrent,
                    throttle_value=0,
                ),
                display_config=DisplayConfig(
                    show_indicator=False,
                    print_results=False,
                    inspect_after_run=False,
                ),
                # One unparseable judge verdict must not abort the whole run
                # after most of the judge spend (R-14); it becomes a null score
                # for that row, which _build_output already reports as missing.
                error_config=ErrorConfig(ignore_errors=True),
            )
            scores_by_index.update(
                self._scores_by_original_index(
                    raw_results=raw,
                    eligible_indices=subset_indices,
                    metric_names=subset_metric_names,
                )
            )

        return self._build_output(
            rows=rows,
            eligibility=eligibility,
            metric_names=metric_names,
            scores_by_index=scores_by_index,
            judge_model=judge_model,
        )

    def _scores_by_original_index(
        self,
        *,
        raw_results: Any,
        eligible_indices: list[int],
        metric_names: list[str],
    ) -> dict[int, dict[str, Any]]:
        test_results = (
            getattr(raw_results, "test_results", None)
            or getattr(raw_results, "results", None)
            or []
        )
        if isinstance(raw_results, list):
            test_results = raw_results

        scores_by_index: dict[int, dict[str, Any]] = {}
        for local_pos, result in enumerate(test_results):
            # R-67: TestResult.index is DeepEval's own submission-order id; it
            # survives async completion-order and skipped cases, whereas raw
            # enumeration position silently mis-attributes scores to the wrong
            # question the moment run_async is enabled or a case is skipped.
            case_pos = getattr(result, "index", None)
            if case_pos is None:
                case_pos = local_pos
            if case_pos >= len(eligible_indices):
                logger.warning("Unmappable DeepEval result #%s dropped", case_pos)
                continue
            original_index = eligible_indices[case_pos]
            row_scores: dict[str, Any] = {name: None for name in metric_names}
            metric_data = (
                getattr(result, "metrics_data", None)
                or getattr(result, "metrics", None)
                or []
            )
            for metric in metric_data:
                key = _metric_key(metric)
                row_scores[key] = _clean_float(getattr(metric, "score", None))
            scores_by_index[original_index] = row_scores
        return scores_by_index
