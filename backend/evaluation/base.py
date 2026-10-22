"""Shared scoring/aggregation machinery for the evaluation runners.

R-68: ``RagasRunner`` and ``DeepEvalRunner`` carried ~161 verbatim-duplicated
lines — ``run``, ``_normalize_row``, ``_build_output`` and the credential gate
were byte-identical apart from the two metric-name frozensets. Duplicated
scoring code is the worst place for drift: a fix applied to one runner and not
the other produces two different numbers for the same answers, with nothing to
flag it. Each runner now owns only its framework-specific ``_evaluate``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from backend.core.logger import logger
from backend.domain.interfaces.evaluation_runner import EvaluationRunner
from backend.evaluation.eligibility import clean_float
from backend.evaluation.outcome_classification import classify_evaluation_outcome


class BaseEvaluationRunner(EvaluationRunner):
    """Common shape for a runner: eligibility, aggregation, output contract.

    Subclasses implement :meth:`_evaluate` (blocking; called in a worker
    thread) and set the two metric-name frozensets.

    Only samples whose generation actually answered are scored; everything
    else is preserved in the output with its status rather than scored as zero.
    """

    #: Decision metrics reported as headline scores.
    HEADLINE_METRIC_NAMES: frozenset[str] = frozenset()
    #: Reported but not gating.
    DIAGNOSTIC_METRIC_NAMES: frozenset[str] = frozenset()
    #: Metrics that require a non-empty ground_truth / reference. On mixed
    #: runs a no-GT row intentionally has no score for these — that is an
    #: expected skip, not a judge failure, so it must not mark the row
    #: "partial" or count as "missing" in metrics_coverage (R-05 follow-up).
    GT_METRIC_NAMES: frozenset[str] = frozenset()

    def _is_intentional_gt_skip(self, metric: str, row: dict[str, Any]) -> bool:
        return metric in self.GT_METRIC_NAMES and not str(
            row.get("ground_truth") or ""
        ).strip()

    async def run(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        if not dataset:
            return {
                "runner": self.name,
                "status": "empty",
                "message": "No samples to evaluate.",
                "samples": 0,
                "metrics": {},
                "per_sample": [],
            }
        # Both libraries' evaluate() are synchronous and blocking; keep the
        # event loop free.
        return await asyncio.to_thread(self._evaluate, dataset)

    def _evaluate(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError

    def _credential_error(
        self, message: str, dataset: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "runner": self.name,
            "status": "error",
            "message": message,
            "samples": len(dataset),
            "metrics": {},
            "per_sample": [],
        }

    @staticmethod
    def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
        # Use "" rather than the literal string "None" when a field is missing,
        # so the evaluator LLM doesn't try to fact-check the word "None".
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        contexts = [str(c) for c in row.get("contexts", [])]
        ground_truth = str(row.get("ground_truth") or "").strip()
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "generation_status": row.get("generation_status") or "answered",
            "finish_reason": row.get("finish_reason"),
            "context_count": row.get("context_count", len(contexts)),
            # Observability passthrough — never fed to scoring.
            "context_details": row.get("context_details") or [],
            "pipeline_debug": row.get("pipeline_debug") or {},
        }

    def _build_output(
        self,
        *,
        rows: list[dict[str, Any]],
        eligibility: list[dict[str, Any]],
        metric_names: list[str],
        scores_by_index: dict[int, dict[str, Any]],
        judge_model: str | None = None,
        embedding_model: str | None = None,
    ) -> dict[str, Any]:
        per_sample: list[dict[str, Any]] = []
        # Accumulate only real (non-null) metric values for the aggregate so a
        # missing metric never drags the mean toward zero.
        metric_totals: dict[str, list[float]] = {m: [] for m in metric_names}
        # Rows where each metric was expected to score (eligible rows minus
        # intentional GT skips) — the coverage denominator.
        metric_applicable: dict[str, int] = {m: 0 for m in metric_names}

        for index, (row, verdict) in enumerate(zip(rows, eligibility)):
            sample: dict[str, Any] = {
                "question": row.get("question"),
                "answer": row.get("answer"),
                "ground_truth": row.get("ground_truth") or "",
                "generation_status": row.get("generation_status", "answered"),
                "finish_reason": row.get("finish_reason"),
                "context_count": row.get(
                    "context_count", len(row.get("contexts") or [])
                ),
                # R-88: the run output used to ship the whole dataset a second
                # time under a top-level "contexts" key. The contexts belong
                # to the sample they were retrieved for.
                "contexts": list(row.get("contexts") or []),
                "context_details": row.get("context_details") or [],
                "pipeline_debug": row.get("pipeline_debug") or {},
            }

            if not verdict["eligible"]:
                for metric in metric_names:
                    sample[metric] = None
                sample["metric_status"] = "not_applicable"
                sample["metric_error"] = verdict["reason"]
                outcome = classify_evaluation_outcome(sample)
                sample["outcome_category"] = outcome["category"]
                sample["is_refusal"] = outcome["is_refusal"]
                per_sample.append(sample)
                continue

            scores = scores_by_index.get(index, {})
            missing: list[str] = []
            expected_metric_count = 0
            for metric in metric_names:
                value = clean_float(scores.get(metric))
                sample[metric] = value
                if value is None and self._is_intentional_gt_skip(metric, row):
                    # No ground truth for this row: a GT-dependent metric is
                    # expected to be absent — neither missing nor scored. A
                    # real score is always counted, GT or not.
                    continue
                metric_applicable[metric] += 1
                expected_metric_count += 1
                if value is None:
                    missing.append(metric)
                else:
                    metric_totals[metric].append(float(value))

            if missing and len(missing) == expected_metric_count:
                sample["metric_status"] = "error"
                sample["metric_error"] = "evaluator returned no score"
            elif missing:
                sample["metric_status"] = "partial"
                sample["metric_error"] = f"missing metrics: {', '.join(missing)}"
            else:
                sample["metric_status"] = "success"
                sample["metric_error"] = None

            outcome = classify_evaluation_outcome(sample)
            sample["outcome_category"] = outcome["category"]
            sample["is_refusal"] = outcome["is_refusal"]
            per_sample.append(sample)

        eligible_count = sum(1 for v in eligibility if v["eligible"])
        metrics_coverage: dict[str, dict[str, int]] = {}
        aggregate: dict[str, Any] = {}
        for metric, values in metric_totals.items():
            scored_count = len(values)
            # Coverage denominator = rows where this metric was expected: all
            # eligible rows, minus intentional GT skips for GT-bound metrics.
            applicable_count = metric_applicable[metric]
            missing_count = applicable_count - scored_count
            metrics_coverage[metric] = {
                "scored_samples": scored_count,
                "eligible_samples": applicable_count,
                "missing_samples": missing_count,
            }
            aggregate[metric] = (sum(values) / scored_count) if values else None
            if missing_count > 0:
                logger.warning(
                    "Judge returned no score for %d/%d applicable samples on %s; "
                    "aggregate mean uses %d rows only (missing rows are not "
                    "scored as 0).",
                    missing_count,
                    applicable_count,
                    metric,
                    scored_count,
                )

        outcome_counts: dict[str, int] = {}
        for sample in per_sample:
            category = str(sample.get("outcome_category") or "not_applicable")
            outcome_counts[category] = outcome_counts.get(category, 0) + 1

        return {
            "runner": self.name,
            "status": "ok",
            "samples": len(rows),
            "evaluated_samples": eligible_count,
            "skipped_samples": len(rows) - eligible_count,
            "judge_model": judge_model,
            "embedding_model": embedding_model,
            "metrics": aggregate,
            "headline_metrics": {
                k: v for k, v in aggregate.items() if k in self.HEADLINE_METRIC_NAMES
            },
            "diagnostic_metrics": {
                k: v
                for k, v in aggregate.items()
                if k in self.DIAGNOSTIC_METRIC_NAMES
            },
            "metrics_coverage": metrics_coverage,
            # R-09 observability: per-category tallies plus a refusal count
            # that includes ONLY intentional model refusals.
            "outcome_counts": outcome_counts,
            "refusal_count": sum(1 for s in per_sample if s.get("is_refusal")),
            "per_sample": per_sample,
        }
