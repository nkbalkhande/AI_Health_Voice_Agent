"""Classify evaluation-sample outcomes for reporting and refusal scorecards.

Refusal statistics must count only intentional model refusals — never
timeouts, API/infrastructure failures, parse errors, or other eval issues
(R-09). Refusal *detection* is delegated to
:mod:`backend.evaluation.eligibility` so there is exactly one implementation of
``looks_like_refusal`` (with the R-33 length anchor) and one set of
``REFUSAL_MARKERS`` kept in lockstep with the system prompt.
"""

from __future__ import annotations

from typing import Any, Literal

from backend.evaluation.eligibility import looks_like_refusal

OutcomeCategory = Literal[
    "answered",
    "model_refusal",
    "timeout",
    "infrastructure_error",
    "generation_error",
    "empty_generation",
    "incomplete_length",
    "parse_validation_failure",
    "evaluation_error",
    "not_applicable",
]

# Genuine model refusals only (API refusal field or mandated no-evidence text).
_MODEL_REFUSAL_STATUSES = frozenset({"refused"})
_MODEL_REFUSAL_REASONS = frozenset({"refused", "refused_no_evidence"})

# Generation-time failure statuses that are NOT refusals. "pipeline_error" is
# the R-13 per-case isolation status from EvaluateRagPipelineUseCase.
_INFRA_STATUSES = frozenset({"infrastructure_error", "pipeline_error"})

_PARSE_METRIC_ERROR_MARKERS = (
    "invalid json",
    "unparseable",
    "json_repair",
    "parse",
    "validation",
    "schema",
)

_EVAL_METRIC_ERROR_MARKERS = (
    "evaluator returned no score",
    "missing metrics:",
)


def is_model_refusal(row: dict[str, Any]) -> bool:
    """True only for intentional model refusals (not infra / empty / errors)."""
    status = str(row.get("generation_status") or "answered")
    if status in _MODEL_REFUSAL_STATUSES:
        return True
    if row.get("metric_error") in _MODEL_REFUSAL_REASONS:
        return True
    if status == "answered" and looks_like_refusal(str(row.get("answer") or "")):
        return True
    return False


def classify_evaluation_outcome(row: dict[str, Any]) -> dict[str, Any]:
    """Map a per-sample eval row to a single outcome category.

    Priority (first match wins):
    1. model refusal
    2. generation-time failure statuses (timeout, infra, API, empty, length)
    3. scoring-time parse / evaluation failures (only when generation answered)
    4. successful answer
    """
    status = str(row.get("generation_status") or "answered")
    metric_status = str(row.get("metric_status") or "")
    metric_error = str(row.get("metric_error") or "")
    metric_error_l = metric_error.lower()

    if is_model_refusal(row):
        if row.get("metric_error") in _MODEL_REFUSAL_REASONS:
            reason = str(row.get("metric_error"))
        elif status == "refused":
            reason = "refused"
        else:
            reason = "refused_no_evidence"
        return {
            "category": "model_refusal",
            "reason": reason,
            "is_refusal": True,
            "is_answered": False,
        }

    if status == "timeout":
        return _outcome("timeout", status)
    if status in _INFRA_STATUSES:
        return _outcome("infrastructure_error", status)
    if status == "generation_error":
        return _outcome("generation_error", status)
    if status == "incomplete_length":
        return _outcome("incomplete_length", status)
    if status in {"empty_generation", "abstained"}:
        return _outcome("empty_generation", status)

    if status == "answered":
        if not str(row.get("answer") or "").strip():
            return _outcome("empty_generation", "empty_generation")
        if metric_status in {"error", "partial"}:
            if any(m in metric_error_l for m in _PARSE_METRIC_ERROR_MARKERS):
                return _outcome(
                    "parse_validation_failure", metric_error or metric_status
                )
            if any(m in metric_error_l for m in _EVAL_METRIC_ERROR_MARKERS):
                return _outcome("evaluation_error", metric_error or metric_status)
            if metric_status == "error":
                return _outcome("evaluation_error", metric_error or metric_status)
        if metric_status == "not_applicable":
            return _outcome("not_applicable", metric_error or metric_status)
        return {
            "category": "answered",
            "reason": None,
            "is_refusal": False,
            "is_answered": True,
        }

    return _outcome("not_applicable", status)


def _outcome(category: OutcomeCategory, reason: str | None) -> dict[str, Any]:
    return {
        "category": category,
        "reason": reason,
        "is_refusal": False,
        "is_answered": False,
    }
