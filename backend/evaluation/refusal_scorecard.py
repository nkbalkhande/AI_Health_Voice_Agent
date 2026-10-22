"""Refusal scorecard scoring against a golden Answerable policy.

Importable so both ``scripts/refusal_scorecard.py`` and the tests use one
implementation (R-61). Counting rules (R-09): a refusal is a refusal-POLICY
decision only — ``generation_status == "refused"``, the mandated refusal
sentence, or ``metric_error`` in ``{refused, refused_no_evidence}``. Rows
whose generation failed for infrastructure reasons (timeout, API error,
empty, truncated, pipeline error) are excluded from the confusion matrix and
tallied separately by outcome category, so an errored trap can never hit the
recall target and an errored genuine question is never a false refusal.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from backend.evaluation.outcome_classification import (
    classify_evaluation_outcome,
    is_model_refusal,
)

#: Generation statuses that enter the refusal confusion matrix. Everything
#: else is an infra/eval failure, excluded and reported under ``categories``.
_MATRIX_STATUSES = frozenset({"answered", "refused"})


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def score(per_sample: list[dict], policy: dict[str, dict]) -> dict[str, Any]:
    """Count refusal outcomes for ``per_sample`` rows against ``policy``.

    ``policy`` maps normalized question text → golden row (must include
    ``Answerable``). Returns the same count keys the scorecard has always
    reported, plus ``categories`` (per-outcome tallies from
    :func:`classify_evaluation_outcome`).
    """
    counts: dict[str, Any] = dict(
        tp=0, fp=0, fn=0, tn=0, unmatched=0, errored=0,
        answered_genuine=0, genuine=0,
    )
    categories: Counter[str] = Counter()

    for row in per_sample:
        outcome = classify_evaluation_outcome(row)
        categories[str(outcome["category"])] += 1

        pol = policy.get(_norm(row.get("question", "")))
        if pol is None:
            counts["unmatched"] += 1
            continue

        if "Answerable" not in pol:
            raise ValueError(
                "Golden CSV has no 'Answerable' column - every question would "
                "silently default to answerable and the refusal numbers would "
                f"be meaningless. Columns found: {sorted(pol)}"
            )
        should_refuse = pol["Answerable"].strip().lower() == "no"

        # R-09: only generation-level outcomes enter the matrix.
        status = str(row.get("generation_status") or "answered")
        if status not in _MATRIX_STATUSES:
            counts["errored"] += 1
            continue
        refused = is_model_refusal(row)

        if should_refuse:
            counts["tp" if refused else "fn"] += 1
        else:
            counts["genuine"] += 1
            if refused:
                counts["fp"] += 1
            else:
                counts["tn"] += 1
                counts["answered_genuine"] += 1

    counts["categories"] = dict(categories)
    return counts


def score_refusal_run(
    run: dict[str, Any],
    policy: dict[str, dict],
) -> dict[str, Any]:
    """Score a full runner result dict (must contain ``per_sample``)."""
    per_sample = list(run.get("per_sample") or [])
    counts = score(per_sample, policy)
    counts["total"] = len(per_sample)
    return counts
