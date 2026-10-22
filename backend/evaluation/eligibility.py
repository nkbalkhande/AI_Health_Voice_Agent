"""Which samples may be scored, and how missing scores are represented.

R-68: this lived in ``ragas_runner`` and was imported from there by
``deepeval_runner``, which made the ragas module a de-facto base module and
forced a one-way import between two peers.
"""

from __future__ import annotations

import math
from typing import Any

# Keep in lockstep with the refusal sentence mandated in
# app/rag/agent_prompt/rag.py. If the prompt's refusal wording ever changes,
# change it here in the same PR — app/tests/test_eval_contracts.py enforces it.
REFUSAL_MARKERS = (
    "i could not find sufficient information in the provided context",
    "could not find sufficient information",
    "no relevant information in the provided context",
)


def looks_like_refusal(answer: str) -> bool:
    """True only when the refusal sentence effectively IS the answer.

    R-33: an unanchored substring match reclassified a substantive answer that
    merely appends the mandated sentence for an unanswerable sub-part as a full
    refusal — excluded from all scoring and counted as a refusal by the
    scorecard.
    """
    normalized = " ".join((answer or "").lower().split())
    if not normalized:
        return False
    return any(
        marker in normalized and len(normalized) <= len(marker) + 40
        for marker in REFUSAL_MARKERS
    )


def classify_for_evaluation(row: dict[str, Any]) -> dict[str, Any]:
    """Decide whether a sample is eligible for scoring.

    Eligible == the generation actually produced a substantive answer.
    Non-answers (refusals — both status-level and the mandated refusal
    sentence — empty, errored, truncated) are excluded with a reason so they
    are reported, not silently scored as failures.
    """
    status = row.get("generation_status") or "answered"
    if status != "answered":
        return {"eligible": False, "reason": status}
    answer = str(row.get("answer") or "").strip()
    if not answer:
        return {"eligible": False, "reason": "empty_generation"}
    if looks_like_refusal(answer):
        return {"eligible": False, "reason": "refused_no_evidence"}
    return {"eligible": True, "reason": None}


def clean_float(value: Any) -> Any:
    """Convert NaN/inf to None (never 0) so unavailable metrics stay unavailable.

    Coercing a missing metric to ``0`` invents a perfect-failure score the
    metric never produced, so we keep it null instead.
    """
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


#: Historical private name, kept because both runners and the tests use it.
_clean_float = clean_float
