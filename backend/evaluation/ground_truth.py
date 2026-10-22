"""Per-sample ground-truth availability helpers for evaluation runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Metric names that require a non-empty ground_truth / reference.
RAGAS_GT_METRIC_NAMES = frozenset(
    {
        "context_recall",
        "context_precision",
        "answer_correctness",
    }
)
DEEPEVAL_GT_METRIC_NAMES = frozenset(
    {
        "contextual_recall",
        "contextual_precision",
    }
)


def row_has_ground_truth(row: dict[str, Any]) -> bool:
    """True when the sample has a non-empty ground-truth string."""
    return bool(str(row.get("ground_truth") or "").strip())


@dataclass(frozen=True)
class GroundTruthPartition:
    with_gt_indices: list[int]
    with_gt_rows: list[dict[str, Any]]
    without_gt_indices: list[int]
    without_gt_rows: list[dict[str, Any]]

    @property
    def all_have_ground_truth(self) -> bool:
        return bool(self.with_gt_indices) and not self.without_gt_indices

    @property
    def none_have_ground_truth(self) -> bool:
        return not self.with_gt_indices

    @property
    def is_mixed(self) -> bool:
        return bool(self.with_gt_indices) and bool(self.without_gt_indices)


def partition_by_ground_truth(
    eligible_indices: list[int],
    eligible_rows: list[dict[str, Any]],
) -> GroundTruthPartition:
    """Split eligible samples by whether each row has ground truth."""
    with_gt_indices: list[int] = []
    with_gt_rows: list[dict[str, Any]] = []
    without_gt_indices: list[int] = []
    without_gt_rows: list[dict[str, Any]] = []
    for index, row in zip(eligible_indices, eligible_rows):
        if row_has_ground_truth(row):
            with_gt_indices.append(index)
            with_gt_rows.append(row)
        else:
            without_gt_indices.append(index)
            without_gt_rows.append(row)
    return GroundTruthPartition(
        with_gt_indices=with_gt_indices,
        with_gt_rows=with_gt_rows,
        without_gt_indices=without_gt_indices,
        without_gt_rows=without_gt_rows,
    )


def merge_scores_by_index(
    *score_maps: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for score_map in score_maps:
        for index, scores in score_map.items():
            merged.setdefault(index, {}).update(scores)
    return merged


def is_intentional_gt_skip(
    *,
    metric: str,
    row: dict[str, Any],
    gt_metric_names: frozenset[str],
) -> bool:
    """GT-dependent metric is N/A for this sample because GT is absent."""
    return metric in gt_metric_names and not row_has_ground_truth(row)


def applicable_sample_count(
    *,
    metric: str,
    rows: list[dict[str, Any]],
    eligibility: list[dict[str, Any]],
    gt_metric_names: frozenset[str],
) -> int:
    """Eligible samples for which ``metric`` is expected to produce a score."""
    count = 0
    for row, verdict in zip(rows, eligibility):
        if not verdict.get("eligible"):
            continue
        if is_intentional_gt_skip(
            metric=metric, row=row, gt_metric_names=gt_metric_names
        ):
            continue
        count += 1
    return count
