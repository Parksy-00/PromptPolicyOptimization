from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass

from prompt_policy.types import Trajectory


@dataclass(frozen=True)
class Decision:
    old_accuracy: float
    new_accuracy: float
    delta_accuracy: float
    success_gain: float
    failure_gain: float | None
    success_sum_gain: float
    success_increase_fraction: float | None
    success_questions: int
    success_trajectories: int
    pool_sha256: str
    score: float
    accepted: bool
    reason: str

    def to_dict(self):
        return asdict(self)


def grouped_mean(rows: list[tuple[Trajectory, float]]) -> float:
    """Each eligible question has equal weight, regardless of success count."""
    groups = defaultdict(list)
    for trajectory, value in rows:
        groups[trajectory.example_id].append(value)
    return math.fsum(math.fsum(v) / len(v) for v in groups.values()) / len(groups) if groups else 0.0


def decide(
    old_scores,
    new_scores,
    success_rows,
    failure_rows,
    sum_rows,
    *,
    pool_sha256,
    mode: str,
    weight: float,
    max_accuracy_drop: float,
    tolerance: float,
) -> Decision:
    if not old_scores or len(old_scores) != len(new_scores):
        raise ValueError("Aligned, nonempty before/after scores required")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in [*old_scores, *new_scores]):
        raise ValueError("Accuracy scores must be finite values in [0, 1]")
    if mode not in {"accuracy", "success_likelihood"}:
        raise ValueError("Unknown objective mode")
    if any(not math.isfinite(v) for _, v in [*success_rows, *failure_rows, *sum_rows]):
        raise ValueError("Non-finite likelihood change")
    old = math.fsum(old_scores) / len(old_scores)
    new = math.fsum(new_scores) / len(new_scores)
    delta, gain = new - old, grouped_mean(success_rows)
    score = delta + (weight * gain if mode == "success_likelihood" else 0)
    if delta < -max_accuracy_drop - tolerance:
        accepted, reason = False, "minibatch accuracy drop exceeds the allowed drop"
    elif score <= tolerance:
        accepted, reason = False, "objective did not improve beyond numerical tolerance"
    else:
        accepted, reason = True, "local objective improved"
    return Decision(
        old,
        new,
        delta,
        gain,
        grouped_mean(failure_rows) if failure_rows else None,
        grouped_mean(sum_rows),
        grouped_mean([(t, float(v > tolerance)) for t, v in success_rows]) if success_rows else None,
        len({t.example_id for t, _ in success_rows}),
        len(success_rows),
        pool_sha256,
        score,
        accepted,
        reason,
    )
