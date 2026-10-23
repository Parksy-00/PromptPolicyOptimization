from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass

from prompt_policy.artifacts import digest
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


def fixed_success_gain(parent, candidate, references, scorer):
    """Reward-weighted sequence gain; failed rollouts stay in each question denominator."""
    if not references or any(t.prompt != parent or t.split != "train" for t in references):
        raise ValueError("Nonempty fixed parent train captures are required")
    grouped, successes = defaultdict(list), []
    for trajectory in references:
        if trajectory.reward not in {0, 1}:
            raise ValueError("Binary rewards required")
        if trajectory.reward == 1:
            old, new = scorer.score(parent, trajectory), scorer.score(candidate, trajectory)
            pair = (new.total - old.total, new.mean - old.mean)
            successes.append((trajectory, *pair))
        else:
            pair = (0.0, 0.0)
        grouped[trajectory.example_id].append(pair)
    if any(not math.isfinite(v) for values in grouped.values() for pair in values for v in pair):
        raise ValueError("Non-finite fixed-trajectory gain")
    return {
        "sequence_gain": math.fsum(
            math.fsum(v[0] for v in values) / len(values) for values in grouped.values()
        )
        / len(grouped),
        "token_mean_gain": math.fsum(
            math.fsum(v[1] for v in values) / len(values) for values in grouped.values()
        )
        / len(grouped),
        "successes": len(successes),
        "questions": len(grouped),
        "success_increases": sum(v > 0 for _, v, _ in successes),
        "pool_sha256": digest([t.to_dict() for t in references if t.reward == 1]),
    }


def textgrad_eligibility(parent, candidate, parent_accuracy, candidate_accuracy, gain, settings):
    if candidate == parent or not candidate.strip():
        return False, "identical or empty prompt"
    if candidate_accuracy < parent_accuracy - settings["max_accuracy_drop"] - settings["tolerance"]:
        return False, "validation accuracy guard"
    if settings["mode"] == "success_likelihood_ascent":
        if gain["successes"] == 0:
            return False, "no parent successes; likelihood ascent is unverified"
        if gain["sequence_gain"] <= settings["tolerance"]:
            return False, "fixed-parent success likelihood did not increase"
    return True, "eligible"


def fixed_directional_gain(parent, candidate, references, scorer, *, settings=None):
    """Fixed parent masks; optionally center mean-LP changes within each question."""
    if not references or any(t.prompt != parent or t.split != "train" for t in references):
        raise ValueError("Nonempty fixed parent train trajectories required")
    settings = settings or {}
    scope = settings.get("likelihood_scope", "full")
    correction = settings.get("drift_correction", "none")
    unit = settings.get("score_unit", "sequence")
    rows, grouped = [], defaultdict(list)
    for t in references:
        if t.reward not in {0, 1}:
            raise ValueError("Binary rewards required")
        old, new = scorer.score(parent, t), scorer.score(candidate, t)
        if scope == "response":
            positions = scorer.backend.response_token_indices(t)
        elif scope == "branch":
            if old.top1_logprobs is None:
                raise ValueError("Branch scoring requires full-vocabulary parent top-1 probabilities")
            cutoff = math.log(settings["branch_probability_threshold"])
            positions = tuple(i for i, v in enumerate(old.top1_logprobs) if v < cutoff)
        else:
            positions = tuple(range(len(t.response_ids)))
        old_sum = math.fsum(old.token_logprobs[i] for i in positions)
        new_sum = math.fsum(new.token_logprobs[i] for i in positions)
        delta = new_sum - old_sum
        mean_delta = delta / len(positions) if positions else None
        raw_gain = delta if unit == "sequence" else mean_delta
        row = {
            "example_id": t.example_id,
            "repeat_index": t.repeat_index,
            "reward": t.reward,
            "parent_logprob": old_sum if positions else None,
            "candidate_logprob": new_sum if positions else None,
            "delta_logprob": delta if positions else None,
            "parent_mean_logprob": old_sum / len(positions) if positions else None,
            "candidate_mean_logprob": new_sum / len(positions) if positions else None,
            "delta_mean_logprob": mean_delta,
            "full_delta_mean_logprob": new.mean - old.mean,
            "selected_token_indices": positions,
            "selected_tokens": len(positions),
            "total_tokens": len(t.response_ids),
            "raw_gain": raw_gain,
            "question_drift": None,
            "corrected_delta_mean_logprob": None,
            "verifiable": bool(positions),
            "unverifiable_reason": None if positions else "no tokens in the fixed scoring mask",
        }
        rows.append(row)
        grouped[t.example_id].append(row)
    for group in grouped.values():
        measured = [r for r in group if r["verifiable"]]
        if correction == "question_mean":
            drift = math.fsum(r["delta_mean_logprob"] for r in measured) / len(measured) if measured else None
            informative = len(measured) >= 2 and {r["reward"] for r in measured} == {0, 1}
            for row in group:
                row["question_drift"] = drift
                if row["verifiable"]:
                    row["corrected_delta_mean_logprob"] = row["delta_mean_logprob"] - drift
                    row["raw_gain"] = row["corrected_delta_mean_logprob"]
                if not informative:
                    row["verifiable"] = False
                    row["unverifiable_reason"] = (
                        "question-mean correction needs observed successes and failures"
                    )
    signed = []
    for t, row in zip(references, rows, strict=True):
        value = row.pop("raw_gain")
        gain = (2 * t.reward - 1) * value if value is not None else None
        if gain is not None and not math.isfinite(gain):
            raise ValueError("Non-finite likelihood change")
        row["signed_gain"] = gain
        row["direction_correct"] = row["verifiable"] and gain is not None and gain > 0
        if row["verifiable"]:
            signed.append((t, gain))
    return {
        "score": grouped_mean(signed),
        "verified": candidate != parent and all(r["direction_correct"] for r in rows),
        "likelihood_scope": scope,
        "drift_correction": correction,
        "score_unit": unit,
        "trajectories": rows,
    }
