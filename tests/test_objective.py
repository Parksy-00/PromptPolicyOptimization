import math

import pytest

from prompt_policy.objective import decide, grouped_mean
from prompt_policy.types import Trajectory


def trajectory(id="a", reward=1.0):
    return Trajectory(id, "q", "train", 0, "parent", "### 1", (1,), (2, 3), reward, "eos")


def decision(old=(0.5,), new=(0.5,), success=None, **kwargs):
    args = dict(
        pool_sha256="pool", mode="success_likelihood", weight=1.0, max_accuracy_drop=0.0, tolerance=1e-8
    )
    args.update(kwargs)
    rows = [(trajectory(), 0.1)] if success is None else success
    return decide(old, new, rows, [], [(t, gain * len(t.response_ids)) for t, gain in rows], **args)


def test_accuracy_tie_can_accept_likelihood_gain():
    assert decision().accepted
    assert not decision(mode="accuracy", weight=0).accepted


def test_likelihood_cannot_override_accuracy_guard():
    value = decision(old=(1,), new=(0,), success=[(trajectory(), 100)])
    assert not value.accepted


def test_empty_success_pool_falls_back_to_accuracy():
    assert decision(old=(0,), new=(0.5,), success=[]).accepted
    assert not decision(success=[]).accepted
    assert decision(success=[]).success_questions == 0


def test_successes_are_balanced_by_question_not_trajectory_count():
    rows = [(trajectory("easy"), 0.1)] * 4 + [(trajectory("hard"), -0.3)]
    assert grouped_mean(rows) == pytest.approx(-0.1)
    assert not decision(success=rows).accepted


def test_numerical_noise_does_not_accept_identical_scores():
    assert not decision(success=[(trajectory(), 1e-10)]).accepted


@pytest.mark.parametrize("old,new", [([], []), ([math.nan], [1]), ([0], [2]), ([0, 1], [1])])
def test_invalid_reward_arrays_are_rejected(old, new):
    with pytest.raises(ValueError):
        decision(old=old, new=new)
