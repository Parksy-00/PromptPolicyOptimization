import math
from dataclasses import replace

import pytest
from conftest import FakeBackend

from prompt_policy.scoring import TrajectoryScorer
from prompt_policy.types import Trajectory


def test_same_prompt_delta_is_zero_and_cache_avoids_repeat_compute(recorder):
    backend = FakeBackend(recorder)
    scorer = TrajectoryScorer(backend, recorder)
    trajectory = Trajectory(
        "a", "q", "train", 0, "parent", "### 1", backend.prompt_ids("parent", "q"), (5, 9), 1, "eos"
    )
    old = scorer.score("parent", trajectory)
    new = scorer.score("parent", trajectory)
    assert old.mean - new.mean == 0
    assert len(backend.score_requests) == 1
    assert old.total == pytest.approx(2 * math.log(0.6))
    child = scorer.score("child", trajectory)
    assert child.mean > old.mean
    assert backend.score_requests[-1][1] == trajectory.response_ids


def test_parent_capture_must_match_rendered_request(recorder):
    backend = FakeBackend(recorder)
    trajectory = Trajectory("a", "q", "train", 0, "parent", "x", (999,), (5,), 1, "eos")
    with pytest.raises(ValueError, match="Re-rendered"):
        TrajectoryScorer(backend, recorder).score("parent", trajectory)


def test_response_ids_must_align_with_logprobs(recorder):
    backend = FakeBackend(recorder)
    backend.logprobs = lambda prefix, response: (-0.1,)
    trajectory = Trajectory(
        "a", "q", "train", 0, "parent", "x", backend.prompt_ids("parent", "q"), (5, 9), 1, "eos"
    )
    with pytest.raises(ValueError, match="align"):
        TrajectoryScorer(backend, recorder).score("parent", trajectory)


def test_capture_roundtrip_preserves_token_ids():
    value = Trajectory("a", "q", "train", 0, "p", "x", (1,), (2,), 1, "eos")
    assert replace(Trajectory.from_dict(value.to_dict())) == value
