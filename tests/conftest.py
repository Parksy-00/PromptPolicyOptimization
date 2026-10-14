from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import pytest

from prompt_policy.artifacts import Recorder
from prompt_policy.config import load_config
from prompt_policy.types import Generation


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[1]
    value, _ = load_config(root / "experiments/qwen3_success_likelihood/experiment.toml")
    value = copy.deepcopy(value)
    value["search"].update(
        max_iterations=1, minibatch_size=2, train_rollouts=2, validation_rollouts=2, max_task_rollouts=100
    )
    value["generation"]["enable_thinking"] = False
    return value


@pytest.fixture
def recorder(tmp_path):
    return Recorder(tmp_path)


class FakeBackend:
    """Deterministic controlled behaviour; never an experimental model backend."""

    identity = {"kind": "test_only"}

    def __init__(self, recorder, *, wrong_child=False):
        self.recorder = recorder
        self.wrong_child = wrong_child
        self.score_requests = []

    def prompt_ids(self, prompt, question):
        return (1 if prompt == "parent" else 2, 10 + len(question))

    def generate(self, prompt, question, *, reflection=False):
        started = time.perf_counter()
        if reflection:
            text, ids = "```\nchild\n```", (7, 8)
        else:
            answer = str(sum(map(int, question.split("+")))) if question.replace("+", "").isdigit() else "5"
            if self.wrong_child and prompt == "child":
                answer = "999"
            text, ids = "### " + answer, (int(answer), 9)
        self.recorder.charge(
            "reflection" if reflection else "task",
            prompt_tokens=2,
            response_tokens=len(ids),
            elapsed=time.perf_counter() - started,
        )
        return Generation(text, self.prompt_ids(prompt, question), ids, "eos")

    def logprobs(self, prefix, response):
        self.score_requests.append((prefix, response))
        probability = 0.6 if prefix[0] == 1 else 0.7
        return tuple(math.log(probability) for _ in response)
