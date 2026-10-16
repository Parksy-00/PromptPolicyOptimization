import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import FakeBackend

pytest.importorskip("textgrad")

from prompt_policy.benchmarks import load_benchmark, reward, task_messages
from prompt_policy.objective import fixed_success_gain, textgrad_eligibility
from prompt_policy.scoring import TrajectoryScorer
from prompt_policy.textgrad_optimizer import search, verify_textgrad
from prompt_policy.types import Generation, Trajectory


class TextGradBackend(FakeBackend):
    def __init__(self, recorder, *, decreasing=False):
        super().__init__(recorder)
        self.updates = 0
        self.decreasing = decreasing

    def prompt_ids(self, prompt, question):
        number = 0 if prompt == "parent" else int(prompt.removeprefix("child"))
        return (number + 1, 10 + len(question))

    def generate(self, prompt, question, *, reflection=False):
        if not reflection:
            generated = super().generate(prompt, question)
            return replace(generated, prompt_ids=self.prompt_ids(prompt, question))
        if "improved variable" in prompt:
            self.updates += 1
            text = f"<IMPROVED_VARIABLE>child{self.updates}</IMPROVED_VARIABLE>"
        else:
            text = "Make the system instruction more accurate while preserving successful reasoning."
        self.recorder.charge("reflection", prompt_tokens=2, response_tokens=3, elapsed=0)
        return Generation(text, (1,), (3, 4, 5), "eos")

    def logprobs(self, prefix, response):
        self.score_requests.append((prefix, response))
        probability = 0.6 + (-1 if self.decreasing else 1) * 0.03 * (prefix[0] - 1)
        return tuple(math.log(probability) for _ in response)


def setup(config):
    config["initial_prompt"] = "parent"
    config["objective"].update(
        mode="success_likelihood_ascent", weight=0, likelihood_feedback=True, score_failures=False
    )
    config["textgrad"] = {"proposals_per_batch": 2}
    config["search"].update(max_iterations=2, train_rollouts=1, validation_rollouts=1)
    root = Path(__file__).parents[1]
    benchmark = load_benchmark(config["benchmark"], root / "experiments/qwen3_success_likelihood")
    return replace(
        benchmark,
        train=tuple(
            replace(e, question=f"{i + 1}+{i + 1}", answer=str(2 * (i + 1)))
            for i, e in enumerate(benchmark.train)
        ),
    )


def test_fixed_pool_uses_failed_rollouts_in_denominator(recorder):
    backend = TextGradBackend(recorder)
    good = Trajectory("q", "1+1", "train", 0, "parent", "### 2", (1, 13), (2, 9), 1, "eos")
    failed = replace(good, repeat_index=1, reward=0)
    other = replace(failed, example_id="other", question="2+2")
    measured = fixed_success_gain(
        "parent", "child1", [good, failed, other], TrajectoryScorer(backend, recorder)
    )
    expected = 2 * math.log(0.63 / 0.6) / 4
    assert measured["sequence_gain"] == pytest.approx(expected)
    assert measured["successes"] == 1
    assert len(backend.score_requests) == 2
    assert all(response == good.response_ids for _, response in backend.score_requests)
    with pytest.raises(ValueError, match="parent train"):
        fixed_success_gain("wrong-parent", "child1", [good], TrajectoryScorer(backend, recorder))
    with pytest.raises(ValueError, match="parent train"):
        fixed_success_gain(
            "parent", "child1", [replace(good, split="validation")], TrajectoryScorer(backend, recorder)
        )


def test_gain_gate_rejects_negative_gain_despite_improved_accuracy(config):
    settings = config["objective"] | {"mode": "success_likelihood_ascent"}
    assert not textgrad_eligibility(
        "parent", "child", 0.5, 1, {"successes": 2, "sequence_gain": -0.1}, settings
    )[0]
    assert not textgrad_eligibility(
        "parent", "child", 0.5, 1, {"successes": 0, "sequence_gain": 0}, settings
    )[0]
    assert not textgrad_eligibility(
        "parent", "child", 1, 0.5, {"successes": 2, "sequence_gain": 1}, settings
    )[0]
    assert textgrad_eligibility(
        "parent", "child", 0.5, 0.5, {"successes": 2, "sequence_gain": 0.1}, settings
    )[0]


def test_true_upstream_backward_refines_and_refreshes_only_after_admission(config, recorder):
    benchmark = setup(config)
    backend = TextGradBackend(recorder)
    result = search(benchmark, backend, recorder, config)
    assert result.best_candidate["system_prompt"] == "child4"
    pools = [json.loads(x) for x in (recorder.run_dir / "textgrad_pools.jsonl").read_text().splitlines()]
    assert len(pools) == 2
    assert {t["prompt"] for t in pools[0]["references"]} == {"parent"}
    assert {t["prompt"] for t in pools[1]["references"]} == {"child2"}
    proposals = [
        json.loads(x) for x in (recorder.run_dir / "textgrad_proposals.jsonl").read_text().splitlines()
    ]
    assert proposals[0]["pool_sha256"] == proposals[1]["pool_sha256"]
    assert proposals[2]["pool_sha256"] == proposals[3]["pool_sha256"]
    calls = [json.loads(x) for x in (recorder.run_dir / "textgrad_calls.jsonl").read_text().splitlines()]
    assert any("negative_reward_weighted_sequence_gain" in c["request"] for c in calls)
    assert all(t["split"] == "train" for pool in pools for t in pool["references"])
    assert verify_textgrad()["revision"] == "75e912e210864b61999781778cdf756d4468120f"


def test_rejected_candidates_never_become_final_or_next_reference(config, recorder):
    benchmark = setup(config)
    backend = TextGradBackend(recorder, decreasing=True)
    result = search(benchmark, backend, recorder, config)
    assert result.best_candidate["system_prompt"] == "parent"
    assert not any(c["accepted"] for c in result.candidates[1:])
    pools = [json.loads(x) for x in (recorder.run_dir / "textgrad_pools.jsonl").read_text().splitlines()]
    assert all(t["prompt"] == "parent" for pool in pools for t in pool["references"])


def test_accuracy_control_does_not_receive_likelihood_feedback(config, recorder):
    benchmark = setup(config)
    config["objective"].update(mode="accuracy", likelihood_feedback=False)
    result = search(benchmark, TextGradBackend(recorder, decreasing=True), recorder, config)
    assert result.best_candidate["system_prompt"] != "parent"
    calls = (recorder.run_dir / "textgrad_calls.jsonl").read_text()
    assert "negative_reward_weighted_sequence_gain" not in calls


def test_dspy_format_preserves_required_fields_and_plain_integer():
    valid = "[[ ## reasoning ## ]]\nReason\n[[ ## answer ## ]]\n42\n[[ ## completed ## ]]"
    assert reward(valid, "42", "dapo_dspy_v1") == 1
    assert reward(valid.replace("\n42\n", "\n\\boxed{42}\n"), "42", "dapo_dspy_v1") == 0
    assert reward("[[ ## answer ## ]]\n42", "42", "dapo_dspy_v1") == 0
    rendered = task_messages("First\nSecond", "1+1", "dapo_dspy_v1")
    assert rendered[0]["content"].endswith("First\n        Second")
    assert task_messages("critic", "feedback")[0]["content"] == "critic"


def test_no_success_batch_is_skipped_and_not_claimed_as_gain(config, recorder):
    benchmark = setup(config)

    class FailedBackend(TextGradBackend):
        def generate(self, prompt, question, *, reflection=False):
            generated = super().generate(prompt, question, reflection=reflection)
            return generated if reflection else replace(generated, text="### 999", response_ids=(999, 9))

    result = search(benchmark, FailedBackend(recorder), recorder, config)
    assert result.best_idx == 0
    assert len(result.candidates) == 1
    decisions = [json.loads(s) for s in (recorder.run_dir / "decisions.jsonl").read_text().splitlines()]
    assert len(decisions) == 2 and all(not d["accepted"] for d in decisions)
    assert not (recorder.run_dir / "textgrad_calls.jsonl").exists()


def test_cli_textgrad_records_upstream_and_freezes_selection_before_test(tmp_path, monkeypatch):
    from test_cli_run import prepare_config

    from prompt_policy import backends
    from prompt_policy.cli import run

    config_path = prepare_config(tmp_path)
    text = config_path.read_text().replace(
        'mode = "success_likelihood"', 'mode = "success_likelihood_ascent"'
    )
    text = text.replace("weight = 1.0", "weight = 0.0").replace(
        "score_failures = true", "score_failures = false"
    )
    text = text.replace("minibatch_size = 2", "minibatch_size = 4")
    config_path.write_text(text + "\n[textgrad]\nproposals_per_batch = 2\n")
    monkeypatch.setattr(backends, "LocalHFBackend", lambda config, recorder: TextGradBackend(recorder))
    output = run(config_path, output=tmp_path / "textgrad-run")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["optimizer"] == "textgrad"
    assert manifest["textgrad"]["revision"] == "75e912e210864b61999781778cdf756d4468120f"
    selected = json.loads((output / "selection.json").read_text())
    candidates = json.loads((output / "candidates.json").read_text())["candidates"]
    assert candidates[selected["candidate_idx"]]["accepted"]
    assert selected["prompt"] == candidates[selected["candidate_idx"]]["prompt"]
    assert all(json.loads(s)["split"] == "test" for s in (output / "test.jsonl").read_text().splitlines())
