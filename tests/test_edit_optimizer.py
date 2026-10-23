import json
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import FakeBackend

from prompt_policy.benchmarks import Benchmark
from prompt_policy.config import validate_config
from prompt_policy.edit_optimizer import search
from prompt_policy.edit_prompts import (
    INSTRUCTION_OUTPUT_RULE,
    PROPOSER_INSTRUCTIONS,
    PROPOSER_INTRO,
    REFINER_INSTRUCTIONS,
    REFINER_INTRO,
)
from prompt_policy.objective import fixed_directional_gain
from prompt_policy.scoring import Likelihood
from prompt_policy.types import Example, Generation, Trajectory


def configure(config):
    config["initial_prompt"] = "parent"
    config["objective"] = {"mode": "directional_likelihood"}
    config["edit_refinement"] = {}
    config["search"]["validation_rollouts"] = 1
    config["search"].update(minibatch_size=1, train_rollouts=2, max_iterations=1, max_task_rollouts=10)
    return Benchmark(
        "test_only",
        "answer_integer_v1",
        (Example("tr", "1+1", "2", "train"),),
        (Example("va", "2+2", "4", "validation"),),
        (Example("te", "3+3", "6", "test"),),
        {},
    )


class EditBackend(FakeBackend):
    def __init__(
        self, recorder, *, outcomes="both", refinement="refined", better_proposal=False, tagged_tasks=False
    ):
        super().__init__(recorder)
        self.outcomes, self.refinement = outcomes, refinement
        self.tagged_tasks = tagged_tasks
        self.better_proposal, self.task_index = better_proposal, 0

    def prompt_ids(self, prompt, question):
        return ({"parent": 1, "proposed": 2, "refined": 3, "suppressed": 4}[prompt], len(question) + 10)

    def generate(self, prompt, question, *, reflection=False):
        if reflection:
            assert prompt == "", "editor must not supply a system instruction"
            text = "proposed" if question.endswith(PROPOSER_INSTRUCTIONS) else self.refinement
            self.recorder.charge("reflection", prompt_tokens=2, response_tokens=2, elapsed=0)
            return Generation(f"<instruction>{text}</instruction>", (7,), (8, 9), "eos")
        if question == "1+1":
            good = self.outcomes == "successes_only" or (self.outcomes == "both" and self.task_index % 2 == 0)
            self.task_index += 1
        else:
            good = getattr(self, "validation_correct", {}).get(prompt, True)
        answer = sum(map(int, question.split("+"))) if good else 999
        self.recorder.charge("task", prompt_tokens=2, response_tokens=2, elapsed=0)
        text = f"<answer>{answer}</answer>" if self.tagged_tasks else f"Answer: {answer}"
        return Generation(text, self.prompt_ids(prompt, question), (answer, 9), "eos")

    def logprobs(self, prefix, response):
        self.score_requests.append((prefix, response))
        good = response[0] != 999
        values = {
            1: -1.0,
            2: (-0.2 if good else -1.4) if self.better_proposal else (-0.5 if good else -0.45),
            3: -0.4 if good else -1.2,
            4: -1.2 if good else -2.0,
        }
        return (values[prefix[0]],) * len(response)


def rows(recorder, name):
    return [json.loads(line) for line in (recorder.run_dir / name).read_text().splitlines()]


@pytest.mark.parametrize("outcomes", ["failures_only", "successes_only", "both"])
def test_minimal_loop_uses_available_rewards_and_refines_with_measured_changes(config, recorder, outcomes):
    benchmark = configure(config)
    backend = EditBackend(recorder, outcomes=outcomes)
    result = search(benchmark, backend, recorder, config)
    assert result.best_candidate["system_prompt"] == "refined"
    assert recorder.cost["task_calls"] == (5 if outcomes == "successes_only" else 4)
    calls = rows(recorder, "optimizer_calls.jsonl")
    assert [r["phase"] for r in calls] == ["propose", "refine"]
    for call, instructions, intro, heading in zip(
        calls, [PROPOSER_INSTRUCTIONS, REFINER_INSTRUCTIONS],
        [PROPOSER_INTRO, REFINER_INTRO], ["Current Instruction:", "Original Instruction:"], strict=True,
    ):
        assert call["system_prompt"] is None
        assert call["messages"] == [{"role": "user", "content": call["request"]}]
        assert call["request"].startswith(intro + "\n\n" + heading)
        assert call["request"].endswith(instructions)
        assert call["request"].count(intro) == 1
        assert call["request"].count(INSTRUCTION_OUTPUT_RULE) == 1
        assert "reward =" not in call["request"]
        assert "Fixed output rule" not in call["request"]
        assert call["request"].index(heading) < call["request"].index("Input:")
        assert call["request"].index("Input:") < call["request"].rindex(instructions)
    propose, refine = calls[0]["request"], calls[1]["request"]
    assert propose.startswith(PROPOSER_INTRO + "\n\nCurrent Instruction:\nparent")
    assert "mean log-prob" not in propose
    assert "mean log-prob change =" in refine
    assert "sequence" not in propose + refine
    assert "Successful response" in propose or "Failed response" in propose
    assert "Original Instruction:\nparent\n\nNew Instruction:\nproposed" in refine
    assert "Likelihood scoring scope" not in refine
    assert PROPOSER_INTRO not in refine
    if outcomes == "successes_only":
        assert "The new instruction increased the likelihood of Successful response 1." in refine
        assert "This is desirable; preserve this increase." in refine
    else:
        index = 1 if outcomes == "failures_only" else 2
        assert f"The new instruction increased the likelihood of Failed response {index}." in refine
        assert "This is undesirable; revise the instruction to decrease that likelihood." in refine
    assert "2+2" not in calls[0]["request"] and "3+3" not in calls[1]["request"]
    assert {ids for _, ids in backend.score_requests} <= {(2, 9), (999, 9)}
    edits = rows(recorder, "edits.jsonl")
    assert len(edits) == 2 and len({e["pool_sha256"] for e in edits}) == 1
    assert rows(recorder, "decisions.jsonl")[0]["accepted"]
    assert result.val_aggregate_scores == [1.0] * len(result.candidates)


def test_keep_better_original_proposal_when_refinement_is_worse(config, recorder):
    benchmark = configure(config)
    result = search(benchmark, EditBackend(recorder, better_proposal=True), recorder, config)
    assert result.best_candidate["system_prompt"] == "proposed"
    assert rows(recorder, "decisions.jsonl")[0]["phase"] == "proposal"


def test_keep_parent_when_neither_edit_moves_all_responses_in_the_right_direction(config, recorder):
    benchmark = configure(config)
    result = search(benchmark, EditBackend(recorder, refinement="suppressed"), recorder, config)
    assert result.best_candidate["system_prompt"] == "parent"
    assert not rows(recorder, "decisions.jsonl")[0]["accepted"]


def test_new_step_generates_from_last_accepted_prompt_and_respects_task_budget(config, recorder):
    benchmark = configure(config)
    config["search"].update(max_iterations=3, max_task_rollouts=9)
    result = search(benchmark, EditBackend(recorder), recorder, config)
    captured = rows(recorder, "trajectories.jsonl")
    assert [r["prompt"] for r in captured] == ["parent", "parent", "refined", "refined"]
    assert recorder.cost["task_calls"] == result.total_metric_calls == 6
    assert result.iterations_completed == 2 and result.termination_reason == "max_task_rollouts"
    assert len(rows(recorder, "optimizer_calls.jsonl")) == 4


@pytest.mark.parametrize("phase", ["propose", "refine"])
def test_incomplete_optimizer_text_is_preserved_and_never_adopted(config, recorder, phase):
    benchmark = configure(config)

    class TruncatedBackend(EditBackend):
        def generate(self, prompt, question, *, reflection=False):
            generated = super().generate(prompt, question, reflection=reflection)
            target = PROPOSER_INSTRUCTIONS if phase == "propose" else REFINER_INSTRUCTIONS
            return (
                replace(generated, finish_reason="length") if reflection and question.endswith(target) else generated
            )

    with pytest.raises(ValueError, match="Incomplete"):
        search(benchmark, TruncatedBackend(recorder), recorder, config)
    assert rows(recorder, "optimizer_calls.jsonl")[-1]["error"]
    assert not (recorder.run_dir / "decisions.jsonl").exists()


@pytest.mark.parametrize(
    "rewards,deltas,verified",
    [
        ([0, 0], [-0.3, -0.1], True),
        ([1, 1], [0.3, 0.1], True),
        ([1, 0], [0.3, -0.1], True),
        ([1, 0], [0.3, 0.1], False),
        ([0, 0], [-0.3, 0.1], False),
        ([1, 1], [0.3, -0.1], False),
        ([0], [0.0], False),
        ([1], [0.0], False),
    ],
)
def test_each_trajectory_direction_is_verified(rewards, deltas, verified):
    refs = [
        Trajectory("q", "question", "train", i, "P", "text", (1,), (i + 2,), r, "eos")
        for i, r in enumerate(rewards)
    ]

    class Scorer:
        def score(self, prompt, t):
            return Likelihood((-2.0 + (deltas[t.repeat_index] if prompt != "P" else 0),))

    result = fixed_directional_gain("P", "C", refs, Scorer())
    assert result["verified"] == verified
    assert result["score"] == pytest.approx(
        sum((2 * r - 1) * d for r, d in zip(rewards, deltas, strict=True)) / len(refs)
    )


def test_question_weight_is_independent_of_response_count():
    refs = [Trajectory("a", "one", "train", i, "P", "text", (1,), (2,), 1, "eos") for i in range(3)]
    refs += [Trajectory("b", "two", "train", 0, "P", "text", (1,), (3,), 0, "eos")]

    class Scorer:
        def score(self, prompt, t):
            return Likelihood((-1 + ((0.2 if t.reward else -0.4) if prompt != "P" else 0),))

    result = fixed_directional_gain("P", "C", refs, Scorer())
    assert result["verified"] and result["score"] == pytest.approx(0.3)


def test_config_and_preview_have_no_search_components_beyond_the_minimal_loop(config):
    from prompt_policy.cli import run

    configure(config)
    validate_config(config)
    config["edit_refinement"]["run_ablations"] = True
    with pytest.raises(ValueError, match="Unknown edit_refinement"):
        validate_config(config)
    root = Path(__file__).parents[1]
    plan = run(root / "experiments/directional_validation_full_raw/experiment.toml", dry_run=True)
    assert plan["task_interface"]["task_format"] == "boxed_integer_v1"
    assert r"\boxed" in plan["task_interface"]["fixed_output_rule"]
    messages = plan["task_interface"]["initial_task_messages_example"]
    assert messages[0]["content"] == plan["config"]["initial_prompt"]
    assert messages[1]["content"].startswith(plan["task_interface"]["fixed_output_rule"] + "\n\nQuestion:\n")
    assert r"\boxed" in plan["task_interface"]["answer_filter"]
    assert plan["optimizer_input_contract"]["roles"] == ["user"]
    assert plan["optimizer_input_contract"]["system_message"] == "omitted"
    assert set(plan["optimizer_prompts"]) == {"proposer_introduction", "refiner_introduction", "proposer_instructions", "refiner_instructions"}
    assert plan["optimizer_output_contract"]["tag"] == "instruction"
    assert plan["final_selection"] == "last_likelihood_and_validation_verified_prompt"
    assert plan["edit_limits_per_outer_iteration"]["optimizer_calls"] == 2
    assert plan["edit_limits_per_outer_iteration"]["measured_candidates"] == 2


def test_cli_selects_validation_verified_prompt_before_test(tmp_path, monkeypatch):
    from prompt_policy import backends
    from prompt_policy.cli import run

    root = Path(__file__).parents[1]
    text = (root / "experiments/directional_edit_dapo_pilot/experiment.toml").read_text()
    model = tmp_path / "model"
    model.mkdir()
    import re

    text = re.sub(r"^name_or_path = .*$", f'name_or_path = "{model}"', text, flags=re.MULTILINE)
    text = re.sub(r"^initial_prompt = .*$", 'initial_prompt = "parent"', text, flags=re.MULTILINE)
    text = text.replace("max_iterations = 5", "max_iterations = 1")
    for split, question, answer in [("train", "1+1", "2"), ("validation", "2+2", "4"), ("test", "3+3", "6")]:
        path = tmp_path / f"{split}.jsonl"
        path.write_text(json.dumps({"id": split, "question": question, "answer": answer}) + "\n")
        text = re.sub(rf"^{split} = .*$", f'{split} = "{path}"', text, flags=re.MULTILINE)
    path = tmp_path / "experiment.toml"
    path.write_text(text)
    monkeypatch.setattr(
        backends, "LocalHFBackend", lambda config, recorder: EditBackend(recorder, tagged_tasks=True)
    )
    output = run(path, output=tmp_path / "run")
    selected = json.loads((output / "selection.json").read_text())
    result = json.loads((output / "result.json").read_text())
    assert (
        selected["criterion"] == "last_likelihood_and_validation_verified_prompt"
        and selected["prompt"] == "refined"
    )
    assert result["validation_accuracy"] == 1 and result["search_cost"]["task_calls"] == 4
    assert result["iterations_completed"] == 1 and result["termination_reason"] == "max_iterations"
    assert all(
        json.loads(line)["split"] == "train"
        for line in (output / "trajectories.jsonl").read_text().splitlines()
    )
    assert json.loads((output / "manifest.json").read_text())["status"] == "complete"


def test_exported_prompts_are_exact_current_templates():
    root = Path(__file__).parents[1] / "examples/contrastive_edit"
    assert (root / "proposer_instructions.txt").read_text() == PROPOSER_INSTRUCTIONS
    assert (root / "refiner_instructions.txt").read_text() == REFINER_INSTRUCTIONS
    assert (root / "proposer_request.txt").read_text().startswith(PROPOSER_INTRO + "\n\nCurrent Instruction:")
    assert (root / "refiner_request.txt").read_text().startswith(REFINER_INTRO + "\n\nOriginal Instruction:")
    assert "likelihood" not in PROPOSER_INSTRUCTIONS
    assert "hypothesis" not in PROPOSER_INSTRUCTIONS + REFINER_INSTRUCTIONS


@pytest.mark.parametrize("drift_correction", ["none", "question_mean"])
def test_refiner_gets_the_users_mean_likelihood_example_with_fixed_trajectory_contents(drift_correction):
    from prompt_policy.edit_prompts import request_payload

    refs = [
        Trajectory("q", "1+1", "train", i, "P", response, (1,), ids, r, "eos")
        for i, (response, ids, r) in enumerate(
            [
                ("Reasoning A. Answer: 2", (2, 2, 9), 1),
                ("Reasoning B. Answer: 3", (3, 9), 0),
            ]
        )
    ]

    class Scorer:
        def score(self, prompt, t):
            value = -0.42 if t.reward else -0.31
            if prompt != "P":
                value += 0.18 if t.reward else 0.24
            return Likelihood((value,) * len(t.response_ids))

    evaluation = fixed_directional_gain("P", "P2", refs, Scorer(), settings={"score_unit": "mean", "drift_correction": drift_correction})
    rollouts = [
        {
            "example_id": "q",
            "question": t.question,
            "gold_answer": "2",
            "response": t.text,
            "reward": t.reward,
        }
        for t in refs
    ]
    request = request_payload("P", rollouts, candidate="P2", evaluation=evaluation)
    assert "Successful response 1:\nReasoning A. Answer: 2" in request
    assert "Failed response 2:\nReasoning B. Answer: 3" in request
    assert "mean log-prob under original instruction = -0.42" in request
    assert "mean log-prob under original instruction = -0.31" in request
    assert "mean log-prob change = +0.18" in request
    assert "mean log-prob change = +0.24" in request
    assert "sequence" not in request
    assert "Likelihood scoring scope" not in request
    assert PROPOSER_INTRO not in request
    assert "Original Instruction:\nP\n\nNew Instruction:\nP2" in request
    assert request.index("New Instruction:") < request.index("Input:")
    assert request.index("mean log-prob change = +0.18") < request.index("Failed response 2:")
    assert request.index("mean log-prob change = +0.24") < request.index("Feedback:")
    assert request.count("mean log-prob change =") == 2
    assert ("Drift-relative mean-log-prob change =" in request) == (drift_correction == "question_mean")
    assert "Measured direction verification:" not in request
    assert "Mean log-prob changes compare" not in request
    assert "required direction" not in request
    if drift_correction == "none":
        assert "The new instruction increased the likelihood of Successful response 1." in request
        assert "The new instruction increased the likelihood of Failed response 2." in request
        assert "revise the instruction to decrease that likelihood." in request
    else:
        assert "The new instruction decreased the drift-relative mean log-prob of Successful response 1." in request
        assert "The new instruction increased the drift-relative mean log-prob of Failed response 2." in request
        assert "decreased the likelihood of Successful response 1" not in request
    assert "semantic component" in request and "suppress failed responses" in REFINER_INSTRUCTIONS
    assert not evaluation["verified"]  # Positive overall gain cannot excuse reinforcing the failure.


def test_refiner_single_class_feedback_never_invents_the_other_class(config, recorder):
    benchmark = configure(config)
    search(benchmark, EditBackend(recorder, outcomes="failures_only"), recorder, config)
    request = rows(recorder, "optimizer_calls.jsonl")[1]["request"]
    assert "Failed response 1:" in request and "Failed response 2:" in request
    assert "Successful response" not in request


@pytest.mark.parametrize("phase", ["propose", "refine"])
def test_optimizer_extracts_only_instruction_block_and_records_full_original(config, recorder, phase):
    from prompt_policy.edit_optimizer import optimizer_call

    configure(config)
    response = (
        "<think><instruction>not the answer</instruction></think>\n"
        "Example x: copied question and gold answer\n"
        "<instruction>refined</instruction>\nFinal instruction: other text"
    )

    class TaggedBackend(EditBackend):
        def generate(self, *args, **kwargs):
            return replace(super().generate(*args, **kwargs), text=response)

    assert optimizer_call(TaggedBackend(recorder), recorder, phase, "request", 0) == "refined"
    record = rows(recorder, "optimizer_calls.jsonl")[0]
    assert record["text"] == response and record["instruction"] == "refined" and record["error"] is None


@pytest.mark.parametrize(
    "response",
    [
        "Final instruction: proposed",
        "<instruction>proposed",
        "<instruction></instruction>",
        "<instruction>...</instruction>",
        "<instruction>…</instruction>",
        "<instruction>--- !!!</instruction>",
        "<instruction>proposed</instruction><instruction>refined</instruction>",
        "<instruction><instruction>proposed</instruction></instruction>",
        "<instruction>proposed</instruction><instruction",
        '<instruction role="new">proposed</instruction>',
        "<think><instruction>proposed</instruction>",
    ],
)
@pytest.mark.parametrize("phase", ["propose", "refine"])
def test_invalid_instruction_tags_are_logged_and_fail_before_admission(config, recorder, response, phase):
    from prompt_policy.edit_optimizer import optimizer_call

    configure(config)

    class MalformedBackend(EditBackend):
        def generate(self, *args, **kwargs):
            return replace(super().generate(*args, **kwargs), text=response)

    with pytest.raises(ValueError, match="malformed <instruction>"):
        optimizer_call(MalformedBackend(recorder), recorder, phase, "request", 0)
    record = rows(recorder, "optimizer_calls.jsonl")[0]
    assert record["instruction"] is None and record["text"] == response and record["error"]


def test_tagged_task_protocol_runs_minimal_loop_without_scoring_answer_substrings(config, recorder):
    benchmark = replace(configure(config), scorer="answer_tag_integer_v1")

    class TaggedTasks(EditBackend):
        def generate(self, *args, **kwargs):
            generated = super().generate(*args, **kwargs)
            if not kwargs.get("reflection", False):
                return replace(generated, text=f"<answer>{generated.response_ids[0]}</answer>")
            return generated

    backend = TaggedTasks(recorder)
    result = search(benchmark, backend, recorder, config)
    assert result.best_candidate["system_prompt"] == "refined"
    assert [t["reward"] for t in rows(recorder, "trajectories.jsonl")] == [1, 0]
    assert all(ids in {(2, 9), (999, 9)} for _, ids in backend.score_requests)
    assert "<answer>" in rows(recorder, "optimizer_calls.jsonl")[0]["request"]


def test_validation_rejects_likelihood_verified_edit(config, recorder):
    benchmark = configure(config)
    backend = EditBackend(recorder)
    backend.validation_correct = {"parent": True, "refined": False}
    result = search(benchmark, backend, recorder, config)
    assert result.best_candidate["system_prompt"] == "parent"
    decision = rows(recorder, "decisions.jsonl")[0]
    assert not decision["accepted"] and decision["validation_candidates"][0]["accuracy"] == 0
    assert decision["reason"] == "validation rejected all eligible edits"
    assert [r["phase"] for r in rows(recorder, "validation_scores.jsonl")] == ["base", "refinement"]
    assert all("2+2" not in r["request"] for r in rows(recorder, "optimizer_calls.jsonl"))


def test_validation_has_priority_over_training_gain(config, recorder):
    benchmark = configure(config)
    backend = EditBackend(recorder, better_proposal=True)
    backend.validation_correct = {"parent": False, "proposed": False, "refined": True}
    result = search(benchmark, backend, recorder, config)
    assert result.best_candidate["system_prompt"] == "refined"
    assert result.val_aggregate_scores == [0, 0, 1]


def test_no_validation_for_likelihood_rejected_edits(config, recorder):
    benchmark = configure(config)
    search(benchmark, EditBackend(recorder, refinement="suppressed"), recorder, config)
    assert [r["phase"] for r in rows(recorder, "validation_scores.jsonl")] == ["base"]


def test_edit_config_defaults_to_five_iterations(config):
    configure(config)
    del config["search"]["max_iterations"]
    del config["search"]["validation_rollouts"]
    validate_config(config)
    assert config["search"]["max_iterations"] == 5
    assert config["search"]["validation_rollouts"] == 1


def test_centered_change_is_invariant_to_shared_drift_and_not_absolute_ascent():
    refs = [
        Trajectory("q", "question", "train", i, "P", "text", (1,), (2,), r, "eos")
        for i, r in enumerate([1, 0])
    ]
    settings = {"score_unit": "mean", "drift_correction": "question_mean"}

    class Scorer:
        def __init__(self, drift):
            self.drift = drift

        def score(self, prompt, t):
            delta = [-0.02, -0.08][t.repeat_index] + self.drift
            return Likelihood((-1 + (delta if prompt != "P" else 0),))

    results = [fixed_directional_gain("P", "C", refs, Scorer(d), settings=settings) for d in [0, -0.2]]
    assert all(e["verified"] for e in results)
    assert results[0]["score"] == pytest.approx(results[1]["score"])
    assert results[0]["trajectories"][0]["delta_mean_logprob"] < 0
    assert results[0]["trajectories"][0]["corrected_delta_mean_logprob"] == pytest.approx(0.03)
    homogeneous = [replace(t, reward=1) for t in refs]
    assert not fixed_directional_gain("P", "C", homogeneous, Scorer(0), settings=settings)["verified"]


def test_branch_mask_is_chosen_under_parent_and_kept_under_candidate():
    import math

    ref = Trajectory("q", "question", "train", 0, "P", "text", (1,), (2, 3), 1, "eos")

    class Scorer:
        def score(self, prompt, t):
            # Candidate changes which position is confident. Only parent's second position counts.
            return (
                Likelihood((-0.01, -0.8), (math.log(0.99), math.log(0.6)))
                if prompt == "P"
                else Likelihood((-2.0, -0.5), (math.log(0.5), math.log(0.99)))
            )

    result = fixed_directional_gain(
        "P",
        "C",
        [ref],
        Scorer(),
        settings={"likelihood_scope": "branch", "branch_probability_threshold": 0.95, "score_unit": "mean"},
    )
    row = result["trajectories"][0]
    assert result["verified"] and row["selected_token_indices"] == (1,)
    assert row["delta_mean_logprob"] == pytest.approx(0.3)
    assert row["full_delta_mean_logprob"] < 0


def test_response_only_keeps_reasoning_in_scoring_context_and_empty_mask_is_unverifiable():
    ref = Trajectory("q", "question", "train", 0, "P", "text", (1,), (2, 3), 1, "eos")

    class Scorer:
        backend = type("Backend", (), {"response_token_indices": lambda self, t: (1,)})()

        def score(self, prompt, t):
            assert t.response_ids == (2, 3)
            return Likelihood((-0.2, -1.0) if prompt == "P" else (-1.0, -0.5))

    scorer = Scorer()
    result = fixed_directional_gain("P", "C", [ref], scorer, settings={"likelihood_scope": "response"})
    assert result["verified"] and result["trajectories"][0]["selected_tokens"] == 1
    scorer.backend = type("Backend", (), {"response_token_indices": lambda self, t: ()})()
    result = fixed_directional_gain("P", "C", [ref], scorer, settings={"likelihood_scope": "response"})
    assert not result["verified"] and result["trajectories"][0]["delta_mean_logprob"] is None


@pytest.mark.parametrize("prefix_open", [False, True])
def test_editor_filters_parse_failures_but_keeps_wrong_answers_and_scores_original_full_ids(
    config, recorder, prefix_open
):
    benchmark = replace(configure(config), scorer="boxed_integer_v1")
    config["search"].update(train_rollouts=3, max_task_rollouts=12)

    class MixedEvidenceBackend(EditBackend):
        def __init__(self, recorder):
            super().__init__(recorder)
            self.train_index = 0

        def generate(self, prompt, question, *, reflection=False):
            generated = super().generate(prompt, question, reflection=reflection)
            if reflection:
                return generated
            if question != "1+1":
                return replace(generated, text=f"\\boxed{{{sum(map(int, question.split('+')))}}}")
            index = self.train_index
            self.train_index += 1
            if index == 2:
                return replace(
                    generated,
                    text=r"<think>PRIVATE_TRUNCATED \boxed{2}",
                    response_ids=(333, 334, 9),
                    finish_reason="length",
                )
            hidden = "PRIVATE_REASONING" + ("" if prefix_open else " with opening")
            text = ("" if prefix_open else "<think>") + hidden + "</think>\n"
            visible = r"VISIBLE_SUCCESS \boxed{2}" if index == 0 else r"VISIBLE_FAILURE \boxed{999}"
            ids = (2, 70, 71, 9) if index == 0 else (999, 72, 9)
            return replace(
                generated, text=text + visible, response_ids=ids, thinking_open_in_prefix=prefix_open
            )

    backend = MixedEvidenceBackend(recorder)
    search(benchmark, backend, recorder, config)
    captured = rows(recorder, "trajectories.jsonl")
    assert len(captured) == 3 and [r["editor_eligible"] for r in captured] == [True, True, False]
    assert [r["reward"] for r in captured] == [1, 0, 0]
    assert all("PRIVATE" in r["text"] for r in captured)
    assert {ids for _, ids in backend.score_requests} == {(2, 70, 71, 9), (999, 72, 9)}
    for call in rows(recorder, "optimizer_calls.jsonl"):
        request = call["request"]
        assert "PRIVATE" not in request and "<think>" not in request
        assert r"Successful response 1:\nVISIBLE_SUCCESS \boxed{2}".replace(r"\n", "\n") in request
        assert r"Failed response 2:\nVISIBLE_FAILURE \boxed{999}".replace(r"\n", "\n") in request
        assert "trajectory 3" not in request
    edits = rows(recorder, "edits.jsonl")
    assert all(len(edit["trajectories"]) == 2 for edit in edits)
    evidence = rows(recorder, "editor_evidence.jsonl")[0]
    assert (evidence["captured"], evidence["included"], evidence["excluded"]) == (3, 2, 1)
    assert evidence["raw_pool_sha256"] != evidence["pool_sha256"]
    assert all(edit["pool_sha256"] == evidence["pool_sha256"] for edit in edits)


def test_all_unparseable_rollouts_skip_each_iteration_without_editor_scoring_or_extra_validation(config, recorder):
    benchmark = replace(configure(config), scorer="boxed_integer_v1")
    config["search"].update(max_iterations=2, max_task_rollouts=10)

    class UnparseableBackend(EditBackend):
        def generate(self, prompt, question, *, reflection=False):
            assert not reflection, "An empty evidence batch must not invoke either editor"
            generated = super().generate(prompt, question, reflection=False)
            if question == "1+1":
                return replace(generated, text=r"<think>unfinished \boxed{2}", finish_reason="length")
            return replace(generated, text=r"\boxed{4}")

    backend = UnparseableBackend(recorder)
    result = search(benchmark, backend, recorder, config)
    assert result.iterations_completed == 2 and result.termination_reason == "max_iterations"
    assert result.best_candidate["system_prompt"] == "parent"
    assert backend.score_requests == []
    assert recorder.cost["task_calls"] == 5  # one base validation plus two batches of two rollouts
    assert not (recorder.run_dir / "optimizer_calls.jsonl").exists()
    assert not (recorder.run_dir / "edits.jsonl").exists()
    assert len(rows(recorder, "validation_scores.jsonl")) == 1
    decisions = rows(recorder, "decisions.jsonl")
    assert len(decisions) == 2
    assert all(not d["accepted"] and d["reason"] == "no parseable training responses" for d in decisions)
    assert all(e["included"] == 0 and e["excluded"] == 2 for e in rows(recorder, "editor_evidence.jsonl"))


@pytest.fixture
def base_validation_source(config, recorder, tmp_path):
    from copy import deepcopy

    from prompt_policy.artifacts import digest

    benchmark = configure(config)
    benchmark = replace(
        benchmark,
        validation=(
            Example("va1", "2+2", "4", "validation"),
            Example("va2", "2+3", "5", "validation"),
            Example("va3", "2+4", "6", "validation"),
        ),
        identity={"fingerprint": "test-split", "scorer": "answer_integer_v1", "scorer_version": 1},
    )
    backend = EditBackend(recorder)
    backend.identity = {
        "files": {"model.safetensors": "test-model-sha"},
        "revision": "test-revision",
        "dtype": "float32",
        "enable_thinking": False,
    }
    source = tmp_path / "source-run"
    source.mkdir()
    original = deepcopy(config)
    (source / "config.json").write_text(json.dumps(original))
    (source / "manifest.json").write_text(json.dumps({
        "status": "complete", "config_sha256": digest(original),
        "benchmark": benchmark.identity, "model": backend.identity,
    }))
    records = []
    for example, text in zip(benchmark.validation, ["Answer: 4", "Answer: 999", "<think>unfinished"], strict=True):
        records.append({
            "iteration": -1, "phase": "base", "example_id": example.id, "question": example.question,
            "split": "validation", "repeat_index": 0, "prompt": "parent", "text": text,
            "prompt_ids": backend.prompt_ids("parent", example.question), "response_ids": [4, 9],
            "finish_reason": "eos", "thinking_open_in_prefix": False,
            "reward": benchmark.score(example, text),
        })
    (source / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (source / "validation_scores.jsonl").write_text(json.dumps({
        "iteration": -1, "phase": "base", "prompt": "parent", "accuracy": 1/3,
        "questions": 3, "rollouts": 3, "repeat_accuracies": [1/3],
    }) + "\n")
    config["search"]["base_validation_run"] = str(source)
    return benchmark, backend, source


@pytest.mark.parametrize("source_status", ["complete", "interrupted"])
def test_base_validation_reuse_preserves_all_rows_and_uses_no_generation_or_scoring(
    config, recorder, base_validation_source, source_status
):
    from prompt_policy.edit_optimizer import reuse_base_validation

    benchmark, backend, source = base_validation_source
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["status"] = source_status
    path.write_text(json.dumps(manifest))
    accuracy = reuse_base_validation(benchmark, backend, recorder, config)
    assert accuracy == 1/3 and not recorder.cost and backend.score_requests == []
    reused = rows(recorder, "validation.jsonl")
    assert len(reused) == 3 and [r["reward"] for r in reused] == [1, 0, 0]
    assert all(r["reused_from_run"] == str(source) for r in reused)
    provenance = json.loads((recorder.run_dir / "base_validation_reuse.json").read_text())
    assert provenance["verified_responses"] == 3 and provenance["generation_calls"] == 0
    assert set(provenance["source_files_sha256"]) == {
        "manifest.json", "config.json", "validation.jsonl", "validation_scores.jsonl"
    }


@pytest.mark.parametrize("mismatch", ["generation", "model", "prompt_ids", "reward", "duplicate", "summary"])
def test_base_validation_reuse_rejects_mismatched_conditions_or_evidence_before_copy(
    config, recorder, base_validation_source, mismatch
):
    from prompt_policy.edit_optimizer import reuse_base_validation

    benchmark, backend, source = base_validation_source
    if mismatch == "generation":
        config["generation"]["max_new_tokens"] += 1
    elif mismatch == "model":
        backend.identity["files"] = {"model.safetensors": "different-sha"}
    elif mismatch == "summary":
        p = source / "validation_scores.jsonl"
        summary = json.loads(p.read_text())
        summary["accuracy"] = 1.0
        p.write_text(json.dumps(summary) + "\n")
    else:
        p = source / "validation.jsonl"
        records = [json.loads(line) for line in p.open()]
        if mismatch == "prompt_ids":
            records[0]["prompt_ids"][0] = 999
        elif mismatch == "reward":
            records[0]["reward"] = 0
        else:
            records[1] = records[0]
        p.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError, match="Cannot reuse base validation"):
        reuse_base_validation(benchmark, backend, recorder, config)
    assert not (recorder.run_dir / "base_validation_reuse.json").exists()
    assert not (recorder.run_dir / "validation.jsonl").exists()
    assert not recorder.cost


def test_search_reuses_base_and_charges_only_fresh_training_and_candidate_validation(
    config, recorder, base_validation_source
):
    benchmark, backend, _ = base_validation_source
    result = search(benchmark, backend, recorder, config)
    assert result.best_candidate["system_prompt"] == "refined"
    assert recorder.cost["task_calls"] == 5  # train2 + candidate validation3; base3 costs zero
    summaries = rows(recorder, "validation_scores.jsonl")
    assert summaries[0]["phase"] == "base" and summaries[0]["reused_from_run"]
    assert len(summaries) == 2 and summaries[1]["phase"] == "refinement"


def probe_config(config, count=8, rounds=2):
    benchmark = configure(config)
    config["edit_refinement"] = {
        "mode": "per_example_likelihood_probe", "prompt_policy": "fixed", "score_unit": "mean",
    }
    config["search"].update(minibatch_size=4, train_rollouts=3, max_iterations=rounds, max_task_rollouts=rounds * 12)
    return replace(benchmark, train=tuple(Example(f"q{i}", f"{i+10}+1", str(i+11), "train") for i in range(count)))


class ProbeBackend(EditBackend):
    def __init__(self, recorder, *, invalid_question=None, malformed_phase=None):
        super().__init__(recorder)
        self.invalid_question, self.malformed_phase = invalid_question, malformed_phase
        self.generated = []

    def generate(self, prompt, question, *, reflection=False):
        if reflection:
            output = super().generate(prompt, question, reflection=True)
            if self.malformed_phase == ("propose" if question.endswith(PROPOSER_INSTRUCTIONS) else "refine"):
                self.malformed_phase = None
                return replace(output, text="missing boundary")
            return output
        assert question not in {"2+2", "3+3"}, "validation/test must not be generated"
        assert prompt == "parent", "probe must never adopt candidates"
        self.generated.append((prompt, question))
        good = self.task_index % 3 != 1
        self.task_index += 1
        answer = sum(map(int, question.split("+"))) if good else 999
        self.recorder.charge("task", prompt_tokens=2, response_tokens=2, elapsed=0)
        text = "unfinished" if question == self.invalid_question else f"Answer: {answer}"
        return Generation(text, self.prompt_ids(prompt, question), (answer, 9), "eos")


def test_probe_independent_question_evidence_fixed_base_and_no_evaluation(config, recorder):
    benchmark = probe_config(config)
    backend = ProbeBackend(recorder)
    result = search(benchmark, backend, recorder, config)
    assert result.best_idx == 0 and result.best_candidate["system_prompt"] == "parent"
    assert result.val_aggregate_scores == [None] * 17
    assert recorder.cost["task_calls"] == 24
    assert result.iterations_completed == 2
    summary = result.probe_summary
    assert summary["proposal_measured"] == summary["refinement_measured"] == 8
    assert summary["proposal_verified"] == 0 and summary["refinement_verified"] == 8
    assert summary["refinement_rescues"] == summary["groups_with_verified_edit"] == 8
    assert summary["accepted_updates"] == 0 and summary["validation"] == summary["test"] == "not_run"
    assert len({q for _, q in backend.generated}) == 8
    for call in rows(recorder, "optimizer_calls.jsonl"):
        question = next(e.question for e in benchmark.train if e.id == call["example_id"])
        assert question in call["request"]
        assert all(e.question not in call["request"] for e in benchmark.train if e.question != question)
        assert call["request"].count("Input:") == 1
    assert all(len(e["trajectories"]) == 3 for e in rows(recorder, "edits.jsonl"))
    assert not (recorder.run_dir / "validation.jsonl").exists()
    assert not (recorder.run_dir / "test.jsonl").exists()
    assert all(not r["accepted"] for r in rows(recorder, "decisions.jsonl"))


@pytest.mark.parametrize("malformed", ["propose", "refine", None])
def test_probe_skips_only_affected_group_or_editor(config, recorder, malformed):
    benchmark = probe_config(config, count=4, rounds=1)
    result = search(benchmark, ProbeBackend(recorder, invalid_question="10+1", malformed_phase=malformed), recorder, config)
    summary = result.probe_summary
    assert summary["groups_total"] == 4 and summary["groups_without_evidence"] == 1
    assert summary["captured_rollouts"] == 12 and summary["retained_rollouts"] == 9
    assert summary["proposal_attempts"] == 3
    assert summary["proposal_measured"] == (2 if malformed == "propose" else 3)
    assert summary["refinement_measured"] == (3 if malformed is None else 2)
    assert result.iterations_completed == 1


def test_probe_cli_never_generates_test_and_records_no_validation(tmp_path, monkeypatch):
    import re

    from prompt_policy import backends
    from prompt_policy.cli import run

    root = Path(__file__).parents[1]
    text = (root / "experiments/directional_edit_dapo_pilot/experiment.toml").read_text()
    model = tmp_path / "model"
    model.mkdir()
    replacements = {"name_or_path": f'"{model}"', "initial_prompt": '"parent"',
                    "max_iterations": "1", "minibatch_size": "4", "train_rollouts": "3", "max_task_rollouts": "12"}
    for key, value in replacements.items():
        text = re.sub(rf"^{key} = .*$", f"{key} = {value}", text, flags=re.MULTILINE)
    text = text.replace('[edit_refinement]', '[edit_refinement]\nmode = "per_example_likelihood_probe"\nprompt_policy = "fixed"')
    for split, items in {
        "train": [{"id": f"q{i}", "question": f"{i+10}+1", "answer": str(i+11)} for i in range(4)],
        "validation": [{"id": "va", "question": "2+2", "answer": "4"}],
        "test": [{"id": "te", "question": "3+3", "answer": "6"}],
    }.items():
        data = tmp_path / f"{split}.jsonl"
        data.write_text("".join(json.dumps(item) + "\n" for item in items))
        text = re.sub(rf"^{split} = .*$", f'{split} = "{data}"', text, flags=re.MULTILINE)
    # This recipe uses answer tags; the controlled backend below renders that protocol.
    class TaggedProbe(ProbeBackend):
        def generate(self, prompt, question, *, reflection=False):
            output = super().generate(prompt, question, reflection=reflection)
            return output if reflection else replace(output, text=f"<answer>{output.response_ids[0]}</answer>")
    monkeypatch.setattr(backends, "LocalHFBackend", lambda config, recorder: TaggedProbe(recorder))
    path = tmp_path / "probe.toml"
    path.write_text(text)
    output = run(path, output=tmp_path / "run")
    result = json.loads((output / "result.json").read_text())
    assert result["test"] == {} and result["validation_accuracy"] is None
    assert result["evaluation_status"] == {"validation": "not_run", "test": "not_run"}
    assert result["probe_summary"]["refinement_measured"] == 4
    assert result["total_cost"]["task_calls"] == 12
    assert json.loads((output / "selection.json").read_text())["criterion"] == "fixed_initial_prompt_probe"
    assert not (output / "test.jsonl").exists() and not (output / "validation.jsonl").exists()


def test_refiner_feedback_identifies_both_harmful_changes_without_generic_verdicts():
    from prompt_policy.edit_prompts import request_payload

    refs = [
        Trajectory("q", "question", "train", i, "P", "response", (1,), (i + 2,), reward, "eos")
        for i, reward in enumerate([1, 0])
    ]
    class Scorer:
        def score(self, prompt, trajectory):
            delta = [-0.2, 0.3][trajectory.repeat_index] if prompt == "C" else 0
            return Likelihood((-1 + delta,))
    evaluation = fixed_directional_gain("P", "C", refs, Scorer())
    evidence = [dict(example_id="q", question="question", gold_answer="2", response="response", reward=t.reward) for t in refs]
    request = request_payload("P", evidence, candidate="C", evaluation=evaluation)
    feedback = request.split("Feedback:\n", 1)[1].split("\n\n", 1)[0]
    assert feedback == (
        "The new instruction decreased the likelihood of Successful response 1. "
        "This is undesirable; revise the instruction to increase that likelihood.\n"
        "The new instruction increased the likelihood of Failed response 2. "
        "This is undesirable; revise the instruction to decrease that likelihood."
    )
    assert "required direction" not in request and "Measured direction verification" not in request


def test_refiner_feedback_separates_unchanged_and_unmeasurable_responses():
    from prompt_policy.edit_prompts import request_payload

    refs = [
        Trajectory("q", "question", "train", i, "P", "response", (1,), (i + 2,), reward, "eos")
        for i, reward in enumerate([1, 0])
    ]
    class Scorer:
        backend = type("Backend", (), {"response_token_indices": lambda self, t: (0,) if t.reward else ()})()
        def score(self, prompt, trajectory):
            return Likelihood((-1.0,))
    evaluation = fixed_directional_gain("P", "C", refs, Scorer(), settings={"likelihood_scope": "response"})
    evidence = [dict(example_id="q", question="question", gold_answer="2", response="response", reward=t.reward) for t in refs]
    request = request_payload("P", evidence, candidate="C", evaluation=evaluation)
    feedback = request.split("Feedback:\n", 1)[1].split("\n\n", 1)[0]
    assert "left the likelihood of Successful response 1 unchanged" in feedback
    assert "Cannot assess the likelihood change of Failed response 2: no tokens in the fixed scoring mask." in feedback
    assert "increased" not in feedback and "decreased" not in feedback


def test_per_example_validation_adopts_one_winner_and_rolls_out_next_parent(config, recorder):
    benchmark = probe_config(config, count=8, rounds=2)
    config["edit_refinement"] = {"mode": "validation_gated_per_example", "score_unit": "mean"}
    config["search"]["max_task_rollouts"] = 100
    validate_config(config)
    result = search(benchmark, EditBackend(recorder), recorder, config)
    assert result.iterations_completed == 2 and result.best_candidate["system_prompt"] == "refined"
    captured = rows(recorder, "trajectories.jsonl")
    assert len(captured) == 24
    assert {t["prompt"] for t in captured if t["iteration"] == 0} == {"parent"}
    assert {t["prompt"] for t in captured if t["iteration"] == 1} == {"refined"}
    assert len(rows(recorder, "optimizer_calls.jsonl")) == 16
    assert len(rows(recorder, "proposal_groups.jsonl")) == 8
    assert len(rows(recorder, "validation_scores.jsonl")) == 3  # base + two distinct candidates
    assert recorder.cost["task_calls"] == 27
    for call in rows(recorder, "optimizer_calls.jsonl"):
        question = next(e.question for e in benchmark.train if e.id == call["example_id"])
        assert question in call["request"]
        assert all(e.question not in call["request"] for e in benchmark.train if e.question != question)
    decisions = rows(recorder, "decisions.jsonl")
    assert decisions[0]["accepted"] and not decisions[1]["accepted"]
    assert all(c["parent_idx"] == 0 for c in result.candidates[1:])


@pytest.mark.parametrize("phase", ["propose", "refine"])
def test_per_example_validation_invalid_output_does_not_drop_other_groups(config, recorder, phase):
    benchmark = probe_config(config, count=4, rounds=1)
    config["edit_refinement"] = {"mode": "validation_gated_per_example", "score_unit": "mean"}
    config["search"]["max_task_rollouts"] = 100
    class OneInvalidBackend(EditBackend):
        def generate(self, prompt, question, *, reflection=False):
            output = super().generate(prompt, question, reflection=reflection)
            target = PROPOSER_INSTRUCTIONS if phase == "propose" else REFINER_INSTRUCTIONS
            if reflection and "Input:\n10+1" in question and question.endswith(target):
                return replace(output, text="<instruction>...</instruction>")
            return output
    result = search(benchmark, OneInvalidBackend(recorder), recorder, config)
    assert result.iterations_completed == 1 and result.best_candidate["system_prompt"] == "refined"
    groups = rows(recorder, "proposal_groups.jsonl")
    assert len(groups) == 4 and sum(bool(g["errors"]) for g in groups) == 1
    assert recorder.cost["task_calls"] == 15


def test_interrupted_base_requires_complete_phase_and_running_source_is_rejected(config, recorder, base_validation_source):
    from prompt_policy.edit_optimizer import reuse_base_validation
    benchmark, backend, source = base_validation_source
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["status"] = "running"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="neither complete nor interrupted"):
        reuse_base_validation(benchmark, backend, recorder, config)
    manifest["status"] = "interrupted"
    path.write_text(json.dumps(manifest))
    records = (source / "validation.jsonl").read_text().splitlines()
    (source / "validation.jsonl").write_text("\n".join(records[:-1]) + "\n")
    with pytest.raises(ValueError, match="base response count differs"):
        reuse_base_validation(benchmark, backend, recorder, config)
    assert not (recorder.run_dir / "base_validation_reuse.json").exists()


def test_rollout_logprobs_survive_training_and_validation_records(config, recorder):
    benchmark = configure(config)

    class CapturedBackend(EditBackend):
        def generate(self, prompt, question, *, reflection=False):
            output = super().generate(prompt, question, reflection=reflection)
            if reflection:
                return output
            return replace(
                output, rollout_token_logprobs=tuple(-.1 for _ in output.response_ids),
                rollout_logprobs_mode='processed_logprobs',
                rollout_sampling={'temperature': .6, 'top_k': 20, 'top_p': .95},
            )

    search(benchmark, CapturedBackend(recorder), recorder, config)
    for name in ['trajectories.jsonl', 'validation.jsonl']:
        for captured in rows(recorder, name):
            assert captured['rollout_logprobs_mode'] == 'processed_logprobs'
            assert len(captured['rollout_token_logprobs']) == len(captured['response_ids'])
            assert captured['rollout_sampling']['temperature'] == .6
    captured = rows(recorder, 'trajectories.jsonl')[0]
    fields = {k: captured[k] for k in Trajectory.__dataclass_fields__}
    restored = Trajectory.from_dict(fields)
    assert restored.rollout_token_logprobs == (-.1, -.1)
