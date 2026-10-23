"""Pinned upstream TextGrad with measured fixed-parent likelihood loss and a verified gate."""

from __future__ import annotations

import importlib.metadata
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import textgrad as tg
from textgrad.autograd.function import BackwardContext
from textgrad.autograd.llm_ops import LLMCall
from textgrad.autograd.string_based_ops import StringBasedFunction
from textgrad.engine import EngineLM

from prompt_policy.artifacts import digest, file_sha256
from prompt_policy.objective import fixed_success_gain, textgrad_eligibility
from prompt_policy.scoring import TrajectoryScorer
from prompt_policy.types import Trajectory

UPSTREAM_REVISION = "75e912e210864b61999781778cdf756d4468120f"


def verify_textgrad():
    distribution = importlib.metadata.distribution("textgrad")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != UPSTREAM_REVISION:
        raise ValueError("TextGrad must use the pinned official Git revision; uv sync --extra textgrad")
    package = Path(tg.__file__).resolve().parent
    inventory = json.loads((Path(__file__).parents[2] / "vendor/textgrad/UPSTREAM.json").read_text())
    for relative, expected in inventory["files_sha256"].items():
        if file_sha256(package / relative) != expected:
            raise ValueError(f"Pinned TextGrad source differs: {relative}")
    return {
        "repository": "https://github.com/zou-group/textgrad",
        "revision": UPSTREAM_REVISION,
        "files_sha256": inventory["files_sha256"],
        "patches": [],
    }


class TextGradEngine(EngineLM):
    def __init__(self, backend, recorder):
        self.backend, self.recorder = backend, recorder
        self.model_string = "pinned-local-model"
        self.phase = "backward"
        self.iteration = self.attempt = 0

    def generate(self, prompt, system_prompt=None, **kwargs):
        if kwargs:
            raise ValueError("TextGrad generation settings must come from experiment TOML")
        generated = self.backend.generate(
            system_prompt or "You provide optimization feedback.", prompt, reflection=True
        )
        self.recorder.append(
            "textgrad_calls.jsonl",
            {
                "iteration": self.iteration,
                "attempt": self.attempt,
                "phase": self.phase,
                "system_prompt": system_prompt,
                "request": prompt,
                **generated.__dict__,
            },
        )
        if generated.finish_reason == "length":
            raise ValueError("TextGrad optimizer hit reflection.max_new_tokens")
        text = ("<think>" if generated.thinking_open_in_prefix else "") + generated.text
        answer = text.rsplit("</think>", 1)[-1].strip()
        if not answer or "<think>" in answer:
            raise ValueError("Empty or unclosed TextGrad optimizer response")
        return answer

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)


class CapturedCall(LLMCall):
    """Attach an actual local rollout to the unchanged upstream LLMCall backward."""

    def forward(self, query, trajectory):
        response = tg.Variable(
            trajectory.text,
            predecessors=[self.system_prompt, query],
            role_description="actual task response including reasoning and final answer",
        )
        response.set_grad_fn(
            BackwardContext(
                backward_fn=self.backward,
                response=response,
                prompt=query.value,
                system_prompt=self.system_prompt.value,
            )
        )
        return response


def capture(benchmark, backend, recorder, prompt, examples, repeats, label):
    rows, scores = [], []
    for example in examples:
        group = []
        for repeat in range(repeats):
            generated = backend.generate(prompt, example.question)
            value = benchmark.score(
                example, generated.text, thinking_open_in_prefix=generated.thinking_open_in_prefix
            )
            trajectory = Trajectory(
                example.id,
                example.question,
                example.split,
                repeat,
                prompt,
                generated.text,
                generated.prompt_ids,
                generated.response_ids,
                value,
                generated.finish_reason,
                generated.thinking_open_in_prefix,
                rollout_token_logprobs=generated.rollout_token_logprobs,
                rollout_logprobs_mode=generated.rollout_logprobs_mode,
                rollout_sampling=generated.rollout_sampling,
            )
            recorder.append("trajectories.jsonl", {"capture": label, **trajectory.to_dict()})
            group.append(value)
            rows.append(trajectory)
        scores.append(statistics.mean(group))
    return rows, statistics.mean(scores)


def backward_losses(
    variable, current_rows, examples, references, parent, benchmark, backend, scorer, engine, settings
):
    losses = []
    example_map = {e.id: e for e in examples}
    call = CapturedCall(engine=engine, system_prompt=variable)

    def task_loss(prediction, answer):
        return json.dumps(
            {
                "task_accuracy": benchmark.score(
                    example_map[trajectory.example_id],
                    prediction.value,
                    thinking_open_in_prefix=trajectory.thinking_open_in_prefix,
                ),
                "gold_answer": answer.value,
            }
        )

    numeric_task = StringBasedFunction(
        task_loss,
        function_purpose=(
            "Maximize verifiable task accuracy. Diagnose the actual prediction against the gold answer. "
            "Preserve the benchmark output format and improve general instructions; do not memorize examples."
        ),
    )
    for trajectory in current_rows:
        query = (
            json.dumps(backend.messages(variable.value, trajectory.question))
            if hasattr(backend, "messages")
            else trajectory.question
        )
        response = call(
            tg.Variable(query, requires_grad=False, role_description="actual task request"), trajectory
        )
        answer = tg.Variable(
            example_map[trajectory.example_id].answer, requires_grad=False, role_description="gold answer"
        )
        # Forward computes immediately, while trajectory still denotes this response.
        losses.append(numeric_task(inputs={"prediction": response, "answer": answer}))
    if settings["likelihood_feedback"]:

        def verified_loss(instruction):
            measured = fixed_success_gain(parent, instruction.value, references, scorer)
            engine.recorder.append(
                "textgrad_losses.jsonl",
                {
                    "iteration": engine.iteration,
                    "attempt": engine.attempt,
                    "prompt": instruction.value,
                    "parent": parent,
                    "negative_sequence_gain": -measured["sequence_gain"],
                    **measured,
                },
            )
            return json.dumps(
                {
                    "negative_reward_weighted_sequence_gain": -measured["sequence_gain"],
                    "measured_gain": measured,
                    "fixed_successes": [
                        {"question": t.question, "response": t.text} for t in references if t.reward == 1
                    ],
                }
            )

        likelihood_loss = StringBasedFunction(
            verified_loss,
            function_purpose=(
                "Minimize negative_reward_weighted_sequence_gain, equivalently maximize measured_gain.sequence_gain. "
                "These exact successful responses were sampled under the fixed parent prompt. Improve the current "
                "instruction so the frozen model assigns higher full-vocabulary T=1 sequence likelihood to those "
                "same responses. Measured gain must be positive. Do not substitute newly generated responses, "
                "copy answers into the instruction, or claim the numerical gain improved without measurement."
            ),
        )
        losses.append(likelihood_loss(inputs={"instruction": variable}))
    engine.phase = "backward"
    tg.sum(losses).backward(engine=engine)


@dataclass
class TextGradResult:
    candidates: list[dict]
    best_idx: int
    total_metric_calls: int

    @property
    def best_candidate(self):
        return {"system_prompt": self.candidates[self.best_idx]["prompt"]}

    @property
    def val_aggregate_scores(self):
        return [c["validation_accuracy"] for c in self.candidates]

    def to_dict(self):
        return {
            "optimizer": "textgrad",
            "candidates": self.candidates,
            "best_idx": self.best_idx,
            "total_metric_calls": self.total_metric_calls,
        }


def search(benchmark, backend, recorder, config):
    settings, search_settings = config["objective"], config["search"]
    scorer = TrajectoryScorer(backend, recorder)
    engine = TextGradEngine(backend, recorder)
    variable = tg.Variable(
        config["initial_prompt"], role_description="task system instruction", requires_grad=True
    )
    optimizer = tg.TextualGradientDescent(
        parameters=[variable],
        engine=engine,
        constraints=[
            "Do not copy training questions or gold answers into the system instruction.",
            "Preserve the required output fields and final-answer format.",
        ],
    )
    _, accuracy = capture(
        benchmark,
        backend,
        recorder,
        variable.value,
        benchmark.validation,
        search_settings["validation_rollouts"],
        "base-validation",
    )
    candidates = [
        {"prompt": variable.value, "validation_accuracy": accuracy, "accepted": True, "parent_idx": None}
    ]
    current_idx = 0
    rng = random.Random(search_settings["search_seed"])
    proposals = config["textgrad"]["proposals_per_batch"]
    for iteration in range(search_settings["max_iterations"]):
        # Reserve the complete outer iteration, never evaluate only part of validation.
        reserve = proposals * (
            search_settings["minibatch_size"] * search_settings["train_rollouts"]
            + len(benchmark.validation) * search_settings["validation_rollouts"]
        )
        if recorder.cost["task_calls"] + reserve > search_settings["max_task_rollouts"]:
            break
        parent, parent_accuracy = variable.value, accuracy
        examples = rng.sample(list(benchmark.train), search_settings["minibatch_size"])
        references, _ = capture(
            benchmark,
            backend,
            recorder,
            parent,
            examples,
            search_settings["train_rollouts"],
            f"{iteration}:parent-train",
        )
        frozen_hash = digest([t.to_dict() for t in references])
        recorder.append(
            "textgrad_pools.jsonl",
            {
                "iteration": iteration,
                "parent_idx": current_idx,
                "pool_sha256": frozen_hash,
                "references": [t.to_dict() for t in references],
            },
        )
        if settings["mode"] == "success_likelihood_ascent" and not any(t.reward == 1 for t in references):
            recorder.append(
                "decisions.jsonl",
                {
                    "iteration": iteration,
                    "accepted": False,
                    "reason": "no parent successes; likelihood ascent is unverified",
                    "pool_sha256": frozen_hash,
                },
            )
            continue
        current_rows = references
        eligible = []
        for attempt in range(proposals):
            engine.iteration, engine.attempt = iteration, attempt
            optimizer.zero_grad()
            backward_losses(
                variable,
                current_rows,
                examples,
                references,
                parent,
                benchmark,
                backend,
                scorer,
                engine,
                settings,
            )
            recorder.append(
                "textgrad_gradients.jsonl",
                {
                    "iteration": iteration,
                    "attempt": attempt,
                    "prompt": variable.value,
                    "feedback": variable.get_gradient_text(),
                    "pool_sha256": frozen_hash,
                },
            )
            engine.phase = "update"
            optimizer.step()
            candidate = variable.value
            gain = (
                fixed_success_gain(parent, candidate, references, scorer)
                if settings["measure_likelihood"]
                else {
                    "sequence_gain": 0.0,
                    "token_mean_gain": 0.0,
                    "successes": 0,
                    "questions": len(examples),
                    "success_increases": 0,
                    "pool_sha256": digest([]),
                }
            )
            _, candidate_accuracy = capture(
                benchmark,
                backend,
                recorder,
                candidate,
                benchmark.validation,
                search_settings["validation_rollouts"],
                f"{iteration}:{attempt}:candidate-validation",
            )
            valid, reason = textgrad_eligibility(
                parent, candidate, parent_accuracy, candidate_accuracy, gain, settings
            )
            record = {
                "iteration": iteration,
                "attempt": attempt,
                "parent_idx": current_idx,
                "prompt": candidate,
                "validation_accuracy": candidate_accuracy,
                "eligible": valid,
                "accepted": False,
                "reason": reason,
                "gain": gain,
                "pool_sha256": frozen_hash,
            }
            candidates.append(record)
            idx = len(candidates) - 1
            recorder.append("textgrad_proposals.jsonl", {"candidate_idx": idx, **record})
            print(
                f"TextGrad iteration={iteration} proposal={attempt} G={gain['sequence_gain']:+.6f} validation={candidate_accuracy:.4f} eligible={valid}",
                flush=True,
            )
            if valid:
                eligible.append(idx)
            if digest([t.to_dict() for t in references]) != frozen_hash:
                raise ValueError("Fixed parent reference pool was mutated")
            if attempt + 1 < proposals:
                current_rows, _ = capture(
                    benchmark,
                    backend,
                    recorder,
                    candidate,
                    examples,
                    search_settings["train_rollouts"],
                    f"{iteration}:{attempt}:candidate-train",
                )
        if eligible:
            criterion = (
                (lambda i: (candidates[i]["gain"]["sequence_gain"], candidates[i]["validation_accuracy"]))
                if settings["mode"] == "success_likelihood_ascent"
                else (lambda i: candidates[i]["validation_accuracy"])
            )
            chosen = max(eligible, key=criterion)
            candidates[chosen]["accepted"] = True
            current_idx, accuracy = chosen, candidates[chosen]["validation_accuracy"]
        variable.set_value(candidates[current_idx]["prompt"])
        optimizer.zero_grad()
        recorder.append(
            "decisions.jsonl",
            {
                "iteration": iteration,
                "accepted": bool(eligible),
                "selected_candidate_idx": current_idx,
                "selected_prompt": variable.value,
                "validation_accuracy": accuracy,
                "pool_sha256": frozen_hash,
            },
        )
    # Only admitted prompts can be selected; likelihood-rejected proposals never enter final selection.
    admitted = [i for i, c in enumerate(candidates) if c["accepted"]]
    best = max(admitted, key=lambda i: (candidates[i]["validation_accuracy"], i))
    return TextGradResult(candidates, best, recorder.cost["task_calls"])
