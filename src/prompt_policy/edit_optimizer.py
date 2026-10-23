"""Minibatch rollouts -> instruction proposal -> likelihood -> one refinement -> update."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from prompt_policy.artifacts import digest, file_sha256
from prompt_policy.backends import generate_batch
from prompt_policy.benchmarks import extract_tag_content, task_messages
from prompt_policy.edit_prompts import request_payload
from prompt_policy.objective import fixed_directional_gain
from prompt_policy.scoring import TrajectoryScorer
from prompt_policy.types import Trajectory


def minibatches(examples, size, iterations, seed):
    rng, remaining = random.Random(seed), []
    for _ in range(iterations):
        if not remaining:
            remaining = list(examples)
            rng.shuffle(remaining)
        batch, remaining = remaining[:size], remaining[size:]
        yield batch


class InvalidInstruction(ValueError):
    """An editor completed, but did not return a valid instruction block."""


def optimizer_call(backend, recorder, phase, request, iteration, *, example_id=None):
    messages = task_messages("", request)
    generated = backend.generate("", request, reflection=True)
    text = ("<think>" if generated.thinking_open_in_prefix else "") + generated.text
    instruction = extract_tag_content(text, "instruction")
    if instruction is not None and not any(c.isalnum() for c in instruction):
        instruction = None
    invalid = generated.finish_reason == "length" or instruction is None
    recorder.append(
        "optimizer_calls.jsonl",
        {
            "iteration": iteration,
            "phase": phase,
            **({"example_id": example_id} if example_id is not None else {}),
            "system_prompt": None,
            "messages": messages,
            "request": request,
            **generated.__dict__,
            "instruction": instruction,
            "error": "incomplete, malformed or punctuation-only <instruction> output" if invalid else None,
        },
    )
    if invalid:
        raise InvalidInstruction("Incomplete or malformed <instruction> output; see optimizer_calls.jsonl")
    return instruction


@dataclass
class EditResult:
    candidates: list[dict]
    best_idx: int
    total_metric_calls: int
    iterations_completed: int = 0
    termination_reason: str = "max_iterations"
    probe_summary: dict | None = None

    @property
    def best_candidate(self):
        return {"system_prompt": self.candidates[self.best_idx]["prompt"]}

    @property
    def val_aggregate_scores(self):
        return [c["validation_accuracy"] for c in self.candidates]

    def to_dict(self):
        return {
            "optimizer": "edit_refinement",
            "candidates": self.candidates,
            "best_idx": self.best_idx,
            "total_metric_calls": self.total_metric_calls,
            "iterations_completed": self.iterations_completed,
            "termination_reason": self.termination_reason,
            **({"probe_summary": self.probe_summary} if self.probe_summary is not None else {}),
        }


def evaluate_validation(benchmark, backend, recorder, prompt, repeats, *, iteration, phase):
    """Fresh task generations on validation only; never expose these to the editor."""
    recorder.write("progress.json", {"iteration": iteration, "phase": "validation", "candidate_phase": phase})
    print(f"Validation {phase}: start ({len(benchmark.validation)} questions x {repeats})", flush=True)
    jobs = [(e, r) for e in benchmark.validation for r in range(repeats)]
    scores = [None] * len(jobs)
    completed = 0

    def record(index, output):
        nonlocal completed
        example, repeat = jobs[index]
        reward = benchmark.score(example, output.text, thinking_open_in_prefix=output.thinking_open_in_prefix)
        scores[index] = reward
        recorder.append(
            "validation.jsonl",
            {
                "iteration": iteration,
                "phase": phase,
                "example_id": example.id,
                "question": example.question,
                "split": "validation",
                "repeat_index": repeat,
                "prompt": prompt,
                "reward": reward,
                **output.__dict__,
            },
        )
        completed += 1
        if completed % 10 == 0 or completed == len(jobs):
            measured = [v for v in scores if v is not None]
            recorder.write(
                "progress.json",
                {
                    "iteration": iteration,
                    "phase": "validation",
                    "candidate_phase": phase,
                    "completed": completed,
                    "total": len(jobs),
                    "partial_accuracy": sum(measured) / len(measured),
                },
            )
            print(
                f"Validation {phase}: {completed}/{len(jobs)} accuracy={sum(measured) / len(measured):.4f}",
                flush=True,
            )

    # VLLM invokes callbacks in its single result-consumer thread, in completion order.
    generate_batch(backend, prompt, [e.question for e, _ in jobs], on_result=record)
    accuracy = sum(scores) / len(scores)
    recorder.append(
        "validation_scores.jsonl",
        {
            "iteration": iteration,
            "phase": phase,
            "prompt": prompt,
            "accuracy": accuracy,
            "questions": len(benchmark.validation),
            "rollouts": len(scores),
            "repeat_accuracies": [
                sum(scores[r::repeats]) / len(benchmark.validation) for r in range(repeats)
            ],
        },
    )
    print(f"Validation {phase}: accuracy={accuracy:.6f}", flush=True)
    return accuracy


def reuse_base_validation(benchmark, backend, recorder, config):
    """Reuse complete baseline evidence only after checking actual inputs, outputs and settings."""
    source = Path(config["search"]["base_validation_run"])
    files = {name: source / name for name in (
        "manifest.json", "config.json", "validation.jsonl", "validation_scores.jsonl"
    )}
    manifest = json.loads(files["manifest.json"].read_text())
    previous = json.loads(files["config.json"].read_text())

    def require(condition, detail):
        if not condition:
            raise ValueError("Cannot reuse base validation: " + detail)

    require(manifest["status"] in {"complete", "interrupted"}, "source run is neither complete nor interrupted")
    require(manifest["config_sha256"] == digest(previous), "source configuration digest differs")
    require(previous["initial_prompt"] == config["initial_prompt"], "initial prompt differs")
    require(previous["generation"] == config["generation"], "generation settings differ")
    require(previous["model"]["max_context_tokens"] == config["model"]["max_context_tokens"], "context differs")
    require(
        previous["search"]["validation_rollouts"] == config["search"]["validation_rollouts"],
        "validation repeat count differs",
    )
    for key in ("fingerprint", "scorer", "scorer_version", "task_format", "fixed_output_rule"):
        require(manifest["benchmark"].get(key) == benchmark.identity.get(key), f"benchmark {key} differs")
    for key in ("files", "revision", "dtype", "enable_thinking"):
        require(key in backend.identity and manifest["model"].get(key) == backend.identity[key], f"model {key} differs")
    require(manifest["model"].get("backend") == backend.identity.get("backend"), "generation backend differs")
    require(
        manifest["model"].get("server", {}).get("vllm_version")
        == backend.identity.get("server", {}).get("vllm_version"),
        "vLLM version differs",
    )
    repeats = config["search"]["validation_rollouts"]
    expected = {(e.id, repeat): e for e in benchmark.validation for repeat in range(repeats)}
    records = [json.loads(line) for line in files["validation.jsonl"].read_text().splitlines()]
    records = [r for r in records if r["phase"] == "base" and r["iteration"] == -1]
    require(len(records) == len(expected), "base response count differs")
    seen, scores = set(), {}
    for record in records:
        key = (record["example_id"], record["repeat_index"])
        require(key in expected and key not in seen, "unknown or repeated validation response")
        example = expected[key]
        require(record["split"] == "validation" and record["question"] == example.question, "question differs")
        require(record["prompt"] == config["initial_prompt"], "recorded prompt differs")
        require(
            tuple(record["prompt_ids"]) == backend.prompt_ids(config["initial_prompt"], example.question),
            "actual rendered prompt token IDs differ",
        )
        value = benchmark.score(
            example, record["text"], thinking_open_in_prefix=record.get("thinking_open_in_prefix", False)
        )
        require(value == record["reward"], "rescored reward differs")
        seen.add(key)
        scores[key] = value
    accuracy = sum(scores.values()) / len(expected)
    summaries = [json.loads(line) for line in files["validation_scores.jsonl"].read_text().splitlines()]
    summaries = [r for r in summaries if r["phase"] == "base" and r["iteration"] == -1]
    require(len(summaries) == 1, "missing or repeated base summary")
    summary = summaries[0]
    require(summary["prompt"] == config["initial_prompt"], "summary prompt differs")
    require(summary["questions"] == len(benchmark.validation) and summary["rollouts"] == len(expected), "summary count differs")
    require(summary["accuracy"] == accuracy, "summary accuracy differs")
    require(
        summary["repeat_accuracies"] == [
            sum(scores[(e.id, r)] for e in benchmark.validation) / len(benchmark.validation)
            for r in range(repeats)
        ],
        "summary repeat accuracies differ",
    )
    provenance = {
        "source_run": str(source.resolve()),
        "source_run_id": source.name,
        "source_run_status": manifest["status"],
        "reuse_scope": "complete base validation only; other source phases are not reused",
        "source_files_sha256": {name: file_sha256(path) for name, path in files.items()},
        "verified_responses": len(records),
        "accuracy": accuracy,
        "generation_calls": 0,
    }
    recorder.write("base_validation_reuse.json", provenance)
    for record in records:
        recorder.append("validation.jsonl", {**record, "reused_from_run": str(source.resolve())})
    recorder.append("validation_scores.jsonl", {**summary, "reused_from_run": str(source.resolve())})
    print(f"Validation base: reused {len(records)} verified responses from {source.name}; accuracy={accuracy:.6f}; no generation", flush=True)
    return accuracy


def capture_training_batch(benchmark, backend, recorder, parent, examples, repeats, iteration):
    captured, references, rollouts, evidence = [], [], [], []
    jobs = [(e, r) for e in examples for r in range(repeats)]
    generated_batch = generate_batch(backend, parent, [e.question for e, _ in jobs])
    for (example, repeat), generated in zip(jobs, generated_batch, strict=True):
        reward = benchmark.score(
            example, generated.text, thinking_open_in_prefix=generated.thinking_open_in_prefix
        )
        t = Trajectory(
            example.id,
            example.question,
            "train",
            repeat,
            parent,
            generated.text,
            generated.prompt_ids,
            generated.response_ids,
            reward,
            generated.finish_reason,
            generated.thinking_open_in_prefix,
            rollout_token_logprobs=generated.rollout_token_logprobs,
            rollout_logprobs_mode=generated.rollout_logprobs_mode,
            rollout_sampling=generated.rollout_sampling,
        )
        visible = benchmark.editor_response(
            generated.text, thinking_open_in_prefix=generated.thinking_open_in_prefix
        )
        captured.append(t)
        evidence.append(
            {
                "example_id": example.id,
                "repeat_index": repeat,
                "included": visible is not None,
                "reason": "answer_parseable" if visible is not None else "answer_unparseable",
            }
        )
        recorder.append(
            "trajectories.jsonl",
            {"iteration": iteration, **t.to_dict(), "editor_eligible": visible is not None},
        )
        if visible is None:
            continue
        references.append(t)
        rollouts.append(
            {
                "example_id": example.id,
                "repeat_index": repeat,
                "question": example.question,
                "gold_answer": example.answer,
                "response": visible,
                "reward": reward,
            }
        )
    return captured, references, rollouts, evidence


def search(benchmark, backend, recorder, config):
    if config["edit_refinement"].get("mode") == "per_example_likelihood_probe":
        return probe_search(benchmark, backend, recorder, config)
    limits, parent = config["search"], config["initial_prompt"]
    settings = config["edit_refinement"]
    per_example = settings.get("mode") == "validation_gated_per_example"
    scorer = TrajectoryScorer(backend, recorder, collect_top1=settings.get("likelihood_scope") == "branch")
    validation_repeats = limits.get("validation_rollouts", 1)
    validation_cost = len(benchmark.validation) * validation_repeats
    if limits.get("base_validation_run"):
        parent_accuracy = reuse_base_validation(benchmark, backend, recorder, config)
    else:
        if recorder.cost["task_calls"] + validation_cost > limits["max_task_rollouts"]:
            raise ValueError("Budget cannot cover complete base validation")
        parent_accuracy = evaluate_validation(
            benchmark, backend, recorder, parent, validation_repeats, iteration=-1, phase="base"
        )
    candidates = [
        {
            "prompt": parent,
            "parent_idx": None,
            "kind": "initial",
            "validation_accuracy": parent_accuracy,
            "accepted": True,
        }
    ]
    current_idx, accepted_updates, completed = 0, 0, 0
    termination = "max_iterations"
    for iteration, examples in enumerate(
        minibatches(
            benchmark.train, limits["minibatch_size"], limits.get("max_iterations", 5), limits["search_seed"]
        )
    ):
        # Reserve enough for both eligible editors; do not stop midway through validation.
        reserved = len(examples) * limits["train_rollouts"] + 2 * (len(examples) if per_example else 1) * validation_cost
        if recorder.cost["task_calls"] + reserved > limits["max_task_rollouts"]:
            termination = "max_task_rollouts"
            break
        print(f"Step {iteration + 1}: rollout start ({len(examples)} questions)", flush=True)
        recorder.write("progress.json", {"iteration": iteration, "phase": "rollout"})
        captured, references, rollouts, evidence = capture_training_batch(
            benchmark, backend, recorder, parent, examples, limits["train_rollouts"], iteration
        )
        print(
            f"Step {iteration + 1}: rewards={[t.reward for t in captured]} "
            f"tokens={[len(t.response_ids) for t in captured]}",
            flush=True,
        )
        pool_hash = digest([t.to_dict() for t in references])
        recorder.append(
            "editor_evidence.jsonl",
            {
                "iteration": iteration,
                "captured": len(captured),
                "included": len(references),
                "excluded": len(captured) - len(references),
                "response_view": "post_thinking_only",
                "raw_pool_sha256": digest([t.to_dict() for t in captured]),
                "pool_sha256": pool_hash,
                "trajectories": evidence,
            },
        )
        print(
            f"Step {iteration + 1}: editor evidence={len(references)}/{len(captured)} parseable responses",
            flush=True,
        )
        if not references:
            completed += 1
            recorder.append(
                "decisions.jsonl",
                {
                    "iteration": iteration,
                    "parent_prompt": parent,
                    "selected_prompt": parent,
                    "accepted": False,
                    "phase": None,
                    "score": 0.0,
                    "pool_sha256": pool_hash,
                    "parent_validation_accuracy": parent_accuracy,
                    "selected_validation_accuracy": parent_accuracy,
                    "validation_candidates": [],
                    "reason": "no parseable training responses",
                },
            )
            recorder.write(
                "progress.json",
                {
                    "iteration": iteration,
                    "phase": "decision",
                    "accepted": False,
                    "accepted_updates": accepted_updates,
                    "validation_accuracy": parent_accuracy,
                    "iterations_completed": completed,
                    "max_iterations": limits.get("max_iterations", 5),
                    "reason": "no parseable training responses",
                },
            )
            print(f"Step {iteration + 1}: skipped; no parseable training responses", flush=True)
            continue
        groups = [
            (
                [t for t in references if t.example_id == example.id],
                [r for r in rollouts if r["example_id"] == example.id],
                {"example_id": example.id, "group_index": index},
            )
            for index, example in enumerate(examples)
        ] if per_example else [(references, rollouts, {})]
        eligible = []
        for group_refs, group_rollouts, metadata in groups:
            group_hash = digest([t.to_dict() for t in group_refs])
            group_record = {
                "iteration": iteration, **metadata, "included": len(group_refs),
                "successes": sum(t.reward == 1 for t in group_refs),
                "failures": sum(t.reward == 0 for t in group_refs),
                "errors": {}, "pool_sha256": group_hash,
            }
            if not group_refs:
                recorder.append("proposal_groups.jsonl", {**group_record, "reason": "no parseable training responses"})
                continue
            proposal = proposal_eval = None
            for phase in ("proposal", "refinement"):
                if phase == "refinement" and proposal is None:
                    break
                recorder.write("progress.json", {"iteration": iteration, "phase": phase, **metadata})
                payload = request_payload(
                    parent, group_rollouts,
                    **({"candidate": proposal, "evaluation": proposal_eval} if phase == "refinement" else {}),
                )
                try:
                    instruction = optimizer_call(
                        backend, recorder, "propose" if phase == "proposal" else "refine", payload,
                        iteration, example_id=metadata.get("example_id"),
                    )
                except InvalidInstruction as error:
                    if not per_example:
                        raise
                    group_record["errors"][phase] = str(error)
                    continue
                evaluation = fixed_directional_gain(parent, instruction, group_refs, scorer, settings=settings)
                recorder.append("edits.jsonl", {
                    "iteration": iteration, **metadata, "phase": phase,
                    "instruction": instruction, "pool_sha256": group_hash, **evaluation,
                })
                group_record[phase + "_verified"] = evaluation["verified"]
                print(
                    f"Step {iteration + 1} {metadata.get('example_id', 'minibatch')}: {phase} "
                    f"verified={evaluation['verified']} mean_deltas="
                    f"{[r['delta_mean_logprob'] for r in evaluation['trajectories']]}", flush=True,
                )
                eligible.append((phase, instruction, evaluation, metadata))
                if phase == "proposal":
                    proposal, proposal_eval = instruction, evaluation
            recorder.append("proposal_groups.jsonl", group_record)
        eligible = [c for c in eligible if c[2]["verified"] and c[2]["score"] > 0]
        evaluated, tested, seen = [], [], set()
        for phase, instruction, evaluation, metadata in eligible:
            if instruction in seen:
                continue
            seen.add(instruction)
            accuracy = evaluate_validation(
                benchmark,
                backend,
                recorder,
                instruction,
                validation_repeats,
                iteration=iteration,
                phase=f"question{metadata['group_index'] + 1}-{phase}" if per_example else phase,
            )
            index = len(candidates)
            candidates.append(
                {
                    "prompt": instruction,
                    "parent_idx": current_idx,
                    "kind": phase,
                    "iteration": iteration,
                    **metadata,
                    "train_signed_gain": evaluation["score"],
                    "validation_accuracy": accuracy,
                    "accepted": False,
                }
            )
            tested.append(
                {
                    "candidate_idx": index,
                    "phase": phase,
                    **metadata,
                    "accuracy": accuracy,
                    "parent_accuracy": parent_accuracy,
                    "passed": accuracy >= parent_accuracy,
                }
            )
            if accuracy >= parent_accuracy:
                evaluated.append((accuracy, evaluation["score"], index))
        old_parent, old_accuracy = parent, parent_accuracy
        winner = max(evaluated, key=lambda c: (c[0], c[1])) if evaluated else None
        if winner:
            parent_accuracy, _, current_idx = winner
            candidates[current_idx]["accepted"] = True
            parent = candidates[current_idx]["prompt"]
            accepted_updates += 1
        completed += 1
        recorder.append(
            "decisions.jsonl",
            {
                "iteration": iteration,
                "parent_prompt": old_parent,
                "selected_prompt": parent,
                "accepted": winner is not None,
                "phase": candidates[current_idx]["kind"] if winner else None,
                "score": winner[1] if winner else 0.0,
                "pool_sha256": pool_hash,
                "parent_validation_accuracy": old_accuracy,
                "selected_validation_accuracy": parent_accuracy,
                "validation_candidates": tested,
                "reason": "validation nondecrease and likelihood verification passed"
                if winner
                else "validation rejected all eligible edits"
                if tested
                else "no likelihood-verified edit",
            },
        )
        recorder.write(
            "progress.json",
            {
                "iteration": iteration,
                "phase": "decision",
                "accepted": winner is not None,
                "accepted_updates": accepted_updates,
                "validation_accuracy": parent_accuracy,
                "iterations_completed": completed,
                "max_iterations": limits.get("max_iterations", 5),
            },
        )
        print(
            f"Step {iteration + 1}: accepted={winner is not None} "
            f"validation={old_accuracy:.4f}->{parent_accuracy:.4f}",
            flush=True,
        )
    return EditResult(candidates, current_idx, recorder.cost["task_calls"], completed, termination)


def probe_search(benchmark, backend, recorder, config):
    """Independent per-question edits against one fixed base; no validation, test or adoption."""
    limits, parent = config["search"], config["initial_prompt"]
    settings = config["edit_refinement"]
    scorer = TrajectoryScorer(backend, recorder, collect_top1=settings.get("likelihood_scope") == "branch")
    candidates = [{
        "prompt": parent, "parent_idx": None, "kind": "initial", "validation_accuracy": None,
        "accepted": True,
    }]
    stats = {key: 0 for key in (
        "rounds_completed", "groups_total", "groups_with_evidence", "groups_without_evidence",
        "captured_rollouts", "retained_rollouts", "retained_successes", "retained_failures",
        "proposal_attempts", "proposal_valid_outputs", "proposal_measured", "proposal_verified",
        "refinement_attempts", "refinement_valid_outputs", "refinement_measured", "refinement_verified",
        "groups_with_verified_edit", "refinement_rescues", "refinement_direction_losses",
    )}
    compositions = {key: 0 for key in ("success_only", "failure_only", "mixed")}
    distinct_measured, distinct_verified = set(), set()
    termination = "max_iterations"

    def snapshot():
        def rate(numerator, denominator):
            return numerator / denominator if denominator else None
        return {
            **stats, "prompt_policy": "fixed", "accepted_updates": 0,
            "validation": "not_run", "test": "not_run", "reward_compositions": dict(compositions),
            "distinct_measured_edits": len(distinct_measured),
            "distinct_verified_edits": len(distinct_verified),
            "evidence_coverage": rate(stats["groups_with_evidence"], stats["groups_total"]),
            "proposal_alignment_rate": rate(stats["proposal_verified"], stats["proposal_measured"]),
            "refinement_alignment_rate": rate(stats["refinement_verified"], stats["refinement_measured"]),
            "group_yield": rate(stats["groups_with_verified_edit"], stats["groups_total"]),
        }

    recorder.write("probe_summary.json", snapshot())
    for iteration, examples in enumerate(minibatches(
        benchmark.train, limits["minibatch_size"], limits["max_iterations"], limits["search_seed"]
    )):
        required = len(examples) * limits["train_rollouts"]
        if recorder.cost["task_calls"] + required > limits["max_task_rollouts"]:
            termination = "max_task_rollouts"
            break
        print(f"Probe round {iteration + 1}: {len(examples)} questions x {limits['train_rollouts']} rollouts", flush=True)
        recorder.write("progress.json", {"iteration": iteration, "phase": "rollout", "questions": len(examples)})
        captured, references, rollouts, evidence = capture_training_batch(
            benchmark, backend, recorder, parent, examples, limits["train_rollouts"], iteration
        )
        for group_index, example in enumerate(examples):
            raw = [t for t in captured if t.example_id == example.id]
            refs = [t for t in references if t.example_id == example.id]
            shown = [r for r in rollouts if r["example_id"] == example.id]
            flags = [r for r in evidence if r["example_id"] == example.id]
            pool_hash = digest([t.to_dict() for t in refs])
            group = {
                "iteration": iteration, "group_index": group_index, "example_id": example.id,
                "parent_prompt": parent, "captured": len(raw), "included": len(refs),
                "successes": sum(t.reward == 1 for t in refs), "failures": sum(t.reward == 0 for t in refs),
                "proposal_verified": None, "refinement_verified": None, "errors": {},
            }
            recorder.append("editor_evidence.jsonl", {
                "iteration": iteration, "group_index": group_index, "example_id": example.id,
                "captured": len(raw), "included": len(refs), "excluded": len(raw) - len(refs),
                "response_view": "post_thinking_only", "raw_pool_sha256": digest([t.to_dict() for t in raw]),
                "pool_sha256": pool_hash, "trajectories": flags,
            })
            stats["groups_total"] += 1
            stats["captured_rollouts"] += len(raw)
            stats["retained_rollouts"] += len(refs)
            stats["retained_successes"] += group["successes"]
            stats["retained_failures"] += group["failures"]

            def measure(phase, instruction, *, refs=refs, iteration=iteration, example=example,
                        group_index=group_index, pool_hash=pool_hash, group=group):
                evaluation = fixed_directional_gain(parent, instruction, refs, scorer, settings=settings)
                index = len(candidates)
                verified = evaluation["verified"] and evaluation["score"] > 0
                candidates.append({
                    "prompt": instruction, "parent_idx": 0, "kind": phase, "iteration": iteration,
                    "example_id": example.id, "train_signed_gain": evaluation["score"],
                    "verified": verified, "validation_accuracy": None, "accepted": False,
                })
                recorder.append("edits.jsonl", {
                    "iteration": iteration, "group_index": group_index, "example_id": example.id,
                    "phase": phase, "instruction": instruction, "pool_sha256": pool_hash, **evaluation,
                })
                stats[phase + "_measured"] += 1
                stats[phase + "_verified"] += int(verified)
                distinct_measured.add((iteration, example.id, instruction))
                if verified:
                    distinct_verified.add((iteration, example.id, instruction))
                group[phase + "_verified"] = verified
                group[phase + "_candidate_idx"] = index
                print(
                    f"Probe round {iteration + 1} question {group_index + 1}: {phase} verified={verified} "
                    f"mean_deltas={[r['delta_mean_logprob'] for r in evaluation['trajectories']]}", flush=True,
                )
                return evaluation

            try:
                if not refs:
                    stats["groups_without_evidence"] += 1
                    group["reason"] = "no parseable training responses"
                    continue
                stats["groups_with_evidence"] += 1
                composition = "mixed" if group["successes"] and group["failures"] else (
                    "success_only" if group["successes"] else "failure_only"
                )
                group["reward_composition"] = composition
                compositions[composition] += 1
                recorder.write("progress.json", {
                    "iteration": iteration, "phase": "proposal", "group_index": group_index,
                    "groups": len(examples), "included_trajectories": len(refs),
                })
                stats["proposal_attempts"] += 1
                try:
                    proposal = optimizer_call(
                        backend, recorder, "propose", request_payload(parent, shown), iteration,
                        example_id=example.id,
                    )
                except InvalidInstruction as error:
                    group["errors"]["proposal"] = str(error)
                    continue
                stats["proposal_valid_outputs"] += 1
                evaluation = measure("proposal", proposal)
                recorder.write("progress.json", {
                    "iteration": iteration, "phase": "refinement", "group_index": group_index,
                    "groups": len(examples),
                })
                stats["refinement_attempts"] += 1
                try:
                    refined = optimizer_call(
                        backend, recorder, "refine",
                        request_payload(parent, shown, candidate=proposal, evaluation=evaluation),
                        iteration, example_id=example.id,
                    )
                except InvalidInstruction as error:
                    group["errors"]["refinement"] = str(error)
                    continue
                stats["refinement_valid_outputs"] += 1
                measure("refinement", refined)
                stats["refinement_rescues"] += int(not group["proposal_verified"] and group["refinement_verified"])
                stats["refinement_direction_losses"] += int(group["proposal_verified"] and not group["refinement_verified"])
            finally:
                group["any_verified"] = bool(group["proposal_verified"] or group["refinement_verified"])
                stats["groups_with_verified_edit"] += int(group["any_verified"])
                recorder.append("probe_groups.jsonl", group)
                recorder.write("probe_summary.json", snapshot())
                print(
                    f"Probe round {iteration + 1} question {group_index + 1}: evidence={len(refs)}/{len(raw)} "
                    f"any_verified={group['any_verified']}", flush=True,
                )
        stats["rounds_completed"] += 1
        recorder.append("decisions.jsonl", {
            "iteration": iteration, "parent_prompt": parent, "selected_prompt": parent,
            "accepted": False, "phase": None, "reason": "fixed base likelihood diagnostic",
            "parent_validation_accuracy": None, "selected_validation_accuracy": None,
            "validation_candidates": [],
        })
        recorder.write("probe_summary.json", snapshot())
        recorder.write("progress.json", {
            "iteration": iteration, "phase": "round_complete", "iterations_completed": stats["rounds_completed"],
            "max_iterations": limits["max_iterations"], "proposal_verified": stats["proposal_verified"],
            "refinement_verified": stats["refinement_verified"],
        })
    return EditResult(candidates, 0, recorder.cost["task_calls"], stats["rounds_completed"], termination, snapshot())
