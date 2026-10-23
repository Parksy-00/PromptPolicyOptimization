from __future__ import annotations

import math
from collections import defaultdict

from gepa import EvaluationBatch, optimize
from gepa.strategies.instruction_proposal import InstructionProposalSignature, parse_proposal

from prompt_policy.artifacts import Recorder, digest
from prompt_policy.benchmarks import Benchmark
from prompt_policy.objective import decide, fixed_directional_gain
from prompt_policy.scoring import TrajectoryScorer
from prompt_policy.types import Backend, Trajectory

REFLECTION_TEMPLATE = """Revise this system instruction to improve task accuracy:
<curr_param>

Observed trajectories and verifiable feedback:
<side_info>

Extract general rules rather than memorizing questions or answers. Successful trajectories are
reference behaviours to preserve. When likelihood measurements are supplied, use the measured
before/after changes to propose a revision that makes these successful behaviours more likely.
Past measurements refer to their recorded minibatch, not to unobserved examples. Preserve the
required final-answer format. Return only the revised instruction within a complete fenced block.
"""


class TaskAdapter:
    propose_new_texts = None

    def __init__(
        self,
        benchmark: Benchmark,
        backend: Backend,
        scorer: TrajectoryScorer,
        recorder: Recorder,
        config: dict,
    ):
        self.benchmark, self.backend, self.scorer = benchmark, backend, scorer
        self.recorder, self.config = recorder, config
        self.history: dict[str, list[dict]] = defaultdict(list)
        self.fatal_error = None

    def evaluate(self, batch, candidate, capture_traces=False):
        if "gepa_refinement" in self.config:
            return self.evaluate_directional(batch, candidate, capture_traces)
        prompt = candidate["system_prompt"]
        repeats = self.config["search"]["train_rollouts" if capture_traces else "validation_rollouts"]
        outputs, scores, trajectories = [], [], []
        for example in batch:
            group = []
            for repeat in range(repeats):
                generated = self.backend.generate(prompt, example.question)
                value = self.benchmark.score(
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
                )
                row = trajectory.to_dict()
                self.recorder.append("trajectories.jsonl", row)
                group.append(row)
            outputs.append(group)
            trajectories.append(group)
            scores.append(math.fsum(x["reward"] for x in group) / repeats)
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
            num_metric_calls=len(batch) * repeats,
        )

    def evaluate_directional(self, batch, candidate, capture_traces):
        """Use the existing concurrent transport; GEPA still owns which batch to evaluate."""
        from prompt_policy.backends import generate_batch

        prompt = candidate["system_prompt"]
        repeats = self.config["search"]["train_rollouts" if capture_traces else "validation_rollouts"]
        jobs = [(e, r) for e in batch for r in range(repeats)]
        rows = [None] * len(jobs)
        completed = 0
        phase = "train_minibatch" if capture_traces else "validation"

        def record(index, generated):
            nonlocal completed
            example, repeat = jobs[index]
            reward = self.benchmark.score(
                example, generated.text, thinking_open_in_prefix=generated.thinking_open_in_prefix
            )
            trajectory = Trajectory(
                example.id, example.question, example.split, repeat, prompt, generated.text,
                generated.prompt_ids, generated.response_ids, reward, generated.finish_reason,
                generated.thinking_open_in_prefix,
            )
            rows[index] = trajectory.to_dict()
            self.recorder.append("trajectories.jsonl", rows[index])
            completed += 1
            if completed % 10 == 0 or completed == len(jobs):
                self.recorder.write("progress.json", {
                    "phase": phase, "prompt_sha256": digest(prompt),
                    "completed": completed, "total": len(jobs),
                })
                print(f"GEPA {phase}: {completed}/{len(jobs)}", flush=True)

        generate_batch(self.backend, prompt, [e.question for e, _ in jobs], on_result=record)
        groups = [rows[i * repeats:(i + 1) * repeats] for i in range(len(batch))]
        scores = [math.fsum(r["reward"] for r in group) / repeats for group in groups]
        self.recorder.append("gepa_evaluations.jsonl", {
            "phase": phase, "prompt": prompt, "example_ids": [e.id for e in batch],
            "scores": scores, "rollouts": len(jobs),
        })
        return EvaluationBatch(
            outputs=groups, scores=scores, trajectories=groups if capture_traces else None,
            num_metric_calls=len(jobs),
        )

    def directional_records(self, candidate, eval_batch):
        if eval_batch.trajectories is None:
            raise ValueError("GEPA likelihood refinement requires captured parent trajectories")
        examples = {e.id: e for e in self.benchmark.train}
        records = []
        for group in eval_batch.trajectories:
            for row in group:
                t = Trajectory.from_dict(row)
                example = examples.get(t.example_id)
                if (example is None or t.prompt != candidate["system_prompt"] or t.split != "train"
                        or t.question != example.question):
                    raise ValueError("GEPA fixed trajectory is not from this parent/train batch")
                visible = self.benchmark.editor_response(
                    t.text, thinking_open_in_prefix=t.thinking_open_in_prefix
                )
                if visible is not None:
                    records.append({
                        "Inputs": {"question": t.question, "gold_answer": example.answer},
                        "Generated Outputs": visible,
                        "Feedback": {"task_reward": t.reward},
                        # Transport evidence for the custom proposer only. Never rendered to the LM.
                        "_fixed_trajectory": t.to_dict(),
                    })
        return records

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        if "gepa_refinement" in self.config:
            try:
                records = self.directional_records(candidate, eval_batch)
            except Exception as error:
                self.fatal_error = error
                raise
            self.recorder.append("editor_evidence.jsonl", {
                "parent_prompt": candidate["system_prompt"],
                "captured": sum(len(g) for g in eval_batch.trajectories),
                "retained": len(records),
                "retained_pool_sha256": digest([r["_fixed_trajectory"] for r in records]),
                "included": [(r["_fixed_trajectory"]["example_id"],
                              r["_fixed_trajectory"]["repeat_index"]) for r in records],
            })
            return {component: records for component in components_to_update}
        if eval_batch.trajectories is None:
            raise ValueError("Reflection requires original captured trajectories")
        records = []
        for group in eval_batch.trajectories:
            for row in group:
                trajectory = Trajectory.from_dict(row)
                record = {
                    "Inputs": {"question": trajectory.question},
                    "Generated Outputs": trajectory.text,
                    "Feedback": {
                        "task_reward": trajectory.reward,
                        "role": "successful reference" if trajectory.reward == 1 else "failure",
                        "finish_reason": trajectory.finish_reason,
                    },
                }
                if self.config["objective"]["likelihood_feedback"] and trajectory.reward == 1:
                    record["Feedback"]["parent_mean_logprob_T1"] = self.scorer.score(
                        candidate["system_prompt"], trajectory
                    ).mean
                records.append(record)
        if self.config["objective"]["likelihood_feedback"]:
            prior = self.history.get(digest(candidate["system_prompt"]), [])
            if prior:
                records.append({"Feedback": {"previous_proposals_for_this_parent": prior[-3:]}})
        return {component: records for component in components_to_update}


class LikelihoodAcceptance:
    def __init__(self, adapter: TaskAdapter):
        self.adapter = adapter

    def should_accept(self, proposal, state):
        parent = state.program_candidates[proposal.parent_program_ids[0]]["system_prompt"]
        child = proposal.candidate["system_prompt"]
        before, after = proposal.eval_before, proposal.eval_after
        if before is None or after is None or before.trajectories is None or after.trajectories is None:
            raise ValueError("Both parent and candidate captures are required")
        if len(before.trajectories) != len(after.trajectories):
            raise ValueError("Mismatched minibatches")
        fixed = []
        for old_group, new_group in zip(before.trajectories, after.trajectories, strict=True):
            old = [Trajectory.from_dict(x) for x in old_group]
            new = [Trajectory.from_dict(x) for x in new_group]
            if not old or not new or len(old) != len(new):
                raise ValueError("Unaligned rollout groups")
            if any(t.prompt != parent or t.example_id != old[0].example_id for t in old):
                raise ValueError("Parent capture identity mismatch")
            if any(
                t.prompt != child
                or (t.example_id, t.question, t.split) != (old[0].example_id, old[0].question, old[0].split)
                for t in new
            ):
                raise ValueError("Candidate evaluated on a different question")
            fixed.extend(old)
        # Freeze old trajectories before any candidate scoring. Never use child responses as references.
        pool_hash = digest([t.to_dict() for t in fixed if t.reward == 1])
        success, failure, sums = [], [], []
        settings = self.adapter.config["objective"]
        for trajectory in fixed:
            if not settings["measure_likelihood"]:
                continue
            if trajectory.reward != 1 and not settings["score_failures"]:
                continue
            old = self.adapter.scorer.score(parent, trajectory)
            new = self.adapter.scorer.score(child, trajectory)
            change = new.mean - old.mean
            if trajectory.reward == 1:
                success.append((trajectory, change))
                sums.append((trajectory, new.total - old.total))
            else:
                failure.append((trajectory, change))
        decision = decide(
            before.scores,
            after.scores,
            success,
            failure,
            sums,
            pool_sha256=pool_hash,
            mode=settings["mode"],
            weight=settings["weight"],
            max_accuracy_drop=settings["max_accuracy_drop"],
            tolerance=settings["tolerance"],
        )
        if child == parent:
            # Fresh sampling noise cannot justify adding an identical instruction.
            from dataclasses import replace

            decision = replace(decision, accepted=False, reason="candidate is identical to its parent")
        record = {
            "iteration": state.i + 1,
            "parent_ids": proposal.parent_program_ids,
            "parent_prompt": parent,
            "candidate_prompt": child,
            "pool": [t.to_dict() for t in fixed if t.reward == 1],
            "decision": decision.to_dict(),
        }
        proposal.metadata["likelihood_decision"] = decision.to_dict()
        self.adapter.recorder.append("decisions.jsonl", record)
        self.adapter.history[digest(parent)].append({"candidate_prompt": child, **decision.to_dict()})
        return decision.accepted

    def reject_reason(self, proposal, state):
        return proposal.metadata["likelihood_decision"]["reason"]


class GEPALikelihoodRefinement:
    """Native GEPA proposal -> fixed-parent measurement -> one refinement, before child rollout."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.proposal_index = 0
        self.selected = {}

    @staticmethod
    def proposal_template():
        from prompt_policy.edit_prompts import GENERAL_LESSONS, INSTRUCTION_OUTPUT_RULE

        template = InstructionProposalSignature.default_prompt_template
        return template.replace(
            "Provide the new instructions within ``` blocks.",
            GENERAL_LESSONS + " " + INSTRUCTION_OUTPUT_RULE,
        )

    def __call__(self, candidate, reflective_dataset, components_to_update, *, metadata=None):
        # Upstream GEPA retries raised proposal errors. Return no child on fatal errors,
        # then re-raise at the next stop check so integrity/transport failures never retry.
        try:
            return self.propose(candidate, reflective_dataset, components_to_update, metadata=metadata)
        except Exception as error:
            self.adapter.fatal_error = error
            self.adapter.recorder.append("gepa_refinement_errors.jsonl", {
                "gepa_anchors": dict(metadata or {}), "error": f"{type(error).__name__}: {error}",
            })
            return {}

    def propose(self, candidate, reflective_dataset, components_to_update, *, metadata=None):
        from prompt_policy.edit_optimizer import InvalidInstruction, optimizer_call
        from prompt_policy.edit_prompts import request_payload

        if components_to_update != ["system_prompt"]:
            raise ValueError("GEPA likelihood refinement supports only the instruction component")
        index = self.proposal_index
        self.proposal_index += 1
        parent = candidate["system_prompt"]
        records = reflective_dataset["system_prompt"]
        references = [Trajectory.from_dict(r["_fixed_trajectory"]) for r in records]
        pool_hash = digest([t.to_dict() for t in references])
        summary = {
            "proposal_index": index, "gepa_anchors": dict(metadata or {}),
            "parent_prompt": parent, "pool_sha256": pool_hash, "retained": len(references),
            "selected_prompt": None, "selected_phase": None, "errors": [],
        }
        recorder, backend = self.adapter.recorder, self.adapter.backend
        if not references:
            summary["reason"] = "no answer-parseable parent trajectories"
            recorder.append("gepa_refinement.jsonl", summary)
            return {}
        public_records = [{k: v for k, v in r.items() if k != "_fixed_trajectory"} for r in records]
        request = InstructionProposalSignature.prompt_renderer({
            "current_instruction_doc": parent,
            "dataset_with_feedback": public_records,
            "prompt_template": self.proposal_template(),
        })
        rollouts = [{
            "example_id": t.example_id, "repeat_index": t.repeat_index, "question": t.question,
            "gold_answer": r["Inputs"]["gold_answer"], "response": r["Generated Outputs"],
            "reward": t.reward,
        } for t, r in zip(references, records, strict=True)]
        eligible = []

        def measure(text, phase):
            recorder.write("progress.json", {"phase": "likelihood", "editor_phase": phase,
                                            "proposal_index": index})
            value = fixed_directional_gain(
                parent, text, references, self.adapter.scorer,
                settings=self.adapter.config["gepa_refinement"],
            )
            recorder.append("edits.jsonl", {
                "proposal_index": index, "gepa_anchors": dict(metadata or {}), "phase": phase,
                "parent_prompt": parent, "instruction": text, "pool_sha256": pool_hash, **value,
            })
            if value["verified"]:
                eligible.append((text, phase, value))
            return value

        try:
            recorder.write("progress.json", {"phase": "proposal", "proposal_index": index})
            proposed = optimizer_call(backend, recorder, "proposal", request, index)
        except InvalidInstruction as error:
            summary["errors"].append({"phase": "proposal", "error": str(error)})
            summary["reason"] = "invalid proposal; no retry"
            recorder.append("gepa_refinement.jsonl", summary)
            return {}
        evaluation = measure(proposed, "proposal")
        refine_request = request_payload(parent, rollouts, candidate=proposed, evaluation=evaluation)
        try:
            recorder.write("progress.json", {"phase": "refinement", "proposal_index": index})
            refined = optimizer_call(backend, recorder, "refinement", refine_request, index)
        except InvalidInstruction as error:
            summary["errors"].append({"phase": "refinement", "error": str(error)})
        else:
            measure(refined, "refinement")
        if eligible:
            # Stable maximum: keep the proposal on an exact gain tie.
            text, phase, value = max(eligible, key=lambda item: item[2]["score"])
            self.selected[(parent, text, pool_hash)] = value
            summary.update(selected_prompt=text, selected_phase=phase,
                           reason="likelihood verified; delegate accuracy and validation to GEPA")
        else:
            summary["reason"] = "no likelihood-verified edit; skip child rollout"
        recorder.append("gepa_refinement.jsonl", summary)
        return {"system_prompt": summary["selected_prompt"]} if eligible else {}


class GEPADirectionalAcceptance:
    """Require proof for this exact parent pool, then keep GEPA's native strict train gate."""

    def __init__(self, adapter, refiner):
        self.adapter, self.refiner = adapter, refiner

    def should_accept(self, proposal, state):
        from gepa.strategies.acceptance import StrictImprovementAcceptance

        parent = state.program_candidates[proposal.parent_program_ids[0]]["system_prompt"]
        child = proposal.candidate["system_prompt"]
        records = self.adapter.directional_records({"system_prompt": parent}, proposal.eval_before)
        pool_hash = digest([r["_fixed_trajectory"] for r in records])
        value = self.refiner.selected.get((parent, child, pool_hash))
        if value is None or not value["verified"]:
            raise ValueError("GEPA child lacks likelihood verification on its exact parent batch")
        accepted = child != parent and StrictImprovementAcceptance().should_accept(proposal, state)
        reason = "likelihood verified and training accuracy strictly improved" if accepted else (
            "training accuracy did not strictly improve"
        )
        decision = {
            "accepted": accepted, "reason": reason, "likelihood_verified": True,
            "likelihood_score": value["score"], "pool_sha256": pool_hash,
            "parent_train_accuracy": math.fsum(proposal.subsample_scores_before) / len(proposal.subsample_scores_before),
            "candidate_train_accuracy": math.fsum(proposal.subsample_scores_after) / len(proposal.subsample_scores_after),
        }
        proposal.metadata["likelihood_decision"] = decision
        self.adapter.recorder.append("decisions.jsonl", {
            "iteration": state.i + 1, "parent_ids": proposal.parent_program_ids,
            "parent_prompt": parent, "candidate_prompt": child, "decision": decision,
        })
        return accepted

    def reject_reason(self, proposal, state):
        return proposal.metadata["likelihood_decision"]["reason"]


class LocalReflection:
    def __init__(self, backend: Backend, recorder: Recorder):
        self.backend, self.recorder = backend, recorder

    def __call__(self, prompt):
        if not isinstance(prompt, str):
            raise ValueError("First version supports text-only reflection")
        generated = self.backend.generate(
            "You revise system instructions. Follow the user's output format.", prompt, reflection=True
        )
        self.recorder.append(
            "reflection.jsonl",
            {
                "request": prompt,
                "response": generated.text,
                "prompt_ids": generated.prompt_ids,
                "response_ids": generated.response_ids,
                "finish_reason": generated.finish_reason,
            },
        )
        if generated.finish_reason == "length":
            raise ValueError("Reflection reached its token limit; increase reflection.max_new_tokens")
        # Qwen's closed thinking block is separate from its proposed instruction.
        raw_answer = ("<think>" if generated.thinking_open_in_prefix else "") + generated.text
        answer = raw_answer.rsplit("</think>", 1)[-1].strip()
        if "<think>" in answer:
            raise ValueError("Reflection has an unterminated thinking block")
        parsed = parse_proposal(InstructionProposalSignature, answer).require()
        if not parsed:
            raise ValueError("Reflection produced an empty instruction")
        return "```\n" + parsed + "\n```"


class SearchStopper:
    def __init__(self, recorder: Recorder, settings: dict, adapter=None):
        self.recorder, self.settings, self.adapter = recorder, settings, adapter

    def __call__(self, state):
        if self.adapter is not None and self.adapter.fatal_error is not None:
            raise self.adapter.fatal_error
        # Own actual-rollout counter is authoritative: GEPA full-val bookkeeping
        # counts question slots in this pinned revision, not all grouped repeats.
        return (
            state.i + 1 >= self.settings["max_iterations"]
            or self.recorder.cost["task_calls"] >= self.settings["max_task_rollouts"]
        )


def search(benchmark: Benchmark, backend: Backend, recorder: Recorder, config: dict, *, reflection_lm=None):
    directional = "gepa_refinement" in config
    if directional and reflection_lm is not None:
        raise ValueError("gepa_refinement uses its own recorded proposal/refinement calls")
    scorer = TrajectoryScorer(
        backend, recorder, collect_top1=config.get("gepa_refinement", {}).get("likelihood_scope") == "branch"
    )
    adapter = TaskAdapter(benchmark, backend, scorer, recorder, config)
    refiner = GEPALikelihoodRefinement(adapter) if directional else None
    result = optimize(
        seed_candidate={"system_prompt": config["initial_prompt"]},
        trainset=list(benchmark.train),
        valset=list(benchmark.validation),
        adapter=adapter,
        reflection_lm=None if directional else reflection_lm or LocalReflection(backend, recorder),
        reflection_prompt_template=None if directional else REFLECTION_TEMPLATE,
        custom_candidate_proposer=refiner,
        acceptance_criterion=GEPADirectionalAcceptance(adapter, refiner) if directional else LikelihoodAcceptance(adapter),
        candidate_selection_strategy="pareto",
        frontier_type="instance",
        skip_perfect_score=False,
        use_merge=False,
        cache_evaluation=False,
        reflection_minibatch_size=config["search"]["minibatch_size"],
        seed=config["search"]["search_seed"],
        stop_callbacks=[SearchStopper(recorder, config["search"], adapter)],
        run_dir=str(recorder.run_dir / "gepa"),
        raise_on_exception=True,
        display_progress_bar=False,
        use_cloudpickle=False,
    )
    if adapter.fatal_error is not None:
        raise adapter.fatal_error
    return result
