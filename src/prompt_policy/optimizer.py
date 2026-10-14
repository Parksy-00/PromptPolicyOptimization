from __future__ import annotations

import math
from collections import defaultdict

from gepa import EvaluationBatch, optimize
from gepa.strategies.instruction_proposal import InstructionProposalSignature, parse_proposal

from prompt_policy.artifacts import Recorder, digest
from prompt_policy.benchmarks import Benchmark
from prompt_policy.objective import decide
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

    def evaluate(self, batch, candidate, capture_traces=False):
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

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
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
    def __init__(self, recorder: Recorder, settings: dict):
        self.recorder, self.settings = recorder, settings

    def __call__(self, state):
        # Own actual-rollout counter is authoritative: GEPA full-val bookkeeping
        # counts question slots in this pinned revision, not all grouped repeats.
        return (
            state.i + 1 >= self.settings["max_iterations"]
            or self.recorder.cost["task_calls"] >= self.settings["max_task_rollouts"]
        )


def search(benchmark: Benchmark, backend: Backend, recorder: Recorder, config: dict, *, reflection_lm=None):
    adapter = TaskAdapter(benchmark, backend, TrajectoryScorer(backend, recorder), recorder, config)
    result = optimize(
        seed_candidate={"system_prompt": config["initial_prompt"]},
        trainset=list(benchmark.train),
        valset=list(benchmark.validation),
        adapter=adapter,
        reflection_lm=reflection_lm or LocalReflection(backend, recorder),
        reflection_prompt_template=REFLECTION_TEMPLATE,
        acceptance_criterion=LikelihoodAcceptance(adapter),
        candidate_selection_strategy="pareto",
        frontier_type="instance",
        skip_perfect_score=False,
        use_merge=False,
        cache_evaluation=False,
        reflection_minibatch_size=config["search"]["minibatch_size"],
        seed=config["search"]["search_seed"],
        stop_callbacks=[SearchStopper(recorder, config["search"])],
        run_dir=str(recorder.run_dir / "gepa"),
        raise_on_exception=True,
        display_progress_bar=False,
        use_cloudpickle=False,
    )
    return result
