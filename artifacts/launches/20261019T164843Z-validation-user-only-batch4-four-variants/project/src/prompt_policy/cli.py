from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from prompt_policy.artifacts import Recorder, digest, file_sha256
from prompt_policy.config import load_config, resolve_model


def verify_gepa(root: Path) -> dict:
    """Reject modified source or an accidentally installed, unpinned PyPI GEPA."""
    import gepa

    manifest = json.loads((root / "vendor/gepa/UPSTREAM.json").read_text())
    for relative, expected in manifest["files_sha256"].items():
        if file_sha256(root / "vendor/gepa" / relative) != expected:
            raise ValueError(f"Vendored GEPA differs from pinned source: {relative}")
    actual = Path(gepa.__file__).resolve().parent
    if actual != (root / "vendor/gepa/src/gepa").resolve():
        raise ValueError("GEPA must be loaded from vendor/gepa; install with uv sync --frozen")
    return {"repository": manifest["repository"], "revision": manifest["revision"], "patches": []}


def source_identity(root: Path) -> dict:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        revision = None
    paths = sorted((root / "src").rglob("*.py")) + [root / "pyproject.toml", root / "uv.lock"]
    return {"git_commit": revision, "files_sha256": {str(p.relative_to(root)): file_sha256(p) for p in paths}}


def evaluate_test(benchmark, backend, recorder, prompt, repeats, label):
    from prompt_policy.backends import generate_batch

    jobs = [(e, r) for e in benchmark.test for r in range(repeats)]
    scores, completed = [None] * len(jobs), 0

    def record(index, generated):
        nonlocal completed
        example, repeat = jobs[index]
        score = benchmark.score(
            example, generated.text, thinking_open_in_prefix=generated.thinking_open_in_prefix
        )
        scores[index] = score
        recorder.append(
            "test.jsonl",
            {
                "condition": label,
                "prompt": prompt,
                "example_id": example.id,
                "question": example.question,
                "split": "test",
                "repeat_index": repeat,
                "reward": score,
                **generated.__dict__,
            },
        )
        completed += 1
        if completed % 10 == 0 or completed == len(jobs):
            recorder.write(
                "progress.json",
                {"phase": "test", "condition": label, "completed": completed, "total": len(jobs)},
            )
            print(f"Test {label}: {completed}/{len(jobs)}", flush=True)

    generate_batch(backend, prompt, [e.question for e, _ in jobs], on_result=record)
    results = [
        {"example_id": e.id, "accuracy": statistics.mean(scores[i * repeats : (i + 1) * repeats])}
        for i, e in enumerate(benchmark.test)
    ]
    repeat_means = [statistics.mean(scores[r::repeats]) for r in range(repeats)]
    return {
        "accuracy": statistics.mean(repeat_means),
        "repeat_accuracies": repeat_means,
        "repeat_std": statistics.stdev(repeat_means) if repeats > 1 else None,
        "questions": len(results),
        "rollouts": len(scores),
        "per_question": results,
    }


def run(config_path: Path, *, dry_run=False, output: Path | None = None):
    config_path = config_path.resolve()
    root = Path(__file__).resolve().parents[2]
    config, benchmark = load_config(config_path)
    config, missing = resolve_model(config, config_path.parent)
    if "edit_refinement" in config:
        from prompt_policy.edit_prompts import (
            PROPOSER_INSTRUCTIONS,
            PROPOSER_INTRO,
            REFINER_INSTRUCTIONS,
            REFINER_INTRO,
        )

        optimizer_name = "edit_refinement"
        upstream = {
            "implementation": "minimal propose-measure-refine/v13-per-example-validation",
            "proposer_intro_sha256": digest(PROPOSER_INTRO),
            "refiner_intro_sha256": digest(REFINER_INTRO),
            "proposer_sha256": digest(PROPOSER_INSTRUCTIONS),
            "refiner_sha256": digest(REFINER_INSTRUCTIONS),
        }
    elif "textgrad" in config:
        from prompt_policy.textgrad_optimizer import verify_textgrad

        upstream = verify_textgrad()
        optimizer_name = "textgrad"
    else:
        upstream = verify_gepa(root)
        optimizer_name = "gepa"
    probe_mode = config.get("edit_refinement", {}).get("mode") == "per_example_likelihood_probe"
    plan = {
        "executable": not missing,
        "missing": missing,
        "config": config,
        "benchmark": benchmark.identity,
        optimizer_name: upstream,
        "optimizer": optimizer_name,
        "objective": (
            "verify each observed success increases and each observed failure decreases; "
            "rank by question-balanced signed likelihood gain; require validation nondecrease; "
            "single-class pools eligible only without question centering"
            if optimizer_name == "edit_refinement"
            else "maximize fixed-parent reward-weighted sequence logprob gain, require G>tolerance; validation guard"
            if config["objective"]["mode"] == "success_likelihood_ascent"
            else "validation accuracy"
            if optimizer_name == "textgrad"
            else "delta_accuracy + weight * question_balanced_mean_delta_logprob_T1"
        ),
        "final_selection": "last_likelihood_and_validation_verified_prompt"
        if optimizer_name == "edit_refinement"
        else "validation_accuracy_only",
        "notes": [
            "No sampling seeds are sent; search_seed controls search minibatches only.",
            (
                "Optimizer reserves each complete iteration; missing-evidence batches can use less."
                if optimizer_name in {"textgrad", "edit_refinement"}
                else "Budget checked between complete iterations; the last iteration can exceed it."
            ),
            "Built-in JSONL scorers do not claim canonical DAPO/GPQA equivalence.",
        ],
    }
    from prompt_policy.benchmarks import TASK_OUTPUT_RULES, task_messages

    task_format = config["benchmark"].get("task_format")
    if task_format in TASK_OUTPUT_RULES:
        plan["task_interface"] = {
            "task_format": task_format,
            "editable_component": "instruction_only",
            "initial_instruction": config["initial_prompt"],
            "fixed_output_rule": TASK_OUTPUT_RULES[task_format],
            "initial_task_messages_example": task_messages(
                config["initial_prompt"], benchmark.train[0].question, task_format
            ),
            "answer_filter": (
                "last balanced \\boxed{...} outside thinking, containing only [+-]?[0-9]+; integer equality; no fallback"
                if task_format == "boxed_integer_v1"
                else "exactly one <answer>[+-]?[0-9]+</answer> block outside thinking; integer equality"
                if task_format in {"answer_tag_integer_v1", "answer_tag_integer_v2"}
                else "last nonempty line outside thinking: Answer: [+-]?[0-9]+; integer equality"
            ),
        }
    if optimizer_name == "edit_refinement":
        from prompt_policy.edit_optimizer import minibatches

        plan["train_minibatch_schedule"] = [
            [e.id for e in batch]
            for batch in minibatches(
                benchmark.train,
                config["search"]["minibatch_size"],
                config["search"]["max_iterations"],
                config["search"]["search_seed"],
            )
        ]
        plan["likelihood_measurement"] = {
            "scope": config["edit_refinement"].get("likelihood_scope", "full"),
            "drift_correction": config["edit_refinement"].get("drift_correction", "none"),
            "score_unit": config["edit_refinement"].get("score_unit", "sequence"),
            "branch_probability_threshold": config["edit_refinement"].get("branch_probability_threshold"),
            "empty_mask": "unverifiable; never treat missing response as probability one",
            "mask_source": "parent, fixed for parent and all candidates",
        }
        plan["optimizer_prompts"] = {
            "proposer_introduction": PROPOSER_INTRO,
            "refiner_introduction": REFINER_INTRO,
            "proposer_instructions": PROPOSER_INSTRUCTIONS,
            "refiner_instructions": REFINER_INSTRUCTIONS,
        }
        plan["optimizer_input_contract"] = {
            "roles": ["user"],
            "content": "purpose, instruction(s), input/responses with inline likelihood changes for refiner, feedback, editing instructions",
            "system_message": "omitted",
        }
        plan["optimizer_output_contract"] = {
            "tag": "instruction",
            "extraction": "one nonempty block outside thinking; discard surrounding text",
            "invalid_output": "record raw output and fail before admission; no fallback or retry",
        }
        plan["pipeline"] = [
            "reuse verified matching base validation when search.base_validation_run is set; otherwise generate it",
            "train minibatch rollouts and task rewards; archive all original responses/IDs",
            "retain answer-parseable trajectories; show only post-thinking response text; skip empty batches",
            "diagnose failures and reproduce successes -> one complete instruction proposal",
            "score the exact fixed parent response IDs under parent and proposal",
            "one refinement using measured changes -> score the same fixed responses",
            "fresh validation for each distinct direction-verified improvement",
            "require validation accuracy >= stored parent validation; select by validation, then train gain; or keep parent",
            "freeze the last accepted prompt before initial/selected test",
        ]
        plan["edit_limits_per_outer_iteration"] = {
            "optimizer_calls": 2,
            "measured_candidates": 2,
            "task_rollouts_reserved": config["search"]["minibatch_size"] * config["search"]["train_rollouts"]
            + 2 * len(benchmark.validation) * config["search"]["validation_rollouts"],
        }
        plan["notes"].extend(
            [
                "Base validation is measured once; distinct likelihood-verified edits get fresh complete validation. Ties with the parent are accepted.",
                "Validation generations and train generations share max_task_rollouts; final test is separate.",
                "Validation responses never enter proposer/refiner; repeated adaptive validation is not held-out test evidence.",
                "Only answer-parseable train trajectories enter editing and likelihood verification; both correct and incorrect answers are retained.",
                "Editors see only post-thinking text; likelihood masks still apply to each retained original full response ID sequence.",
                "A minibatch without parseable responses counts as one skipped iteration; no retry or resampling.",
                "Each retained success must increase and each retained failure decrease; all reward compositions are used.",
                "The proposer receives task/reward evidence; the refiner additionally receives numerical changes.",
                "T=1 full-vocabulary sequence likelihood differs from the truncated sampling policy.",
            ]
        )
    if config.get("edit_refinement", {}).get("mode") == "validation_gated_per_example":
        count = config["search"]["minibatch_size"]
        plan["edit_limits_per_outer_iteration"] = {
            "optimizer_calls": 2 * count,
            "measured_candidates": 2 * count,
            "task_rollouts_reserved": count * config["search"]["train_rollouts"]
                + 2 * count * len(benchmark.validation) * config["search"]["validation_rollouts"],
        }
        plan["pipeline"][3:6] = [
            "independently propose and refine once for each question using only its retained responses",
            "verify each proposal/refinement against that question's exact original response IDs",
            "record invalid editor outputs and skip only the affected question/phase; no retry",
        ]
        plan["notes"].extend([
            "All questions in a round roll out under the same parent; adopt at most one prompt after all candidate validations.",
            "Validate each distinct eligible instruction once per round; identical instructions use the first qualifying source in schedule order.",
            "Punctuation-only instruction bodies are invalid and never measured or validated.",
            "A complete base validation may be reused from an interrupted run only after all 300 records and the complete summary pass identity checks.",
        ])
        plan["optimizer_output_contract"]["invalid_output"] = "record raw output and skip affected question/phase; no fallback or retry"
    if probe_mode:
        plan["objective"] = "measure per-question directional likelihood alignment against a fixed base; no adoption or accuracy gate"
        plan["final_selection"] = "fixed_initial_prompt_probe"
        plan["pipeline"] = [
            "generate train minibatch rollouts under the fixed initial prompt",
            "group by question; retain parseable responses and show only post-thinking text",
            "independently propose one instruction per question, then measure its fixed response IDs",
            "refine once using measured mean-logprob changes; remeasure the same fixed responses",
            "record individual success-increase/failure-decrease checks; never adopt candidates",
            "repeat for max_iterations; do not generate validation or test responses",
        ]
        plan["edit_limits_per_outer_iteration"] = {
            "optimizer_calls": 2 * config["search"]["minibatch_size"],
            "measured_candidates": 2 * config["search"]["minibatch_size"],
            "task_rollouts_reserved": config["search"]["minibatch_size"] * config["search"]["train_rollouts"],
        }
        plan["notes"] = [
            "Fixed base throughout; each edit is verified only on its own question's retained trajectories.",
            "Validation and test are not run; no accuracy improvement claim or candidate adoption.",
            "Each retained success must increase and each retained failure decrease individually, strictly above zero.",
            "Only answer-parseable train trajectories enter editing and verification; editors see post-thinking text only.",
            "Original full response IDs are scored under T=1 full-vocabulary likelihood, distinct from rollout sampling.",
            "Empty evidence groups and invalid editor outputs are recorded and skipped independently; no retry or resampling.",
            "No request-level sampling seeds; search_seed controls the minibatch schedule only.",
        ]
        plan["optimizer_output_contract"]["invalid_output"] = "record raw output and skip this editor/group; no fallback or retry"
    if dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return plan
    if missing:
        raise ValueError("Missing configuration: " + "; ".join(missing))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = (output or root / "artifacts/runs" / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    recorder = Recorder(run_dir)
    started = time.perf_counter()
    manifest = {
        "schema": "prompt_policy_run/v1",
        "run_id": run_dir.name,
        "status": "running",
        "config_sha256": digest(config),
        "benchmark": benchmark.identity,
        "source": source_identity(root),
        optimizer_name: upstream,
        "optimizer": optimizer_name,
        "environment": {
            "python": sys.version,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        },
        "limits": plan["notes"],
    }
    if "task_interface" in plan:
        manifest["task_interface"] = plan["task_interface"]
    import shutil

    for relative in manifest["source"]["files_sha256"]:
        target = run_dir / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    recorder.write("plan.json", plan)
    recorder.write("config.json", config)
    recorder.write("manifest.json", manifest)
    try:
        from prompt_policy.backends import LocalHFBackend, VLLMBackend

        if optimizer_name == "edit_refinement":
            from prompt_policy.edit_optimizer import search
        elif optimizer_name == "textgrad":
            from prompt_policy.textgrad_optimizer import search
        else:
            from prompt_policy.optimizer import search

        backend = (VLLMBackend if "serving" in config else LocalHFBackend)(config, recorder)
        manifest["model"] = backend.identity
        recorder.write("manifest.json", manifest)
        result = search(benchmark, backend, recorder, config)
        recorder.write("candidates.json", result.to_dict())
        selected = result.best_candidate["system_prompt"]
        # Selection is frozen and written before the test split is touched.
        recorder.write(
            "selection.json",
            {
                "candidate_idx": result.best_idx,
                "prompt": selected,
                "criterion": "fixed_initial_prompt_probe"
                if probe_mode
                else "last_likelihood_and_validation_verified_prompt"
                if optimizer_name == "edit_refinement"
                else "validation_accuracy",
            },
        )
        search_cost = dict(recorder.cost)
        tests = {}
        if probe_mode:
            print("Probe complete: fixed base retained; validation and test not run", flush=True)
        else:
            recorder.write("progress.json", {"phase": "test", "selected_candidate_idx": result.best_idx})
            print(f"Search complete: selected candidate {result.best_idx}; starting frozen-prompt test", flush=True)
            for label, prompt in [("initial", config["initial_prompt"]), ("selected", selected)]:
                if label == "selected" and selected == config["initial_prompt"]:
                    tests[label] = {**tests["initial"], "reused_initial": True}
                else:
                    tests[label] = evaluate_test(
                        benchmark, backend, recorder, prompt, config["search"]["test_rollouts"], label
                    )
        recorder.write(
            "result.json",
            {
                "best_candidate_idx": result.best_idx,
                "selected_prompt": selected,
                "validation_accuracy": result.val_aggregate_scores[result.best_idx],
                **(
                    {
                        "iterations_completed": result.iterations_completed,
                        "termination_reason": result.termination_reason,
                    }
                    if optimizer_name == "edit_refinement"
                    else {}
                ),
                **({"probe_summary": result.probe_summary,
                    "evaluation_status": {"validation": "not_run", "test": "not_run"}}
                   if probe_mode else {}),
                "test": tests,
                "search_cost": search_cost,
                "total_cost": dict(recorder.cost),
                (
                    f"{optimizer_name}_total_metric_calls"
                    if optimizer_name in {"textgrad", "edit_refinement"}
                    else "gepa_total_metric_calls"
                ): result.total_metric_calls,
                "budget_counter": "search_cost.task_calls (actual individual task rollouts)",
            },
        )
        manifest["status"] = "complete"
        recorder.write("progress.json", {"phase": "complete", "selected_candidate_idx": result.best_idx})
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["cost"] = dict(recorder.cost)
        recorder.write("manifest.json", manifest)
    print(f"Completed: {run_dir}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Independent prompt policy optimization")
    parser.add_argument("config", type=Path, help="Standalone experiment TOML")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and report; no model load or generation"
    )
    parser.add_argument("--output", type=Path, help="New run directory; existing directories are refused")
    args = parser.parse_args()
    try:
        run(args.config, dry_run=args.dry_run, output=args.output)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"error: {error}\n")
