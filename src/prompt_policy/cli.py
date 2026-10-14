from __future__ import annotations

import argparse
import importlib.metadata
import json
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
    paths = sorted((root / "src").rglob("*.py")) + [root / "pyproject.toml"]
    return {"git_commit": revision, "files_sha256": {str(p.relative_to(root)): file_sha256(p) for p in paths}}


def evaluate_test(benchmark, backend, recorder, prompt, repeats, label):
    results, per_repeat = [], [[] for _ in range(repeats)]
    for example in benchmark.test:
        values = []
        for repeat in range(repeats):
            generated = backend.generate(prompt, example.question)
            score = benchmark.score(
                example, generated.text, thinking_open_in_prefix=generated.thinking_open_in_prefix
            )
            values.append(score)
            per_repeat[repeat].append(score)
            recorder.append(
                "test.jsonl",
                {
                    "condition": label,
                    "prompt": prompt,
                    "example_id": example.id,
                    "question": example.question,
                    "split": "test",
                    "repeat_index": repeat,
                    "text": generated.text,
                    "reward": score,
                    "finish_reason": generated.finish_reason,
                    "thinking_open_in_prefix": generated.thinking_open_in_prefix,
                    "prompt_ids": generated.prompt_ids,
                    "response_ids": generated.response_ids,
                },
            )
        results.append({"example_id": example.id, "accuracy": statistics.mean(values)})
    repeat_means = [statistics.mean(values) for values in per_repeat]
    return {
        "accuracy": statistics.mean(repeat_means),
        "repeat_accuracies": repeat_means,
        "repeat_std": statistics.stdev(repeat_means) if repeats > 1 else None,
        "questions": len(results),
        "rollouts": len(results) * repeats,
        "per_question": results,
    }


def run(config_path: Path, *, dry_run=False, output: Path | None = None):
    config_path = config_path.resolve()
    root = Path(__file__).resolve().parents[2]
    config, benchmark = load_config(config_path)
    config, missing = resolve_model(config, config_path.parent)
    upstream = verify_gepa(root)
    plan = {
        "executable": not missing,
        "missing": missing,
        "config": config,
        "benchmark": benchmark.identity,
        "gepa": upstream,
        "objective": "delta_accuracy + weight * question_balanced_mean_delta_logprob_T1",
        "final_selection": "validation_accuracy_only",
        "notes": [
            "No sampling seeds are sent; search_seed controls GEPA search only.",
            "Budget checked between complete iterations; the last iteration can exceed it.",
            "Built-in JSONL scorers do not claim canonical DAPO/GPQA equivalence.",
        ],
    }
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
        "gepa": upstream,
        "environment": {
            "python": sys.version,
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        },
        "limits": plan["notes"],
    }
    recorder.write("config.json", config)
    recorder.write("manifest.json", manifest)
    try:
        from prompt_policy.backends import LocalHFBackend
        from prompt_policy.optimizer import search

        backend = LocalHFBackend(config, recorder)
        manifest["model"] = backend.identity
        recorder.write("manifest.json", manifest)
        result = search(benchmark, backend, recorder, config)
        recorder.write("candidates.json", result.to_dict())
        selected = result.best_candidate["system_prompt"]
        # Selection is frozen and written before the test split is touched.
        recorder.write(
            "selection.json",
            {"candidate_idx": result.best_idx, "prompt": selected, "criterion": "validation_accuracy"},
        )
        search_cost = dict(recorder.cost)
        tests = {}
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
                "test": tests,
                "search_cost": search_cost,
                "total_cost": dict(recorder.cost),
                "gepa_total_metric_calls": result.total_metric_calls,
                "budget_counter": "search_cost.task_calls (actual individual task rollouts)",
            },
        )
        manifest["status"] = "complete"
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
    parser = argparse.ArgumentParser(description="Independent GEPA prompt policy optimization")
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
