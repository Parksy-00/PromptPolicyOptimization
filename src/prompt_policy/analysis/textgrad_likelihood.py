"""Post-hoc TextGrad update likelihood audit using immutable source artifacts.

Run as python -m prompt_policy.analysis.textgrad_likelihood --config <analysis.toml>.
No generation, source-artifact mutation or external project imports.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

from prompt_policy.artifacts import digest, file_sha256


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def flat_ids(value):
    if hasattr(value, "keys"):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Expected one tokenized request")
        value = value[0]
    return list(value)


def reconstruct(row, tokenizer):
    """Decode-preserving text reconstruction, with explicit ambiguity/count audit.

    EOS is never appended. The original response IDs are unavailable. Count
    agreement does not establish exact token identity or original sampler replay.
    """
    reason, final = row.get("reasoning_content"), row.get("completion") or ""
    if reason is not None:
        if "<think>" in reason or "</think>" in reason:
            raise ValueError("Unexpected delimiters in the separated reasoning field")
        if final:
            raw, rule = "<think>" + reason + "</think>" + final, "closed_thinking"
        elif row["finish_reason"] == "length":
            raw, rule = "<think>" + reason, "length_open_thinking"
        else:
            raw, rule = "<think>" + reason + "</think>", "closed_thinking_empty_final"
    else:
        raw, rule = final, "content_only"
    ids = tokenizer.encode(raw, add_special_tokens=False)
    if not ids:
        raise ValueError("No preserved response text to score")
    saved = (row.get("usage") or {}).get("completion_tokens")
    delta = saved - len(ids) if isinstance(saved, int) else None
    return ids, {
        "rule": rule,
        "text_sha256": digest(raw),
        "response_tokens": len(ids),
        "saved_completion_tokens": saved,
        "saved_minus_reconstructed": delta,
        "expected_count_match": delta == (1 if row["finish_reason"] == "stop" else 0),
        "response_ids_originally_saved": False,
        "eos_appended": False,
    }


def native_reward(completion, answer, benchmark):
    """Reproduce each saved run's declared native parser, not a new canonical scorer."""
    import re

    text = completion or ""
    if benchmark == "GPQABench":
        match = re.search(r"Answer\s*:\s*([A-D])", text, re.IGNORECASE)
        return float(bool(match) and ord(match[1].upper()) - ord("A") == int(answer))
    words = [word for word in text.split() if any(c.isdigit() for c in word)]
    if not words:
        return 0.0
    digits = "".join(c for c in words[-1].split(".", 1)[0] if c.isdigit())
    try:
        return float(bool(digits) and int(digits) == int(answer))
    except ValueError:
        return 0.0


def prepare(config, tokenizer):
    sources, transitions, jobs, links, audits = [], [], {}, [], []
    total_tokens = 0
    for source in config["sources"]:
        root = Path(source["root"]).resolve()
        native = root / "native"
        evaluation = root / "evaluations" / source["repeat_evaluation"]
        required = [
            native / name
            for name in [
                "config.json",
                "splits.json",
                "candidates.jsonl",
                "proposals.jsonl",
                "steps.jsonl",
                "responses.jsonl",
                "gradients.jsonl",
                "summary.json",
            ]
        ]
        required += [evaluation / name for name in ["manifest.json", "validation_summary.json"]]
        inventory = {str(p.relative_to(root)): file_sha256(p) for p in required}
        cfg = json.loads((native / "config.json").read_text())
        summary = json.loads((native / "summary.json").read_text())
        if summary["status"] != "complete":
            raise ValueError("Only complete source runs are included")
        prov = cfg["provenance"]
        if prov.get("task_format") != "textgrad_native_v1":
            raise ValueError("Source is not a native-format TextGrad run")
        if prov["task_model"]["revision"] != config["model"]["revision"]:
            raise ValueError("Target model revision differs from source")
        sample = prov["task_model"]
        if (sample["temperature"], sample["top_p"], sample["extra_body"]["top_k"]) != (0.6, 0.95, 20):
            raise ValueError("Unexpected original sampling settings")
        splits = json.loads((native / "splits.json").read_text())
        examples = {s: {x["sample_id"]: x for x in xs} for s, xs in splits.items()}
        candidates = {x["candidate_id"]: x for x in read_jsonl(native / "candidates.jsonl")}
        proposals = read_jsonl(native / "proposals.jsonl")
        steps = {x["step"]: x for x in read_jsonl(native / "steps.jsonl")}
        rows = read_jsonl(native / "responses.jsonl")
        by_train, by_validation = defaultdict(list), defaultdict(list)
        for row in rows:
            if row["split"] not in {"train", "validation"}:
                raise ValueError("Optimization source unexpectedly contains test generations")
            ex = examples[row["split"]][row["sample_id"]]
            if native_reward(row.get("completion"), ex["answer"], source["benchmark"]) != row["score"]:
                raise ValueError("Saved reward differs from original native scoring rule")
            if row["split"] == "train":
                by_train[row["step"]].append(row)
            else:
                by_validation[row["candidate_id"]].append(row)
        for cid, candidate in candidates.items():
            vals = by_validation[cid]
            if {x["sample_id"] for x in vals} != set(examples["validation"]) or len(vals) != len(
                examples["validation"]
            ):
                raise ValueError("Incomplete original full-validation capture")
            actual = sum(x["score"] for x in vals) / len(vals)
            if abs(actual - candidate["validation_score"]) > 1e-12:
                raise ValueError("Candidate validation score mismatch")
        em = json.loads((evaluation / "manifest.json").read_text())
        if em["source_run_id"] != root.name:
            raise ValueError("Repeated evaluation attribution mismatch")
        for name, checksum in em["source_sha256"].items():
            if file_sha256(native / name) != checksum:
                raise ValueError("Repeat evaluation source checksum mismatch: " + name)
        repeat = json.loads((evaluation / "validation_summary.json").read_text())
        if repeat["status"] != "complete":
            raise ValueError("Repeated validation evaluation is incomplete")
        repeat_by = {x["candidate_id"]: x for x in repeat["candidates"]}
        sources.append(
            {
                "run_id": root.name,
                "benchmark": source["benchmark"],
                "root": str(root),
                "files_sha256": inventory,
                "seed": cfg["seed"],
                "task_model": {
                    k: sample[k] for k in ["model", "revision", "temperature", "top_p", "extra_body"]
                },
                "repeat_validation": repeat,
            }
        )
        for prop in proposals:
            cid, pid = prop["candidate_id"], prop["parent_id"]
            child, parent = candidates[cid], candidates[pid]
            if child["parent_ids"] != [pid] or child["instruction"] != prop["instruction"]:
                raise ValueError("Proposal lineage mismatch")
            if child["accepted"] != steps[prop["step"]]["accepted"]:
                raise ValueError("Accepted/reverted state mismatch")
            expected = child["validation_score"] >= parent["validation_score"]
            if expected != child["accepted"]:
                raise ValueError("Saved acceptance does not match the original validation-revert rule")
            batch = by_train[prop["step"]]
            if {x["sample_id"] for x in batch} != set(child["train_sample_ids"]):
                raise ValueError("Gradient train batch mismatch")
            transition_id = source["name"] + ":" + cid
            transitions.append(
                {
                    "transition_id": transition_id,
                    "source": source["name"],
                    "run_id": root.name,
                    "step": prop["step"],
                    "parent_id": pid,
                    "candidate_id": cid,
                    "accepted": child["accepted"],
                    "parent_validation": parent["validation_score"],
                    "candidate_validation": child["validation_score"],
                    "delta_validation": child["validation_score"] - parent["validation_score"],
                    "repeat_parent": repeat_by[pid],
                    "repeat_candidate": repeat_by[cid],
                }
            )
            for cohort, cohort_rows in [("train", batch), ("validation", by_validation[pid])]:
                for row in cohort_rows:
                    if row["candidate_id"] != pid or row["messages"][0]["content"] != parent["instruction"]:
                        raise ValueError("Reference trajectory did not come from the actual parent")
                    row_key = source["name"] + ":" + str(row["call_id"])
                    if row_key not in jobs:
                        response_ids, audit = reconstruct(row, tokenizer)
                        prefix = flat_ids(
                            tokenizer.apply_chat_template(
                                row["messages"],
                                tokenize=True,
                                add_generation_prompt=True,
                                enable_thinking=True,
                            )
                        )
                        if len(prefix) != row["usage"]["prompt_tokens"]:
                            raise ValueError("Reconstructed parent input length differs from the API record")
                        audit.update(
                            row_key=row_key,
                            source=source["name"],
                            call_id=row["call_id"],
                            split=cohort,
                            reward=row["score"],
                            finish_reason=row["finish_reason"],
                        )
                        audits.append(audit)
                        jobs[row_key] = {
                            "row_key": row_key,
                            "response_ids": response_ids,
                            "messages": row["messages"],
                            "contexts": {},
                            "source": source["name"],
                            "reward": row["score"],
                            "sample_id": row["sample_id"],
                            "split": cohort,
                            "finish_reason": row["finish_reason"],
                            "audit": audit,
                        }
                    job = jobs[row_key]
                    for candidate in [parent, child]:
                        context = candidate["candidate_id"]
                        if context not in job["contexts"]:
                            messages = [dict(m) for m in job["messages"]]
                            messages[0]["content"] = candidate["instruction"]
                            prefix = flat_ids(
                                tokenizer.apply_chat_template(
                                    messages, tokenize=True, add_generation_prompt=True, enable_thinking=True
                                )
                            )
                            if len(prefix) + len(job["response_ids"]) > config["model"]["max_context_tokens"]:
                                raise ValueError("Context overflow; no truncation is permitted")
                            job["contexts"][context] = prefix
                            total_tokens += len(prefix) + len(job["response_ids"])
                    links.append(
                        {
                            "transition_id": transition_id,
                            "cohort": cohort,
                            "row_key": row_key,
                            "parent_id": pid,
                            "candidate_id": cid,
                            "sample_id": row["sample_id"],
                            "reward": row["score"],
                            "finish_reason": row["finish_reason"],
                            "reconstruction_count_match": job["audit"]["expected_count_match"],
                        }
                    )
    plan = {
        "sources": sources,
        "transitions": transitions,
        "reference_trajectories": len(jobs),
        "scoring_jobs": sum(len(j["contexts"]) for j in jobs.values()),
        "input_tokens_across_jobs": total_tokens,
        "comparison_rows": len(links),
        "reconstruction": dict(Counter((a["rule"], a["expected_count_match"]) for a in audits)),
    }
    plan["reconstruction"] = {str(k): v for k, v in plan["reconstruction"].items()}
    return plan, jobs, links, audits


def filtered_logprobs(logits, labels, temperature=0.6, top_k=20, top_p=0.95):
    """Sequential top-k then top-p replay convention; not a claim about old backend identity.

    Detect top-k boundary ties explicitly; leave their replay value undefined.
    Other tokens outside support receive -inf. Full-vocabulary metrics remain valid.
    """
    import torch

    values, indices = torch.topk(logits.float() / temperature, top_k + 1, dim=-1)
    ties = values[:, top_k - 1] == values[:, top_k]
    values, indices = values[:, :top_k], indices[:, :top_k]
    # Ascending cumulative <=1-p follows the native sequential top-k/top-p convention.
    ascending = values.flip(-1)
    discard = ascending.softmax(-1).cumsum(-1) <= 1 - top_p
    discard[:, -1] = False
    values = ascending.masked_fill(discard, -torch.inf).flip(-1)
    lp = values.log_softmax(-1)
    found = indices == labels[:, None]
    target = lp.masked_fill(~found, -torch.inf).amax(-1)
    target[ties] = torch.nan
    return target


def score_context(model, prompt_ids, response_ids, head_chunk):
    import numpy as np
    import torch

    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        hidden = model.base_model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
        offset = len(prompt_ids) - 1
        hidden = hidden[0, offset : offset + len(response_ids)]
        labels = torch.tensor(response_ids, device=device)
        traces = [[], [], []]
        for start in range(0, len(response_ids), head_chunk):
            stop = min(start + head_chunk, len(response_ids))
            logits = model.get_output_embeddings()(hidden[start:stop]).float()
            chosen = labels[start:stop]
            for target, temperature in zip(traces[:2], [1.0, 0.6], strict=True):
                lp = (logits / temperature).log_softmax(-1)
                target.append(lp.gather(-1, chosen[:, None]).flatten())
            traces[2].append(filtered_logprobs(logits, chosen))
        arrays = torch.stack([torch.cat(t) for t in traces]).cpu().numpy()
    if not np.isfinite(arrays[:2]).all():
        raise ValueError("Invalid full-vocabulary score")
    return arrays


def summarize_arrays(arrays):
    import numpy as np

    result = {}
    for i, name in enumerate(["t1", "t06", "filtered"]):
        values = arrays[i]
        valid = np.isfinite(values).all()
        result[name] = {
            "sum": float(values.sum(dtype=np.float64)) if valid else None,
            "mean": float(values.mean(dtype=np.float64)) if valid else None,
            "outside_support": int(np.isneginf(values).sum()),
            "boundary_ties": int(np.isnan(values).sum()),
        }
    return result


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(output, plan, links, cache, config):
    import numpy as np

    transitions = {t["transition_id"]: t for t in plan["transitions"]}
    comparison = []
    grouped = defaultdict(list)
    for link in links:
        old = cache[link["row_key"] + ":" + link["parent_id"]]
        new = cache[link["row_key"] + ":" + link["candidate_id"]]
        row = {**link}
        for metric in ["t1", "t06", "filtered"]:
            for kind in ["sum", "mean"]:
                a, b = old[metric][kind], new[metric][kind]
                row[f"delta_{metric}_{kind}"] = b - a if a is not None and b is not None else None
            row[f"parent_{metric}_outside"] = old[metric]["outside_support"]
            row[f"child_{metric}_outside"] = new[metric]["outside_support"]
        comparison.append(row)
        grouped[(link["transition_id"], link["cohort"])].append(row)
    write_csv(output / "trajectory_changes.csv", comparison)
    aggregate = []
    rng = np.random.default_rng(config["statistics"]["bootstrap_seed"])
    for (tid, cohort), rows in grouped.items():
        tr = transitions[tid]
        good = [r for r in rows if r["reward"] == 1]
        bad = [r for r in rows if r["reward"] == 0]
        result = {
            k: tr[k]
            for k in [
                "transition_id",
                "source",
                "step",
                "parent_id",
                "candidate_id",
                "accepted",
                "parent_validation",
                "candidate_validation",
                "delta_validation",
            ]
        }
        result.update(cohort=cohort, questions=len(rows), successful_questions=len(good))
        for metric in ["t1", "t06"]:
            for kind in ["sum", "mean"]:
                values = np.array([r[f"delta_{metric}_{kind}"] * r["reward"] for r in rows])
                result[f"reward_weighted_{metric}_{kind}"] = float(values.mean())
                result[f"success_{metric}_{kind}"] = (
                    float(np.mean([r[f"delta_{metric}_{kind}"] for r in good])) if good else None
                )
                result[f"failure_{metric}_{kind}"] = (
                    float(np.mean([r[f"delta_{metric}_{kind}"] for r in bad])) if bad else None
                )
        values = np.array([r["delta_t1_sum"] * r["reward"] for r in rows])
        boot = np.array(
            [
                rng.choice(values, size=len(values), replace=True).mean()
                for _ in range(config["statistics"]["bootstrap_repeats"])
            ]
        )
        result["surrogate_t1_ci_low"], result["surrogate_t1_ci_high"] = map(
            float, np.quantile(boot, [0.025, 0.975])
        )
        result["success_trajectory_increase_fraction"] = (
            sum(r["delta_t1_sum"] > 0 for r in good) / len(good) if good else None
        )
        result["parent_filtered_unsupported_trajectories"] = sum(
            r["parent_filtered_outside"] > 0 for r in rows
        )
        result["child_filtered_unsupported_trajectories"] = sum(r["child_filtered_outside"] > 0 for r in rows)
        result["filtered_valid_success_pairs"] = sum(r["delta_filtered_sum"] is not None for r in good)
        # Repeat fields vary by historical schema; preserve full originals in plan.json.
        for role in ["parent", "candidate"]:
            repeat = tr["repeat_" + role]
            for key in ["mean_success_rate", "mean", "mean_accuracy", "accuracy", "success_rate"]:
                if isinstance(repeat.get(key), (int, float)):
                    result["repeat_" + role + "_accuracy"] = repeat[key]
                    break
        if "repeat_parent_accuracy" in result and "repeat_candidate_accuracy" in result:
            result["delta_repeat_validation"] = (
                result["repeat_candidate_accuracy"] - result["repeat_parent_accuracy"]
            )
        clean = [r for r in rows if r["finish_reason"] == "stop" and r["reconstruction_count_match"]]
        clean_good = [r for r in clean if r["reward"] == 1]
        result["stop_countmatched_questions"] = len(clean)
        result["stop_countmatched_successes"] = len(clean_good)
        result["reward_weighted_t1_sum_stop_countmatched"] = (
            float(np.mean([r["reward"] * r["delta_t1_sum"] for r in clean])) if clean else None
        )
        result["success_t1_sum_stop_countmatched"] = (
            float(np.mean([r["delta_t1_sum"] for r in clean_good])) if clean_good else None
        )
        aggregate.append(result)
    write_csv(output / "update_summary.csv", aggregate)
    counts = []
    for source in sorted({r["source"] for r in aggregate}):
        for cohort in ["train", "validation"]:
            rows = [r for r in aggregate if r["source"] == source and r["cohort"] == cohort]
            for accepted in [True, False]:
                selected = [r for r in rows if r["accepted"] == accepted]
                counts.append(
                    {
                        "source": source,
                        "cohort": cohort,
                        "accepted": accepted,
                        "updates": len(selected),
                        "positive_reward_weighted_likelihood_updates": sum(
                            r["reward_weighted_t1_sum"] > 0 for r in selected
                        ),
                        "positive_mean_logprob_updates": sum(
                            r["reward_weighted_t1_mean"] > 0 for r in selected
                        ),
                    }
                )
    write_json(
        output / "summary.json",
        {
            "update_counts": counts,
            "updates": aggregate,
            "limits": [
                "Retokenized separated reasoning/final text; original token IDs and stop token unavailable.",
                "No EOS appended. Scored values describe the preserved response prefix.",
                "Filtered values are sequential top-k/top-p replay diagnostics, not certified original sampler logprobs.",
                "Validation guided original prompt selection; it is not held-out generalization evidence.",
                "17 correlated prompt proposals; extra validation rows do not create independent optimizer runs.",
                "Question bootstrap is conditional on recorded responses and does not propagate generation uncertainty.",
            ],
        },
    )
    make_plot(output, aggregate)
    return counts


def make_plot(output, aggregate):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, cohort in zip(axes, ["train", "validation"], strict=True):
        for source, marker in [("dapo", "o"), ("gpqa", "s")]:
            for accepted, color in [(True, "#247b4a"), (False, "#be4646")]:
                rows = [
                    r
                    for r in aggregate
                    if r["source"] == source and r["cohort"] == cohort and r["accepted"] == accepted
                ]
                if rows:
                    ax.scatter(
                        [r["reward_weighted_t1_sum"] for r in rows],
                        [100 * r["delta_validation"] for r in rows],
                        color=color,
                        marker=marker,
                        label=f"{source} {'accepted' if accepted else 'rejected'}",
                        s=48,
                    )
        ax.axvline(0, color="0.6", linewidth=0.8)
        ax.axhline(0, color="0.6", linewidth=0.8)
        ax.set_title(f"{cohort}: fixed parent trajectories")
        ax.set_xlabel("Reward-weighted sequence logprob change (T=1)")
        ax.set_ylabel("Original validation accuracy change (pp)")
        ax.legend(fontsize=7)
    fig.savefig(output / "likelihood_vs_validation.png", dpi=180)
    fig.savefig(output / "likelihood_vs_validation.pdf")
    plt.close(fig)


def run(config_path, dry_run=False, resume=False):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config_path = Path(config_path).resolve()
    config = tomllib.loads(config_path.read_text())
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("Use CUDA_VISIBLE_DEVICES=1; no other GPU may be exposed")
    if config["execution"]["physical_gpu"] != 1:
        raise ValueError("This approved analysis is restricted to physical GPU 1")
    torch.set_num_threads(config["execution"]["cpu_threads"])
    model_path = Path(config["model"]["path"]).resolve()
    if model_path.name != config["model"]["revision"]:
        raise ValueError("Local model path does not identify the pinned snapshot")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    started = time.perf_counter()
    plan, jobs, links, audits = prepare(config, tokenizer)
    plan["config_sha256"] = file_sha256(config_path)
    plan["analysis_source_sha256"] = file_sha256(Path(__file__))
    plan["model"] = config["model"]
    plan["design"] = config["design"]
    output = Path(config["execution"]["output"]).resolve()
    if dry_run:
        print(json.dumps({k: v for k, v in plan.items() if k not in {"sources", "transitions"}}, indent=2))
        return plan
    if resume:
        if json.loads((output / "plan.json").read_text()) != plan:
            raise ValueError("Resume plan/input identities differ")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "analysis.toml").write_bytes(config_path.read_bytes())
        write_json(output / "plan.json", plan)
        write_csv(output / "reconstruction_audit.csv", audits)
        write_json(output / "links.json", links)
    cache_path = output / "score_index.jsonl"
    cache = {r["key"]: r["metrics"] for r in read_jsonl(cache_path)} if cache_path.exists() else {}
    native = output / "tokens"
    native.mkdir(exist_ok=True)
    model_files = {
        str(p.relative_to(model_path)): file_sha256(p)
        for p in sorted(model_path.iterdir())
        if p.is_file() and p.suffix in {".json", ".jinja", ".safetensors"}
    }
    manifest = {
        "status": "running",
        "model_files_sha256": model_files,
        "config": config,
        "physical_gpu": 1,
        "visible_device": "cuda:0",
        "torch": torch.__version__,
    }
    write_json(output / "manifest.json", manifest)
    print(
        f"PLAN: {len(plan['transitions'])} proposals; {plan['scoring_jobs']} scoring contexts; "
        f"{plan['input_tokens_across_jobs']:,} input tokens; {len(cache)} cached",
        flush=True,
    )
    try:
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_path, dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=False
            )
            .to("cuda:0")
            .eval()
        )
        model.requires_grad_(False)
        work = [(j, cid, prefix) for j in jobs.values() for cid, prefix in j["contexts"].items()]
        # Train diagnostics first, then short-to-long validation scoring. Never drop long responses.
        work.sort(key=lambda w: (w[0]["split"] != "train", len(w[2]) + len(w[0]["response_ids"])))
        done = len(cache)
        for job, cid, prefix in work:
            key = job["row_key"] + ":" + cid
            if key in cache:
                continue
            arrays = score_context(model, prefix, job["response_ids"], config["execution"]["head_chunk"])
            token_path = native / (digest(key) + ".npz")
            np.savez_compressed(
                token_path,
                scores=arrays,
                response_ids=np.asarray(job["response_ids"], dtype=np.int32),
                prompt_ids=np.asarray(prefix, dtype=np.int32),
            )
            metrics = summarize_arrays(arrays)
            record = {
                "key": key,
                "metrics": metrics,
                "token_file": str(token_path.relative_to(output)),
                "sha256": file_sha256(token_path),
                "response_sha256": digest(job["response_ids"]),
            }
            with cache_path.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            cache[key] = metrics
            done += 1
            if done % 20 == 0 or done == len(work):
                elapsed = time.perf_counter() - started
                write_json(
                    output / "progress.json", {"done": done, "total": len(work), "elapsed_seconds": elapsed}
                )
                print(f"PROGRESS {done}/{len(work)} elapsed={elapsed:.1f}s", flush=True)
        counts = report(output, plan, links, cache, config)
        manifest["status"] = "complete"
        manifest["counts"] = counts
        print(json.dumps(counts), flush=True)
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["completed_scoring_jobs"] = len(cache)
        write_json(output / "manifest.json", manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.config, args.dry_run, args.resume)


if __name__ == "__main__":
    main()
