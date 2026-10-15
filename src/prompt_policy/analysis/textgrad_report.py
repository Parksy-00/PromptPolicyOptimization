"""Verify a completed TextGrad likelihood audit and write its cross-run report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from prompt_policy.analysis.textgrad_likelihood import native_reward, read_jsonl, write_csv, write_json
from prompt_policy.artifacts import digest, file_sha256


def audit_validation(plan):
    """Check actual requests and reward identity; preserve original vs repeat distinctions."""
    result = []
    for source in plan["sources"]:
        root = Path(source["root"])
        native = root / "native"
        summaries = [
            root / name for name in source["files_sha256"] if name.endswith("/validation_summary.json")
        ]
        if len(summaries) != 1:
            raise ValueError("Expected one declared repeat-validation input")
        repeat_root = summaries[0].parent
        for name, checksum in source["files_sha256"].items():
            if file_sha256(root / name) != checksum:
                raise ValueError("Source input changed after scoring: " + name)
        repeat_summary = json.loads(summaries[0].read_text())
        cands = {r["candidate_id"]: r for r in read_jsonl(native / "candidates.jsonl")}
        original, repeated = defaultdict(Counter), defaultdict(Counter)
        questions, settings, repeat_settings = {}, {}, {}
        mismatches = Counter()
        for role, path, counts in [
            ("original", native / "responses.jsonl", original),
            ("repeat", repeat_root / "responses.jsonl", repeated),
        ]:
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["split"] != "validation":
                        continue
                    cid = row["candidate_id"]
                    counts[cid].update(
                        total=1,
                        correct=int(row["score"] == 1),
                        format_success=int(row["format_success"] == 1),
                        length=int(row["finish_reason"] == "length"),
                    )
                    mismatches[role + "_prompt"] += row["messages"][0]["content"] != cands[cid]["instruction"]
                    if role == "original":
                        questions[row["sample_id"]] = row["messages"][1]["content"]
                    else:
                        mismatches["repeat_question"] += (
                            row["messages"][1]["content"] != questions[row["sample_id"]]
                        )
                        if (
                            native_reward(row.get("completion"), row["correct_answer"], source["benchmark"])
                            != row["score"]
                        ):
                            raise ValueError("Repeat reward does not reproduce the source native scorer")
                    request = row["provider_call"]["request"]
                    mismatches[role + "_request_messages"] += request.get("messages") != row["messages"]
                    scientific = {
                        k: request.get(k)
                        for k in [
                            "temperature",
                            "top_p",
                            "top_k",
                            "max_tokens",
                            "chat_template_kwargs",
                            "presence_penalty",
                            "frequency_penalty",
                            "repetition_penalty",
                        ]
                    }
                    target = settings if role == "original" else repeat_settings
                    target[digest(scientific)] = scientific
                    counts[cid]["requests_with_sampling_seed"] += request.get("seed") is not None
        if any(mismatches.values()):
            raise ValueError("Original/repeated validation attribution mismatch")
        if set(repeated) != set(cands):
            raise ValueError("Repeat candidate inventory differs from optimization candidates")
        repeat_by = {r["candidate_id"]: r for r in repeat_summary["candidates"]}
        expected_total = repeat_summary["num_examples"] * repeat_summary["repeats"]
        for cid, counts in repeated.items():
            if (
                counts["total"] != expected_total
                or abs(counts["correct"] / counts["total"] - repeat_by[cid]["mean_success_rate"]) > 1e-12
            ):
                raise ValueError("Repeat response rewards disagree with archived summary")
        result.append(
            {
                "run_id": root.name,
                "benchmark": source["benchmark"],
                "original": dict(original),
                "repeat8": dict(repeated),
                "original_sampling": list(settings.values()),
                "repeat_sampling": list(repeat_settings.values()),
                "mismatches": dict(mismatches),
                "source_sha256": {
                    "native_responses": file_sha256(native / "responses.jsonl"),
                    "repeat_responses": file_sha256(repeat_root / "responses.jsonl"),
                },
                "note": "Original requests were unseeded; historical repeat8 requests carried seeds. "
                "The original sampler implementation is not certified by these artifacts.",
            }
        )
    return result


def verify_tokens(run):
    index = read_jsonl(run / "score_index.jsonl")
    fixed, seen = {}, set()
    count, tokens = 0, 0
    for row in index:
        if row["key"] in seen:
            raise ValueError("Duplicate scoring context in token index")
        seen.add(row["key"])
        path = run / row["token_file"]
        if file_sha256(path) != row["sha256"]:
            raise ValueError("Token cache checksum mismatch")
        with np.load(path) as data:
            scores, response = data["scores"], data["response_ids"]
            if scores.shape != (3, len(response)) or not np.isfinite(scores[:2]).all():
                raise ValueError("Invalid token score shape or values")
            response_hash = digest(response.tolist())
            key = row["key"].rsplit(":", 1)[0]
            if (
                response_hash != row["response_sha256"]
                or fixed.setdefault(key, response_hash) != response_hash
            ):
                raise ValueError("A reference response changed across scoring prompts")
            for i, metric in enumerate(["t1", "t06"]):
                independently_summed = math.fsum(float(x) for x in scores[i])
                if abs(independently_summed - row["metrics"][metric]["sum"]) > 1e-7:
                    raise ValueError("Token aggregation mismatch")
                if abs(independently_summed / len(response) - row["metrics"][metric]["mean"]) > 1e-10:
                    raise ValueError("Mean token aggregation mismatch")
            tokens += len(response)
            count += 1
    return {
        "verified_files": count,
        "verified_response_token_instances": tokens,
        "fixed_reference_trajectories": len(fixed),
        "all_file_hashes_match": True,
        "response_ids_identical_across_contexts": True,
        "independent_token_aggregation_matches": True,
    }


def ranks(values):
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        result[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return result


def descriptive_correlations(rows):
    result = []
    for source in sorted({r["source"] for r in rows}):
        validation = [r for r in rows if r["source"] == source and r["cohort"] == "validation"]
        n = int(validation[0]["questions"])
        for cohort in ["train", "validation"]:
            selected = [r for r in rows if r["source"] == source and r["cohort"] == cohort]
            if not selected:
                continue
            x = np.array([float(r["reward_weighted_t1_sum"]) for r in selected])
            for target, scale in [("delta_validation", n), ("delta_repeat_validation", n * 8)]:
                # Reconstruct integer success-count differences to preserve exact rank ties.
                y = np.array([round(float(r[target]) * scale) for r in selected])
                rho = float(np.corrcoef(ranks(x), ranks(y))[0, 1]) if np.ptp(x) and np.ptp(y) else None
                result.append(
                    {
                        "source": source,
                        "cohort": cohort,
                        "target": target,
                        "proposals": len(selected),
                        "spearman_descriptive": rho,
                        "inferential_pvalue": None,
                        "note": "Descriptive only; proposals share parents and validation questions.",
                    }
                )
    return result


def generate_report(run, report_root):
    run, report_root = Path(run).resolve(), Path(report_root).resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Only a completed audit can be reported")
    plan = json.loads((run / "plan.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    from prompt_policy.analysis import textgrad_likelihood

    if file_sha256(Path(textgrad_likelihood.__file__)) != plan["analysis_source_sha256"]:
        raise ValueError("Analysis implementation differs from the scored run")
    with (run / "update_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    proof = verify_tokens(run)
    if proof["verified_files"] != plan["scoring_jobs"]:
        raise ValueError("Incomplete score inventory")
    links = json.loads((run / "links.json").read_text())
    expected_keys = {
        link["row_key"] + ":" + link[role] for link in links for role in ["parent_id", "candidate_id"]
    }
    actual_keys = {r["key"] for r in read_jsonl(run / "score_index.jsonl")}
    if actual_keys != expected_keys:
        raise ValueError("Scoring contexts differ from the declared comparisons")
    audit = audit_validation(plan)
    correlations = descriptive_correlations(rows)
    report_root.mkdir(parents=True, exist_ok=False)
    write_json(report_root / "verification.json", proof)
    write_json(report_root / "validation_evaluation_audit.json", audit)
    write_csv(report_root / "descriptive_correlations.csv", correlations)
    with (run / "trajectory_changes.csv").open() as stream:
        trajectories = list(csv.DictReader(stream))
    source_by_transition = {t["transition_id"]: t["source"] for t in plan["transitions"]}
    for trajectory in trajectories:
        trajectory["source"] = source_by_transition[trajectory["transition_id"]]
    comparison_counts = []
    for source in sorted({r["source"] for r in trajectories}):
        for cohort in ["train", "validation"]:
            selected = [r for r in trajectories if r["source"] == source and r["cohort"] == cohort]
            good = [r for r in selected if float(r["reward"]) == 1]
            comparison_counts.append(
                {
                    "source": source,
                    "cohort": cohort,
                    "trajectory_comparisons": len(selected),
                    "successful_comparisons": len(good),
                    "successful_t1_sequence_increases": sum(float(r["delta_t1_sum"]) > 0 for r in good),
                    "note": "Comparisons share reference trajectories across proposals.",
                }
            )
    write_csv(report_root / "trajectory_sign_counts.csv", comparison_counts)
    sensitivity = []
    for source in sorted({r["source"] for r in rows}):
        for cohort in ["train", "validation"]:
            selected = [r for r in rows if r["source"] == source and r["cohort"] == cohort]
            item = {"source": source, "cohort": cohort, "updates": len(selected)}
            for metric in [
                "reward_weighted_t1_sum",
                "reward_weighted_t06_sum",
                "reward_weighted_t1_mean",
                "reward_weighted_t1_sum_stop_countmatched",
            ]:
                valid = [r for r in selected if r[metric] != ""]
                item[metric + "_valid_updates"] = len(valid)
                item[metric + "_positive_updates"] = sum(float(r[metric]) > 0 for r in valid)
            sensitivity.append(item)
    write_csv(report_root / "sensitivity_counts.csv", sensitivity)
    facts = {
        "run": str(run),
        "run_manifest_sha256": file_sha256(run / "manifest.json"),
        "analysis_source_sha256": plan["analysis_source_sha256"],
        "report_source_sha256": file_sha256(Path(__file__)),
        "counts": summary["update_counts"],
        "verification": proof,
        "correlations": correlations,
        "sensitivity_counts": sensitivity,
        "trajectory_comparison_counts": comparison_counts,
    }
    write_json(report_root / "completion_verified.json", facts)
    reconstruction = list(csv.DictReader((run / "reconstruction_audit.csv").open()))
    count_matched = sum(r["expected_count_match"] == "True" for r in reconstruction)
    cohort_counts = Counter(r["cohort"] for r in json.loads((run / "links.json").read_text()))
    lines = [
        "# TextGrad update likelihood: train and optimization-validation",
        "",
        "Completed on local physical GPU 1 with frozen Qwen3-8B revision "
        f"`{plan['model']['revision']}`. No new responses were generated.",
        "",
        "The primary statistic is mean original task reward times the change in preserved-response "
        "sequence logprob under child versus actual parent, at T=1. All accepted and rejected "
        "proposals are included. Parent-generated responses are fixed across scoring prompts.",
        "",
        "| Source | Cohort | Original decision | Updates | Positive weighted sequence change | "
        "Positive weighted token-mean change |",
        "|---|---|---|---:|---:|---:|",
    ]
    for c in summary["update_counts"]:
        lines.append(
            f"| {c['source']} | {c['cohort']} | {'accepted' if c['accepted'] else 'rejected'} | "
            f"{c['updates']} | {c['positive_reward_weighted_likelihood_updates']} | "
            f"{c['positive_mean_logprob_updates']} |"
        )
    lines += [
        "",
        "Individual successful-response increase counts are recorded in "
        "[trajectory signs](trajectory_sign_counts.csv); these comparisons share references.",
        "",
        "T=0.6 and normal-stop/count-matched sensitivity sign counts are recorded in "
        "[sensitivity counts](sensitivity_counts.csv). All sequence changes use natural-log units "
        "(nats) per question, averaged over successes and failures with failed rewards set to zero.",
        "",
        "## Actual accepted updates",
        "",
        "| Source / proposal | Train weighted sequence change | Validation weighted sequence change | "
        "Original validation change (pp) | Repeat8 validation change (pp) |",
        "|---|---:|---:|---:|---:|",
    ]
    keyed = {(r["transition_id"], r["cohort"]): r for r in rows}
    for r in rows:
        if r["cohort"] != "train" or r["accepted"] != "True":
            continue
        val = keyed[(r["transition_id"], "validation")]
        lines.append(
            f"| {r['transition_id']} | {float(r['reward_weighted_t1_sum']):+.4f} | "
            f"{float(val['reward_weighted_t1_sum']):+.4f} | {100 * float(r['delta_validation']):+.3f} | "
            f"{100 * float(r['delta_repeat_validation']):+.3f} |"
        )
    lines += [
        "",
        "## Interpretation and limitations",
        "",
        "Original validation decided whether the optimizer accepted or reverted each proposal; "
        "it is part of the optimization process. The separate repeat8 evaluation did not drive "
        "those decisions and must not replace the original scores when reconstructing the path.",
        "",
        f"There are {len(plan['transitions'])} correlated prompt proposals, {cohort_counts['train']} "
        f"train trajectory comparisons and {cohort_counts['validation']} full "
        f"validation trajectory comparisons. The {plan['reference_trajectories']} unique references "
        f"are scored in {plan['scoring_jobs']} "
        "contexts. More validation examples improve trajectory coverage, not the number of "
        "independent optimization runs.",
        "",
        "The repeat8 requests included historical request seeds; the original optimization "
        "requests did not. Recorded prompts, questions and declared temperature/top-p/top-k/token "
        "limits agree, but sampler/backend identity is not fully recorded. Several original "
        "candidate evaluations have much higher length termination rates than repeat8. Do not "
        "attribute their large accuracy differences to sampling variance alone or treat repeat8 "
        "as a certified replication of the original sampler.",
        "",
        "Response token IDs were not archived. Reasoning and final content were reconstructed "
        f"without whitespace trimming; no missing EOS was appended. {count_matched}/{len(reconstruction)} "
        "references agree with recorded token counts under the expected stop-token difference; "
        f"{len(reconstruction) - count_matched} do not. Count "
        "agreement is not proof of original token identity. Normal-stop/count-matched sensitivity "
        "is included in the update table.",
        "",
        "T=1 full-vocabulary likelihood, T=0.6 full-vocabulary likelihood and filtered sampler "
        "probabilities are different quantities. Filtered replay is deliberately undefined at "
        "top-k boundary ties and records tokens outside support; invalid trajectories are not "
        "silently averaged as finite sampler logprobs. This audit does not reconstruct an exact "
        "historical PPO loss or introduce new advantages/clipping parameters.",
        "",
        "Negative fixed-success likelihood changes do not imply that task accuracy must decrease. "
        "A finite prompt edit may replace one successful reasoning style with another. These "
        "measurements diagnose alignment with the proposed update signal; they do not prove "
        "that enforcing it improves future optimization. Counterfactual optimization paths "
        "cannot be recovered by filtering this recorded path alone.",
        "",
        "Original native scoring semantics are preserved, including DAPO's last-number parser "
        "and GPQA's first Answer:A-D match. Conditional question bootstrap intervals do not "
        "capture across-run, regeneration or model/backend uncertainty. Validation is not "
        "held-out test generalization evidence.",
        "",
        f"[All update metrics]({run}/update_summary.csv), "
        f"[trajectory changes]({run}/trajectory_changes.csv), "
        "[validation audit](validation_evaluation_audit.json), "
        "[verification](verification.json).",
        "",
        f"![Likelihood versus original validation]({run}/likelihood_vs_validation.png)",
        "",
    ]
    (report_root / "README.md").write_text("\n".join(lines))
    print(json.dumps(facts, indent=2))
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    generate_report(args.run, args.report)


if __name__ == "__main__":
    main()
