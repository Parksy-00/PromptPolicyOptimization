"""CPU audit of archived top-k logprobs under the rollout sampling policy.

python -m prompt_policy.analysis.sampling_policy --config analysis.toml --output RUN_DIR

Uses immutable exact-ID vLLM response archives, without model calls. Top-k ties
outside the archive can prevent exact normalization. A residual-mass bound marks
support decisions certain/ambiguous rather than silently treating top-k as complete.
Finite logprobs are explicitly nominal (no missing cutoff ties), not sampler replay.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from prompt_policy.artifacts import file_sha256


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _retained_distribution(weights, missing_count, cutoff_weight, top_p):
    """Aggregate missing cutoff ties without materializing a vocabulary-sized array."""
    groups = Counter(weights)
    groups[cutoff_weight] += missing_count
    total = math.fsum(w * n for w, n in groups.items())
    allowance = (1 - top_p) * total
    removed, removed_count = 0.0, 0
    for weight, number in sorted(groups.items()):
        count = min(number, max(0, math.floor((allowance - removed) / weight)))
        removed += count * weight
        removed_count += count
        if count < number:
            break
    return total - removed, sum(groups.values()) - removed_count


def distribution_statistics(row, *, temperature, top_k, top_p, vocab_size,
                            probability_threshold, mass_tolerance=1e-5, cdf_tolerance=1e-6):
    """Bound max probability over all feasible missing top-k boundary ties.

    Adding tokens at the cutoff cannot reduce the final retained normalizer:
    they either remain in the removed low tail, become retained themselves, or
    make another existing token remain retained. Thus the endpoint normalizers
    bound pmax after nucleus renormalization as well as before it.
    """
    values = sorted((float(v["logprob"]) for v in row.values()), reverse=True)
    if len(values) < top_k:
        raise ValueError("Insufficient archived alternatives for branch analysis")
    cutoff, maximum = values[top_k - 1], values[0]
    known_mass = math.fsum(math.exp(v) for v in values)
    if known_mass > 1 + mass_tolerance:
        raise ValueError("Archived raw probabilities exceed one")
    missing = min(vocab_size - len(values), math.floor(
        max(0.0, 1 - known_mass + mass_tolerance) / math.exp(cutoff)
    ))
    weights = [math.exp((v - maximum) / temperature) for v in values if v >= cutoff]
    cutoff_weight = math.exp((cutoff - maximum) / temperature)
    z0, n0 = _retained_distribution(weights, 0, cutoff_weight, top_p)
    z1, n1 = _retained_distribution(weights, missing, cutoff_weight, top_p)
    if z1 < z0 - 1e-10:
        raise ValueError("Missing-tie normalizer bound is not monotone")
    lower, upper = 1 / z1, 1 / z0
    uncertain = lower <= probability_threshold + cdf_tolerance and upper >= probability_threshold - cdf_tolerance
    return {
        "pmax_nominal": upper, "pmax_lower": lower, "pmax_upper": upper,
        "branch_nominal": upper < probability_threshold,
        "branch_uncertain": uncertain,
        "support_size_nominal": n0, "support_size_upper": n1,
        "max_missing_cutoff_ties": missing,
    }


def compare_branch_masks(rows):
    """Count position changes relative to the same prompt's T1 full-vocabulary mask."""
    summary = {}
    for scope, subset in [("full", rows), ("thinking", [r for r in rows if r["scope"] == "thinking"]),
                          ("response", [r for r in rows if r["scope"] == "response"])]:
        base = [r["branch_distributions"]["T1_full"]["branch_nominal"] for r in subset]
        summary[scope] = {}
        for name in subset[0]["branch_distributions"]:
            values = [r["branch_distributions"][name] for r in subset]
            masks = [v["branch_nominal"] for v in values]
            summary[scope][name] = {
                "tokens": len(subset), "branch_tokens": sum(masks),
                "branch_fraction": sum(masks) / len(subset),
                "uncertain_positions": sum(v.get("branch_uncertain", False) for v in values),
                "removed_from_T1": sum(a and not b for a, b in zip(base, masks, strict=True)),
                "added_to_T1": sum(not a and b for a, b in zip(base, masks, strict=True)),
                "singleton_support_nominal": sum(v.get("support_size_nominal") == 1 for v in values),
                "mean_support_size_nominal": (
                    math.fsum(v["support_size_nominal"] for v in values) / len(values)
                    if name != "T1_full" else None
                ),
            }
    return summary


def policy_token(row, token_id, *, temperature, top_k, top_p, vocab_size,
                 mass_tolerance=1e-5, cdf_tolerance=1e-6):
    """Bound support, then compute a labeled top-k-archive nominal logprob.

    vLLM 0.11.1 native order: temperature, top-k cutoff (retaining ties),
    ascending cumulative probability <= 1-top_p masked, then renormalization.
    LPs are raw full-vocabulary LPs; their common additive offset cancels.
    Unknown tokens at the kth value are bounded by the remaining raw probability.
    Equal-value nucleus-boundary tokens are conservatively ambiguous.
    """
    if not 0 < temperature or not 0 < top_p <= 1 or not 1 <= top_k <= vocab_size:
        raise ValueError("Invalid sampling policy")
    entries = {int(k): float(v["logprob"]) for k, v in row.items()}
    if token_id not in entries or len(entries) < top_k:
        raise ValueError("Archive must contain actual token and at least top_k alternatives")
    if any(not math.isfinite(v) or v > 0 for v in entries.values()):
        raise ValueError("Expected finite raw full-vocabulary logprobs")
    ordered = sorted(entries.values(), reverse=True)
    cutoff, lp, maximum = ordered[top_k - 1], entries[token_id], ordered[0]
    known_mass = math.fsum(math.exp(v) for v in entries.values())
    if known_mass > 1 + mass_tolerance:
        raise ValueError("Archived logprobs exceed full-vocabulary probability mass")
    tie_bound = min(vocab_size - len(entries), math.floor(
        max(0.0, 1 - known_mass + mass_tolerance) / math.exp(cutoff)
    ))
    kept_k = {k: math.exp((v - maximum) / temperature)
              for k, v in entries.items() if v >= cutoff}
    z = math.fsum(kept_k.values())
    extra = tie_bound * math.exp((cutoff - maximum) / temperature)
    if lp < cutoff:
        status, cause, cdf_low, cdf_high = "excluded", "top_k", None, None
    else:
        weight = kept_k[token_id]
        lower = math.fsum(w for k, w in kept_k.items() if entries[k] < lp)
        equal = math.fsum(w for k, w in kept_k.items() if entries[k] == lp)
        cdf_low = weight / (z + extra) if lp == cutoff else (lower + weight) / z
        cdf_high = (lower + equal + extra) / (z + extra)
        if cdf_high < 1 - top_p - cdf_tolerance:
            status, cause = "excluded", "top_p"
        elif cdf_low > 1 - top_p + cdf_tolerance:
            status, cause = "retained", None
        else:
            status, cause = "ambiguous", "missing_cutoff_ties_or_nucleus_boundary"
    nominal, cumulative = {}, 0.0
    for k, w in sorted(kept_k.items(), key=lambda item: (item[1], item[0])):
        cumulative += w
        if cumulative / z > 1 - top_p:
            nominal[k] = w
    nominal_sum = math.fsum(nominal.values())
    nominal_lp = math.log(nominal[token_id] / nominal_sum) if token_id in nominal else None
    return {
        "support": status, "exclusion_reason": cause,
        "nominal_logprob": nominal_lp, "nominal_zero": nominal_lp is None,
        "raw_logprob": lp, "cutoff_logprob": cutoff,
        "max_missing_cutoff_ties": tie_bound,
        "inclusive_cdf_lower": cdf_low, "inclusive_cdf_upper": cdf_high,
    }


def scope_summary(rows):
    counts = Counter(r["support"] for r in rows)
    zero = any(r["nominal_zero"] for r in rows)
    return {
        "tokens": len(rows), "support_counts": dict(counts),
        "sequence_logprob": "-inf" if counts["excluded"] else None,
        "sequence_status": "zero_under_reconstructed_distribution" if counts["excluded"]
        else "requires_complete_cutoff_ties_for_exact_finite_score",
        "nominal_sequence_logprob": "-inf" if zero else math.fsum(r["nominal_logprob"] for r in rows),
        "nominal_mean_logprob": "-inf" if zero else math.fsum(r["nominal_logprob"] for r in rows) / len(rows),
        "excluded_positions": [r["position"] for r in rows if r["support"] == "excluded"],
    }


def run(config_path, output):
    config = tomllib.loads(config_path.read_text())
    archive = Path(config["archive_run"])
    old_manifest = json.loads((archive / "manifest.json").read_text())
    if old_manifest["status"] != "complete":
        raise ValueError("Complete source archive required")
    scientific = old_manifest["scientific_config"]
    source = Path(config["source_run"])
    if source.name != scientific["source_run_id"]:
        raise ValueError("Archive source identity mismatch")
    captured = next(r for r in map(json.loads, (source / "trajectories.jsonl").open())
                    if r["example_id"] == scientific["example_id"]
                    and r["repeat_index"] == scientific["repeat_index"])
    original = json.loads((source / "config.json").read_text())
    policy = config["policy"]
    if any(policy[k] != original["generation"][k] for k in ("temperature", "top_k", "top_p")):
        raise ValueError("Policy differs from the original rollout policy")
    if original["model"]["revision"] != scientific["model_revision"]:
        raise ValueError("Model revision mismatch")
    ids = captured["response_ids"]
    model_path = Path(original["model"]["name_or_path"])
    vocab_size = json.loads((model_path / "config.json").read_text())["vocab_size"]
    tokenizer = json.loads((model_path / "tokenizer_config.json").read_text())
    special = {v["content"]: int(k) for k, v in tokenizer["added_tokens_decoder"].items()}
    close = [i for i, t in enumerate(ids) if t == special["</think>"]]
    if not close:
        raise ValueError("This archive requires a closed thinking boundary")
    response_start = close[-1] + 1
    output.mkdir(parents=True, exist_ok=False)
    inventory = {str(p): file_sha256(p) for p in [
        config_path, archive / "manifest.json", source / "trajectories.jsonl", source / "config.json",
    ]}
    manifest = {"status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
                "inputs": inventory, "model_revision": scientific["model_revision"],
                "new_model_calls": 0, "new_rollouts": 0}
    write_json(output / "manifest.json", manifest)
    (output / "analysis_config.toml").write_text(config_path.read_text())
    snapshot = output / "analysis_source.py"
    snapshot.write_text(Path(__file__).read_text())
    inventory[str(snapshot)] = file_sha256(snapshot)
    variants, branch_comparisons = {}, {}
    for label in config["variants"]:
        req_path = archive / f"{label}_request.json"
        raw_path = archive / f"{label}_raw_response.json.gz"
        inventory.update({str(p): file_sha256(p) for p in [req_path, raw_path]})
        request = json.loads(req_path.read_text())
        with gzip.open(raw_path, "rt") as f:
            choice = json.load(f)["choices"][0]
        prefix_size = len(request["prompt"]) - len(ids)
        if choice["prompt_token_ids"] != request["prompt"] or request["prompt"][prefix_size:] != ids:
            raise ValueError("Exact original response IDs must remain unchanged")
        if label == "parent" and request["prompt"][:prefix_size] != captured["prompt_ids"]:
            raise ValueError("Parent prefix differs from original rollout")
        if len(choice["prompt_logprobs"]) != len(request["prompt"]):
            raise ValueError("Misaligned prompt logprobs")
        rows = []
        for i, token_id in enumerate(ids):
            row = choice["prompt_logprobs"][prefix_size + i]
            value = policy_token(row, token_id, vocab_size=vocab_size, **policy,
                                 **config["numerics"])
            detail = {"position": i, "token_id": token_id,
                      "token_text": row[str(token_id)]["decoded_token"],
                      "scope": "thinking" if i < response_start else "response", **value}
            if "branch_analysis" in config:
                threshold = config["branch_analysis"]["probability_threshold"]
                raw_pmax = math.exp(max(v["logprob"] for v in row.values()))
                detail["branch_distributions"] = {
                    "T1_full": {"pmax_nominal": raw_pmax, "branch_nominal": raw_pmax < threshold}
                }
                for name, temperature, top_p in [
                    ("T1_topk", 1.0, 1.0),
                    ("sample_topk", policy["temperature"], 1.0),
                    ("sample_topk_topp", policy["temperature"], policy["top_p"]),
                ]:
                    detail["branch_distributions"][name] = distribution_statistics(
                        row, temperature=temperature, top_k=policy["top_k"], top_p=top_p,
                        vocab_size=vocab_size, probability_threshold=threshold, **config["numerics"],
                    )
            rows.append(detail)
        variants[label] = {scope: scope_summary(subset) for scope, subset in [
            ("full", rows), ("thinking", rows[:response_start]), ("response", rows[response_start:]),
        ]}
        write_json(output / f"{label}_tokens.json", rows)
        if "branch_analysis" in config:
            branch_comparisons[label] = compare_branch_masks(rows)
    summary = {"source_run_id": source.name, "archive_run_id": archive.name,
               "example_id": captured["example_id"], "repeat_index": captured["repeat_index"],
               "reward": captured["reward"], "policy": policy, "variants": variants,
               "parent_replay_consistent": not variants["parent"]["full"]["support_counts"].get("excluded"),
               "limits": [
                   "Post-hoc cached teacher-forcing distributions, not saved generation-time processed logits.",
                   "Top-k boundary ties may be missing; finite logprobs use a labeled nominal reconstruction.",
                   "Definite exclusions are bounded over missing tied tokens and nucleus tie order.",
                   "-inf minus -inf is undefined; never floor probabilities or discard zero-probability tokens.",
                   "Response-only scores condition on the unchanged original thinking, even if it is unreachable.",
               ]}
    if branch_comparisons:
        summary["branch_comparisons"] = branch_comparisons
        summary["branch_probability_threshold"] = config["branch_analysis"]["probability_threshold"]
    write_json(output / "summary.json", summary)
    lines = ["# Archived trajectory sampling-policy audit", "",
             f"Source: `{source.name}`; archive: `{archive.name}`.", "",
             f"One original successful geometry trajectory, unchanged {len(ids)} response IDs; zero model calls.", "",
             "| Prompt | Scope | Tokens | Definitely excluded | Ambiguous | Nominal mean logprob |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for label, scopes in variants.items():
        for scope, values in scopes.items():
            counts = values["support_counts"]
            lines.append(f"| {label} | {scope} | {values['tokens']} | {counts.get('excluded', 0)} | "
                         f"{counts.get('ambiguous', 0)} | {values['nominal_mean_logprob']} |")
    lines += ["", *summary["limits"], "", "Parent replay consistent: " + str(summary["parent_replay_consistent"])]
    if branch_comparisons:
        lines += ["", "## Branch mask comparison on the same fixed response", "",
                  "Branch = maximum token probability < " + str(config["branch_analysis"]["probability_threshold"]),
                  "Counts use nominal archived top-k distributions; uncertain cutoff-tie/boundary positions listed separately.",
                  "No test/validation evaluation or prompt update. Other prompts use the same original response, not new trajectories.",
                  "", "| Prompt | Scope | Distribution | Branch / tokens | Uncertain | Removed / added vs T1 | Singleton support |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
        for label, scopes in branch_comparisons.items():
            for scope, conditions in scopes.items():
                for condition, value in conditions.items():
                    lines.append(f"| {label} | {scope} | {condition} | {value['branch_tokens']}/{value['tokens']} | "
                                 f"{value['uncertain_positions']} | {value['removed_from_T1']} / {value['added_to_T1']} | "
                                 f"{value['singleton_support_nominal']} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    manifest.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat(), inputs=inventory)
    write_json(output / "manifest.json", manifest)
    print((output / "report.md").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
