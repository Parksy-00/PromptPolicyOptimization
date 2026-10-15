import numpy as np
import pytest

from prompt_policy.analysis.textgrad_endpoints import (
    cached_context,
    instruction_wrapper,
    legacy_dapo_score,
    make_origin_report,
    origin_contrast_row,
    summarize_endpoint,
)
from prompt_policy.analysis.textgrad_likelihood import summarize_arrays
from prompt_policy.artifacts import digest, file_sha256


def test_endpoint_weights_failures_zero_but_keeps_question_denominator():
    rows = []
    for reward, change in [(1, 2), (1, -1), (0, -100)]:
        row = {"reward": reward, "finish_reason": "stop", "count_match": True}
        for metric in ["t1", "t06"]:
            for kind in ["sum", "mean"]:
                row[f"delta_{metric}_{kind}"] = change
        rows.append(row)
    result = summarize_endpoint(rows, 100, np.random.default_rng(1729))
    assert result["reward_weighted_t1_sum"] == pytest.approx(1 / 3)
    assert result["successful_sequence_increases"] == 1
    assert result["successful_questions"] == 2
    assert result["stop_countmatched_weighted_t1_sum"] == pytest.approx(1 / 3)


def test_cache_reuse_checks_request_ids_even_when_cache_name_matches(tmp_path):
    scores = np.array([[-1, -2], [-2, -3], [-1, -np.inf]], dtype=np.float32)
    path = tmp_path / "cache.npz"
    response, prompt = [3, 4], [1, 2]
    np.savez_compressed(path, scores=scores, response_ids=response, prompt_ids=prompt)
    record = {
        "token_file": path.name,
        "sha256": file_sha256(path),
        "metrics": summarize_arrays(scores),
        "response_sha256": digest(response),
    }
    assert cached_context(record, tmp_path, prompt, response) == path
    with pytest.raises(ValueError, match="prompt IDs"):
        cached_context(record, tmp_path, [1, 5], response)
    with pytest.raises(ValueError, match="response IDs"):
        cached_context(record, tmp_path, prompt, [3, 5])


def test_historical_parser_requires_both_fields_and_plain_integer():
    valid = "[[ ## reasoning ## ]]\nA derivation.\n[[ ## answer ## ]]\n42\n[[ ## completed ## ]]"
    assert legacy_dapo_score(valid, "42") == (1.0, 1.0)
    assert legacy_dapo_score(valid, "43") == (0.0, 1.0)
    assert legacy_dapo_score(valid.replace("\n42\n", "\n\\boxed{42}\n"), "42") == (0.0, 0.0)
    assert legacy_dapo_score("[[ ## answer ## ]]\n42", "42") == (0.0, 0.0)
    assert legacy_dapo_score(None, "42") == (0.0, 0.0)


def test_historical_parser_preserves_first_answer_field():
    text = "[[ ## reasoning ## ]]\nDerivation\n[[ ## answer ## ]]\n41\n[[ ## answer ## ]]\n42"
    assert legacy_dapo_score(text, "42") == (0.0, 1.0)


def test_captured_dspy_wrapper_preserves_indentation_and_checks_instruction():
    wrapper = "Structured fields\nObjective:\n        "
    instruction = "First line\n\nSecond line"
    system = wrapper + instruction.replace("\n", "\n        ")
    assert instruction_wrapper(system, instruction) == wrapper
    with pytest.raises(ValueError, match="declared DSPy instruction"):
        instruction_wrapper(system, "Wrong instruction")


def test_origin_and_base_contrasts_can_have_opposite_signs():
    job = {
        "source": "dapo",
        "row_key": "call1",
        "sample_id": "q1",
        "origin_id": "2",
        "reward": 1,
        "finish_reason": "stop",
        "audit": {"expected_count_match": True},
    }
    cache = {}
    for cid, value in [("0", -10), ("2", -7), ("17", -8)]:
        cache["call1:" + cid] = {
            metric: {kind: value for kind in ["sum", "mean"]} for metric in ["t1", "t06"]
        }
    assert origin_contrast_row(job, "0", "17", cache)["delta_t1_sum"] == 2
    assert origin_contrast_row(job, "2", "17", cache)["delta_t1_sum"] == -1


def test_origin_report_keeps_pools_separate_and_marks_final_self_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr("prompt_policy.analysis.textgrad_endpoints.plot_origin_report", lambda *_: None)
    endpoint = {
        "source": "dapo",
        "selected_id": "17",
        "response_origins": ["0", "2", "17"],
        "selected_validation": 1.0,
        "origin_candidates": {
            cid: {"accepted": True, "on_selected_path": True, "validation_accuracy": 1.0}
            for cid in ["0", "2", "17"]
        },
    }
    jobs, cache = [], {}
    for origin, scores in [
        ("0", {"0": -5, "17": -6}),
        ("2", {"0": -10, "2": -7, "17": -8}),
        ("17", {"0": -10, "17": -5}),
    ]:
        key = "call" + origin
        jobs.append(
            {
                "source": "dapo",
                "row_key": key,
                "sample_id": "q1",
                "origin_id": origin,
                "reward": 1,
                "finish_reason": "stop",
                "audit": {"expected_count_match": True},
                "contexts": {cid: [] for cid in scores},
            }
        )
        for cid, value in scores.items():
            cache[key + ":" + cid] = {
                metric: {kind: value for kind in ["sum", "mean"]} for metric in ["t1", "t06"]
            }
    result = make_origin_report(
        tmp_path, {"endpoints": [endpoint]}, jobs, cache, {"bootstrap_seed": 1729, "bootstrap_repeats": 10}
    )
    keyed = {(r["origin_id"], r["contrast"]): r for r in result}
    assert keyed[("2", "base_to_final")]["reward_weighted_t1_sum"] == 2
    assert keyed[("2", "origin_to_final")]["reward_weighted_t1_sum"] == -1
    assert keyed[("17", "origin_to_final")]["reward_weighted_t1_sum"] == 0
    assert (
        keyed[("0", "base_to_final")]["conditional_ci_low"]
        == keyed[("0", "origin_to_final")]["conditional_ci_low"]
    )
    assert all(r["questions"] == 1 for r in result)
