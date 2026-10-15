import numpy as np
import pytest

from prompt_policy.analysis.textgrad_endpoints import cached_context, summarize_endpoint
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
