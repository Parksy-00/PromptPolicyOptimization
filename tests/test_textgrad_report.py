import json

import numpy as np
import pytest

from prompt_policy.analysis.textgrad_report import descriptive_correlations, ranks, verify_tokens
from prompt_policy.artifacts import digest, file_sha256


def test_rank_ties_use_average_rank():
    assert ranks(np.array([2, 1, 2])).tolist() == [1.5, 0.0, 1.5]


def test_correlations_recover_exact_validation_count_ties():
    rows = []
    for cohort in ["train", "validation"]:
        for gain, delta in [(1, 1 / 300), (2, 0.003333333333333355), (3, 2 / 300)]:
            rows.append(
                {
                    "source": "dapo",
                    "cohort": cohort,
                    "questions": "300",
                    "reward_weighted_t1_sum": gain,
                    "delta_validation": delta,
                    "delta_repeat_validation": delta,
                }
            )
    result = descriptive_correlations(rows)
    assert result[0]["spearman_descriptive"] == pytest.approx(np.sqrt(3) / 2)
    assert result[0]["inferential_pvalue"] is None


def test_verification_recomputes_token_aggregates_and_checks_hashes(tmp_path):
    scores = np.array([[-1.0, -2.0], [-0.5, -1.5], [-0.1, -np.inf]], dtype=np.float32)
    response = np.array([5, 6], dtype=np.int32)
    path = tmp_path / "tokens.npz"
    np.savez_compressed(path, scores=scores, response_ids=response, prompt_ids=np.array([1, 2]))
    metrics = {"t1": {"sum": -3.0, "mean": -1.5}, "t06": {"sum": -2.0, "mean": -1.0}}
    row = {
        "key": "dapo:1:0",
        "token_file": path.name,
        "sha256": file_sha256(path),
        "response_sha256": digest(response.tolist()),
        "metrics": metrics,
    }
    (tmp_path / "score_index.jsonl").write_text(json.dumps(row) + "\n")
    assert verify_tokens(tmp_path)["response_ids_identical_across_contexts"]
    row["metrics"]["t1"]["sum"] = -4
    (tmp_path / "score_index.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="aggregation"):
        verify_tokens(tmp_path)
