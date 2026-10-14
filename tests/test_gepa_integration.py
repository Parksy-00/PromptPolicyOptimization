import json

from conftest import FakeBackend

from prompt_policy.benchmarks import Benchmark
from prompt_policy.optimizer import search
from prompt_policy.types import Example


def benchmark():
    return Benchmark(
        "controlled",
        "numeric",
        (Example("a", "2+3", "5", "train"), Example("b", "1+4", "5", "train")),
        (Example("v", "3+2", "5", "validation"),),
        (Example("t", "4+1", "5", "test"),),
        {"kind": "test"},
    )


def test_real_gepa_accepts_accuracy_tie_and_keeps_global_scores_accuracy_only(config, recorder):
    config["initial_prompt"] = "parent"
    backend = FakeBackend(recorder)
    result = search(benchmark(), backend, recorder, config, reflection_lm=lambda p: "```\nchild\n```")
    assert len(result.candidates) == 2
    assert result.val_aggregate_scores == [1.0, 1.0]
    assert result.best_idx == 0  # Accuracy-only final selection retains original on a tie.
    row = json.loads((recorder.run_dir / "decisions.jsonl").read_text())
    assert row["decision"]["accepted"]
    assert row["decision"]["delta_accuracy"] == 0
    assert row["decision"]["success_gain"] > 0
    assert row["decision"]["success_trajectories"] == 4
    assert all(t["prompt"] == "parent" for t in row["pool"])
    assert all(response == (5, 9) for _, response in backend.score_requests)
    assert recorder.cost["task_calls"] == 12  # 2 seed-val + 4 parent + 4 child + 2 child-val.


def test_accuracy_baseline_rejects_same_candidate_with_same_success_pool(config, recorder):
    config["initial_prompt"] = "parent"
    config["objective"].update(mode="accuracy", weight=0)
    result = search(
        benchmark(), FakeBackend(recorder), recorder, config, reflection_lm=lambda p: "```\nchild\n```"
    )
    assert len(result.candidates) == 1
    decision = json.loads((recorder.run_dir / "decisions.jsonl").read_text())["decision"]
    assert not decision["accepted"]
    assert decision["success_gain"] > 0


def test_candidate_failure_does_not_replace_reference_pool(config, recorder):
    config["initial_prompt"] = "parent"
    backend = FakeBackend(recorder, wrong_child=True)
    result = search(benchmark(), backend, recorder, config, reflection_lm=lambda p: "```\nchild\n```")
    assert len(result.candidates) == 1
    row = json.loads((recorder.run_dir / "decisions.jsonl").read_text())
    assert not row["decision"]["accepted"]
    assert row["decision"]["delta_accuracy"] == -1
    assert all(response == (5, 9) for _, response in backend.score_requests)
