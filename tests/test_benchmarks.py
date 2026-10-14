import json

import pytest

from prompt_policy.benchmarks import Benchmark, extract_answer, load_benchmark, reward
from prompt_policy.types import Example


@pytest.mark.parametrize("text", ["### 5", "Answer: 5", r"work \boxed{5}", "<think>### 999</think>\n### 5"])
def test_final_answer_formats(text):
    assert reward(text, "5", "numeric") == 1


def test_unclosed_thinking_and_unclosed_box_are_failures():
    assert extract_answer("<think>### 5") is None
    assert extract_answer(r"\boxed{5") is None
    assert extract_answer("5") is None


def test_thinking_started_in_request_is_not_mistaken_for_final_answer():
    benchmark = Benchmark("toy", "numeric", (), (), (), {})
    example = Example("a", "q", "5", "train")
    assert benchmark.score(example, "### 5", thinking_open_in_prefix=True) == 0
    assert benchmark.score(example, "thought </think>\n### 5", thinking_open_in_prefix=True) == 1


def test_numeric_is_exact_rational_not_float_approximation():
    assert reward("### 0.5", "1/2", "numeric") == 1
    assert reward("### 0.33333333", "1/3", "numeric") == 0
    assert reward("### nan", "5", "numeric") == 0


def test_split_overlap_fails_before_model_load(tmp_path):
    for split in ("train", "validation", "test"):
        (tmp_path / f"{split}.jsonl").write_text(
            json.dumps({"id": split, "question": "same question", "answer": "1"}) + "\n"
        )
    config = {
        "name": "test",
        "scorer": "numeric",
        **{s: f"{s}.jsonl" for s in ("train", "validation", "test")},
    }
    with pytest.raises(ValueError, match="Duplicate"):
        load_benchmark(config, tmp_path)
