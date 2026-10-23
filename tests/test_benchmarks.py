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


@pytest.mark.parametrize(
    "text,gold",
    [
        ("Reasoning\nAnswer: 35", "35"),
        ("Answer: -35", "-35"),
        ("Answer: +0035\n\n", "35"),
        ("<think>Answer: 999</think>\nReasoning\nAnswer: 35", "35"),
    ],
)
def test_fixed_answer_integer_accepts_only_the_final_answer_outside_thinking(text, gold):
    assert reward(text, gold, "answer_integer_v1") == 1


@pytest.mark.parametrize(
    "text",
    [
        "Reasoning mentions 35.",
        "Answer: 35\nMore reasoning",
        r"Answer: \boxed{35}",
        "Answer: 35.0",
        "Answer: 35/1",
        "Answer: 35 or 36",
        "Answer: 35.",
        "answer: 35",
        "Answer: ３５",
        "<think>Answer: 35",
        "<think>Answer: 35</think>\nNo final answer",
        "### 35",
    ],
)
def test_fixed_answer_integer_rejects_ambiguous_malformed_or_reasoning_only_answers(text):
    assert reward(text, "35", "answer_integer_v1") == 0


def test_fixed_answer_integer_preserves_negative_sign_and_prefix_thinking():
    benchmark = Benchmark("test", "answer_integer_v1", (), (), (), {})
    example = Example("a", "q", "35", "train")
    assert benchmark.score(example, "Answer: -35") == 0
    assert benchmark.score(example, "Answer: 35", thinking_open_in_prefix=True) == 0
    assert benchmark.score(example, "reasoning </think>\nAnswer: 35", thinking_open_in_prefix=True) == 1


def test_fixed_rule_is_outside_editable_instruction_and_does_not_affect_plain_reflection():
    from prompt_policy.benchmarks import ANSWER_INTEGER_RULE, task_messages

    parent = "Solve the problem."
    candidate = "Solve the problem and check your reasoning."
    before = task_messages(parent, "question", "answer_integer_v1")
    after = task_messages(candidate, "question", "answer_integer_v1")
    assert before[0]["content"] == parent + "\n\n" + ANSWER_INTEGER_RULE
    assert after[0]["content"] == candidate + "\n\n" + ANSWER_INTEGER_RULE
    assert before[1] == after[1] == {"role": "user", "content": "question"}
    assert task_messages("critic", "request")[0]["content"] == "critic"


def test_fixed_rule_and_scorer_are_paired_and_recorded_in_identity(tmp_path):
    from prompt_policy.benchmarks import ANSWER_INTEGER_RULE

    for split in ("train", "validation", "test"):
        (tmp_path / f"{split}.jsonl").write_text(
            json.dumps({"id": split, "question": f"{split} question", "answer": "35"}) + "\n"
        )
    config = {
        "name": "test",
        "scorer": "answer_integer_v1",
        "task_format": "answer_integer_v1",
        **{s: f"{s}.jsonl" for s in ("train", "validation", "test")},
    }
    benchmark = load_benchmark(config, tmp_path)
    assert benchmark.identity["fixed_output_rule"] == ANSWER_INTEGER_RULE
    assert benchmark.identity["task_format"] == "answer_integer_v1"
    for wrong in (config | {"task_format": "plain"}, config | {"scorer": "numeric"}):
        with pytest.raises(ValueError, match="used together"):
            load_benchmark(wrong, tmp_path)
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"id": "train", "question": "tr", "answer": "35.0"}) + "\n"
    )
    with pytest.raises(ValueError, match="integer ground truth"):
        load_benchmark(config, tmp_path)


@pytest.mark.parametrize(
    "text,gold",
    [
        ("Reasoning\n<answer>35</answer>", "35"),
        ("<answer>-35</answer>", "-35"),
        ("<answer>\n+0035\n</answer>", "35"),
        ("<think><answer>999</answer></think>\n<answer>35</answer>", "35"),
        ("Answer: 999\n<answer>35</answer>\nOther text says 888.", "35"),
    ],
)
def test_answer_tag_scores_only_the_tag_body_outside_thinking(text, gold):
    assert reward(text, gold, "answer_tag_integer_v1") == 1


@pytest.mark.parametrize(
    "text",
    [
        "Answer: 35",
        r"\boxed{35}",
        "35",
        "<answer>35",
        "35</answer>",
        "<answer></answer>",
        "<answer> </answer>",
        "<answer>35.0</answer>",
        "<answer>35/1</answer>",
        r"<answer>\boxed{35}</answer>",
        "<answer>**35**</answer>",
        "<answer>３５</answer>",
        "<answer>35 or 36</answer>",
        "<answer>35</answer><answer>35</answer>",
        "<answer><answer>35</answer></answer>",
        '<answer x="1">35</answer>',
        "<ANSWER>35</ANSWER>",
        "<answer>35</answer >",
        "<answer>35</answer><answer",
        "<think><answer>35</answer>",
        "<think><answer>35</answer></think>no final answer",
        "<answer>35</answer><think>still thinking",
        "<answer>999</answer>\nAnswer: 35",
    ],
)
def test_answer_tag_never_falls_back_to_unwrapped_numbers_or_ambiguous_tags(text):
    assert reward(text, "35", "answer_tag_integer_v1") == 0


def test_answer_tag_respects_thinking_started_in_prefix():
    benchmark = Benchmark("tagged", "answer_tag_integer_v1", (), (), (), {})
    example = Example("q", "question", "35", "train")
    assert benchmark.score(example, "<answer>35</answer>", thinking_open_in_prefix=True) == 0
    assert benchmark.score(example, "reason </think><answer>35</answer>", thinking_open_in_prefix=True) == 1


def test_answer_tag_rendering_identity_and_historical_protocol_remain_distinct(tmp_path):
    from prompt_policy.benchmarks import ANSWER_TAG_INTEGER_RULE, task_messages

    for split in ("train", "validation", "test"):
        (tmp_path / f"{split}.jsonl").write_text(
            json.dumps({"id": split, "question": f"question {split}", "answer": "35"}) + "\n"
        )
    config = {
        "name": "tagged",
        "scorer": "answer_tag_integer_v1",
        "task_format": "answer_tag_integer_v1",
        **{s: f"{s}.jsonl" for s in ("train", "validation", "test")},
    }
    tagged = load_benchmark(config, tmp_path)
    assert tagged.identity["fixed_output_rule"] == ANSWER_TAG_INTEGER_RULE
    assert task_messages("P", "x", config["task_format"]) == [
        {"role": "system", "content": "P\n\n" + ANSWER_TAG_INTEGER_RULE},
        {"role": "user", "content": "x"},
    ]
    old = load_benchmark(
        {**config, "scorer": "answer_integer_v1", "task_format": "answer_integer_v1"}, tmp_path
    )
    assert old.identity["fingerprint"] != tagged.identity["fingerprint"]
    assert reward("Answer: 35", "35", "answer_integer_v1") == 1
    assert reward("<answer>35</answer>", "35", "answer_integer_v1") == 0
    with pytest.raises(ValueError, match="must be used together"):
        load_benchmark({**config, "scorer": "answer_integer_v1"}, tmp_path)


def test_user_answer_constraint_precedes_question_and_system_is_only_editable_instruction(tmp_path):
    from prompt_policy.benchmarks import ANSWER_TAG_INTEGER_RULE, task_messages

    question = "What is 1 + 1?\nExplain your calculation."
    parent = task_messages("Solve carefully.", question, "answer_tag_integer_v2")
    candidate = task_messages("Verify intermediate calculations.", question, "answer_tag_integer_v2")
    assert parent == [
        {"role": "system", "content": "Solve carefully."},
        {"role": "user", "content": ANSWER_TAG_INTEGER_RULE + "\n\nQuestion:\n" + question},
    ]
    assert candidate[0]["content"] == "Verify intermediate calculations."
    assert parent[1] == candidate[1]  # fixed rule/question stay identical across likelihood comparisons
    assert reward("Outside number: 999. <answer>2</answer>", "2", "answer_tag_integer_v2") == 1
    assert reward("Answer: 2", "2", "answer_tag_integer_v2") == 0
    assert reward("<answer>2</answer><answer>3</answer>", "2", "answer_tag_integer_v2") == 0

    for split in ("train", "validation", "test"):
        (tmp_path / f"{split}.jsonl").write_text(
            json.dumps({"id": split, "question": f"{question} {split}", "answer": "2"}) + "\n"
        )
    sources = {s: f"{s}.jsonl" for s in ("train", "validation", "test")}
    old = load_benchmark(
        {
            "name": "example",
            **sources,
            "scorer": "answer_tag_integer_v1",
            "task_format": "answer_tag_integer_v1",
        },
        tmp_path,
    )
    new = load_benchmark(
        {
            "name": "example",
            **sources,
            "scorer": "answer_tag_integer_v2",
            "task_format": "answer_tag_integer_v2",
        },
        tmp_path,
    )
    assert old.identity["fingerprint"] != new.identity["fingerprint"]
    assert new.identity["task_format"] == "answer_tag_integer_v2"
    assert old.identity["fixed_output_rule"] == new.identity["fixed_output_rule"]
    assert task_messages("P", "x", "answer_tag_integer_v1")[0]["content"] == "P\n\n" + ANSWER_TAG_INTEGER_RULE


def test_explicit_canonical_duplicate_policy_preserves_within_split_but_rejects_leakage(tmp_path):
    rows = {
        "train": [{"id": "tr", "question": "train question", "answer": "1"}],
        "validation": [{"id": "va", "question": "validation question", "answer": "2"}],
        "test": [
            {"id": "te1", "question": "same  test", "answer": "3"},
            {"id": "te2", "question": "same test", "answer": "3"},
        ],
    }
    for split, examples in rows.items():
        (tmp_path / f"{split}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in examples))
    config = {"name": "canonical", "scorer": "numeric", **{s: f"{s}.jsonl" for s in rows}}
    with pytest.raises(ValueError, match="Duplicate"):
        load_benchmark(config, tmp_path)
    config["allow_within_split_question_duplicates"] = True
    result = load_benchmark(config, tmp_path)
    assert len(result.test) == 2
    assert result.identity["within_split_question_duplicates"] == [
        {"split": "test", "first_id": "te1", "duplicate_id": "te2"}
    ]
    (tmp_path / "test.jsonl").write_text(
        json.dumps({"id": "te3", "question": "train question", "answer": "1"}) + "\n"
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_benchmark(config, tmp_path)


@pytest.mark.parametrize(
    "text,gold",
    [
        (r"Reasoning. Final answer: $\boxed{35}$", "35"),
        (r"**Answer: $\boxed{35}$**", "35"),
        (r"\boxed{-35}", "-35"),
        (r"\boxed{ +0035 }", "35"),
        (r"Intermediate \boxed{10}; final \boxed{35}", "35"),
        (r"<think>\boxed{999}</think>\boxed{35}", "35"),
    ],
)
def test_boxed_integer_scores_the_last_visible_box_body(text, gold):
    assert reward(text, gold, "boxed_integer_v1") == 1


@pytest.mark.parametrize(
    "text",
    [
        "Answer: 35",
        "<answer>35</answer>",
        "35",
        r"\boxed{}",
        r"\boxed{35.0}",
        r"\boxed{\frac{35}{1}}",
        r"\boxed{３５}",
        r"\boxed{35 or 36}",
        r"\boxed{35} then \boxed{36}",
        r"\boxed{35} then \boxed{",
        r"\boxed{999} then Answer: 35",
        r"<think>\boxed{35}",
        r"<think>\boxed{35}</think>No final answer.",
    ],
)
def test_boxed_integer_never_falls_back_to_other_numbers_or_an_earlier_box(text):
    assert reward(text, "35", "boxed_integer_v1") == 0


def test_boxed_integer_respects_thinking_started_in_prefix():
    benchmark = Benchmark("boxed", "boxed_integer_v1", (), (), (), {})
    example = Example("a", "q", "35", "train")
    assert benchmark.score(example, r"\boxed{35}", thinking_open_in_prefix=True) == 0
    assert benchmark.score(example, r"reason </think>\boxed{35}", thinking_open_in_prefix=True) == 1


def test_boxed_integer_protocol_keeps_system_editable_and_changes_benchmark_identity(tmp_path):
    from prompt_policy.benchmarks import BOXED_INTEGER_RULE, task_messages

    for split in ("train", "validation", "test"):
        (tmp_path / f"{split}.jsonl").write_text(
            json.dumps({"id": split, "question": f"{split} question", "answer": "35"}) + "\n"
        )
    config = {
        "name": "boxed",
        "scorer": "boxed_integer_v1",
        "task_format": "boxed_integer_v1",
        **{s: f"{s}.jsonl" for s in ("train", "validation", "test")},
    }
    benchmark = load_benchmark(config, tmp_path)
    old = load_benchmark(
        config | {"scorer": "answer_tag_integer_v2", "task_format": "answer_tag_integer_v2"}, tmp_path
    )
    assert benchmark.identity["fingerprint"] != old.identity["fingerprint"]
    assert benchmark.identity["fixed_output_rule"] == BOXED_INTEGER_RULE
    assert task_messages("P", "q", "boxed_integer_v1") == [
        {"role": "system", "content": "P"},
        {"role": "user", "content": BOXED_INTEGER_RULE + "\n\nQuestion:\nq"},
    ]
    with pytest.raises(ValueError, match="must be used together"):
        load_benchmark(config | {"scorer": "answer_tag_integer_v2"}, tmp_path)


@pytest.mark.parametrize(
    "text,prefix_open,expected",
    [
        (r"<think>private \boxed{99}</think>Visible work. \boxed{35}", False, r"Visible work. \boxed{35}"),
        (r"private reasoning</think>Wrong calculation. \boxed{99}", True, r"Wrong calculation. \boxed{99}"),
        (r"Visible work. \boxed{-35}", False, r"Visible work. \boxed{-35}"),
        (r"<think>unfinished reasoning \boxed{35}", False, None),
        (r"unfinished reasoning \boxed{35}", True, None),
        (r"<think>done</think>Answer: 35", False, None),
        (r"<think>done</think>\boxed{35, 36}", False, None),
        ("<think>done</think>   ", False, None),
    ],
)
def test_editor_response_requires_a_parseable_answer_and_removes_internal_thinking(text, prefix_open, expected):
    benchmark = Benchmark("boxed", "boxed_integer_v1", (), (), (), {})
    assert benchmark.editor_response(text, thinking_open_in_prefix=prefix_open) == expected


def test_plain_reflection_can_omit_system_without_changing_task_roles():
    from prompt_policy.benchmarks import task_messages

    assert task_messages("", "editor instructions and evidence") == [
        {"role": "user", "content": "editor instructions and evidence"}
    ]
    assert task_messages("task instruction", "question") == [
        {"role": "system", "content": "task instruction"},
        {"role": "user", "content": "question"},
    ]
