import math

import pytest

from prompt_policy.analysis.textgrad_likelihood import (
    filtered_logprobs,
    native_reward,
    reconstruct,
    score_context,
    summarize_arrays,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(x) for x in text]


def test_reconstruction_keeps_whitespace_and_does_not_invent_eos():
    text = "<think>\nreason\n</think>\n\nAnswer: 5"
    row = {
        "reasoning_content": "\nreason\n",
        "completion": "\n\nAnswer: 5",
        "finish_reason": "stop",
        "usage": {"completion_tokens": len(text) + 1},
    }
    ids, audit = reconstruct(row, CharacterTokenizer())
    assert ids == list(map(ord, text))
    assert audit["expected_count_match"]
    assert not audit["eos_appended"]
    assert not audit["response_ids_originally_saved"]


def test_length_terminated_thinking_has_no_fabricated_closing_delimiter():
    row = {
        "reasoning_content": "unfinished",
        "completion": None,
        "finish_reason": "length",
        "usage": {"completion_tokens": len("<think>unfinished")},
    }
    ids, audit = reconstruct(row, CharacterTokenizer())
    assert "".join(map(chr, ids)) == "<think>unfinished"
    assert audit["expected_count_match"]


def test_native_parsers_match_historical_rules_not_new_canonical_scoring():
    assert native_reward("Answer: B", 1, "GPQABench") == 1
    assert native_reward("Answer: A\nAnswer: B", 1, "GPQABench") == 0
    assert native_reward("Answer: -2.8", 2, "DAPOMathBench") == 1
    assert native_reward(None, 0, "DAPOMathBench") == 0
    assert native_reward("Answer: 44²₁4", 44, "DAPOMathBench") == 0
    assert native_reward("Answer : B", 1, "GPQABench") == 1


def test_filtered_replay_matches_full_vocabulary_reference():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[5.0, 4.2, 3.1, 2.4, 1.0]]).repeat(5, 1)
    labels = torch.arange(5)
    result = filtered_logprobs(logits, labels, temperature=0.6, top_k=3, top_p=0.8)
    scaled = logits / 0.6
    sorted_values, order = scaled.sort(dim=-1)
    sorted_values[:, :2] = -torch.inf
    discard = sorted_values.softmax(-1).cumsum(-1) <= 0.2
    discard[:, -1] = False
    sorted_values[discard] = -torch.inf
    filtered = sorted_values.scatter(1, order, sorted_values)
    expected = filtered.log_softmax(-1).gather(-1, labels[:, None]).flatten()
    torch.testing.assert_close(result, expected)
    assert torch.isneginf(result[-1])


def test_filtered_topk_boundary_ties_are_reported_as_undefined():
    torch = pytest.importorskip("torch")
    value = filtered_logprobs(torch.tensor([[3.0, 2.0, 2.0, 1.0]]), torch.tensor([0]), top_k=2)
    assert torch.isnan(value).all()


def test_teacher_forcing_temperatures_match_direct_causal_logits():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
        )
    ).eval()
    prefix, response = [1, 4, 5], [8, 9, 2]
    actual = score_context(model, prefix, response, 2)
    with torch.inference_mode():
        logits = model(torch.tensor([prefix + response])).logits[0, 2:5].float()
        for row, temperature in enumerate([1.0, 0.6]):
            expected = (
                (logits / temperature).log_softmax(-1).gather(-1, torch.tensor(response)[:, None]).flatten()
            )
            torch.testing.assert_close(torch.tensor(actual[row]), expected, atol=1e-6, rtol=1e-6)


def test_unsupported_sampler_tokens_are_not_silently_averaged_away():
    np = pytest.importorskip("numpy")
    value = summarize_arrays(np.array([[-1.0, -2.0], [-0.5, -1.5], [-0.1, -math.inf]]))
    assert value["t1"]["sum"] == -3
    assert value["filtered"]["sum"] is None
    assert value["filtered"]["outside_support"] == 1
