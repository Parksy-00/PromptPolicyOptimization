import pytest

from prompt_policy.scoring import token_logprobs

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.local


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
        ),
    ],
)
def test_qwen3_teacher_forcing_matches_full_causal_lm_and_chunks(device):
    # Random tiny model: numerical verification, not a research experiment.
    torch.manual_seed(123)
    config = transformers.Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    model = transformers.Qwen3ForCausalLM(config).to(device).eval()
    prefix, response = (1, 7, 8), (4, 5, 6, 2)
    values = token_logprobs(model, prefix, response, head_chunk=2)
    with torch.inference_mode():
        logits = model(torch.tensor([prefix + response], device=device)).logits.float()
        expected = (
            torch.log_softmax(logits[0, len(prefix) - 1 : -1], -1)
            .gather(-1, torch.tensor(response, device=device)[:, None])
            .flatten()
            .cpu()
        )
    torch.testing.assert_close(torch.tensor(values), expected, atol=1e-6, rtol=1e-6)
    assert values == pytest.approx(token_logprobs(model, prefix, response, head_chunk=1), abs=1e-6)
    assert values == pytest.approx(token_logprobs(model, prefix, response), abs=1e-6)
    assert values == token_logprobs(model, prefix, response, head_chunk=2)
    with pytest.raises(ValueError, match="eval mode"):
        token_logprobs(model.train(), prefix, response)
