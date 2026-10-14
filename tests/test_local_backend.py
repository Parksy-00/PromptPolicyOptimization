import pytest

from prompt_policy.backends import LocalHFBackend

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

pytestmark = pytest.mark.local


def test_real_local_backend_load_generate_score_and_actual_eos(tmp_path, config, recorder, monkeypatch):
    # Self-contained checkpoint/tokenizer; no network or pretrained weights.
    model_path = tmp_path / "tiny_model"
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            eos_token_id=2,
            pad_token_id=1,
        )
    ).eval()
    model.save_pretrained(model_path)
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {
                "[UNK]": 0,
                "[PAD]": 1,
                "[EOS]": 2,
                "<think>": 3,
                "</think>": 4,
                "parent": 5,
                "q": 6,
                "<assistant>": 7,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    hf_tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        chat_template="{% for m in messages %}{{ m.content }} {% endfor %}<assistant> "
        "{% if enable_thinking %}<think>{% endif %}",
    )
    hf_tokenizer.save_pretrained(model_path)
    config["model"].update(
        name_or_path=str(model_path), device="cpu", dtype="float32", max_context_tokens=128
    )
    config["generation"].update(temperature=0, max_new_tokens=2, enable_thinking=False)
    config["reflection"]["max_new_tokens"] = 4
    backend = LocalHFBackend(config, recorder)
    generation = backend.generate("parent", "q")
    assert generation.prompt_ids == backend.prompt_ids("parent", "q")
    assert len(generation.response_ids) in {1, 2}
    values = backend.logprobs(generation.prompt_ids, generation.response_ids)
    assert len(values) == len(generation.response_ids)
    assert all(value <= 0 for value in values)
    assert "chat_template.jinja" in backend.identity["files"]
    assert not any(p.requires_grad for p in backend.model.parameters())

    def eos_generate(**kwargs):
        prefix = kwargs["input_ids"]
        return torch.cat([prefix, torch.tensor([[2]], device=prefix.device)], dim=1)

    monkeypatch.setattr(backend.model, "generate", eos_generate)
    backend.config["generation"]["enable_thinking"] = True
    completed = backend.generate("parent", "q")
    assert completed.response_ids == (2,)
    assert completed.finish_reason == "eos"
    assert completed.thinking_open_in_prefix
    assert completed.text == ""  # No synthetic think or EOS in the actual completion text.
    assert len(backend.logprobs(completed.prompt_ids, completed.response_ids)) == 1
    with pytest.raises(ValueError, match="truncation"):
        backend.logprobs((1,) * 128, (2,))
