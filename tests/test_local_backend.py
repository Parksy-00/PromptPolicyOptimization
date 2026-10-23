import pytest

from prompt_policy.backends import LocalHFBackend

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

pytestmark = pytest.mark.local


@pytest.mark.parametrize(
    "task_format", ["plain", "answer_integer_v1", "answer_tag_integer_v1", "answer_tag_integer_v2", "boxed_integer_v1"]
)
def test_real_local_backend_load_generate_score_and_actual_eos(
    tmp_path, config, recorder, monkeypatch, task_format
):
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
    config["benchmark"]["task_format"] = task_format
    backend = LocalHFBackend(config, recorder)
    generation = backend.generate("parent", "q")
    assert generation.prompt_ids == backend.prompt_ids("parent", "q")
    assert len(generation.response_ids) in {1, 2}
    values = backend.logprobs(generation.prompt_ids, generation.response_ids)
    assert len(values) == len(generation.response_ids)
    assert all(value <= 0 for value in values)
    assert "chat_template.jinja" in backend.identity["files"]
    assert not any(p.requires_grad for p in backend.model.parameters())

    seen_sampling = []

    def eos_generate(**kwargs):
        seen_sampling.append(kwargs)
        prefix = kwargs["input_ids"]
        return torch.cat([prefix, torch.tensor([[2]], device=prefix.device)], dim=1)

    monkeypatch.setattr(backend.model, "generate", eos_generate)
    backend.config["generation"].update(enable_thinking=True, temperature=0.6, top_p=0.95, top_k=20)
    completed = backend.generate("parent", "q")
    assert seen_sampling[-1]["temperature"] == 0.6
    assert seen_sampling[-1]["top_p"] == 0.95
    assert seen_sampling[-1]["top_k"] == 20
    assert completed.response_ids == (2,)
    assert completed.finish_reason == "eos"
    assert completed.thinking_open_in_prefix
    assert completed.text == ""  # No synthetic think or EOS in the actual completion text.
    assert len(backend.logprobs(completed.prompt_ids, completed.response_ids)) == 1
    from prompt_policy.benchmarks import TASK_OUTPUT_RULES, task_messages

    if task_format in {"answer_tag_integer_v2", "boxed_integer_v1"}:
        assert backend.messages("parent", "q") == [
            {"role": "system", "content": "parent"},
            {"role": "user", "content": TASK_OUTPUT_RULES[task_format] + "\n\nQuestion:\nq"},
        ]
    elif task_format in TASK_OUTPUT_RULES:
        assert backend.messages("parent", "q")[0]["content"] == "parent\n\n" + TASK_OUTPUT_RULES[task_format]
    backend.config["reflection"].update(temperature=0.6, top_p=0.95, top_k=20)
    reflection = backend.generate("critic", "request", reflection=True)
    assert seen_sampling[-1]["top_k"] == 20
    del backend.config["reflection"]["top_k"]
    backend.generate("critic", "request", reflection=True)
    assert seen_sampling[-1]["top_k"] == 0  # Explicit backward compatibility for older TOMLs.
    assert reflection.prompt_ids == backend._encode_messages(task_messages("critic", "request"))
    user_only = backend.generate("", "editor instructions and evidence", reflection=True)
    assert user_only.prompt_ids == backend._encode_messages(
        [{"role": "user", "content": "editor instructions and evidence"}]
    )
    assert tuple(seen_sampling[-1]["input_ids"][0].tolist()) == user_only.prompt_ids
    with pytest.raises(ValueError, match="truncation"):
        backend.logprobs((1,) * 128, (2,))


@pytest.mark.parametrize(
    "ids,opened,expected",
    [
        ((3, 8, 4, 7, 2), False, (3, 4)),
        ((8, 4, 7, 2), True, (2, 3)),
        ((8, 7), True, ()),
        ((3, 8), False, ()),
        ((7, 2), False, (0, 1)),
        ((4, 7, 3, 8), False, ()),
    ],
)
def test_response_scoring_mask_uses_original_thinking_token_boundaries(ids, opened, expected):
    from types import SimpleNamespace

    backend = LocalHFBackend.__new__(LocalHFBackend)
    backend.tokenizer = SimpleNamespace(get_vocab=lambda: {"<think>": 3, "</think>": 4})
    assert (
        backend.response_token_indices(SimpleNamespace(response_ids=ids, thinking_open_in_prefix=opened))
        == expected
    )
