from __future__ import annotations

import time
from pathlib import Path

from prompt_policy.artifacts import Recorder, file_sha256
from prompt_policy.scoring import token_logprobs
from prompt_policy.types import Generation


class LocalHFBackend:
    """One frozen local causal LM for task generation, reflection and scoring."""

    def __init__(self, config: dict, recorder: Recorder):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config, self.recorder = config, recorder
        source = config["model"]["name_or_path"]
        root = Path(source).expanduser()
        if not root.is_dir():
            root = Path(snapshot_download(source, revision=config["model"]["revision"]))
        root = root.resolve()
        files = sorted(
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".jinja"}
        )
        if not any(p.suffix in {".safetensors", ".bin"} for p in files):
            raise ValueError("Local model has no weight files")
        self.identity = {
            "source": source,
            "resolved_path": str(root),
            "revision": config["model"]["revision"],
            "files": {str(p.relative_to(root)): file_sha256(p) for p in files},
            "dtype": config["model"]["dtype"],
            "scoring_temperature": 1.0,
            "enable_thinking": config["generation"]["enable_thinking"],
        }
        dtype = getattr(torch, config["model"]["dtype"])
        self.tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(root, dtype=dtype, trust_remote_code=False)
        self.model.to(config["model"]["device"]).eval()
        self.model.requires_grad_(False)
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = next(iter(sorted(self.eos_ids)), None)
        if self.pad_id is None:
            raise ValueError("Tokenizer/model must supply a pad or EOS token")

    def messages(self, prompt: str, question: str) -> list[dict]:
        return [{"role": "system", "content": prompt}, {"role": "user", "content": question}]

    def prompt_ids(self, prompt: str, question: str) -> tuple[int, ...]:
        ids = self.tokenizer.apply_chat_template(
            self.messages(prompt, question),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.config["generation"]["enable_thinking"],
        )
        # Transformers versions may return either flat IDs or BatchEncoding.
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            if len(ids) != 1:
                raise ValueError("Expected exactly one request")
            ids = ids[0]
        return tuple(ids)

    def _check_length(self, size: int):
        model_limit = getattr(self.model.config, "max_position_embeddings", None)
        limits = [self.config["model"]["max_context_tokens"]]
        if isinstance(model_limit, int):
            limits.append(model_limit)
        if size > min(limits):
            raise ValueError(f"Request needs {size} tokens; truncation is not permitted")

    def generate(self, prompt: str, question: str, *, reflection: bool = False) -> Generation:
        import torch

        settings = self.config["reflection"] if reflection else self.config["generation"]
        prefix = self.prompt_ids(prompt, question)
        self._check_length(len(prefix) + settings["max_new_tokens"])
        inputs = torch.tensor([prefix], device=next(self.model.parameters()).device, dtype=torch.long)
        kwargs = {
            "max_new_tokens": settings["max_new_tokens"],
            "pad_token_id": self.pad_id,
            "do_sample": settings["temperature"] > 0,
            "num_beams": 1,
            "repetition_penalty": 1.0,
            "use_cache": True,
        }
        if settings["temperature"] > 0:
            kwargs.update(temperature=settings["temperature"], top_p=settings["top_p"], top_k=0)
        started = time.perf_counter()
        with torch.inference_mode():
            sequences = self.model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs), **kwargs
            )
        response = tuple(sequences[0, len(prefix) :].cpu().tolist())
        if not response:
            raise ValueError("Model produced an empty response")
        finish = "eos" if response[-1] in self.eos_ids else "length"
        # Strip only actual terminal EOS from display text; keep it in scored IDs.
        displayed = response[:-1] if finish == "eos" else response
        text = self.tokenizer.decode(displayed, skip_special_tokens=False)
        # Preserve completion text; separately record thinking begun by the request.
        prefix_text = self.tokenizer.decode(prefix, skip_special_tokens=False)
        thinking_open = prefix_text.rstrip().endswith("<think>")
        kind = "reflection" if reflection else "task"
        self.recorder.charge(
            kind,
            prompt_tokens=len(prefix),
            response_tokens=len(response),
            elapsed=time.perf_counter() - started,
        )
        return Generation(text, prefix, response, finish, thinking_open)

    def logprobs(self, prompt_ids: tuple[int, ...], response_ids: tuple[int, ...]) -> tuple[float, ...]:
        self._check_length(len(prompt_ids) + len(response_ids))
        return token_logprobs(
            self.model, prompt_ids, response_ids, head_chunk=self.config["model"]["head_chunk"]
        )
