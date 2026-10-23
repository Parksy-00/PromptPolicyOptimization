from __future__ import annotations

import math
import time
from pathlib import Path

from prompt_policy.artifacts import Recorder, file_sha256
from prompt_policy.benchmarks import task_messages
from prompt_policy.scoring import token_logprobs
from prompt_policy.types import Generation


def local_model_identity(config):
    from huggingface_hub import snapshot_download

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
    identity = {
        "source": source,
        "resolved_path": str(root),
        "revision": config["model"]["revision"],
        "files": {str(p.relative_to(root)): file_sha256(p) for p in files},
        "dtype": config["model"]["dtype"],
        "scoring_temperature": 1.0,
        "enable_thinking": config["generation"]["enable_thinking"],
    }
    return root, identity


class LocalHFBackend:
    """One frozen local causal LM for task generation, reflection and scoring."""

    def __init__(self, config: dict, recorder: Recorder):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config, self.recorder = config, recorder
        root, self.identity = local_model_identity(config)
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
        return task_messages(prompt, question, self.config.get("benchmark", {}).get("task_format", "plain"))

    def prompt_ids(self, prompt: str, question: str) -> tuple[int, ...]:
        return self._encode_messages(self.messages(prompt, question))

    def _encode_messages(self, messages):
        ids = self.tokenizer.apply_chat_template(
            messages,
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
        prefix = (
            self._encode_messages(task_messages(prompt, question))
            if reflection
            else self.prompt_ids(prompt, question)
        )
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
            kwargs.update(
                temperature=settings["temperature"],
                top_p=settings["top_p"],
                top_k=settings.get("top_k", 0),
            )
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

    def decode_tokens(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def response_token_indices(self, trajectory):
        """Score after the closing thinking token; retain original reasoning as context."""
        ids = trajectory.response_ids
        vocabulary = self.tokenizer.get_vocab()
        opening, closing = vocabulary.get("<think>"), vocabulary.get("</think>")
        closes = [i for i, token in enumerate(ids) if token == closing]
        opens = [i for i, token in enumerate(ids) if token == opening]
        if opens and (not closes or opens[-1] > closes[-1]):
            return ()
        if trajectory.thinking_open_in_prefix and not closes:
            return ()
        start = closes[-1] + 1 if closes else 0
        return tuple(range(start, len(ids)))

    def logprob_statistics(self, prompt_ids, response_ids):
        self._check_length(len(prompt_ids) + len(response_ids))
        return token_logprobs(
            self.model,
            prompt_ids,
            response_ids,
            head_chunk=self.config["model"]["head_chunk"],
            return_top1=True,
        )

    def logprobs(self, prompt_ids: tuple[int, ...], response_ids: tuple[int, ...]) -> tuple[float, ...]:
        self._check_length(len(prompt_ids) + len(response_ids))
        return token_logprobs(
            self.model, prompt_ids, response_ids, head_chunk=self.config["model"]["head_chunk"]
        )


def validate_rollout_logprobs_mode(manifest, requested):
    """Bind the requested probability meaning to the recorded server startup command."""
    command = manifest.get("command")
    if not isinstance(command, list) or not all(isinstance(v, str) for v in command):
        raise ValueError("Rollout logprob capture requires the server startup command in its manifest")
    modes = []
    for index, argument in enumerate(command):
        if argument == "--logprobs-mode":
            if index + 1 >= len(command):
                raise ValueError("Missing server --logprobs-mode value")
            modes.append(command[index + 1])
        elif argument.startswith("--logprobs-mode="):
            modes.append(argument.split("=", 1)[1])
    if len(modes) > 1:
        raise ValueError("Ambiguous server --logprobs-mode arguments")
    actual = modes[0] if modes else "raw_logprobs"
    if manifest.get("logprobs_mode", actual) != actual or actual != requested:
        raise ValueError("Requested rollout logprobs mode differs from the server startup command")
    return actual


class VLLMBackend(LocalHFBackend):
    """Local chat rendering, exact-ID completions and T=1 prompt scoring over HTTP."""

    def __init__(self, config: dict, recorder: Recorder):
        import json

        from transformers import AutoConfig, AutoTokenizer, GenerationConfig

        self.config, self.recorder = config, recorder
        root, self.identity = local_model_identity(config)
        self.tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=False)
        model_config = AutoConfig.from_pretrained(root, trust_remote_code=False)
        if config["model"]["max_context_tokens"] > model_config.max_position_embeddings:
            raise ValueError("Server context must not exceed the pinned model context")
        eos = GenerationConfig.from_pretrained(root).eos_token_id
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos or [self.tokenizer.eos_token_id])
        self.serving = config["serving"]
        manifest_path = Path(self.serving["server_manifest"])
        manifest = json.loads(manifest_path.read_text())
        expected = {
            "model_path": str(root),
            "revision": config["model"]["revision"],
            "dtype": config["model"]["dtype"],
            "max_context_tokens": config["model"]["max_context_tokens"],
            "served_model": self.serving["model"],
            "base_url": self.serving["base_url"],
        }
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError("Serving manifest differs from the requested model/protocol")
        advertised = self._request("/models", None)
        matches = [m for m in advertised["data"] if m["id"] == self.serving["model"]]
        if len(matches) != 1 or matches[0].get("root") != str(root):
            raise ValueError("Live server does not advertise the pinned local model path")
        if matches[0].get("max_model_len") != config["model"]["max_context_tokens"]:
            raise ValueError("Live server context differs from the requested context")
        version = self._request("/version", None, version=True)["version"]
        if version != manifest["vllm_version"]:
            raise ValueError("Live server version differs from serving manifest")
        self.identity.update(
            backend="vllm_openai_completions",
            server=manifest,
            server_manifest_sha256=file_sha256(manifest_path),
        )

        self.rollout_logprobs_mode = self.serving.get("rollout_logprobs")
        if self.rollout_logprobs_mode is not None:
            validate_rollout_logprobs_mode(manifest, self.rollout_logprobs_mode)
            self.identity["rollout_logprobs_mode"] = self.rollout_logprobs_mode

        self.replicas = []
        for peer in self.serving.get("replicas", []):
            import copy

            peer_config = copy.deepcopy(config)
            peer_config["serving"].pop("replicas", None)
            peer_config["serving"].update(peer)
            replica = VLLMBackend(peer_config, recorder)
            if replica.identity["files"] != self.identity["files"]:
                raise ValueError("Replica weights differ from primary")
            self.replicas.append(replica)
        if self.replicas:
            self.identity["replica_servers"] = [peer.identity["server"] for peer in self.replicas]

    def _request(self, path, payload, *, version=False):
        import json
        import urllib.error
        import urllib.request

        base = self.serving["base_url"].removesuffix("/v1") if version else self.serving["base_url"]
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request(base + path, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.serving["timeout_seconds"]) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise ValueError(f"vLLM HTTP {error.code}: {error.read().decode()}") from error

    def _check_length(self, size):
        if size > self.config["model"]["max_context_tokens"]:
            raise ValueError(f"Request needs {size} tokens; truncation is not permitted")

    def _completion(self, prefix, **settings):
        result = self._request(
            "/completions",
            {
                "model": self.serving["model"],
                "prompt": list(prefix),
                "return_token_ids": True,
                "skip_special_tokens": False,
                "repetition_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                **settings,
            },
        )
        if len(result["choices"]) != 1:
            raise ValueError("Expected exactly one completion")
        choice = result["choices"][0]
        if tuple(choice.get("prompt_token_ids") or ()) != tuple(prefix):
            raise ValueError("Server changed the supplied prompt token IDs")
        return choice

    def generate(self, prompt, question, *, reflection=False):
        settings = self.config["reflection"] if reflection else self.config["generation"]
        prefix = (
            self._encode_messages(task_messages(prompt, question))
            if reflection
            else self.prompt_ids(prompt, question)
        )
        self._check_length(len(prefix) + settings["max_new_tokens"])
        capture_mode = self.serving.get("rollout_logprobs") if not reflection else None
        if capture_mode == "processed_logprobs" and settings["temperature"] <= 0:
            raise ValueError("Processed rollout logprobs require stochastic generation")
        started = time.perf_counter()
        choice = self._completion(
            prefix,
            max_tokens=settings["max_new_tokens"],
            temperature=settings["temperature"],
            top_p=settings["top_p"],
            top_k=settings.get("top_k", 0) or -1,
            **({"logprobs": 0, "return_tokens_as_token_ids": True, "min_p": 0.0} if capture_mode else {}),
        )
        ids = choice.get("token_ids")
        if not ids or any(type(t) is not int for t in ids):
            raise ValueError("vLLM must return actual generated token IDs; text retokenization is forbidden")
        response = tuple(ids)
        if choice["finish_reason"] == "length":
            finish = "length"
        elif choice["finish_reason"] == "stop" and response[-1] in self.eos_ids:
            finish = "eos"
        else:
            raise ValueError("Completion stop did not retain an actual EOS token")
        token_logprobs = None
        sampling = None
        if capture_mode:
            logprobs = choice.get("logprobs") or {}
            values = logprobs.get("token_logprobs")
            if not isinstance(values, list) or len(values) != len(response):
                raise ValueError("Rollout token logprobs must align with every response ID including EOS")
            if logprobs.get("tokens") != [f"token_id:{token}" for token in response]:
                raise ValueError("Rollout logprob token IDs differ from the actual sampled response IDs")
            if any(type(v) not in (float, int) or not math.isfinite(v) or v > 0 or v <= -9999 for v in values):
                raise ValueError("Invalid, missing or API-clipped sampled-token logprob")
            token_logprobs = tuple(float(v) for v in values)
            sampling = {
                "temperature": settings["temperature"], "top_p": settings["top_p"],
                "top_k": settings.get("top_k", 0) or -1,
                "repetition_penalty": 1.0, "presence_penalty": 0.0,
                "frequency_penalty": 0.0, "min_p": 0.0,
                "max_new_tokens": settings["max_new_tokens"],
            }
        displayed = response[:-1] if finish == "eos" else response
        text = self.tokenizer.decode(displayed, skip_special_tokens=False)
        self.recorder.charge(
            "reflection" if reflection else "task",
            prompt_tokens=len(prefix),
            response_tokens=len(response),
            elapsed=time.perf_counter() - started,
        )
        return Generation(
            text, prefix, response, finish, self.decode_tokens(prefix).rstrip().endswith("<think>"),
            rollout_token_logprobs=token_logprobs,
            rollout_logprobs_mode=capture_mode,
            rollout_sampling=sampling,
        )

    def generate_many(self, prompt, questions, *, on_result=None):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        engines = [self, *getattr(self, "replicas", [])]
        outputs = [None] * len(questions)
        with ThreadPoolExecutor(max_workers=self.serving["max_concurrency"]) as pool:
            futures = {
                pool.submit(engines[i % len(engines)].generate, prompt, question): i
                for i, question in enumerate(questions)
            }
            for future in as_completed(futures):
                index = futures[future]
                outputs[index] = future.result()
                if on_result is not None:
                    on_result(index, outputs[index])
        return outputs

    def logprobs(self, prompt_ids, response_ids):
        return self._fixed_logprobs(prompt_ids, response_ids, top1=False)[0]

    def logprob_statistics(self, prompt_ids, response_ids):
        return self._fixed_logprobs(prompt_ids, response_ids, top1=True)

    def _fixed_logprobs(self, prompt_ids, response_ids, *, top1):
        # One dummy continuation is discarded; only supplied, fixed response IDs are scored.
        joined = tuple(prompt_ids) + tuple(response_ids)
        self._check_length(len(joined) + 1)
        choice = self._completion(
            joined,
            max_tokens=1,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            prompt_logprobs=1 if top1 else 0,
        )
        rows = choice.get("prompt_logprobs")
        if rows is None or len(rows) != len(joined):
            raise ValueError("vLLM prompt logprobs must cover every supplied token position")
        result, maxima = [], []
        for position, token_id in enumerate(response_ids, len(prompt_ids)):
            row = rows[position]
            if row is None or str(token_id) not in row:
                raise ValueError("Missing fixed-response token logprob at its causal position")
            result.append(float(row[str(token_id)]["logprob"]))
            if top1:
                maxima.append(max(float(entry["logprob"]) for entry in row.values()))
        return tuple(result), tuple(maxima) if top1 else None


def generate_batch(backend, prompt, questions, *, on_result=None):
    """Preserve input order; stream completed jobs for durable validation progress."""
    if hasattr(backend, "generate_many"):
        if on_result is None:
            return backend.generate_many(prompt, questions)
        return backend.generate_many(prompt, questions, on_result=on_result)
    results = []
    for index, question in enumerate(questions):
        output = backend.generate(prompt, question)
        results.append(output)
        if on_result is not None:
            on_result(index, output)
    return results
