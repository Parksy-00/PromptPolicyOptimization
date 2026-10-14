"""Exact full-vocabulary teacher forcing. GPU dependencies load only on use."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from prompt_policy.artifacts import Recorder, digest
from prompt_policy.types import Backend, Trajectory


def token_logprobs(model, prompt_ids, response_ids, *, head_chunk: int = 256) -> tuple[float, ...]:
    """Score actual response IDs, including any sampled EOS; add no tokens.

    Causal prediction alignment follows PromptConsolidation's response_hidden:
    the final prefix position predicts response[0]. Serial, unpadded backbone;
    chunked FP32 full-vocabulary log-softmax avoids a response×vocab allocation.
    """
    import torch

    if not prompt_ids or not response_ids or head_chunk < 1:
        raise ValueError("nonempty prefix/response and positive head_chunk required")
    if model.training:
        raise ValueError("teacher-forcing model must be in eval mode")
    device = next(model.parameters()).device
    inputs = torch.tensor([list(prompt_ids) + list(response_ids)], device=device, dtype=torch.long)
    with torch.inference_mode():
        # AutoModelForCausalLM.base_model is the decoder for Qwen3 and other supported decoder LMs.
        hidden = model.base_model(input_ids=inputs, use_cache=False, return_dict=True).last_hidden_state
        start = len(prompt_ids) - 1
        states = hidden[0, start : start + len(response_ids)]
        labels = torch.tensor(response_ids, device=device, dtype=torch.long)
        traces = []
        for offset in range(0, len(response_ids), head_chunk):
            stop = min(offset + head_chunk, len(response_ids))
            logits = model.get_output_embeddings()(states[offset:stop]).float()
            lp = torch.log_softmax(logits, dim=-1)
            traces.append(lp.gather(-1, labels[offset:stop, None]).flatten())
        values = tuple(torch.cat(traces).cpu().tolist())
    if any(not math.isfinite(x) or x > 0 for x in values):
        raise ValueError("non-finite or positive token logprob")
    return values


@dataclass(frozen=True)
class Likelihood:
    token_logprobs: tuple[float, ...]

    @property
    def total(self) -> float:
        return math.fsum(self.token_logprobs)

    @property
    def mean(self) -> float:
        return self.total / len(self.token_logprobs)


class TrajectoryScorer:
    def __init__(self, backend: Backend, recorder: Recorder):
        self.backend = backend
        self.recorder = recorder
        self.cache: dict[str, Likelihood] = {}

    def score(self, prompt: str, trajectory: Trajectory) -> Likelihood:
        prefix = self.backend.prompt_ids(prompt, trajectory.question)
        if prompt == trajectory.prompt and prefix != trajectory.prompt_ids:
            raise ValueError("Re-rendered parent request differs from its captured token IDs")
        key = digest([self.backend.identity, prefix, trajectory.response_ids, "T1_full_vocab_v1"])
        hit = key in self.cache
        started = time.perf_counter()
        if not hit:
            values = tuple(self.backend.logprobs(prefix, trajectory.response_ids))
            if len(values) != len(trajectory.response_ids) or not values:
                raise ValueError("Response IDs and token logprobs must align exactly")
            if any(not math.isfinite(x) or x > 0 for x in values):
                raise ValueError("Invalid token logprobs")
            self.cache[key] = Likelihood(values)
            self.recorder.charge(
                "teacher_forcing",
                prompt_tokens=len(prefix),
                response_tokens=len(values),
                elapsed=time.perf_counter() - started,
            )
        result = self.cache[key]
        self.recorder.append(
            "likelihood.jsonl",
            {
                "example_id": trajectory.example_id,
                "repeat_index": trajectory.repeat_index,
                "split": trajectory.split,
                "reward": trajectory.reward,
                "response_sha256": digest(trajectory.response_ids),
                "prompt": prompt,
                "prompt_ids": prefix,
                "response_ids": trajectory.response_ids,
                "token_logprobs": result.token_logprobs,
                "sum_logprob": result.total,
                "mean_logprob": result.mean,
                "cache_hit": hit,
                "cache_key": key,
            },
        )
        return result
