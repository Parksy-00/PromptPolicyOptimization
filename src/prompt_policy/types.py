from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    answer: str
    split: str


@dataclass(frozen=True)
class Generation:
    text: str
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    finish_reason: str
    thinking_open_in_prefix: bool = False


@dataclass(frozen=True)
class Trajectory:
    example_id: str
    question: str
    split: str
    repeat_index: int
    prompt: str
    text: str
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    reward: float
    finish_reason: str
    thinking_open_in_prefix: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> Trajectory:
        value = dict(value)
        value["prompt_ids"] = tuple(value["prompt_ids"])
        value["response_ids"] = tuple(value["response_ids"])
        return cls(**value)


class Backend(Protocol):
    identity: dict

    def prompt_ids(self, prompt: str, question: str) -> tuple[int, ...]: ...

    def generate(self, prompt: str, question: str, *, reflection: bool = False) -> Generation: ...

    def logprobs(self, prompt_ids: tuple[int, ...], response_ids: tuple[int, ...]) -> tuple[float, ...]: ...
