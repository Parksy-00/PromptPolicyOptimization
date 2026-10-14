from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from prompt_policy.artifacts import digest, file_sha256
from prompt_policy.types import Example


def extract_answer(text: str) -> str | None:
    # Ignore reasoning before the final closed thinking block.
    text = text.rsplit("</think>", 1)[-1]
    if "<think>" in text:
        return None
    start = text.rfind("\\boxed{")
    if start >= 0:
        start += len("\\boxed{")
        depth = 1
        for i in range(start, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                return text[start:i].strip() or None
        return None
    answers = re.findall(r"^\s*(?:###|Answer:)\s*(.*?)\s*$", text, re.MULTILINE | re.IGNORECASE)
    return answers[-1].strip() if answers and answers[-1].strip() else None


def reward(text: str, answer: str, scorer: str) -> float:
    predicted = extract_answer(text)
    if predicted is None:
        return 0.0
    if scorer == "exact_match":
        return float(predicted == answer.strip())
    if scorer == "numeric":
        try:
            return float(Fraction(predicted) == Fraction(answer.strip()))
        except (ValueError, ZeroDivisionError):
            return 0.0
    raise ValueError(f"Unknown scorer: {scorer}")


@dataclass(frozen=True)
class Benchmark:
    name: str
    scorer: str
    train: tuple[Example, ...]
    validation: tuple[Example, ...]
    test: tuple[Example, ...]
    identity: dict

    def score(self, example: Example, text: str, *, thinking_open_in_prefix=False) -> float:
        if thinking_open_in_prefix:
            text = "<think>" + text
        return reward(text, example.answer, self.scorer)


def load_benchmark(config: dict, base: Path) -> Benchmark:
    if config["scorer"] not in {"exact_match", "numeric"}:
        raise ValueError("scorer must be exact_match or numeric")
    splits, sources, ids, questions = {}, {}, set(), set()
    for split in ("train", "validation", "test"):
        path = (base / config[split]).resolve()
        rows = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != {"id", "question", "answer"}:
                raise ValueError(f"{path}:{number}: expected exactly id, question, answer")
            if any(not isinstance(value, str) or not value.strip() for value in row.values()):
                raise ValueError(f"{path}:{number}: values must be nonempty strings")
            question = " ".join(row["question"].split())
            if row["id"] in ids or question in questions:
                raise ValueError(f"Duplicate ID or question across/within splits: {row['id']}")
            ids.add(row["id"])
            questions.add(question)
            if config["scorer"] == "numeric":
                Fraction(row["answer"])
            rows.append(Example(**row, split=split))
        if not rows:
            raise ValueError(f"Empty {split} split")
        splits[split] = tuple(rows)
        sources[split] = {"path": str(path), "sha256": file_sha256(path), "count": len(rows)}
    identity = {
        "name": config["name"],
        "scorer": config["scorer"],
        "scorer_version": 1,
        "sources": sources,
        "fingerprint": digest({s: [asdict(x) for x in rows] for s, rows in splits.items()}),
    }
    return Benchmark(config["name"], config["scorer"], identity=identity, **splits)
