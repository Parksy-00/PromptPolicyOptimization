from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Recorder:
    """Append-only scientific records; callers create an exclusive run directory."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.cost = Counter()

    def append(self, name: str, value: dict):
        with (self.run_dir / name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

    def write(self, name: str, value: dict):
        path = self.run_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(path)

    def charge(self, kind: str, *, prompt_tokens: int, response_tokens: int, elapsed: float):
        self.cost[f"{kind}_calls"] += 1
        self.cost[f"{kind}_prompt_tokens"] += prompt_tokens
        self.cost[f"{kind}_response_tokens"] += response_tokens
        self.cost[f"{kind}_seconds"] += elapsed
