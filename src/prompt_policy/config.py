from __future__ import annotations

import copy
import math
import re
import tomllib
from pathlib import Path

from prompt_policy.benchmarks import load_benchmark

SCHEMA = {
    "benchmark": {"name", "scorer", "train", "validation", "test"},
    "model": {"name_or_path", "revision", "device", "dtype", "max_context_tokens", "head_chunk"},
    "generation": {"temperature", "top_p", "max_new_tokens", "enable_thinking"},
    "reflection": {"temperature", "top_p", "max_new_tokens"},
    "search": {
        "minibatch_size",
        "train_rollouts",
        "validation_rollouts",
        "test_rollouts",
        "max_iterations",
        "max_task_rollouts",
        "search_seed",
    },
    "objective": {
        "mode",
        "weight",
        "max_accuracy_drop",
        "tolerance",
        "measure_likelihood",
        "likelihood_feedback",
        "score_failures",
    },
}


def validate_config(config: dict):
    if set(config) != {*SCHEMA, "initial_prompt"}:
        raise ValueError("Config requires initial_prompt and exactly: " + ", ".join(SCHEMA))
    if not isinstance(config["initial_prompt"], str) or not config["initial_prompt"].strip():
        raise ValueError("initial_prompt must be nonempty")
    for section, keys in SCHEMA.items():
        if not isinstance(config[section], dict) or set(config[section]) != keys:
            raise ValueError(f"{section}: expected exactly {sorted(keys)}")
    for section, keys in {
        "model": ["max_context_tokens", "head_chunk"],
        "generation": ["max_new_tokens"],
        "reflection": ["max_new_tokens"],
        "search": [
            "minibatch_size",
            "train_rollouts",
            "validation_rollouts",
            "test_rollouts",
            "max_iterations",
            "max_task_rollouts",
        ],
    }.items():
        for key in keys:
            value = config[section][key]
            if type(value) is not int or value < 1:
                raise ValueError(f"{section}.{key} must be a positive integer")
    seed = config["search"]["search_seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("search_seed must be a nonnegative integer; it controls search, not sampling")
    for section in ("generation", "reflection"):
        for key in ("temperature", "top_p"):
            value = config[section][key]
            if type(value) not in {int, float} or not math.isfinite(value):
                raise ValueError(f"{section}.{key} must be finite")
        if config[section]["temperature"] < 0 or not 0 < config[section]["top_p"] <= 1:
            raise ValueError(f"Invalid {section} temperature/top_p")
    if type(config["generation"]["enable_thinking"]) is not bool:
        raise ValueError("enable_thinking must be boolean")
    objective = config["objective"]
    if objective["mode"] not in {"accuracy", "success_likelihood"}:
        raise ValueError("Unknown objective mode")
    for key in ("weight", "max_accuracy_drop", "tolerance"):
        value = objective[key]
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            raise ValueError(f"objective.{key} must be finite and nonnegative")
    if objective["max_accuracy_drop"] > 1:
        raise ValueError("max_accuracy_drop must be <= 1")
    for key in ("measure_likelihood", "likelihood_feedback", "score_failures"):
        if type(objective[key]) is not bool:
            raise ValueError(f"objective.{key} must be boolean")
    if objective["mode"] == "success_likelihood" and (
        objective["weight"] <= 0 or not objective["measure_likelihood"]
    ):
        raise ValueError("success_likelihood requires positive weight and measure_likelihood=true")
    if objective["mode"] == "accuracy" and objective["weight"] != 0:
        raise ValueError("accuracy objective requires weight=0")
    if objective["likelihood_feedback"] and not objective["measure_likelihood"]:
        raise ValueError("likelihood_feedback requires measure_likelihood=true")
    model = config["model"]
    if model["dtype"] not in {"float32", "bfloat16", "float16"}:
        raise ValueError("Unsupported dtype")
    if not re.fullmatch(r"cpu|cuda(?::\d+)?", model["device"]):
        raise ValueError("device must be cpu or an explicit cuda device")
    for key in ("name_or_path", "revision"):
        if not isinstance(model[key], str):
            raise ValueError(f"model.{key} must be a string")


def load_config(path: Path):
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    benchmark = load_benchmark(config["benchmark"], path.parent)
    if config["search"]["minibatch_size"] > len(benchmark.train):
        raise ValueError("minibatch_size exceeds train split size")
    return config, benchmark


def resolve_model(config: dict, base: Path) -> tuple[dict, list[str]]:
    result = copy.deepcopy(config)
    model = result["model"]
    missing = []
    if not model["name_or_path"].strip():
        missing.append("model.name_or_path (local directory or Hugging Face repo ID)")
    else:
        path = Path(model["name_or_path"]).expanduser()
        path = path if path.is_absolute() else base / path
        if path.is_dir():
            model["name_or_path"] = str(path.resolve())
        elif not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            missing.append("model.revision (immutable 40-character commit for a remote model)")
    return result, missing
