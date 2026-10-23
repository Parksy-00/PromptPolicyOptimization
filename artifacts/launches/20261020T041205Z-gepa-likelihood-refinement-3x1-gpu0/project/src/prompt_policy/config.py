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
    if set(config) - {"textgrad", "edit_refinement", "gepa_refinement", "serving"} != {*SCHEMA, "initial_prompt"}:
        raise ValueError("Config requires initial_prompt and exactly: " + ", ".join(SCHEMA))
    if sum(name in config for name in ("textgrad", "edit_refinement", "gepa_refinement")) > 1:
        raise ValueError("Choose only one optimizer extension: textgrad, edit_refinement or gepa_refinement")
    if not isinstance(config["initial_prompt"], str) or not config["initial_prompt"].strip():
        raise ValueError("initial_prompt must be nonempty")
    simple_edit = "edit_refinement" in config
    directional = simple_edit or "gepa_refinement" in config
    if simple_edit:
        config["search"].setdefault("max_iterations", 5)
        config["search"].setdefault("validation_rollouts", 1)
    for section, keys in SCHEMA.items():
        if directional and section == "objective":
            keys = {"mode"}
        if (
            not isinstance(config[section], dict)
            or set(config[section])
            - (
                {"task_format", "allow_within_split_question_duplicates"}
                if section == "benchmark"
                else {"top_k"}
                if section in {"generation", "reflection"}
                else {"base_validation_run"}
                if simple_edit and section == "search"
                else set()
            )
            != keys
        ):
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
    if "base_validation_run" in config["search"]:
        source_run = config["search"]["base_validation_run"]
        if not isinstance(source_run, str) or not source_run.strip():
            raise ValueError("search.base_validation_run must be a nonempty run directory")
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
        top_k = config[section].get("top_k", 0)
        if type(top_k) is not int or top_k < 0:
            raise ValueError(f"{section}.top_k must be a nonnegative integer (0 disables truncation)")
    if type(config["generation"]["enable_thinking"]) is not bool:
        raise ValueError("enable_thinking must be boolean")
    if directional:
        settings = config["edit_refinement" if simple_edit else "gepa_refinement"]
        allowed = {"mode", "prompt_policy", "likelihood_scope", "drift_correction", "branch_probability_threshold", "score_unit"}
        if not simple_edit:
            allowed -= {"mode", "prompt_policy"}
        if not isinstance(settings, dict) or set(settings) - allowed:
            raise ValueError(
                f"Unknown {'edit_refinement' if simple_edit else 'gepa_refinement'} setting; "
                "old recipes require their captured source snapshot"
            )
        mode = settings.get("mode", "validation_gated")
        if mode not in {"validation_gated", "validation_gated_per_example", "per_example_likelihood_probe"}:
            raise ValueError("Unknown edit_refinement mode")
        if mode == "per_example_likelihood_probe":
            if settings.get("prompt_policy") != "fixed":
                raise ValueError("Likelihood probe requires explicit prompt_policy=fixed")
            if "base_validation_run" in config["search"]:
                raise ValueError("Likelihood probe does not use base_validation_run")
        elif "prompt_policy" in settings:
            raise ValueError("prompt_policy is only valid for per-example likelihood probes")
        if settings.get("likelihood_scope", "full") not in {"full", "response", "branch"}:
            raise ValueError("likelihood_scope must be full, response or branch")
        if settings.get("drift_correction", "none") not in {"none", "question_mean"}:
            raise ValueError("drift_correction must be none or question_mean")
        if settings.get("score_unit", "sequence") not in {"sequence", "mean"}:
            raise ValueError("score_unit must be sequence or mean")
        if settings.get("drift_correction", "none") == "question_mean":
            if config["search"]["train_rollouts"] < 2 or settings.get("score_unit", "sequence") != "mean":
                raise ValueError("question_mean requires multiple train_rollouts and score_unit=mean")
        threshold = settings.get("branch_probability_threshold")
        if settings.get("likelihood_scope", "full") == "branch":
            if type(threshold) not in {int, float} or not math.isfinite(threshold) or not 0 < threshold < 1:
                raise ValueError("branch scope requires branch_probability_threshold in (0, 1)")
        elif threshold is not None:
            raise ValueError("branch_probability_threshold is only valid for branch scope")
        if config["objective"]["mode"] != "directional_likelihood":
            raise ValueError("Likelihood refinement uses directional_likelihood")
        if config["benchmark"].get("task_format", "plain") == "dapo_dspy_v1":
            raise ValueError("Simple editing uses plain or a supported integer task format")
    else:
        objective = config["objective"]
        if objective["mode"] not in {
            "accuracy",
            "success_likelihood",
            "success_likelihood_ascent",
        }:
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
        if "textgrad" in config:
            settings = config["textgrad"]
            if (
                set(settings) != {"proposals_per_batch"}
                or type(settings["proposals_per_batch"]) is not int
                or settings["proposals_per_batch"] < 1
            ):
                raise ValueError("textgrad requires positive integer proposals_per_batch")
            if objective["score_failures"]:
                raise ValueError("TextGrad first version requires score_failures=false")
            if objective["mode"] not in {"accuracy", "success_likelihood_ascent"} or objective["weight"] != 0:
                raise ValueError(
                    "TextGrad uses accuracy or success_likelihood_ascent, without an additive weight"
                )
            if objective["mode"] == "success_likelihood_ascent" and not (
                objective["measure_likelihood"] and objective["likelihood_feedback"]
            ):
                raise ValueError("TextGrad likelihood ascent requires measurement and numerical feedback")
            if objective["mode"] == "accuracy" and objective["likelihood_feedback"]:
                raise ValueError("Accuracy-only TextGrad must not receive likelihood feedback")
        elif objective["mode"] == "success_likelihood_ascent":
            raise ValueError("success_likelihood_ascent requires the textgrad optimizer")
    if "serving" in config:
        from urllib.parse import urlparse

        serving = config["serving"]
        if set(serving) - {"replicas"} != {
            "base_url",
            "model",
            "max_concurrency",
            "timeout_seconds",
            "server_manifest",
        }:
            raise ValueError(
                "serving requires base_url, model, max_concurrency, timeout_seconds, server_manifest"
            )
        replicas = serving.get("replicas", [])
        if not isinstance(replicas, list) or any(
            not isinstance(peer, dict) or set(peer) != {"base_url", "server_manifest"} for peer in replicas
        ):
            raise ValueError("serving.replicas must list base_url/server_manifest pairs")
        endpoints = [serving, *replicas]
        if len({e["base_url"] for e in endpoints}) != len(endpoints):
            raise ValueError("Replica endpoints must be distinct")
        for peer in replicas:
            address = urlparse(peer["base_url"])
            if (
                address.scheme != "http"
                or address.hostname not in {"127.0.0.1", "localhost", "::1"}
                or address.path != "/v1"
            ):
                raise ValueError("Replica endpoints must be local http /v1 URLs")
            if not isinstance(peer["server_manifest"], str) or not peer["server_manifest"].strip():
                raise ValueError("Replica server_manifest must be nonempty")
        url = urlparse(serving["base_url"])
        if url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"} or url.path != "/v1":
            raise ValueError("serving.base_url must be a local http OpenAI-compatible /v1 endpoint")
        for key in ("max_concurrency", "timeout_seconds"):
            if type(serving[key]) is not int or serving[key] < 1:
                raise ValueError(f"serving.{key} must be a positive integer")
        for key in ("model", "server_manifest"):
            if not isinstance(serving[key], str) or not serving[key].strip():
                raise ValueError(f"serving.{key} must be nonempty")
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
    if (
        ("textgrad" in config or "edit_refinement" in config or "gepa_refinement" in config)
        and not config["search"].get("base_validation_run")
        and config.get("edit_refinement", {}).get("mode") != "per_example_likelihood_probe"
        and config["search"]["max_task_rollouts"] < len(
        benchmark.validation
        ) * config["search"]["validation_rollouts"]
    ):
        raise ValueError("Optimizer budget cannot cover the initial complete validation")
    return config, benchmark


def resolve_model(config: dict, base: Path) -> tuple[dict, list[str]]:
    result = copy.deepcopy(config)
    model = result["model"]
    if result["search"].get("base_validation_run"):
        path = Path(result["search"]["base_validation_run"]).expanduser()
        result["search"]["base_validation_run"] = str((path if path.is_absolute() else base / path).resolve())
    if "serving" in result:
        path = Path(result["serving"]["server_manifest"]).expanduser()
        result["serving"]["server_manifest"] = str((path if path.is_absolute() else base / path).resolve())
        for peer in result["serving"].get("replicas", []):
            path = Path(peer["server_manifest"]).expanduser()
            peer["server_manifest"] = str((path if path.is_absolute() else base / path).resolve())
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
