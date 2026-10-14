import copy
from pathlib import Path

import pytest

from prompt_policy.cli import run
from prompt_policy.config import resolve_model, validate_config


def test_dry_run_never_creates_output_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    destination = tmp_path / "new-run"
    plan = run(
        root / "experiments/qwen3_success_likelihood/experiment.toml", dry_run=True, output=destination
    )
    assert not plan["executable"]
    assert "model.name_or_path" in plan["missing"][0]
    assert not destination.exists()


def test_refuse_unpinned_remote_model(config, tmp_path):
    config["model"].update(name_or_path="Qwen/Qwen3-8B", revision="main")
    _, missing = resolve_model(config, tmp_path)
    assert missing


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("search", "sampling_seed", 0),
        ("objective", "weight", float("nan")),
        ("search", "train_rollouts", 0),
        ("generation", "enable_thinking", "yes"),
        ("objective", "measure_likelihood", False),
    ],
)
def test_invalid_conditions_rejected(config, section, key, value):
    value_config = copy.deepcopy(config)
    value_config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(value_config)
