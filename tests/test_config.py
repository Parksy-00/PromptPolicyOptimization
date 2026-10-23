import copy
from pathlib import Path

import pytest

from prompt_policy.cli import run
from prompt_policy.config import resolve_model, validate_config


def test_removed_gepa_cannot_silently_run_or_create_output(tmp_path):
    root = Path(__file__).resolve().parents[1]
    destination = tmp_path / "new-run"
    with pytest.raises(ValueError, match="Standalone GEPA framework support was removed"):
        run(root / "tests/fixtures/base.toml", dry_run=True, output=destination)
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


@pytest.mark.parametrize("section", ["generation", "reflection"])
@pytest.mark.parametrize("value", [-1, True, 20.0])
def test_invalid_top_k_is_rejected(config, section, value):
    config[section]["top_k"] = value
    with pytest.raises(ValueError, match="top_k"):
        validate_config(config)


def test_removed_gepa_section_is_rejected(config):
    config["gepa_refinement"] = {"enabled": True}
    with pytest.raises(ValueError, match="Config requires"):
        validate_config(config)
