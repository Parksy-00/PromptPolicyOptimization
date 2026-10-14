import json
from pathlib import Path

import pytest
from conftest import FakeBackend

from prompt_policy.cli import run


def prepare_config(tmp_path):
    root = Path(__file__).resolve().parents[1]
    text = (root / "experiments/qwen3_success_likelihood/experiment.toml").read_text()
    text = text.replace(
        'initial_prompt = "Solve the problem carefully. Put your final answer on a line starting with ###."',
        'initial_prompt = "parent"',
    )
    model = tmp_path / "model"
    model.mkdir()
    text = text.replace('name_or_path = ""', f'name_or_path = "{model}"')
    text = text.replace("max_iterations = 20", "max_iterations = 1")
    text = text.replace("enable_thinking = true", "enable_thinking = false")
    for split in ("train", "validation", "test"):
        text = text.replace(f"../../examples/data/{split}.jsonl", str(root / f"examples/data/{split}.jsonl"))
    path = tmp_path / "experiment.toml"
    path.write_text(text)
    return path


def test_cli_writes_complete_run_and_refuses_overwrite(tmp_path, monkeypatch):
    from prompt_policy import backends

    monkeypatch.setattr(backends, "LocalHFBackend", lambda config, recorder: FakeBackend(recorder))
    config = prepare_config(tmp_path)
    output = tmp_path / "run"
    assert run(config, output=output) == output
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["model"]["kind"] == "test_only"
    selected = json.loads((output / "selection.json").read_text())
    result = json.loads((output / "result.json").read_text())
    assert selected["criterion"] == "validation_accuracy"
    assert selected["prompt"] == result["selected_prompt"]
    assert result["test"]["selected"]["reused_initial"]
    assert result["search_cost"]["task_calls"] < result["total_cost"]["task_calls"]
    assert all(
        json.loads(line)["split"] == "test" for line in (output / "test.jsonl").read_text().splitlines()
    )
    with pytest.raises(FileExistsError):
        run(config, output=output)


def test_backend_failure_keeps_failed_manifest(tmp_path, monkeypatch):
    from prompt_policy import backends

    def fail(config, recorder):
        raise RuntimeError("controlled failure")

    monkeypatch.setattr(backends, "LocalHFBackend", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="controlled"):
        run(prepare_config(tmp_path), output=output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "controlled failure" in manifest["error"]
