# Copyright 2026 Google LLC
from pathlib import Path
from click.testing import CliRunner
from google.models.cli.eval.eval_utils import (
    evaluate_candidate_models,
    load_golden_dataset,
    update_project_with_winner,
)
from google.models.cli.main import app

runner = CliRunner()


def test_load_golden_dataset_fallback(tmp_path):
    dataset = load_golden_dataset(tmp_path / "non_existent.jsonl")
    assert len(dataset) > 0


def test_evaluate_candidate_models_mock():
    dataset = [{"messages": [{"role": "user", "content": "hello"}], "expected_output": "world"}]
    models = ["google/gemma-4-31B-it", "google/gemma-2-9b-it"]
    results = evaluate_candidate_models(models, dataset, ["accuracy", "quality"], mock=True)

    assert len(results) == 2
    # 31B should rank #1 due to higher quality/accuracy
    assert results[0]["model"] == "google/gemma-4-31B-it"


def test_update_project_with_winner(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text('MODEL_ID="old-model"\nOTHER_VAR="123"\n')

    updated = update_project_with_winner("google/gemma-4-31B-it", tmp_path)
    assert updated is True

    assert 'MODEL_ID="google/gemma-4-31B-it"' in env_file.read_text()


def test_eval_command_mock(tmp_path):
    dataset_file = tmp_path / "golden.jsonl"
    dataset_file.write_text('{"messages": [{"role": "user", "content": "hi"}], "expected_output": "bye"}\n')

    result = runner.invoke(
        app,
        [
            "eval",
            "--models",
            "google/gemma-4-31B-it,google/gemma-2-9b-it",
            "--dataset",
            str(dataset_file),
            "--mock",
        ],
    )
    assert result.exit_code == 0
    assert "Model Evaluation Scorecard" in result.stdout
    assert "Winner:" in result.stdout
