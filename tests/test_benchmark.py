# Copyright 2026 Google LLC
from pathlib import Path
from typer.testing import CliRunner
import yaml
from google.models.cli.benchmark.benchmark_utils import (
    generate_benchmark_config,
    generate_markdown_report,
    parse_lifecycle_metrics,
)
from google.models.cli.main import app

runner = CliRunner()


def test_generate_benchmark_config_mock(tmp_path):
    out_yaml = tmp_path / "config.yaml"
    generate_benchmark_config(
        template_config_path=None,
        output_config_path=out_yaml,
        model_name="google/gemma-4-31B-it",
        mock=True,
    )
    assert out_yaml.is_file()
    with open(out_yaml, "r") as f:
        data = yaml.safe_load(f)
    assert data["server"]["type"] == "mock"


def test_parse_lifecycle_metrics(tmp_path):
    metrics = parse_lifecycle_metrics(tmp_path)
    assert "total_requests" in metrics
    assert "ttft_ms" in metrics
    assert "tpot_ms" in metrics


def test_generate_markdown_report(tmp_path):
    metrics = parse_lifecycle_metrics(tmp_path)
    report_file = generate_markdown_report(tmp_path, "google/gemma-4-31B-it", metrics)
    assert report_file.is_file()
    content = report_file.read_text()
    assert "Performance Benchmark Report" in content
    assert "google/gemma-4-31B-it" in content


def test_benchmark_command_mock(tmp_path):
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--model",
            "google/gemma-4-31B-it",
            "--output-dir",
            str(tmp_path / "bench-reports"),
            "--mock",
        ],
    )
    assert result.exit_code == 0
    assert "Inference Benchmark Summary" in result.stdout
