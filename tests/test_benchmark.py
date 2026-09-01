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


def test_find_inference_perf_command_alongside_python(tmp_path, monkeypatch):
    import sys
    from google.models.cli.benchmark.benchmark_utils import find_inference_perf_command

    fake_bin = tmp_path / "inference-perf"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)

    fake_python = tmp_path / "python"
    fake_python.touch()

    monkeypatch.setattr(sys, "executable", str(fake_python))
    cmd = find_inference_perf_command()
    assert cmd == [str(fake_bin)]


def test_find_inference_perf_command_in_path(tmp_path, monkeypatch):
    import sys
    from google.models.cli.benchmark.benchmark_utils import find_inference_perf_command

    # Python dir has no inference-perf
    fake_python = tmp_path / "python"
    fake_python.touch()
    monkeypatch.setattr(sys, "executable", str(fake_python))

    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/inference-perf")
    cmd = find_inference_perf_command()
    assert cmd == ["/usr/local/bin/inference-perf"]


def test_benchmark_live_missing_binary_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "google.models.cli.benchmark.cmd_benchmark.find_inference_perf_command",
        lambda: None,
    )
    monkeypatch.setattr(
        "google.models.cli.benchmark.cmd_benchmark.ensure_authenticated",
        lambda **kwargs: True,
    )

    result = runner.invoke(
        app,
        [
            "benchmark",
            "--model",
            "google/gemma-4-31B-it",
            "--endpoint",
            "http://localhost:8080",
            "--output-dir",
            str(tmp_path / "live-bench"),
        ],
    )
    assert result.exit_code == 1
    assert "inference-perf is required to benchmark a live endpoint" in result.output

