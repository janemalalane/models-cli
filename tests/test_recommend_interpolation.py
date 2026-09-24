# Copyright 2026 Google LLC

from click.testing import CliRunner

from google.models.cli.recommend.cmd_recommend import recommend_cmd
from google.models.cli.recommend.interpolate_utils import (
    interpolate_profile_metrics,
    interpolate_recommendations,
)


def test_interpolate_profile_metrics_idw():
    baselines = [
        {
            "average_input_length": 128,
            "average_output_length": 128,
            "ttft_ms": 75,
            "ntpot_ms": 64,
            "itl_ms": 40,
            "output_tokens_per_sec": 4509,
        },
        {
            "average_input_length": 8192,
            "average_output_length": 256,
            "ttft_ms": 394,
            "ntpot_ms": 29,
            "itl_ms": 23,
            "output_tokens_per_sec": 249,
        },
    ]

    # Target close to the 8192/256 profile
    res = interpolate_profile_metrics(baselines, target_input_tokens=8000, target_output_tokens=250)
    assert res["ttft_ms"] is not None
    # TTFT should be much closer to 394 than 75
    assert 300 < res["ttft_ms"] <= 394
    assert res["estimated_request_latency_ms"] is not None

def test_interpolate_recommendations_marks_interpolated():
    recs = [
        {
            "machine_type": "a4-highgpu-8g",
            "accelerator_type": "nvidia-b200",
            "accelerator_count": 8,
            "chip_name": "nvidia-b200 (8x)",
            "average_input_length": 128,
            "average_output_length": 128,
            "ttft_ms": 75,
            "ntpot_ms": 64,
            "itl_ms": 40,
            "output_tokens_per_sec": 4509,
            "model_server": "vllm",
            "engine_params": {"tensor_parallel_size": 8},
            "input_cost_per_m": None,
            "output_cost_per_m": None,
        }
    ]

    interp = interpolate_recommendations(recs, target_input_tokens=4000, target_output_tokens=500)
    assert len(interp) == 1
    item = interp[0]
    assert item["is_interpolated"] is True
    assert item["profile_status"] == "interpolated"
    assert item["custom_input_tokens"] == 4000
    assert item["custom_output_tokens"] == 500
    assert item["custom_ratio"] == 8.0
    assert "Custom" in item["use_case"]

def test_cli_recommend_with_custom_distribution(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "google.models.cli.recommend.cmd_recommend.ensure_authenticated",
        lambda **kwargs: True,
    )
    mock_recs = [
        {
            "machine_type": "a4-highgpu-8g",
            "accelerator_type": "nvidia-b200",
            "accelerator_count": 8,
            "chip_name": "nvidia-b200 (8x)",
            "input_cost_per_m": 0.5,
            "output_cost_per_m": 1.5,
            "ttft_ms": 75,
            "ntpot_ms": 64,
            "itl_ms": 40,
            "output_tokens_per_sec": 4509,
            "average_input_length": 128,
            "average_output_length": 128,
            "model_server": "vllm",
            "engine_params": {"tensor_parallel_size": 8},
        }
    ]
    monkeypatch.setattr(
        "google.models.cli.recommend.cmd_recommend.get_recommendations",
        lambda **kwargs: interpolate_recommendations(mock_recs, kwargs["input_tokens"], kwargs["output_tokens"]),
    )

    result = runner.invoke(
        recommend_cmd,
        [
            "--model", "Qwen/Qwen3-235B-A22B",
            "--input-tokens", "4000",
            "--output-tokens", "500",
        ],
    )
    assert result.exit_code == 0
    assert "Custom Workload Distribution" in result.output
    assert "4,000" in result.output
    assert "500" in result.output
    assert "Ratio: 8:1" in result.output
    assert "8:1.0" not in result.output
    assert "interpolated" in result.output


def test_cli_recommend_ratio_formatting_generation_workload(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "google.models.cli.recommend.cmd_recommend.ensure_authenticated",
        lambda **kwargs: True,
    )
    mock_recs = [
        {
            "machine_type": "g4-standard-48",
            "accelerator_type": "nvidia-rtx-pro-6000",
            "accelerator_count": 1,
            "chip_name": "nvidia-rtx-pro-6000 (1x)",
            "input_cost_per_m": 0.5,
            "output_cost_per_m": 1.5,
            "ttft_ms": 75,
            "ntpot_ms": 64,
            "itl_ms": 40,
            "output_tokens_per_sec": 4509,
            "average_input_length": 128,
            "average_output_length": 128,
            "model_server": "vllm",
            "engine_params": {"tensor_parallel_size": 1},
        }
    ]
    monkeypatch.setattr(
        "google.models.cli.recommend.cmd_recommend.get_recommendations",
        lambda **kwargs: interpolate_recommendations(mock_recs, kwargs["input_tokens"], kwargs["output_tokens"]),
    )

    result = runner.invoke(
        recommend_cmd,
        [
            "--model", "google/gemma-4-31B-it",
            "--input-tokens", "500",
            "--output-tokens", "2000",
        ],
    )
    assert result.exit_code == 0
    assert "Ratio: 1:4" in result.output
    assert "1:4.0" not in result.output

def test_cli_recommend_validation_only_one_token_flag():
    runner = CliRunner()
    result = runner.invoke(
        recommend_cmd,
        ["--model", "Qwen/Qwen3-235B-A22B", "--input-tokens", "4000"],
    )
    assert result.exit_code == 1
    assert "Both --input-tokens" in result.output


def test_interpolate_recommendations_scales_cost():
    # Baseline with known throughput and cost
    recs = [
        {
            "machine_type": "a3-highgpu-4g",
            "accelerator_type": "nvidia-h100-80gb",
            "accelerator_count": 4,
            "chip_name": "nvidia-h100-80gb (4x)",
            "average_input_length": 131,
            "average_output_length": 126,
            "input_cost_per_m": 0.2799,
            "output_cost_per_m": 1.1195,
            "output_input_cost_ratio": 4.0,
            "output_tokens_per_sec": 8698,
            "ttft_ms": 38,
            "ntpot_ms": 17,
            "itl_ms": 12,
            "model_server": "vllm",
            "engine_params": {"tensor_parallel_size": 4},
        }
    ]

    # Hourly cost of baseline:
    # 0.2799 * (131 / 126 + 4.0) * 8698 * 3600 / 1e6 ~= 44.17 $/hr
    # Interpolating with custom distribution (e.g., target 4000 in, 500 out -> ratio 8.0)
    interp = interpolate_recommendations(recs, target_input_tokens=4000, target_output_tokens=500)
    assert len(interp) == 1
    res = interp[0]
    assert res["input_cost_per_m"] is not None
    assert res["output_cost_per_m"] is not None
    # Output cost should be exactly 4x input cost
    assert round(res["output_cost_per_m"] / res["input_cost_per_m"], 2) == 4.0
    # Higher input prompt length should result in scaled cost matching hardware rate
    assert res["input_cost_per_m"] > 0.1

