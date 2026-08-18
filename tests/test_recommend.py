# Copyright 2026 Google LLC
from pathlib import Path
import yaml
from google.models.cli.common.constants import AcceleratorFamily, OptimizationMetric
from google.models.cli.recommend.recommend_utils import (
    apply_recommendation,
    estimate_model_memory_gb,
    extract_parameter_count_billions,
    generate_recommendations,
)


def test_extract_parameter_count():
    assert extract_parameter_count_billions("google/gemma-4-31B-it") == 31.0
    assert extract_parameter_count_billions("google/gemma-2-9b-it") == 9.0
    assert extract_parameter_count_billions("meta-llama/Llama-3.1-70B-Instruct") == 70.0
    assert extract_parameter_count_billions("custom-model") == 31.0


def test_estimate_model_memory():
    mem_fp8 = estimate_model_memory_gb(31.0, precision="fp8")
    assert mem_fp8 > 31.0
    mem_fp16 = estimate_model_memory_gb(31.0, precision="fp16")
    assert mem_fp16 > mem_fp8


def test_generate_recommendations_gemma_31b():
    recs = generate_recommendations(
        model_id="google/gemma-4-31B-it",
        objective=OptimizationMetric.COST,
        family=AcceleratorFamily.ANY,
    )
    assert len(recs) > 0
    # Ensure all recommended machines have >= 31GB total VRAM
    for r in recs:
        assert r["vram_gb"] >= 35.0


def test_generate_recommendations_tpu_only():
    recs = generate_recommendations(
        model_id="google/gemma-4-31B-it",
        objective=OptimizationMetric.TTFT,
        family=AcceleratorFamily.TPU,
    )
    assert len(recs) > 0
    for r in recs:
        assert r["family"] == "tpu"


def test_apply_recommendation(tmp_path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True)
    deploy_file = cfg_dir / "deployment_spec.yaml"
    deploy_file.write_text("deployment:\n  machine_type: null\n")

    engine_file = cfg_dir / "engine_config.yaml"
    engine_file.write_text("engine: vllm\ntensor_parallel_size: 1\n")

    recs = generate_recommendations("google/gemma-4-31B-it", objective=OptimizationMetric.COST)
    updated = apply_recommendation(recs[0], tmp_path)
    assert updated is True

    with open(deploy_file, "r") as f:
        deploy_data = yaml.safe_load(f)
    assert deploy_data["deployment"]["machine_type"] == recs[0]["machine_type"]
