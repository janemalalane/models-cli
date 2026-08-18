# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recommendation heuristics and hardware matching for models-cli."""

import re
from pathlib import Path
from typing import Any, Optional
import yaml
from google.models.cli.common.constants import (
    HARDWARE_SPECS,
    AcceleratorFamily,
    InferenceEngine,
    OptimizationMetric,
)


def extract_parameter_count_billions(model_id: str) -> float:
    """Extracts approximate parameter count in billions from model repo name.

    Examples:
        'google/gemma-4-31B-it' -> 31.0
        'google/gemma-2-9b-it' -> 9.0
        'meta-llama/Llama-3.1-70B-Instruct' -> 70.0
        'mistralai/Mistral-7B-v0.1' -> 7.0
    """
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", model_id)
    if match:
        return float(match.group(1))
    # Default fallback
    return 31.0


def estimate_model_memory_gb(param_count_b: float, precision: str = "fp8") -> float:
    """Estimates required VRAM (weights + KV cache + runtime headroom) in GB.

    Args:
        param_count_b: Parameter count in billions.
        precision: Weight precision ('fp16', 'bf16', 'fp8', 'int4').

    Returns:
        Estimated required VRAM in gigabytes.
    """
    bytes_per_param = {
        "fp16": 2.0,
        "bf16": 2.0,
        "fp8": 1.0,
        "int4": 0.5,
    }.get(precision.lower(), 1.0)

    weights_gb = param_count_b * bytes_per_param
    # Headroom for KV-cache (4K-8K context) and CUDA/TPU runtime context (~30%)
    total_required_gb = weights_gb * 1.35
    return total_required_gb


def generate_recommendations(
    model_id: str,
    objective: OptimizationMetric = OptimizationMetric.COST,
    family: AcceleratorFamily = AcceleratorFamily.ANY,
    max_budget_hourly: Optional[float] = None,
    engine: InferenceEngine = InferenceEngine.VLLM,
) -> list[dict[str, Any]]:
    """Generates ranked hardware configurations and engine parameters for a given model.

    Args:
        model_id: Hugging Face model repository identifier.
        objective: Optimization target (ttft, tpot, throughput, cost).
        family: Filter by GPU, TPU, or ANY.
        max_budget_hourly: Optional maximum USD hourly rate ceiling.
        engine: Target serving engine (vllm or sglang).

    Returns:
        List of recommendation dictionaries sorted by suitability for the objective.
    """
    param_b = extract_parameter_count_billions(model_id)
    required_vram_gb = estimate_model_memory_gb(param_b, precision="fp8")

    candidates: list[dict[str, Any]] = []

    for machine_type, spec in HARDWARE_SPECS.items():
        # Check accelerator family filter
        if family != AcceleratorFamily.ANY and spec["family"] != family:
            continue

        # Check budget filter
        if max_budget_hourly and spec["hourly_cost_usd"] > max_budget_hourly:
            continue

        # Check if hardware has enough total VRAM
        total_vram = spec["vram_gb"]
        if total_vram < required_vram_gb:
            continue

        # Determine Tensor Parallelism
        chip_count = spec.get("accelerator_count", 1)
        tp_size = max(1, chip_count)

        # Generate engine parameters
        engine_params = {
            "tensor_parallel_size": tp_size,
            "kv_cache_dtype": "fp8",
            "max_model_len": 4096 if param_b >= 30 else 8192,
            "gpu_memory_utilization": 0.90,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
        }

        # Estimate performance scores
        bandwidth = spec.get("memory_bandwidth_gb_s", 500)
        cost = spec["hourly_cost_usd"]

        # Relative scoring heuristics
        est_ttft_ms = round(max(35.0, (param_b * 1200.0) / bandwidth), 1)
        est_tpot_ms = round(max(8.0, (param_b * 450.0) / bandwidth), 1)
        est_throughput_tokens_sec = round((bandwidth / (param_b * 1.0)) * 0.75, 1)

        candidate = {
            "machine_type": machine_type,
            "chip_name": spec["chip_name"],
            "family": spec["family"].value,
            "vram_gb": spec["vram_gb"],
            "hourly_cost_usd": cost,
            "est_ttft_ms": est_ttft_ms,
            "est_tpot_ms": est_tpot_ms,
            "est_throughput_tokens_sec": est_throughput_tokens_sec,
            "engine_params": engine_params,
            "description": spec["description"],
        }
        candidates.append(candidate)

    # Ranking logic
    if objective == OptimizationMetric.COST:
        candidates.sort(key=lambda x: (x["hourly_cost_usd"], x["est_ttft_ms"]))
    elif objective == OptimizationMetric.TTFT:
        candidates.sort(key=lambda x: (x["est_ttft_ms"], x["hourly_cost_usd"]))
    elif objective == OptimizationMetric.TPOT:
        candidates.sort(key=lambda x: (x["est_tpot_ms"], x["hourly_cost_usd"]))
    elif objective == OptimizationMetric.THROUGHPUT:
        candidates.sort(key=lambda x: (-x["est_throughput_tokens_sec"], x["hourly_cost_usd"]))

    return candidates


def apply_recommendation(
    top_candidate: dict[str, Any],
    project_dir: Path,
) -> bool:
    """Updates deployment_spec.yaml and engine_config.yaml with the recommended settings."""
    updated = False
    spec = HARDWARE_SPECS.get(top_candidate["machine_type"], {})

    # 1. Update config/deployment_spec.yaml
    deploy_file = project_dir / "config" / "deployment_spec.yaml"
    if deploy_file.is_file():
        try:
            with open(deploy_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            dep = data.setdefault("deployment", {})
            dep["machine_type"] = top_candidate["machine_type"]
            dep["accelerator_type"] = spec.get("accelerator_type")
            dep["accelerator_count"] = spec.get("accelerator_count")
            with open(deploy_file, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)
            updated = True
        except Exception:
            pass

    # 2. Update config/engine_config.yaml
    engine_file = project_dir / "config" / "engine_config.yaml"
    if engine_file.is_file():
        try:
            with open(engine_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            params = top_candidate.get("engine_params", {})
            if "tensor_parallel_size" in params:
                data["tensor_parallel_size"] = params["tensor_parallel_size"]
            if "max_model_len" in params:
                data["max_model_len"] = params["max_model_len"]
            with open(engine_file, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)
            updated = True
        except Exception:
            pass

    return updated
