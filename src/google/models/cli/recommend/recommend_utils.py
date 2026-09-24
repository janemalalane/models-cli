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

"""Recommendation utilities powered strictly by the Google Cloud GKE Recommender service."""

from pathlib import Path
from typing import Any, Optional
import yaml
from google.protobuf.json_format import MessageToDict
import google.auth.exceptions
import google.api_core.exceptions

from google.models.cli.common.constants import (
    DEFAULT_PRICING_MODEL,
    HARDWARE_SPECS,
    AcceleratorFamily,
)


class ModelNotSupportedError(Exception):
    """Raised when a model is not supported by the GKE Recommender service."""

    def __init__(self, model_id: str, supported_models: list[str]):
        super().__init__(
            f"Model '{model_id}' is not supported by the GKE Recommender service."
        )
        self.model_id = model_id
        self.supported_models = supported_models


def fetch_supported_models(client: Optional[Any] = None) -> list[str]:
    """Fetches the list of supported model repositories from GKE Recommender."""
    if client is None:
        import google.cloud.gkerecommender_v1 as gke_rec

        client = gke_rec.GkeInferenceQuickstartClient()

    import google.cloud.gkerecommender_v1 as gke_rec

    request = gke_rec.FetchModelsRequest()
    try:
        response = client.fetch_models(request=request)
        return sorted([m for m in response])
    except (
        google.auth.exceptions.GoogleAuthError,
        google.api_core.exceptions.Unauthenticated,
        google.api_core.exceptions.PermissionDenied,
    ):
        raise
    except Exception as exc:
        from google.models.cli.common.auth import is_auth_error

        if is_auth_error(exc):
            raise
        return []


def _extract_workload_spec(p: Any) -> dict[str, Any]:
    """Extracts WorkloadSpec (use_case, average_input_length, average_output_length).

    Supports:
    1. Direct attribute access (forward compatibility with future SDK releases containing workload_spec).
    2. Dict access (when raw dictionary or MessageToDict is used).
    3. Decoding Field 7 from the raw protobuf wire bytes preserved on p._pb.
    """
    # 1. Direct attribute (future SDK versions)
    if hasattr(p, "workload_spec") and p.workload_spec:
        ws = p.workload_spec
        return {
            "use_case": getattr(ws, "use_case", ""),
            "average_input_length": getattr(ws, "average_input_length", 0),
            "average_output_length": getattr(ws, "average_output_length", 0),
        }

    # 2. Dictionary access
    if isinstance(p, dict):
        ws = p.get("workloadSpec", {})
        return {
            "use_case": ws.get("useCase", ""),
            "average_input_length": ws.get("averageInputLength", 0),
            "average_output_length": ws.get("averageOutputLength", 0),
        }

    # 3. Decode Field 7 from raw protobuf wire bytes preserved on the message
    pb = getattr(p, "_pb", None) or (p if hasattr(p, "SerializeToString") else None)
    if pb is None:
        return {}

    try:
        pb_bytes = pb.SerializeToString()
        pos = 0
        workload_spec: dict[str, Any] = {}
        while pos < len(pb_bytes):
            key = 0
            shift = 0
            while pos < len(pb_bytes):
                b = pb_bytes[pos]
                pos += 1
                key |= (b & 0x7F) << shift
                shift += 7
                if not (b & 0x80):
                    break
            field_num = key >> 3
            wire_type = key & 0x07
            if wire_type == 0:
                while pos < len(pb_bytes) and (pb_bytes[pos] & 0x80):
                    pos += 1
                pos += 1
            elif wire_type == 1:
                pos += 8
            elif wire_type == 2:
                length = 0
                shift = 0
                while pos < len(pb_bytes):
                    b = pb_bytes[pos]
                    pos += 1
                    length |= (b & 0x7F) << shift
                    shift += 7
                    if not (b & 0x80):
                        break
                val_bytes = pb_bytes[pos : pos + length]
                pos += length
                if field_num == 7:
                    sub_pos = 0
                    while sub_pos < len(val_bytes):
                        sub_key = 0
                        sub_shift = 0
                        while sub_pos < len(val_bytes):
                            sb = val_bytes[sub_pos]
                            sub_pos += 1
                            sub_key |= (sb & 0x7F) << sub_shift
                            sub_shift += 7
                            if not (sb & 0x80):
                                break
                        sub_field = sub_key >> 3
                        sub_wire = sub_key & 0x07
                        if sub_wire == 0:
                            v = 0
                            s = 0
                            while sub_pos < len(val_bytes):
                                sb = val_bytes[sub_pos]
                                sub_pos += 1
                                v |= (sb & 0x7F) << s
                                s += 7
                                if not (sb & 0x80):
                                    break
                            if sub_field == 1:
                                workload_spec["average_input_length"] = v
                            elif sub_field == 2:
                                workload_spec["average_output_length"] = v
                        elif sub_wire == 2:
                            v_len = 0
                            s = 0
                            while sub_pos < len(val_bytes):
                                sb = val_bytes[sub_pos]
                                sub_pos += 1
                                v_len |= (sb & 0x7F) << s
                                s += 7
                                if not (sb & 0x80):
                                    break
                            s_val = val_bytes[sub_pos : sub_pos + v_len].decode(
                                "utf-8", errors="replace"
                            )
                            sub_pos += v_len
                            if sub_field == 3:
                                workload_spec["use_case"] = s_val
                        elif sub_wire == 1:
                            sub_pos += 8
                        elif sub_wire == 5:
                            sub_pos += 4
            elif wire_type == 5:
                pos += 4
        return workload_spec
    except Exception:
        return {}


# Standard GKE Inference Recommender use cases and their default average token distributions
# (Matching `gcloud container ai profiles use-case list`)
KNOWN_USE_CASE_LENGTHS: dict[str, tuple[int, int]] = {
    "deep research": (256, 4096),
    "text summarization": (1024, 128),
    "multi agent large document summarization": (7936, 64),
    "advanced customer support": (8912, 256),
    "code completion": (512, 32),
    "chatbot (sharegpt)": (128, 128),
    "chatbot": (128, 128),
    "text generation": (512, 2048),
}


def _match_use_case(profile_use_case: str, target_use_case: Optional[str]) -> bool:
    """Matches a workload use-case using case-insensitive shorthand keywords."""
    if not target_use_case:
        return True
    if not profile_use_case:
        # GKE Recommender SDK proto does not always include workloadSpec field
        return True
    target_lower = target_use_case.strip().lower()
    profile_lower = profile_use_case.lower()

    if target_lower in profile_lower:
        return True

    # Shorthand aliases
    aliases = {
        "chatbot": ["chatbot", "sharegpt"],
        "chat": ["chatbot", "sharegpt"],
        "summarization": ["summarization"],
        "summary": ["summarization"],
        "code": ["code completion"],
        "code-completion": ["code completion"],
        "text-generation": ["text generation"],
        "generation": ["text generation"],
        "deep-research": ["deep research"],
        "research": ["deep research"],
        "customer-support": ["customer support"],
    }
    keywords = aliases.get(target_lower, [target_lower])
    return any(kw in profile_lower for kw in keywords)


def get_recommendations(
    model_id: str,
    model_server: Optional[str] = None,
    model_server_version: Optional[str] = None,
    target_cost_per_million_input_tokens: Optional[float] = None,
    target_cost_per_million_output_tokens: Optional[float] = None,
    output_input_cost_ratio: Optional[float] = None,
    pricing_model: Optional[str] = DEFAULT_PRICING_MODEL,
    target_ttft_milliseconds: Optional[int] = None,
    target_ntpot_milliseconds: Optional[int] = None,
    use_case: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    family: AcceleratorFamily = AcceleratorFamily.ANY,
    sort_by: str = "cost",
    client: Optional[Any] = None,
) -> list[dict[str, Any]]:
    """Retrieves and ranks genuine hardware configurations from GKE Recommender service.

    Args:
        model_id: Hugging Face model repository identifier.
        model_server: Serving engine target (defaults to 'vllm').
        model_server_version: Optional server version.
        target_cost_per_million_input_tokens: Target cost per 1M input tokens in USD.
        target_cost_per_million_output_tokens: Target cost per 1M output tokens in USD.
        output_input_cost_ratio: Output-to-input token pricing ratio (e.g. 4.0).
        pricing_model: Pricing model ('on-demand', 'spot', '1-year-cud', '3-years-cud').
        target_ttft_milliseconds: Maximum Time to First Token in milliseconds.
        target_ntpot_milliseconds: Maximum Normalized Time per Output Token in milliseconds.
        use_case: Workload traffic pattern filter.
        family: Filter by GPU, TPU, or ANY.
        sort_by: Ranking metric ('cost', 'throughput', 'ttft', 'ntpot').
        client: Optional GkeInferenceQuickstartClient instance for dependency injection.

    Returns:
        Ranked list of recommendation dictionaries.

    Raises:
        ModelNotSupportedError: If model is not supported by GKE Recommender service.
    """
    if client is None:
        import google.cloud.gkerecommender_v1 as gke_rec

        client = gke_rec.GkeInferenceQuickstartClient()
    else:
        import google.cloud.gkerecommender_v1 as gke_rec

    # 1. Build target Cost
    cost_kwargs = {}
    if target_cost_per_million_input_tokens is not None:
        in_units = int(target_cost_per_million_input_tokens)
        in_nanos = int(
            round((target_cost_per_million_input_tokens - in_units) * 1_000_000_000)
        )
        cost_kwargs["cost_per_million_input_tokens"] = gke_rec.Amount(
            units=in_units, nanos=in_nanos
        )

    if target_cost_per_million_output_tokens is not None:
        out_units = int(target_cost_per_million_output_tokens)
        out_nanos = int(
            round((target_cost_per_million_output_tokens - out_units) * 1_000_000_000)
        )
        cost_kwargs["cost_per_million_output_tokens"] = gke_rec.Amount(
            units=out_units, nanos=out_nanos
        )

    if output_input_cost_ratio is not None:
        cost_kwargs["output_input_cost_ratio"] = float(output_input_cost_ratio)

    target_pm = pricing_model or DEFAULT_PRICING_MODEL
    cost_kwargs["pricing_model"] = target_pm

    cost_obj = gke_rec.Cost(**cost_kwargs) if cost_kwargs else None

    # 2. Build PerformanceRequirements
    perf_kwargs = {}
    if cost_obj:
        perf_kwargs["target_cost"] = cost_obj
    if target_ttft_milliseconds is not None:
        perf_kwargs["target_ttft_milliseconds"] = int(target_ttft_milliseconds)
    if target_ntpot_milliseconds is not None:
        perf_kwargs["target_ntpot_milliseconds"] = int(target_ntpot_milliseconds)

    perf_req = gke_rec.PerformanceRequirements(**perf_kwargs) if perf_kwargs else None

    # 3. Build FetchProfilesRequest
    req_kwargs: dict[str, Any] = {"model": model_id}
    if model_server:
        req_kwargs["model_server"] = model_server
    if model_server_version:
        req_kwargs["model_server_version"] = model_server_version
    if perf_req:
        req_kwargs["performance_requirements"] = perf_req

    request = gke_rec.FetchProfilesRequest(**req_kwargs)

    try:
        profiles_pager = client.fetch_profiles(request=request)
        raw_profiles = list(profiles_pager)
    except (
        google.auth.exceptions.GoogleAuthError,
        google.api_core.exceptions.Unauthenticated,
        google.api_core.exceptions.PermissionDenied,
    ):
        raise
    except Exception as exc:
        from google.models.cli.common.auth import is_auth_error

        if is_auth_error(exc):
            raise
        try:
            supported = fetch_supported_models(client)
        except Exception:
            supported = []
        if supported and model_id not in supported:
            raise ModelNotSupportedError(model_id, supported) from exc
        raise

    if not raw_profiles:
        # Verify if model is supported at all
        try:
            supported = fetch_supported_models(client)
        except Exception:
            supported = []
        if supported and model_id not in supported:
            raise ModelNotSupportedError(model_id, supported)
        return []

    candidates: list[dict[str, Any]] = []

    for p in raw_profiles:
        d = (
            MessageToDict(p._pb)
            if hasattr(p, "_pb")
            else (p if isinstance(p, dict) else MessageToDict(p))
        )

        accelerator_type = d.get("acceleratorType", "")
        instance_type = d.get("instanceType", "")
        accelerator_count = d.get("resourcesUsed", {}).get("acceleratorCount", 1)

        is_tpu = "tpu" in accelerator_type.lower() or instance_type.lower().startswith(
            "ct"
        )
        profile_family = AcceleratorFamily.TPU if is_tpu else AcceleratorFamily.GPU

        # Accelerator family filter
        if family != AcceleratorFamily.ANY and profile_family != family:
            continue

        # Workload use-case filter
        workload_spec = _extract_workload_spec(p)
        profile_use_case = workload_spec.get("use_case", "")
        if not _match_use_case(profile_use_case, use_case):
            continue

        # Performance and cost metrics
        stats_list = d.get("performanceStats", [])
        stat0 = stats_list[0] if stats_list else {}
        cost_list = stat0.get("cost", [])

        target_pricing_str = target_pm.lower().replace("_", "-")
        cost0 = {}
        if cost_list:
            matched = [
                c
                for c in cost_list
                if c.get("pricingModel", "").lower().replace("_", "-")
                == target_pricing_str
            ]
            cost0 = matched[0] if matched else cost_list[0]

        in_nanos_raw = cost0.get("costPerMillionInputTokens", {}).get("nanos", 0)
        in_units_raw = cost0.get("costPerMillionInputTokens", {}).get("units", 0)
        in_nanos = int(in_nanos_raw) if in_nanos_raw is not None else 0
        in_units = int(in_units_raw) if in_units_raw is not None else 0
        input_cost = (
            (in_units + in_nanos / 1_000_000_000) if (in_nanos or in_units) else None
        )

        out_nanos_raw = cost0.get("costPerMillionOutputTokens", {}).get("nanos", 0)
        out_units_raw = cost0.get("costPerMillionOutputTokens", {}).get("units", 0)
        out_nanos = int(out_nanos_raw) if out_nanos_raw is not None else 0
        out_units = int(out_units_raw) if out_units_raw is not None else 0
        output_cost = (
            (out_units + out_nanos / 1_000_000_000)
            if (out_nanos or out_units)
            else None
        )

        ratio = cost0.get("outputInputCostRatio")
        pricing = cost0.get("pricingModel") or target_pm

        server_info = d.get("modelServerInfo", {})

        candidate = {
            "machine_type": instance_type,
            "accelerator_type": accelerator_type,
            "accelerator_count": accelerator_count,
            "chip_name": f"{accelerator_type} ({accelerator_count}x)",
            "family": profile_family.value,
            "input_cost_per_m": (
                round(input_cost, 4) if input_cost is not None else None
            ),
            "output_cost_per_m": (
                round(output_cost, 4) if output_cost is not None else None
            ),
            "output_input_cost_ratio": ratio,
            "pricing_model": pricing,
            "ttft_ms": stat0.get("ttftMilliseconds"),
            "ntpot_ms": stat0.get("ntpotMilliseconds"),
            "itl_ms": stat0.get("itlMilliseconds"),
            "output_tokens_per_sec": stat0.get("outputTokensPerSecond"),
            "queries_per_sec": stat0.get("queriesPerSecond"),
            "use_case": profile_use_case or (use_case or ""),
            "average_input_length": (
                workload_spec.get("average_input_length")
                or KNOWN_USE_CASE_LENGTHS.get(
                    (profile_use_case or use_case or "").strip().lower(), (None, None)
                )[0]
            ),
            "average_output_length": (
                workload_spec.get("average_output_length")
                or KNOWN_USE_CASE_LENGTHS.get(
                    (profile_use_case or use_case or "").strip().lower(), (None, None)
                )[1]
            ),
            "model_server": server_info.get("modelServer") or model_server or "vllm",
            "model_server_version": server_info.get("modelServerVersion", ""),
            "engine_params": {
                "tensor_parallel_size": accelerator_count,
            },
        }
        candidates.append(candidate)

    # If custom token distribution is requested, interpolate performance metrics
    if input_tokens is not None and output_tokens is not None:
        from google.models.cli.recommend.interpolate_utils import (
            interpolate_recommendations,
        )

        candidates = interpolate_recommendations(
            recommendations=candidates,
            target_input_tokens=input_tokens,
            target_output_tokens=output_tokens,
        )

    # Sort results
    sort_key = sort_by.lower().strip()
    if sort_key == "throughput":
        candidates.sort(key=lambda x: -(x["output_tokens_per_sec"] or 0))
    elif sort_key == "ttft":
        candidates.sort(
            key=lambda x: (x["ttft_ms"] if x["ttft_ms"] is not None else 999999)
        )
    elif sort_key == "ntpot":
        candidates.sort(
            key=lambda x: (x["ntpot_ms"] if x["ntpot_ms"] is not None else 999999)
        )
    else:  # Default to cost
        candidates.sort(
            key=lambda x: (
                x["input_cost_per_m"] if x["input_cost_per_m"] is not None else 999999,
                (
                    x["output_cost_per_m"]
                    if x["output_cost_per_m"] is not None
                    else 999999
                ),
            )
        )

    return candidates


def apply_recommendation(
    top_candidate: dict[str, Any],
    project_dir: Path,
) -> bool:
    """Updates deployment_spec.yaml and engine_config.yaml with the recommended settings."""
    updated = False

    machine_type = top_candidate["machine_type"]
    accel_type = top_candidate.get("accelerator_type")
    accel_count = top_candidate.get("accelerator_count")

    # Fallback to HARDWARE_SPECS if not directly present in candidate
    if not accel_type or not accel_count:
        spec = HARDWARE_SPECS.get(machine_type, {})
        accel_type = accel_type or spec.get("accelerator_type")
        accel_count = accel_count or spec.get("accelerator_count", 1)

    # 1. Update config/deployment_spec.yaml
    deploy_file = project_dir / "config" / "deployment_spec.yaml"
    if deploy_file.is_file():
        try:
            with open(deploy_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            data["machine_type"] = machine_type
            data["accelerator_type"] = accel_type
            data["accelerator_count"] = accel_count
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
            with open(engine_file, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)
            updated = True
        except Exception:
            pass

    return updated
