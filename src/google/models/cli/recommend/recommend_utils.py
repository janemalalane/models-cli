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

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import google.api_core.exceptions
import google.auth.exceptions
from google.protobuf.json_format import MessageToDict
import yaml

from google.models.cli.common.constants import (
    DEFAULT_PRICING_MODEL,
    MODELS_SPECS,
    AcceleratorFamily,
    ModelArchitecture,
)

if TYPE_CHECKING:
    from google.cloud.gkerecommender_v1 import GkeInferenceQuickstartClient


class ModelNotSupportedError(Exception):
    """Raised when a model is not supported by the GKE Recommender service."""

    def __init__(self, model_id: str, supported_models: list[str]):
        super().__init__(
            f"Model '{model_id}' is not supported by the GKE Recommender service."
        )
        self.model_id = model_id
        self.supported_models = supported_models


def _create_recommender_client() -> GkeInferenceQuickstartClient:
    """Creates a GkeInferenceQuickstartClient with quota_project_id (x-goog-user-project) set."""
    import google.cloud.gkerecommender_v1 as gke_rec

    from google.models.cli._gcp_project import resolve_gcp_project
    from google.models.cli.common.config import Settings, settings

    quota_project = resolve_gcp_project(
        override_project=settings.google_cloud_project
        or Settings().google_cloud_project
    )
    if quota_project:
        return gke_rec.GkeInferenceQuickstartClient(
            client_options={"quota_project_id": quota_project}
        )
    return gke_rec.GkeInferenceQuickstartClient()


def fetch_supported_models(
    client: GkeInferenceQuickstartClient | None = None,
) -> list[str]:
    """Fetches the list of supported model repositories from GKE Recommender."""
    if client is None:
        client = _create_recommender_client()

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


def _match_use_case(profile_use_case: str, target_use_case: str | None) -> bool:
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


def lookup_model_spec(
    model_id: str,
    models_specs: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Looks up a model in MODELS_SPECS by full repo ID or short model_name."""
    catalog = models_specs if models_specs is not None else MODELS_SPECS
    cleaned = model_id.strip()
    if not cleaned:
        return None

    if cleaned in catalog:
        return cleaned, catalog[cleaned]

    lowered = cleaned.lower()
    for repo_id, spec in catalog.items():
        if repo_id.lower() == lowered:
            return repo_id, spec

    for repo_id, spec in catalog.items():
        if str(spec.get("model_name", "")).lower() == lowered:
            return repo_id, spec

    return None


def find_most_similar_supported_model(
    model_id: str,
    models_specs: dict[str, dict[str, Any]] | None = None,
    shortlist_size: int = 3,
) -> str | None:
    """Finds the most similar GKE Recommender-supported model in MODELS_SPECS.

    Similarity hierarchy:
    1. ``architecture`` (``dense`` vs ``moe``)
    2. ``total_parameters_b``
    3. If ``architecture`` is ``moe``, ``active_parameters_b``
    4. Within a shortlist of 2 to 3 similar models, prefer the candidate from the same ``origin``

    Args:
        model_id: Target Hugging Face repository ID or model name.
        models_specs: Optional catalog override (defaults to ``MODELS_SPECS``).
        shortlist_size: Maximum size of the top similar models shortlist (default 3).

    Returns:
        Canonical repository ID of the most similar supported model, or ``None``
        if ``model_id`` is not in ``models_specs`` or no supported candidates exist.
    """
    catalog = models_specs if models_specs is not None else MODELS_SPECS
    found = lookup_model_spec(model_id, catalog)
    if found is None:
        return None

    target_repo_id, target_spec = found
    if target_spec.get("gke_recommender_supported", False):
        return target_repo_id

    supported_pool = [
        (repo_id, spec)
        for repo_id, spec in catalog.items()
        if spec.get("gke_recommender_supported", False) and repo_id != target_repo_id
    ]
    if not supported_pool:
        return None

    # 1. Hierarchy Level 1: Match model architecture ('dense' vs 'moe')
    target_arch = target_spec.get("architecture")
    arch_matched = [
        (repo_id, spec)
        for repo_id, spec in supported_pool
        if spec.get("architecture") == target_arch
    ]
    candidates = arch_matched if arch_matched else supported_pool

    target_total = max(float(target_spec.get("total_parameters_b") or 1.0), 1e-6)
    target_active = max(
        float(target_spec.get("active_parameters_b") or target_total), 1e-6
    )
    target_quant = target_spec.get("quantization")
    target_origin = target_spec.get("origin")
    is_moe = target_arch == ModelArchitecture.MOE

    # 2 & 3. Hierarchy Levels 2 & 3: total_parameters_b (primary) and active_parameters_b (secondary for MoE)
    scored: list[dict[str, Any]] = []
    for repo_id, spec in candidates:
        cand_total = max(float(spec.get("total_parameters_b") or 1.0), 1e-6)
        total_dist = abs(math.log2(cand_total / target_total))

        if is_moe:
            cand_active = max(
                float(spec.get("active_parameters_b") or cand_total), 1e-6
            )
            active_dist = abs(math.log2(cand_active / target_active))
            # Weight total_parameters_b higher (0.75) than active_parameters_b (0.25)
            param_dist = (0.75 * total_dist) + (0.25 * active_dist)
        else:
            param_dist = total_dist

        quant_mismatch = 0 if spec.get("quantization") == target_quant else 1
        scored.append(
            {
                "repo_id": repo_id,
                "spec": spec,
                "param_dist": param_dist,
                "total_dist": total_dist,
                "quant_mismatch": quant_mismatch,
                "timestamp": str(spec.get("timestamp") or ""),
            }
        )

    # Sort ascending by parameter distance, then matching quantization, newer timestamp, and repo name descending
    scored.sort(
        key=lambda item: (
            round(item["param_dist"], 6),
            round(item["total_dist"], 6),
            item["quant_mismatch"],
        )
    )
    # Break exact ties deterministically in favor of newer model versions (e.g. gemma-3 over gemma-2)
    # using Python's stable sort in reverse first:
    scored_desc = sorted(
        scored,
        key=lambda item: (item["timestamp"], item["repo_id"]),
        reverse=True,
    )
    scored_desc.sort(
        key=lambda item: (
            round(item["param_dist"], 6),
            round(item["total_dist"], 6),
            item["quant_mismatch"],
        )
    )

    # Build shortlist of 2 to 3 genuinely similar models (avoiding extreme scale outliers > ~2x away from best)
    k = max(1, shortlist_size)
    best_dist = scored_desc[0]["param_dist"]
    max_allowed_dist = max(best_dist + 0.85, 1.0)
    shortlist = [
        item
        for idx, item in enumerate(scored_desc[:k])
        if idx == 0 or item["param_dist"] <= max_allowed_dist
    ]

    # 4. Hierarchy Level 4: Within the shortlist of 2-3 similar models, prefer the same 'origin'
    same_origin = [
        item for item in shortlist if item["spec"].get("origin") == target_origin
    ]
    if same_origin:
        return str(same_origin[0]["repo_id"])

    return str(shortlist[0]["repo_id"])


def get_recommendations(
    model_id: str,
    *,
    model_server: str | None = None,
    model_server_version: str | None = None,
    target_cost_per_million_input_tokens: float | None = None,
    target_cost_per_million_output_tokens: float | None = None,
    output_input_cost_ratio: float | None = None,
    pricing_model: str | None = DEFAULT_PRICING_MODEL,
    target_ttft_milliseconds: int | None = None,
    target_ntpot_milliseconds: int | None = None,
    use_case: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    family: AcceleratorFamily = AcceleratorFamily.ANY,
    sort_by: str = "cost",
    client: GkeInferenceQuickstartClient | None = None,
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
        client = _create_recommender_client()
    import google.cloud.gkerecommender_v1 as gke_rec

    # 1. Build target Cost
    cost_kwargs: dict[str, Any] = {}
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
    perf_kwargs: dict[str, Any] = {}
    if cost_obj:
        perf_kwargs["target_cost"] = cost_obj
    if target_ttft_milliseconds is not None:
        perf_kwargs["target_ttft_milliseconds"] = int(target_ttft_milliseconds)
    if target_ntpot_milliseconds is not None:
        perf_kwargs["target_ntpot_milliseconds"] = int(target_ntpot_milliseconds)

    perf_req = gke_rec.PerformanceRequirements(**perf_kwargs) if perf_kwargs else None

    # Resolve proxy model when chosen model has gke_recommender_supported=False in MODELS_SPECS
    effective_model_id = model_id
    proxy_model_id: str | None = None
    spec_entry = lookup_model_spec(model_id)
    if spec_entry is not None:
        canonical_repo_id, model_spec = spec_entry
        if not model_spec.get("gke_recommender_supported", True):
            similar_model = find_most_similar_supported_model(canonical_repo_id)
            if similar_model:
                effective_model_id = similar_model
                proxy_model_id = similar_model
        else:
            effective_model_id = canonical_repo_id

    # 3. Build FetchProfilesRequest
    req_kwargs: dict[str, Any] = {"model": effective_model_id}
    if model_server:
        req_kwargs["model_server"] = model_server
    if model_server_version:
        req_kwargs["model_server_version"] = model_server_version
    if perf_req:
        req_kwargs["performance_requirements"] = perf_req

    request = gke_rec.FetchProfilesRequest(**req_kwargs)

    try:
        profiles_pager = client.fetch_profiles(request=request)
        raw_profiles: list[Any] = list(profiles_pager)
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
        if supported and effective_model_id not in supported:
            raise ModelNotSupportedError(effective_model_id, supported) from exc
        raise

    if not raw_profiles:
        # Verify if model is supported at all
        try:
            supported = fetch_supported_models(client)
        except Exception:
            supported = []
        if supported and effective_model_id not in supported:
            raise ModelNotSupportedError(effective_model_id, supported)
        return []

    candidates: list[dict[str, Any]] = []

    for p in raw_profiles:
        if isinstance(p, dict):
            d: dict[str, Any] = p
        else:
            pb_msg: Any = getattr(p, "_pb", p)
            d = MessageToDict(pb_msg)

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
        if proxy_model_id:
            candidate["proxy_model"] = proxy_model_id
            candidate["requested_model"] = model_id
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

    # 1. Update config/deployment_spec.yaml
    deploy_file = project_dir / "config" / "deployment_spec.yaml"
    if deploy_file.is_file():
        try:
            with open(deploy_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            data["machine_type"] = machine_type
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
