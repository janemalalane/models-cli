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

"""Profile interpolation utilities for estimating latency and throughput for custom token distributions."""

import math
from typing import Any


def interpolate_profile_metrics(
    baselines: list[dict[str, Any]],
    target_input_tokens: int,
    target_output_tokens: int,
    power: float = 2.0,
) -> dict[str, Any]:
    """Interpolates performance metrics for a custom input/output distribution from baselines.

    Uses Log-Space Inverse Distance Weighting (IDW) combined with LLM prefill/decode
    decomposition to predict TTFT, ITL, NTPOT, and Output Tokens/sec.

    Args:
        baselines: List of empirical recommendation profile dictionaries for the same accelerator.
        target_input_tokens: Desired average input prompt sequence length.
        target_output_tokens: Desired average output generation sequence length.
        power: IDW power parameter (default 2.0).

    Returns:
        Dictionary of interpolated metrics.
    """
    if not baselines:
        return {}

    # Target coordinates in log2 space
    t_x = math.log2(max(target_input_tokens, 1))
    t_y = math.log2(max(target_output_tokens, 1))

    # Calculate distances
    weights: list[float] = []
    exact_match: dict[str, Any] | None = None

    for b in baselines:
        b_in = b.get("average_input_length") or 512
        b_out = b.get("average_output_length") or 128
        dx = math.log2(max(b_in, 1)) - t_x
        dy = math.log2(max(b_out, 1)) - t_y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1e-6:
            exact_match = b
            break
        weights.append(1.0 / (dist ** power))

    if exact_match is not None:
        ttft = exact_match.get("ttft_ms")
        ntpot = exact_match.get("ntpot_ms")
        itl = exact_match.get("itl_ms")
        tput = exact_match.get("output_tokens_per_sec")
        qps = exact_match.get("queries_per_sec")
    else:
        total_weight = sum(weights)
        norm_weights = [w / total_weight for w in weights] if total_weight > 0 else [1.0 / len(weights)] * len(weights)

        def weighted_avg(key: str) -> float | None:
            valid_pairs = [(w, b.get(key)) for w, b in zip(norm_weights, baselines) if b.get(key) is not None]
            if not valid_pairs:
                return None
            w_sum = sum(w for w, _ in valid_pairs)
            if w_sum <= 0:
                return None
            return sum(w * val for w, val in valid_pairs) / w_sum

        raw_ttft = weighted_avg("ttft_ms")
        raw_itl = weighted_avg("itl_ms")
        raw_ntpot = weighted_avg("ntpot_ms")
        raw_tput = weighted_avg("output_tokens_per_sec")
        raw_qps = weighted_avg("queries_per_sec")

        ttft = int(round(raw_ttft)) if raw_ttft is not None else None
        itl = int(round(raw_itl)) if raw_itl is not None else None
        ntpot = int(round(raw_ntpot)) if raw_ntpot is not None else None
        tput = int(round(raw_tput)) if raw_tput is not None else None
        qps = round(raw_qps, 2) if raw_qps is not None else None

    # Physics check & per-request latency calculation
    # E2E Latency = TTFT + target_output_tokens * ITL
    est_latency_ms: float | None = None
    if ttft is not None:
        eff_itl = itl if itl is not None else (ntpot if ntpot is not None else 20)
        est_latency_ms = ttft + (target_output_tokens * eff_itl)

    return {
        "ttft_ms": ttft,
        "ntpot_ms": ntpot,
        "itl_ms": itl,
        "output_tokens_per_sec": tput,
        "queries_per_sec": qps,
        "estimated_request_latency_ms": round(est_latency_ms, 1) if est_latency_ms is not None else None,
    }


def interpolate_recommendations(
    recommendations: list[dict[str, Any]],
    target_input_tokens: int,
    target_output_tokens: int,
) -> list[dict[str, Any]]:
    """Groups empirical recommendations by accelerator and generates interpolated candidate profiles.

    Args:
        recommendations: Full list of empirical profile dictionaries.
        target_input_tokens: Target input token length.
        target_output_tokens: Target output token length.

    Returns:
        List of candidate dictionaries with interpolated performance metrics and is_interpolated = True.
    """
    if not recommendations:
        return []

    # Group by unique hardware / engine configuration
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in recommendations:
        key = (
            r.get("machine_type", ""),
            r.get("accelerator_type", ""),
            r.get("accelerator_count", 1),
            r.get("model_server", ""),
        )
        groups.setdefault(key, []).append(r)

    ratio = round(target_input_tokens / max(target_output_tokens, 1), 1)
    interpolated_recs: list[dict[str, Any]] = []

    for key, group in groups.items():
        base = group[0]  # Base hardware attributes
        interpolated_metrics = interpolate_profile_metrics(
            group, target_input_tokens, target_output_tokens
        )

        candidate = dict(base)
        candidate.update(interpolated_metrics)

        # Scale input and output token costs using the interpolated throughput:
        # Machine hourly rate H = InCost * (InLen/OutLen + R) * Tput * 3600 / 1e6.
        # Across workloads on identical hardware, H is approximately constant.
        # Therefore, at the new target distribution and interpolated throughput:
        # InCost_new = (H * 1e6) / (3600 * Tput_interp * (target_in / target_out + R)).
        # OutCost_new = R * InCost_new.
        interp_tput = interpolated_metrics.get("output_tokens_per_sec")
        if interp_tput and interp_tput > 0:
            hourly_costs: list[float] = []
            cost_ratios: list[float] = []

            for b in group:
                b_in_cost = b.get("input_cost_per_m")
                b_out_cost = b.get("output_cost_per_m")
                b_tput = b.get("output_tokens_per_sec")
                b_in_len = b.get("average_input_length")
                b_out_len = b.get("average_output_length")

                if (
                    b_in_cost is not None
                    and b_out_cost is not None
                    and b_tput
                    and b_tput > 0
                    and b_in_len
                    and b_out_len
                    and b_out_len > 0
                ):
                    r_ratio = (
                        b.get("output_input_cost_ratio")
                        or (b_out_cost / b_in_cost if b_in_cost > 0 else 4.0)
                    )
                    cost_ratios.append(r_ratio)
                    # Hourly machine rate
                    h = b_in_cost * (b_in_len / b_out_len + r_ratio) * b_tput * 3600 / 1_000_000
                    hourly_costs.append(h)

            if hourly_costs:
                avg_hourly = sum(hourly_costs) / len(hourly_costs)
                effective_ratio = (
                    sum(cost_ratios) / len(cost_ratios) if cost_ratios else 4.0
                )
                len_ratio = target_input_tokens / max(target_output_tokens, 1)

                scaled_in_cost = (avg_hourly * 1_000_000) / (
                    3600 * interp_tput * (len_ratio + effective_ratio)
                )
                scaled_out_cost = scaled_in_cost * effective_ratio

                candidate["input_cost_per_m"] = round(scaled_in_cost, 4)
                candidate["output_cost_per_m"] = round(scaled_out_cost, 4)
                candidate["output_input_cost_ratio"] = round(effective_ratio, 2)

        candidate["is_interpolated"] = True
        candidate["profile_status"] = "interpolated"
        candidate["custom_input_tokens"] = target_input_tokens
        candidate["custom_output_tokens"] = target_output_tokens
        candidate["custom_ratio"] = ratio
        candidate["use_case"] = f"Custom ({target_input_tokens:,} in / {target_output_tokens:,} out)"

        interpolated_recs.append(candidate)

    return interpolated_recs
