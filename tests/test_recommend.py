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

from unittest.mock import MagicMock
import pytest
import yaml
from click.testing import CliRunner

from google.models.cli.recommend.cmd_recommend import recommend_cmd
from google.models.cli.recommend.recommend_utils import (
    ModelNotSupportedError,
    apply_recommendation,
    fetch_supported_models,
    get_recommendations,
)


def _make_mock_profile(
    instance_type="g2-standard-12",
    accelerator_type="nvidia-l4",
    accelerator_count=1,
    cost_in_nanos=19461894,
    cost_out_nanos=77847577,
    ratio=4.0,
    ttft_ms=602,
    ntpot_ms=104,
    tok_s=1059,
    qps=6.09,
    use_case="Chatbot (ShareGPT)",
    model_server="vllm",
    model_server_version="v0.9.2",
):
    """Creates a mock Profile object compatible with MessageToDict."""
    mock_p = MagicMock()
    # Provide dictionary structure for tests
    data = {
        "instanceType": instance_type,
        "acceleratorType": accelerator_type,
        "resourcesUsed": {"acceleratorCount": accelerator_count},
        "performanceStats": [
            {
                "queriesPerSecond": qps,
                "outputTokensPerSecond": tok_s,
                "ntpotMilliseconds": ntpot_ms,
                "ttftMilliseconds": ttft_ms,
                "cost": [
                    {
                        "costPerMillionOutputTokens": {"nanos": cost_out_nanos},
                        "costPerMillionInputTokens": {"nanos": cost_in_nanos},
                        "pricingModel": "spot",
                        "outputInputCostRatio": ratio,
                    }
                ],
            }
        ],
        "workloadSpec": {"useCase": use_case},
        "modelServerInfo": {
            "model": "google/gemma-3-4b-it",
            "modelServer": model_server,
            "modelServerVersion": model_server_version,
        },
    }
    mock_p._pb = MagicMock()
    # Allow MessageToDict to fallback or return data
    mock_p.instance_type = instance_type
    mock_p.accelerator_type = accelerator_type
    return data


def test_fetch_supported_models():
    mock_client = MagicMock()
    mock_client.fetch_models.return_value = [
        "meta-llama/Llama-3.3-70B-Instruct",
        "google/gemma-3-4b-it",
    ]
    models = fetch_supported_models(mock_client)
    assert models == [
        "google/gemma-3-4b-it",
        "meta-llama/Llama-3.3-70B-Instruct",
    ]


def test_get_recommendations_success():
    mock_client = MagicMock()
    mock_client.fetch_profiles.return_value = [
        _make_mock_profile(
            instance_type="g2-standard-12",
            accelerator_type="nvidia-l4",
            accelerator_count=1,
            cost_in_nanos=19461894,
            cost_out_nanos=77847577,
            ratio=4.0,
            ttft_ms=602,
            ntpot_ms=104,
            tok_s=1059,
        )
    ]

    recs = get_recommendations(
        model_id="google/gemma-3-4b-it",
        client=mock_client,
    )

    assert len(recs) == 1
    r = recs[0]
    assert r["machine_type"] == "g2-standard-12"
    assert r["accelerator_type"] == "nvidia-l4"
    assert r["accelerator_count"] == 1
    assert r["input_cost_per_m"] == 0.0195
    assert r["output_cost_per_m"] == 0.0778
    assert r["output_input_cost_ratio"] == 4.0
    assert r["ttft_ms"] == 602
    assert r["ntpot_ms"] == 104
    assert r["output_tokens_per_sec"] == 1059
    assert r["engine_params"]["tensor_parallel_size"] == 1


def test_get_recommendations_target_construction():
    mock_client = MagicMock()
    mock_client.fetch_profiles.return_value = [
        _make_mock_profile()
    ]

    get_recommendations(
        model_id="google/gemma-3-4b-it",
        target_cost_per_million_input_tokens=0.02,
        target_cost_per_million_output_tokens=0.08,
        output_input_cost_ratio=4.0,
        target_ttft_milliseconds=500,
        target_ntpot_milliseconds=100,
        client=mock_client,
    )

    mock_client.fetch_profiles.assert_called_once()
    call_args = mock_client.fetch_profiles.call_args[1]
    req = call_args["request"]

    assert req.model == "google/gemma-3-4b-it"
    assert req.performance_requirements.target_cost.output_input_cost_ratio == 4.0
    assert req.performance_requirements.target_cost.cost_per_million_input_tokens.nanos == 20000000
    assert req.performance_requirements.target_cost.cost_per_million_output_tokens.nanos == 80000000
    assert req.performance_requirements.target_ttft_milliseconds == 500
    assert req.performance_requirements.target_ntpot_milliseconds == 100


def test_get_recommendations_sort_options():
    p1 = _make_mock_profile(
        instance_type="g2-standard-12",
        accelerator_type="nvidia-l4",
        cost_in_nanos=19000000,
        tok_s=1000,
        ttft_ms=600,
    )
    p2 = _make_mock_profile(
        instance_type="a2-highgpu-1g",
        accelerator_type="nvidia-tesla-a100",
        cost_in_nanos=20000000,
        tok_s=4000,
        ttft_ms=100,
    )

    mock_client = MagicMock()
    mock_client.fetch_profiles.return_value = [p1, p2]

    # Sort by cost (default)
    by_cost = get_recommendations("google/gemma-3-4b-it", sort_by="cost", client=mock_client)
    assert by_cost[0]["machine_type"] == "g2-standard-12"

    # Sort by throughput
    by_tp = get_recommendations("google/gemma-3-4b-it", sort_by="throughput", client=mock_client)
    assert by_tp[0]["machine_type"] == "a2-highgpu-1g"

    # Sort by TTFT
    by_ttft = get_recommendations("google/gemma-3-4b-it", sort_by="ttft", client=mock_client)
    assert by_ttft[0]["machine_type"] == "a2-highgpu-1g"


def test_unsupported_model_raises_clean_error():
    mock_client = MagicMock()
    mock_client.fetch_profiles.side_effect = Exception("400 no latency profiles found")
    mock_client.fetch_models.return_value = ["google/gemma-3-4b-it"]

    with pytest.raises(ModelNotSupportedError) as exc_info:
        get_recommendations("custom/unsupported-model", client=mock_client)

    assert exc_info.value.model_id == "custom/unsupported-model"
    assert "google/gemma-3-4b-it" in exc_info.value.supported_models


def test_apply_recommendation(tmp_path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True)
    deploy_file = cfg_dir / "deployment_spec.yaml"
    deploy_file.write_text("machine_type: null\n")

    engine_file = cfg_dir / "engine_config.yaml"
    engine_file.write_text("engine: vllm\ntensor_parallel_size: 1\n")

    top = {
        "machine_type": "g2-standard-12",
        "accelerator_type": "nvidia-l4",
        "accelerator_count": 1,
        "engine_params": {"tensor_parallel_size": 1},
    }

    updated = apply_recommendation(top, tmp_path)
    assert updated is True

    with open(deploy_file, "r") as f:
        deploy_data = yaml.safe_load(f)
    assert deploy_data["machine_type"] == "g2-standard-12"
    assert "accelerator_type" not in deploy_data
    assert "accelerator_count" not in deploy_data


def test_cli_list_models(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "google.models.cli.recommend.cmd_recommend.ensure_authenticated",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        "google.models.cli.recommend.cmd_recommend.fetch_supported_models",
        lambda: ["google/gemma-3-4b-it", "Qwen/Qwen3-32B"],
    )

    result = runner.invoke(recommend_cmd, ["--list-models"])
    assert result.exit_code == 0
    assert "Supported models in GKE Recommender" in result.output
    assert "google/gemma-3-4b-it" in result.output
    assert "Qwen/Qwen3-32B" in result.output


def test_cli_recommend_json_output(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "google.models.cli.recommend.cmd_recommend.ensure_authenticated",
        lambda **kwargs: True,
    )
    sample_candidate = {
        "machine_type": "g2-standard-12",
        "accelerator_type": "nvidia-l4",
        "accelerator_count": 1,
        "chip_name": "nvidia-l4 (1x)",
        "family": "gpu",
        "input_cost_per_m": 0.019,
        "output_cost_per_m": 0.078,
        "output_input_cost_ratio": 4.0,
        "pricing_model": "spot",
        "ttft_ms": 602,
        "ntpot_ms": 104,
        "itl_ms": None,
        "output_tokens_per_sec": 1059,
        "queries_per_sec": 6.09,
        "use_case": "Chatbot (ShareGPT)",
        "model_server": "vllm",
        "model_server_version": "v0.9.2",
        "engine_params": {"tensor_parallel_size": 1},
    }

    monkeypatch.setattr(
        "google.models.cli.recommend.cmd_recommend.get_recommendations",
        lambda **kwargs: [sample_candidate],
    )

    result = runner.invoke(
        recommend_cmd,
        [
            "--model",
            "google/gemma-3-4b-it",
            "--format",
            "json",
            "--sort-by",
            "cost",
            "--output-input-cost-ratio",
            "4.0",
        ],
    )
    assert result.exit_code == 0
    assert '"model": "google/gemma-3-4b-it"' in result.output
    assert '"g2-standard-12"' in result.output
    assert '"output_input_cost_ratio": 4.0' in result.output


def test_auth_error_not_masked_as_unsupported_model():
    import google.auth.exceptions

    mock_client = MagicMock()
    mock_client.fetch_profiles.side_effect = google.auth.exceptions.RefreshError("reauthentication required")

    with pytest.raises(google.auth.exceptions.RefreshError):
        get_recommendations("google/gemma-4-31B-it", client=mock_client)


def test_supported_model_api_error_not_masked_as_unsupported():
    mock_client = MagicMock()
    mock_client.fetch_profiles.side_effect = RuntimeError("backend connection failed")
    mock_client.fetch_models.return_value = ["google/gemma-4-31B-it"]

    with pytest.raises(RuntimeError):
        get_recommendations("google/gemma-4-31B-it", client=mock_client)


def test_cli_auth_error_displays_helpful_message(monkeypatch):
    import google.auth.exceptions

    runner = CliRunner()
    monkeypatch.setattr(
        "google.models.cli.recommend.cmd_recommend.get_recommendations",
        MagicMock(side_effect=google.auth.exceptions.RefreshError("reauthentication required")),
    )

    result = runner.invoke(recommend_cmd, ["--model", "google/gemma-4-31B-it"])
    assert result.exit_code == 1
    assert "Google Cloud authentication required" in result.output
    assert "gcloud auth application-default login" in result.output


def test_cli_gce_metadata_503_displays_helpful_message(monkeypatch):
    import google.api_core.exceptions

    runner = CliRunner()
    metadata_503 = google.api_core.exceptions.ServiceUnavailable(
        "503 Getting metadata from plugin failed with error: ('Failed to retrieve "
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token "
        "from the Google Compute Engine metadata service. Status: 404 Response:\\nb\\'\"No service account scopes specified.\"\\'')"
    )
    monkeypatch.setattr(
        "google.models.cli.recommend.cmd_recommend.get_recommendations",
        MagicMock(side_effect=metadata_503),
    )

    result = runner.invoke(recommend_cmd, ["--model", "google/gemma-4-31B-it"])
    assert result.exit_code == 1
    assert "Google Cloud authentication required" in result.output
    assert "gcloud auth application-default login" in result.output


def test_cli_recommend_default_pricing_model_on_demand(monkeypatch):
    runner = CliRunner()
    mock_get_recs = MagicMock(return_value=[{
        "machine_type": "g2-standard-12",
        "accelerator_type": "nvidia-l4",
        "accelerator_count": 1,
        "chip_name": "nvidia-l4 (1x)",
        "family": "gpu",
        "input_cost_per_m": 0.035,
        "output_cost_per_m": 0.140,
        "pricing_model": "on-demand",
        "ttft_ms": 500,
        "ntpot_ms": 50,
        "output_tokens_per_sec": 1200,
        "model_server": "vllm",
        "engine_params": {"tensor_parallel_size": 1},
    }])
    monkeypatch.setattr(
        "google.models.cli.recommend.cmd_recommend.ensure_authenticated",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        "google.models.cli.recommend.cmd_recommend.get_recommendations",
        mock_get_recs,
    )

    result = runner.invoke(recommend_cmd, ["--model", "google/gemma-3-4b-it"])
    assert result.exit_code == 0
    assert "Pricing: ON-DEMAND" in result.output
    assert mock_get_recs.call_args[1]["pricing_model"] == "on-demand"


def test_cli_recommend_select_specific_rank(tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True)
    deploy_file = cfg_dir / "deployment_spec.yaml"
    deploy_file.write_text("machine_type: null\n")

    recs = [
        {
            "machine_type": "a4-highgpu-8g",
            "accelerator_type": "nvidia-b200",
            "accelerator_count": 8,
            "chip_name": "nvidia-b200 (8x)",
            "family": "gpu",
            "input_cost_per_m": None,
            "output_cost_per_m": None,
            "ttft_ms": 68,
            "ntpot_ms": 10,
            "output_tokens_per_sec": 127,
            "model_server": "vllm",
            "engine_params": {"tensor_parallel_size": 8},
        },
        {
            "machine_type": "a3-ultragpu-8g",
            "accelerator_type": "nvidia-h200-141gb",
            "accelerator_count": 8,
            "chip_name": "nvidia-h200-141gb (8x)",
            "family": "gpu",
            "input_cost_per_m": None,
            "output_cost_per_m": None,
            "ttft_ms": 56,
            "ntpot_ms": 10,
            "output_tokens_per_sec": 128,
            "model_server": "vllm",
            "engine_params": {"tensor_parallel_size": 8},
        },
    ]

    monkeypatch.setattr(
        "google.models.cli.recommend.cmd_recommend.ensure_authenticated",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        "google.models.cli.recommend.cmd_recommend.get_recommendations",
        lambda **kwargs: recs,
    )

    # User inputs '2' to select the second recommendation
    result = runner.invoke(recommend_cmd, ["--model", "moonshotai/Kimi-K2.5"], input="2\n")
    assert result.exit_code == 0
    assert "Updated config/deployment_spec.yaml with machine type 'a3-ultragpu-8g'" in result.output

    with open(deploy_file, "r") as f:
        data = yaml.safe_load(f)
    assert data["machine_type"] == "a3-ultragpu-8g"


def test_cli_recommend_select_top_default(tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True)
    deploy_file = cfg_dir / "deployment_spec.yaml"
    deploy_file.write_text("machine_type: null\n")

    recs = [
        {
            "machine_type": "a4-highgpu-8g",
            "accelerator_type": "nvidia-b200",
            "accelerator_count": 8,
            "chip_name": "nvidia-b200 (8x)",
            "family": "gpu",
            "input_cost_per_m": None,
            "output_cost_per_m": None,
            "ttft_ms": 68,
            "ntpot_ms": 10,
            "output_tokens_per_sec": 127,
            "model_server": "vllm",
            "engine_params": {"tensor_parallel_size": 8},
        },
        {
            "machine_type": "a3-ultragpu-8g",
            "accelerator_type": "nvidia-h200-141gb",
            "accelerator_count": 8,
            "chip_name": "nvidia-h200-141gb (8x)",
            "family": "gpu",
            "input_cost_per_m": None,
            "output_cost_per_m": None,
            "ttft_ms": 56,
            "ntpot_ms": 10,
            "output_tokens_per_sec": 128,
            "model_server": "vllm",
            "engine_params": {"tensor_parallel_size": 8},
        },
    ]

    monkeypatch.setattr(
        "google.models.cli.recommend.cmd_recommend.ensure_authenticated",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        "google.models.cli.recommend.cmd_recommend.get_recommendations",
        lambda **kwargs: recs,
    )

    # User inputs 'y' to accept the top recommendation
    result = runner.invoke(recommend_cmd, ["--model", "moonshotai/Kimi-K2.5"], input="y\n")
    assert result.exit_code == 0
    assert "Updated config/deployment_spec.yaml with machine type 'a4-highgpu-8g'" in result.output

    with open(deploy_file, "r") as f:
        data = yaml.safe_load(f)
    assert data["machine_type"] == "a4-highgpu-8g"


def test_cli_recommend_displays_token_lengths_in_use_case(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "google.models.cli.recommend.cmd_recommend.ensure_authenticated",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        "google.models.cli.recommend.cmd_recommend.get_recommendations",
        lambda **kwargs: [{
            "machine_type": "a4-highgpu-8g",
            "accelerator_type": "nvidia-b200",
            "accelerator_count": 8,
            "chip_name": "nvidia-b200 (8x)",
            "family": "gpu",
            "input_cost_per_m": None,
            "output_cost_per_m": None,
            "ttft_ms": 68,
            "ntpot_ms": 10,
            "output_tokens_per_sec": 127,
            "model_server": "vllm",
            "engine_params": {"tensor_parallel_size": 8},
            "use_case": "Chatbot (ShareGPT)",
            "average_input_length": 128,
            "average_output_length": 128,
        }],
    )

    result = runner.invoke(recommend_cmd, ["--model", "moonshotai/Kimi-K2.5"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "Chatbot (ShareGPT) (128/128)" in result.output
    assert "Use Case (In/Out)" in result.output


def test_create_recommender_client_uses_quota_project(monkeypatch):
    import google.cloud.gkerecommender_v1 as gke_rec
    from google.models.cli.common.config import settings
    from google.models.cli.recommend.recommend_utils import _create_recommender_client

    mock_cls = MagicMock()
    monkeypatch.setattr(gke_rec, "GkeInferenceQuickstartClient", mock_cls)
    monkeypatch.setattr(settings, "google_cloud_project", "my-scaffolded-project")

    _create_recommender_client()

    mock_cls.assert_called_once_with(
        client_options={"quota_project_id": "my-scaffolded-project"}
    )


def test_create_recommender_client_without_project(monkeypatch, tmp_path):
    import google.cloud.gkerecommender_v1 as gke_rec
    from google.models.cli.common.config import settings
    from google.models.cli.recommend.recommend_utils import _create_recommender_client

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setattr(settings, "google_cloud_project", None)
    monkeypatch.setattr(
        "google.models.cli._gcp_project.resolve_gcp_project",
        lambda override_project=None, **kwargs: override_project or "",
    )
    mock_cls = MagicMock()
    monkeypatch.setattr(gke_rec, "GkeInferenceQuickstartClient", mock_cls)

    _create_recommender_client()

    mock_cls.assert_called_once_with()


def test_find_most_similar_supported_model_dense_prefers_same_origin_in_shortlist():
    from google.models.cli.recommend.recommend_utils import (
        find_most_similar_supported_model,
    )

    # Qwen3.5-9B (9B Dense, Qwen) has Llama-3-8B (8B, Meta) and Qwen3-8B-FP8 (6.95B, Qwen)
    # in its shortlist of similar Dense models -> prefers Qwen3-8B-FP8 from the same origin.
    assert find_most_similar_supported_model("Qwen/Qwen3.5-9B") == "Qwen/Qwen3-8B-FP8"
    # Also resolves short model name without org prefix
    assert find_most_similar_supported_model("Qwen3.5-9B") == "Qwen/Qwen3-8B-FP8"


def test_find_most_similar_supported_model_moe_hierarchy():
    from google.models.cli.recommend.recommend_utils import (
        find_most_similar_supported_model,
    )

    # 1. Architecture dominates origin: gemma-4-26B-A4B-it (25.2B / 3.8B MoE, Google)
    # matches gpt-oss-20b (21B / 3.6B MoE) rather than a Dense Google Gemma model.
    assert (
        find_most_similar_supported_model("google/gemma-4-26B-A4B-it")
        == "openai/gpt-oss-20b"
    )

    # 2. Parameter closeness forms the top-3 shortlist before origin preference:
    # Qwen3.5-122B-A10B-GPTQ-Int4 (122B / 10B MoE, Qwen) has top-3 shortlist
    # [gpt-oss-120b (117B), Llama-4-Scout (109B), Mixtral-8x22B (141B)] -> picks #1 gpt-oss-120b.
    assert (
        find_most_similar_supported_model("Qwen/Qwen3.5-122B-A10B-GPTQ-Int4")
        == "openai/gpt-oss-120b"
    )

    # 3. Large MoE with same-origin candidate in shortlist:
    assert (
        find_most_similar_supported_model("Qwen/Qwen3-235B-A22B")
        == "Qwen/Qwen3.5-397B-A17B"
    )

    # 4. Small/medium MoE with same-origin A3B candidate in top-3 shortlist:
    assert (
        find_most_similar_supported_model("Qwen/Qwen3.6-35B-A3B")
        == "Qwen/Qwen3-Coder-Next"
    )


def test_find_most_similar_supported_model_active_params_for_moe():
    from google.models.cli.common.constants import (
        ModelArchitecture,
        ModelOrigin,
        ModelQuantization,
    )
    from google.models.cli.recommend.recommend_utils import (
        find_most_similar_supported_model,
    )

    custom_specs = {
        "org-a/target-moe": {
            "model_name": "target-moe",
            "origin": ModelOrigin.GOOGLE,
            "architecture": ModelArchitecture.MOE,
            "total_parameters_b": 100.0,
            "active_parameters_b": 8.0,
            "quantization": ModelQuantization.NONE,
            "gke_recommender_supported": False,
        },
        "org-b/moe-high-active": {
            "model_name": "moe-high-active",
            "origin": ModelOrigin.META,
            "architecture": ModelArchitecture.MOE,
            "total_parameters_b": 100.0,
            "active_parameters_b": 32.0,
            "quantization": ModelQuantization.NONE,
            "gke_recommender_supported": True,
        },
        "org-c/moe-close-active": {
            "model_name": "moe-close-active",
            "origin": ModelOrigin.MISTRAL,
            "architecture": ModelArchitecture.MOE,
            "total_parameters_b": 102.0,
            "active_parameters_b": 8.0,
            "quantization": ModelQuantization.NONE,
            "gke_recommender_supported": True,
        },
    }

    assert (
        find_most_similar_supported_model("org-a/target-moe", models_specs=custom_specs)
        == "org-c/moe-close-active"
    )


def test_find_most_similar_supported_model_rejects_extreme_outlier_in_shortlist():
    from google.models.cli.common.constants import (
        ModelArchitecture,
        ModelOrigin,
        ModelQuantization,
    )
    from google.models.cli.recommend.recommend_utils import (
        find_most_similar_supported_model,
    )

    custom_specs = {
        "Qwen/target-10b": {
            "model_name": "target-10b",
            "origin": ModelOrigin.QWEN,
            "architecture": ModelArchitecture.DENSE,
            "total_parameters_b": 10.0,
            "active_parameters_b": 10.0,
            "quantization": ModelQuantization.NONE,
            "gke_recommender_supported": False,
        },
        "meta-llama/exact-10b": {
            "model_name": "exact-10b",
            "origin": ModelOrigin.META,
            "architecture": ModelArchitecture.DENSE,
            "total_parameters_b": 10.0,
            "active_parameters_b": 10.0,
            "quantization": ModelQuantization.NONE,
            "gke_recommender_supported": True,
        },
        "Qwen/huge-500b": {
            "model_name": "huge-500b",
            "origin": ModelOrigin.QWEN,
            "architecture": ModelArchitecture.DENSE,
            "total_parameters_b": 500.0,
            "active_parameters_b": 500.0,
            "quantization": ModelQuantization.NONE,
            "gke_recommender_supported": True,
        },
    }

    # 500B is 50x larger and should not be considered "similar" to 10B just for matching origin
    assert (
        find_most_similar_supported_model("Qwen/target-10b", models_specs=custom_specs)
        == "meta-llama/exact-10b"
    )


def test_get_recommendations_uses_proxy_model_when_unsupported_in_catalog():
    mock_client = MagicMock()
    mock_client.fetch_profiles.return_value = [_make_mock_profile()]

    recs = get_recommendations("Qwen/Qwen3.5-9B", client=mock_client)

    mock_client.fetch_profiles.assert_called_once()
    req = mock_client.fetch_profiles.call_args[1]["request"]
    assert req.model == "Qwen/Qwen3-8B-FP8"
    assert len(recs) == 1
    assert recs[0]["proxy_model"] == "Qwen/Qwen3-8B-FP8"
    assert recs[0]["requested_model"] == "Qwen/Qwen3.5-9B"


def test_cli_recommend_displays_proxy_notice_and_json_field(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "google.models.cli.recommend.cmd_recommend.ensure_authenticated",
        lambda **kwargs: True,
    )
    sample_rec = {
        "machine_type": "g2-standard-12",
        "accelerator_type": "nvidia-l4",
        "accelerator_count": 1,
        "chip_name": "nvidia-l4 (1x)",
        "family": "gpu",
        "input_cost_per_m": 0.019,
        "output_cost_per_m": 0.078,
        "output_input_cost_ratio": 4.0,
        "pricing_model": "on-demand",
        "ttft_ms": 120,
        "ntpot_ms": 18,
        "output_tokens_per_sec": 1059,
        "model_server": "vllm",
        "engine_params": {"tensor_parallel_size": 1},
        "proxy_model": "Qwen/Qwen3-8B-FP8",
        "requested_model": "Qwen/Qwen3.5-9B",
    }
    monkeypatch.setattr(
        "google.models.cli.recommend.cmd_recommend.get_recommendations",
        lambda **kwargs: [sample_rec],
    )

    table_res = runner.invoke(recommend_cmd, ["--model", "Qwen/Qwen3.5-9B"])
    assert table_res.exit_code == 0
    assert "Similar Model Proxy" in table_res.output
    assert "Qwen/Qwen3-8B-FP8" in table_res.output

    json_res = runner.invoke(
        recommend_cmd, ["--model", "Qwen/Qwen3.5-9B", "--format", "json"]
    )
    assert json_res.exit_code == 0
    assert '"proxy_model": "Qwen/Qwen3-8B-FP8"' in json_res.output


def test_recommend_utils_preserves_lazy_gkerecommender_import():
    import subprocess
    import sys

    code = (
        "import sys; "
        "import google.models.cli.main; "
        "assert 'google.models.cli.recommend.recommend_utils' not in sys.modules; "
        "assert 'google.cloud.gkerecommender_v1' not in sys.modules; "
        "import google.models.cli.recommend.recommend_utils; "
        "assert 'google.cloud.gkerecommender_v1' not in sys.modules"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr


