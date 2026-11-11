# Copyright 2026 Google LLC
import pytest
from google.models.cli.common.config import Settings, is_single_region
from google.models.cli.common.constants import DEFAULT_MODEL_REPO


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_STORAGE_BUCKET", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_STORAGE_BUCKET_BASE_PATH", raising=False)
    monkeypatch.delenv("ARTIFACT_BASE_PATH", raising=False)
    monkeypatch.delenv("BASE_PATH", raising=False)
    monkeypatch.delenv("MODEL_ID", raising=False)
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.delenv("HF_MODEL_REPO", raising=False)
    settings = Settings()
    assert settings.google_cloud_project is None
    assert settings.google_cloud_location is None
    assert settings.google_cloud_storage_bucket is None
    assert settings.artifact_base_path is None
    assert settings.base_path is None
    assert settings.google_cloud_storage_bucket_base_path is None
    assert settings.model_id == DEFAULT_MODEL_REPO
    assert settings.model == DEFAULT_MODEL_REPO


def test_settings_model_id_priority(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_STORAGE_BUCKET_BASE_PATH", raising=False)
    monkeypatch.delenv("ARTIFACT_BASE_PATH", raising=False)
    monkeypatch.delenv("BASE_PATH", raising=False)
    monkeypatch.setenv("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
    settings = Settings()
    assert settings.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert settings.model_name == "Llama-3.1-8B-Instruct"
    assert settings.artifact_base_path is None
    assert settings.base_path is None
    assert settings.google_cloud_storage_bucket_base_path is None


def test_settings_custom_artifact_base_path(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_STORAGE_BUCKET_BASE_PATH", raising=False)
    monkeypatch.delenv("BASE_PATH", raising=False)
    monkeypatch.setenv("MODEL_ID", "google/gemma-4-31B-it")
    monkeypatch.setenv("ARTIFACT_BASE_PATH", "custom/weights/folder")
    settings = Settings()
    assert settings.artifact_base_path == "custom/weights/folder"
    assert settings.base_path == "custom/weights/folder"
    assert settings.google_cloud_storage_bucket_base_path == "custom/weights/folder"


def test_settings_custom_base_path(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_STORAGE_BUCKET_BASE_PATH", raising=False)
    monkeypatch.delenv("ARTIFACT_BASE_PATH", raising=False)
    monkeypatch.setenv("MODEL_ID", "google/gemma-4-31B-it")
    monkeypatch.setenv("BASE_PATH", "custom/base/folder")
    settings = Settings()
    assert settings.base_path == "custom/base/folder"
    assert settings.artifact_base_path == "custom/base/folder"
    assert settings.google_cloud_storage_bucket_base_path == "custom/base/folder"


def test_settings_custom_google_cloud_storage_bucket_base_path(monkeypatch):
    monkeypatch.delenv("BASE_PATH", raising=False)
    monkeypatch.delenv("ARTIFACT_BASE_PATH", raising=False)
    monkeypatch.setenv("MODEL_ID", "google/gemma-4-31B-it")
    monkeypatch.setenv("GOOGLE_CLOUD_STORAGE_BUCKET_BASE_PATH", "custom/gcs/folder")
    settings = Settings()
    assert settings.google_cloud_storage_bucket_base_path == "custom/gcs/folder"
    assert settings.base_path == "custom/gcs/folder"
    assert settings.artifact_base_path == "custom/gcs/folder"


def test_single_region_validation():
    assert is_single_region("us-central1") is True
    assert is_single_region("europe-west4") is True
    assert is_single_region("asia-southeast1") is True
    assert is_single_region("US") is False
    assert is_single_region("eu") is False
    assert is_single_region("global") is False

    with pytest.raises(ValueError, match="Must be a single region like 'us-central1'"):
        Settings(google_cloud_location="US")


def test_settings_validate_deployment_missing(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_STORAGE_BUCKET", raising=False)
    settings = Settings()
    with pytest.raises(ValueError, match="Missing required environment variables"):
        settings.validate_deployment_env()


def test_settings_validate_deployment_valid():
    settings = Settings(
        google_cloud_project="test-proj",
        google_cloud_location="us-central1",
        google_cloud_storage_bucket="my-model-bucket",
    )
    res = settings.validate_deployment_env()
    assert res == {
        "project_id": "test-proj",
        "location": "us-central1",
        "bucket": "my-model-bucket",
    }


def test_models_specs_catalog_and_enums():
    from google.models.cli.common.constants import (
        DEFAULT_MODEL_REPO,
        MODELS_SPECS,
        ModelArchitecture,
        ModelOrigin,
        ModelQuantization,
    )

    assert len(MODELS_SPECS) == 32
    assert DEFAULT_MODEL_REPO in MODELS_SPECS

    required_keys = {
        "model_name",
        "origin",
        "architecture",
        "total_parameters_b",
        "active_parameters_b",
        "total_experts",
        "activated_experts",
        "shared_experts",
        "quantization",
        "gke_recommender_supported",
        "rank",
        "timestamp",
        "model_card_url",
    }

    for repo_id, spec in MODELS_SPECS.items():
        assert "/" in repo_id
        assert set(spec.keys()) == required_keys
        assert isinstance(spec["origin"], ModelOrigin)
        assert isinstance(spec["architecture"], ModelArchitecture)
        assert isinstance(spec["quantization"], ModelQuantization)
        assert isinstance(spec["gke_recommender_supported"], bool)
        assert spec["total_parameters_b"] >= spec["active_parameters_b"] > 0

        if spec["architecture"] == ModelArchitecture.DENSE:
            assert spec["total_parameters_b"] == spec["active_parameters_b"]
            assert spec["total_experts"] is None
            assert spec["activated_experts"] is None
            assert spec["shared_experts"] is None
        else:
            assert spec["architecture"] == ModelArchitecture.MOE
            assert spec["total_experts"] is not None and spec["total_experts"] > 0
            assert spec["activated_experts"] is not None and spec["activated_experts"] > 0


def test_models_specs_matches_models_reference_csv():
    import csv
    from pathlib import Path
    from google.models.cli.common.constants import (
        MODELS_SPECS,
        ModelArchitecture,
        ModelOrigin,
        ModelQuantization,
    )

    csv_path = Path(__file__).resolve().parent.parent / "models_reference.csv"
    if not csv_path.is_file():
        pytest.skip("models_reference.csv not present")

    origin_map = {
        "Google": ModelOrigin.GOOGLE,
        "Qwen": ModelOrigin.QWEN,
        "Meta": ModelOrigin.META,
        "Mistral": ModelOrigin.MISTRAL,
        "Deepseek-ai": ModelOrigin.DEEPSEEK_AI,
        "Moonshotai": ModelOrigin.MOONSHOTAI,
        "OpenAI": ModelOrigin.OPENAI,
    }
    arch_map = {
        "Dense": ModelArchitecture.DENSE,
        "MOE": ModelArchitecture.MOE,
    }
    quant_map = {
        "": ModelQuantization.NONE,
        "fp8": ModelQuantization.FP8,
        "GPTQ 4-bit": ModelQuantization.GPTQ_4BIT,
        "MXFP4": ModelQuantization.MXFP4,
    }

    def _parse_params_b(raw: str) -> float:
        val = raw.strip()
        if val.endswith("T"):
            return float(val[:-1]) * 1000.0
        if val.endswith("B"):
            return float(val[:-1])
        return float(val)

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == len(MODELS_SPECS)

    for row in rows:
        url = row["Model card "].strip()
        repo_id = url.removeprefix("https://huggingface.co/")
        assert repo_id in MODELS_SPECS

        m, d, y = row["Timestamp"].strip().split("/")
        iso_ts = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

        expected = {
            "model_name": row["Model name"].strip(),
            "origin": origin_map[row["Origin"].strip()],
            "architecture": arch_map[row["Dense/MOE"].strip()],
            "total_parameters_b": _parse_params_b(row["# Total Parameters"]),
            "active_parameters_b": _parse_params_b(row["# Active Parameters"]),
            "total_experts": int(row["# Total Experts"]) if row["# Total Experts"].strip() else None,
            "activated_experts": int(row["# Activated Experts"]) if row["# Activated Experts"].strip() else None,
            "shared_experts": int(row["# Shared Experts"]) if row["# Shared Experts"].strip() else None,
            "quantization": quant_map[row["Quantization"].strip()],
            "gke_recommender_supported": row["GKE Recommend API included"].strip() == "Yes",
            "rank": int(row["Rank #"]) if row["Rank #"].strip() else None,
            "timestamp": iso_ts,
            "model_card_url": url,
        }
        assert MODELS_SPECS[repo_id] == expected


