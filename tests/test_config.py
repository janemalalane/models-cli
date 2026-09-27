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
