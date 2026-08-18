# Copyright 2026 Google LLC
import pytest
from google.models.cli.common.config import Settings
from google.models.cli.common.constants import DEFAULT_MODEL_REPO, DEFAULT_REGION


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("MODEL_ID", raising=False)
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.delenv("HF_MODEL_REPO", raising=False)
    settings = Settings()
    assert settings.google_cloud_location == DEFAULT_REGION
    assert settings.model_id == DEFAULT_MODEL_REPO
    assert settings.model == DEFAULT_MODEL_REPO


def test_settings_model_id_priority(monkeypatch):
    monkeypatch.setenv("MODEL_ID", "custom-model-id-123")
    settings = Settings()
    assert settings.model_id == "custom-model-id-123"


def test_settings_validate_deployment_missing():
    settings = Settings(google_cloud_project=None, service_account_email=None)
    with pytest.raises(ValueError, match="Missing required environment variables"):
        settings.validate_deployment_env()


def test_settings_validate_deployment_valid():
    settings = Settings(
        google_cloud_project="test-proj",
        google_cloud_location="us-central1",
        service_account_email="sa@test-proj.iam.gserviceaccount.com",
    )
    res = settings.validate_deployment_env()
    assert res["project_id"] == "test-proj"
    assert res["location"] == "us-central1"
    assert res["service_account"] == "sa@test-proj.iam.gserviceaccount.com"
