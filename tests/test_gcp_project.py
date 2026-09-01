# Copyright 2026 Google LLC
import pytest
from google.models.cli._gcp_project import resolve_gcp_project


def test_resolve_gcp_project_override():
    project = resolve_gcp_project(override_project="custom-project-999")
    assert project == "custom-project-999"


def test_resolve_gcp_project_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project-123")
    project = resolve_gcp_project()
    assert project == "env-project-123"


def test_resolve_gcp_project_required_raises(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    monkeypatch.delenv("PROJECT_ID", raising=False)
    monkeypatch.delenv("CLOUDSDK_CORE_PROJECT", raising=False)
    
    # Mock ADC and gcloud to return None
    monkeypatch.setattr("google.models.cli._gcp_project._get_project_from_adc", lambda: None)
    monkeypatch.setattr("google.models.cli._gcp_project._get_project_from_gcloud", lambda: None)

    with pytest.raises(ValueError, match="Could not determine active Google Cloud project"):
        resolve_gcp_project(required=True)


def test_list_accessible_gcp_projects(monkeypatch):
    from unittest.mock import MagicMock
    from google.models.cli._gcp_project import list_accessible_gcp_projects

    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/gcloud")
    mock_run = MagicMock(returncode=0, stdout="proj-alpha\nproj-beta\n")
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: mock_run)

    projects = list_accessible_gcp_projects()
    assert projects == ["proj-alpha", "proj-beta"]


def test_get_project_from_gcloud_unset(monkeypatch):
    from unittest.mock import MagicMock
    from google.models.cli._gcp_project import _get_project_from_gcloud

    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/gcloud")
    mock_run = MagicMock(returncode=0, stdout="(unset)\n")
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: mock_run)

    assert _get_project_from_gcloud() is None


