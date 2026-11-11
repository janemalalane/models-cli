import click
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
    
    # Mock ADC to return None
    monkeypatch.setattr("google.models.cli._gcp_project._get_project_from_adc", lambda: None)

    with pytest.raises(click.ClickException, match="Could not determine GCP project"):
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


def test_get_gcp_project_number_numeric():
    from google.models.cli._gcp_project import get_gcp_project_number

    assert get_gcp_project_number("123456789") == "123456789"
    assert get_gcp_project_number("") is None


def test_get_gcp_project_number_resourcemanager(monkeypatch):
    from unittest.mock import MagicMock
    from google.models.cli._gcp_project import get_gcp_project_number

    mock_client = MagicMock()
    mock_proj = MagicMock()
    mock_proj.name = "projects/9876543210"
    mock_client.get_project.return_value = mock_proj

    import google.cloud.resourcemanager_v3 as rm_v3
    monkeypatch.setattr(rm_v3, "ProjectsClient", lambda: mock_client)

    assert get_gcp_project_number("my-project") == "9876543210"
    mock_client.get_project.assert_called_once_with(name="projects/my-project")


def test_check_gke_recommender_api_enabled_empty_project():
    from google.models.cli._gcp_project import check_gke_recommender_api_enabled

    enabled, msg = check_gke_recommender_api_enabled("")
    assert enabled is None
    assert "not set" in msg


def test_check_gke_recommender_api_enabled_gcloud_true(monkeypatch):
    from unittest.mock import MagicMock
    from google.models.cli._gcp_project import check_gke_recommender_api_enabled

    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/gcloud")
    mock_run = MagicMock(returncode=0, stdout="gkerecommender.googleapis.com\n")
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: mock_run)

    enabled, msg = check_gke_recommender_api_enabled("my-project")
    assert enabled is True
    assert "gkerecommender.googleapis.com" in msg


def test_check_gke_recommender_api_enabled_gcloud_false(monkeypatch):
    from unittest.mock import MagicMock
    from google.models.cli._gcp_project import check_gke_recommender_api_enabled

    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/gcloud")
    mock_run = MagicMock(returncode=0, stdout="")
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: mock_run)

    enabled, msg = check_gke_recommender_api_enabled("my-project")
    assert enabled is False
    assert msg == "Not enabled"


def test_check_gke_recommender_api_enabled_rest_fallback(monkeypatch):
    from unittest.mock import MagicMock
    import google.auth
    import google.auth.transport.requests as ga_requests
    from google.models.cli._gcp_project import check_gke_recommender_api_enabled

    monkeypatch.setattr("shutil.which", lambda cmd: None)
    monkeypatch.setattr(google.auth, "default", lambda scopes=None: (MagicMock(), "my-project"))

    mock_session = MagicMock()
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"state": "ENABLED"}
    mock_session.get.return_value = mock_resp
    monkeypatch.setattr(ga_requests, "AuthorizedSession", lambda creds: mock_session)

    enabled, msg = check_gke_recommender_api_enabled("my-project")
    assert enabled is True
    assert "gkerecommender.googleapis.com" in msg


def test_enable_gke_recommender_api_gcloud_success(monkeypatch):
    from unittest.mock import MagicMock
    from google.models.cli._gcp_project import enable_gke_recommender_api

    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/gcloud")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    ok, msg = enable_gke_recommender_api("my-project")
    assert ok is True
    assert "gkerecommender.googleapis.com" in captured["cmd"]
    assert "--project=my-project" in captured["cmd"]
    assert "Enabled GKE Recommender API" in msg


def test_enable_gke_recommender_api_rest_fallback_success(monkeypatch):
    from unittest.mock import MagicMock
    import google.auth
    import google.auth.transport.requests as ga_requests
    from google.models.cli._gcp_project import enable_gke_recommender_api

    monkeypatch.setattr("shutil.which", lambda cmd: None)
    monkeypatch.setattr(google.auth, "default", lambda scopes=None: (MagicMock(), "my-project"))

    mock_session = MagicMock()
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"name": "operations/noop.DONE_OPERATION", "done": True}
    mock_session.post.return_value = mock_resp
    monkeypatch.setattr(ga_requests, "AuthorizedSession", lambda creds: mock_session)

    ok, msg = enable_gke_recommender_api("my-project")
    assert ok is True
    assert "Enabled GKE Recommender API" in msg


def test_enable_gke_recommender_api_failure(monkeypatch):
    from unittest.mock import MagicMock
    import google.auth
    from google.models.cli._gcp_project import enable_gke_recommender_api

    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/gcloud")
    mock_run = MagicMock(returncode=1, stdout="", stderr="PERMISSION_DENIED: Caller does not have permission")
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: mock_run)
    monkeypatch.setattr(google.auth, "default", MagicMock(side_effect=Exception("No ADC")))

    ok, msg = enable_gke_recommender_api("my-project")
    assert ok is False
    assert "PERMISSION_DENIED" in msg



