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

"""GCP project resolution and lookup utilities."""

import os
import shutil
import subprocess
import click


def _get_project_from_gcloud() -> str | None:
    """Retrieves the active project configured in gcloud CLI."""
    if not shutil.which("gcloud"):
        return None
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            project = result.stdout.strip()
            if project and project != "(unset)":
                return project
    except Exception:
        pass
    return None


def _get_account_from_gcloud() -> str | None:
    """Retrieves the active account logged into gcloud CLI."""
    if not shutil.which("gcloud"):
        return None
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "account"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            account = result.stdout.strip()
            if account and account != "(unset)":
                return account
    except Exception:
        pass
    return None


def list_accessible_gcp_projects(limit: int = 15) -> list[str]:
    """Attempts to list accessible GCP project IDs via gcloud CLI."""
    if not shutil.which("gcloud"):
        return []
    try:
        result = subprocess.run(
            [
                "gcloud",
                "projects",
                "list",
                f"--limit={limit}",
                "--format=value(projectId)",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            projects = [
                p.strip() for p in result.stdout.strip().splitlines() if p.strip()
            ]
            return projects
    except Exception:
        pass
    return []


def _get_adc_project() -> str | None:
    """Retrieves project from Google Application Default Credentials."""
    try:
        import google.auth

        _, project_id = google.auth.default()
        if project_id:
            return project_id
    except Exception:
        pass
    return None


_get_project_from_adc = _get_adc_project


def get_active_gcp_account() -> str | None:
    """Retrieves active GCP account email."""
    account = os.environ.get("GOOGLE_CLOUD_ACCOUNT")
    if account:
        return account
    return _get_account_from_gcloud()


def get_gcp_access_token() -> str | None:
    """Dynamically fetches or refreshes an active GCP OAuth2 access token."""
    # 1. Try google.auth default credentials
    try:
        import google.auth
        from google.auth.transport.requests import Request

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        if not credentials.valid:
            credentials.refresh(Request())
        if credentials.token:
            return credentials.token
    except Exception:
        pass

    # 2. Fallback to gcloud auth print-access-token
    if shutil.which("gcloud"):
        try:
            result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if result.returncode == 0:
                tok = result.stdout.strip()
                if tok:
                    return tok
        except Exception:
            pass

    return None


def resolve_gcp_project(
    override_project: str | None = None, *, required: bool = False
) -> str:
    """Resolves the GCP project ID to use.

    The project ID is resolved in the following order of precedence:

    1.  The ``override_project`` argument if provided.
        It's expected this would come from a --project command line argument.
    2.  The ``GOOGLE_CLOUD_PROJECT`` environment variable.
    3.  Application Default Credentials via :func:`google.auth.default`,
        which itself checks (in order):

        a.  ``GOOGLE_APPLICATION_CREDENTIALS`` service account JSON file.
        b.  The gcloud SDK ADC file
            (``gcloud auth application-default login``); when this file
            exists but lacks a project, the gcloud SDK falls back to
            ``gcloud config get-value project``.
        c.  GAE metadata service.

    Returns:
        The resolved GCP project ID, or an empty string if no project is found.
    """
    if override_project:
        return override_project
    env_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if env_project:
        return env_project

    project = _get_project_from_adc() or ""
    if required and not project:
        raise click.ClickException(
            "Could not determine GCP project. Set one with:\n"
            "  * pass --project <PROJECT_ID>\n"
            "  * export GOOGLE_CLOUD_PROJECT=<PROJECT_ID>\n"
            "  * gcloud config set project <PROJECT_ID>"
        )
    return project


def get_gcp_project_number(project_id: str) -> str | None:
    """Get numeric GCP project number for a project ID or project number using Resource Manager API.

    Args:
        project_id: GCP project ID or project number (e.g., 'my-project' or '123456789').

    Returns:
        Project number string, or None if lookup fails.
    """
    if not project_id:
        return None

    if project_id.isdigit():
        return project_id

    try:
        from google.cloud import resourcemanager_v3

        client = resourcemanager_v3.ProjectsClient()
        project = client.get_project(name=f"projects/{project_id}")
        if project.name and "/" in project.name:
            number = project.name.split("/")[-1]
            if number.isdigit():
                return number
    except Exception:
        pass

    return None


def _format_api_check_error(project_id: str, raw_error: str) -> str:
    """Formats a concise human-readable reason when API status check fails."""
    lowered = raw_error.lower()
    if "not found" in lowered:
        return f"GCP project '{project_id}' not found"
    if "permission" in lowered or "denied" in lowered or "403" in lowered:
        return f"Permission denied checking API status for '{project_id}'"
    return "Unable to verify API status"


def check_gke_recommender_api_enabled(project_id: str) -> tuple[bool | None, str]:
    """Checks whether the GKE Recommender API (gkerecommender.googleapis.com) is enabled.

    Args:
        project_id: Target GCP project ID.

    Returns:
        Tuple of (enabled, detail_message) where ``enabled`` is:
        - ``True`` if confirmed enabled,
        - ``False`` if confirmed disabled,
        - ``None`` if status could not be determined (e.g. unauthenticated, missing permissions, or project not found).
    """
    from google.models.cli.common.constants import GKE_RECOMMENDER_SERVICE

    if not project_id:
        return None, "GCP project not set"

    gcloud_err = ""
    if shutil.which("gcloud"):
        try:
            res = subprocess.run(
                [
                    "gcloud",
                    "services",
                    "list",
                    "--enabled",
                    f"--project={project_id}",
                    f"--filter=config.name:{GKE_RECOMMENDER_SERVICE}",
                    "--format=value(config.name)",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if res.returncode == 0:
                enabled_services = [
                    line.strip()
                    for line in res.stdout.splitlines()
                    if line.strip()
                ]
                if GKE_RECOMMENDER_SERVICE in enabled_services:
                    return True, f"Enabled ({GKE_RECOMMENDER_SERVICE})"
                return False, "Not enabled"
            gcloud_err = res.stderr.strip()
        except Exception as exc:
            gcloud_err = str(exc)

    # Fallback to Service Usage v1 REST API via Application Default Credentials
    try:
        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        session = AuthorizedSession(credentials)
        url = (
            f"https://serviceusage.googleapis.com/v1/projects/{project_id}"
            f"/services/{GKE_RECOMMENDER_SERVICE}"
        )
        resp = session.get(
            url,
            headers={"x-goog-user-project": project_id},
            timeout=10,
        )
        if resp.status_code == 200:
            state = resp.json().get("state", "")
            if state == "ENABLED":
                return True, f"Enabled ({GKE_RECOMMENDER_SERVICE})"
            return False, "Not enabled"
        if resp.status_code == 404:
            return None, f"GCP project '{project_id}' not found"
        if resp.status_code == 403:
            return None, f"Permission denied checking API status for '{project_id}'"
    except Exception as exc:
        if not gcloud_err:
            gcloud_err = str(exc)

    return None, _format_api_check_error(project_id, gcloud_err)


def enable_gke_recommender_api(project_id: str) -> tuple[bool, str]:
    """Enables the GKE Recommender API (gkerecommender.googleapis.com) in the given project.

    Args:
        project_id: Target GCP project ID.

    Returns:
        Tuple of (success, message).
    """
    from google.models.cli.common.constants import GKE_RECOMMENDER_SERVICE

    if not project_id:
        return False, "GCP project ID is required."

    gcloud_err = ""
    if shutil.which("gcloud"):
        try:
            res = subprocess.run(
                [
                    "gcloud",
                    "services",
                    "enable",
                    GKE_RECOMMENDER_SERVICE,
                    f"--project={project_id}",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            if res.returncode == 0:
                return (
                    True,
                    f"Enabled GKE Recommender API ({GKE_RECOMMENDER_SERVICE}) in '{project_id}'.",
                )
            gcloud_err = res.stderr.strip()
        except Exception as exc:
            gcloud_err = str(exc)

    # Fallback to Service Usage v1 REST API via Application Default Credentials
    try:
        import time
        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        session = AuthorizedSession(credentials)
        url = (
            f"https://serviceusage.googleapis.com/v1/projects/{project_id}"
            f"/services/{GKE_RECOMMENDER_SERVICE}:enable"
        )
        resp = session.post(
            url,
            headers={"x-goog-user-project": project_id},
            json={},
            timeout=60,
        )
        if resp.status_code == 200:
            op = resp.json()
            op_name = op.get("name")
            done = op.get("done", False)
            if (
                op_name
                and not done
                and "DONE_OPERATION" not in str(op_name)
            ):
                op_url = f"https://serviceusage.googleapis.com/v1/{op_name}"
                for _ in range(15):
                    time.sleep(2)
                    op_resp = session.get(
                        op_url,
                        headers={"x-goog-user-project": project_id},
                        timeout=10,
                    )
                    if op_resp.status_code == 200:
                        op_data = op_resp.json()
                        if op_data.get("done"):
                            if "error" in op_data:
                                err_msg = op_data["error"].get(
                                    "message", "Operation failed"
                                )
                                return False, err_msg
                            break
            return (
                True,
                f"Enabled GKE Recommender API ({GKE_RECOMMENDER_SERVICE}) in '{project_id}'.",
            )
        err_body = resp.text.strip()
        if not gcloud_err and err_body:
            gcloud_err = err_body.splitlines()[0]
    except Exception as exc:
        if not gcloud_err:
            gcloud_err = str(exc)

    first_line = gcloud_err.splitlines()[0] if gcloud_err else f"Failed to enable {GKE_RECOMMENDER_SERVICE}."
    return False, first_line
