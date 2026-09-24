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

import click
import os
import shutil
import subprocess
from typing import Optional
import subprocess


def _get_project_from_gcloud() -> Optional[str]:
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


def _get_account_from_gcloud() -> Optional[str]:
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


def _get_adc_project() -> Optional[str]:
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


def get_active_gcp_account() -> Optional[str]:
    """Retrieves active GCP account email."""
    account = os.environ.get("GOOGLE_CLOUD_ACCOUNT")
    if account:
        return account
    return _get_account_from_gcloud()


def get_gcp_access_token() -> Optional[str]:
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
