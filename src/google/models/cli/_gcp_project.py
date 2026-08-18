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
from typing import Optional


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


def _get_project_from_adc() -> Optional[str]:
    """Retrieves project from Google Application Default Credentials."""
    try:
        import google.auth
        _, project_id = google.auth.default()
        if project_id:
            return project_id
    except Exception:
        pass
    return None


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
    override_project: Optional[str] = None,
    *,
    required: bool = False,
) -> Optional[str]:
    """Resolves the GCP project ID to use.

    Order of precedence:
    1. The `override_project` argument (e.g. from --project CLI flag).
    2. The `GOOGLE_CLOUD_PROJECT` or `GCP_PROJECT_ID` environment variable.
    3. Application Default Credentials via `google.auth.default()`.
    4. Active project in `gcloud config get-value project`.

    Args:
        override_project: Explicit project ID override.
        required: If True and project cannot be resolved, raises ValueError.

    Returns:
        Resolved project ID string or None.
    """
    if override_project:
        return override_project.strip()

    # Check environment variables
    for env_key in ("GOOGLE_CLOUD_PROJECT", "GCP_PROJECT_ID", "PROJECT_ID", "CLOUDSDK_CORE_PROJECT"):
        val = os.environ.get(env_key)
        if val and val.strip():
            return val.strip()

    # Check ADC
    adc_project = _get_project_from_adc()
    if adc_project:
        return adc_project.strip()

    # Check gcloud CLI
    gcloud_project = _get_project_from_gcloud()
    if gcloud_project:
        return gcloud_project.strip()

    if required:
        raise ValueError(
            "Could not determine active Google Cloud project.\n"
            "Please specify via --project, set the GOOGLE_CLOUD_PROJECT environment variable, "
            "or configure gcloud via `gcloud config set project <PROJECT_ID>`."
        )

    return None
