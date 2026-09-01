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

"""Configuration management and environment variable loader."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from google.models.cli._gcp_project import resolve_gcp_project
from google.models.cli.common.constants import DEFAULT_MODEL_REPO, DEFAULT_REGION


def find_dotenv() -> Optional[Path]:
    """Search upwards for .env file starting from current working directory."""
    current = Path.cwd()
    for directory in [current, *current.parents]:
        env_file = directory / ".env"
        if env_file.is_file():
            return env_file
    return None


# Auto load .env if found
_dotenv_path = find_dotenv()
if _dotenv_path:
    load_dotenv(dotenv_path=_dotenv_path)
else:
    load_dotenv()


class Settings(BaseSettings):
    """Runtime settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        extra="ignore",
    )

    # Google Cloud Configuration
    google_cloud_project: Optional[str] = None
    google_cloud_location: str = (
        os.environ.get("GOOGLE_CLOUD_LOCATION") or DEFAULT_REGION
    )
    google_cloud_storage_bucket: Optional[str] = os.environ.get(
        "GOOGLE_CLOUD_STORAGE_BUCKET"
    )
    service_account_email: Optional[str] = os.environ.get("SERVICE_ACCOUNT_EMAIL")

    # Container / Docker Registry
    docker_repository: Optional[str] = os.environ.get("DOCKER_REPOSITORY")
    device_type: Optional[str] = os.environ.get("DEVICE_TYPE")

    # Model Configuration (Single Source of Truth)
    hf_token: Optional[str] = os.environ.get("HF_TOKEN")
    model_id: str = (
        os.environ.get("MODEL_ID")
        or os.environ.get("MODEL")
        or os.environ.get("HF_MODEL_REPO")
        or DEFAULT_MODEL_REPO
    )

    @property
    def model(self) -> str:
        """Alias for model_id."""
        return self.model_id

    @property
    def hf_model_repo(self) -> str:
        """Alias for model_id for backwards compatibility."""
        return self.model_id

    # Endpoint & Auth
    auth_token: Optional[str] = os.environ.get("AUTH_TOKEN")
    endpoint_url: Optional[str] = os.environ.get("ENDPOINT_URL")

    def get_project_id(self, override_project: Optional[str] = None) -> Optional[str]:
        """Resolves active GCP project ID using project resolution chain."""
        return resolve_gcp_project(
            override_project=override_project or self.google_cloud_project
        )

    def validate_deployment_env(
        self, override_project: Optional[str] = None
    ) -> dict[str, str]:
        """Validates that essential deployment variables are set."""
        project_id = self.get_project_id(override_project)
        missing = []
        if not project_id:
            missing.append("GOOGLE_CLOUD_PROJECT")
        if not self.google_cloud_location:
            missing.append("GOOGLE_CLOUD_LOCATION")
        if not self.service_account_email:
            missing.append("SERVICE_ACCOUNT_EMAIL")

        if (
            not project_id
            or not self.google_cloud_location
            or not self.service_account_email
        ):
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Please configure them in your .env file or environment."
            )

        return {
            "project_id": project_id,
            "location": self.google_cloud_location,
            "service_account": self.service_account_email,
        }


# Global settings singleton
settings = Settings()
