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
import re
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from google.models.cli.common.constants import DEFAULT_MODEL_REPO

# Single GCP region pattern, e.g. us-central1, europe-west4, asia-southeast1
# Rejects multi-regions like 'us', 'eu', 'asia', 'global', 'nam4', 'eur4'.
SINGLE_REGION_PATTERN = re.compile(r"^[a-z]+-[a-z]+\d+$")


def is_single_region(location: str) -> bool:
    """Returns True if location is a valid single GCP region (e.g. us-central1), not a multi-region."""
    if not location:
        return False
    return bool(SINGLE_REGION_PATTERN.match(location.strip().lower()))


def validate_single_region(location: str) -> str:
    """Validates that a location string is a single GCP region and returns it normalized."""
    cleaned = location.strip().lower()
    if not is_single_region(cleaned):
        raise ValueError(
            f"Invalid GOOGLE_CLOUD_LOCATION '{location}'. "
            "Must be a single region like 'us-central1' (not a multi-region like 'US', 'EU', or 'ASIA')."
        )
    return cleaned


def extract_model_name(model_id: str) -> str:
    """Extracts the model name (second part of 'org/model-name', or full string if no slash)."""
    parts = model_id.strip().strip("/").split("/")
    if len(parts) >= 2:
        return parts[1]
    return parts[0]


def find_dotenv() -> Path | None:
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

    # Compulsory Google Cloud Configuration (validated via validate_deployment_env)
    google_cloud_project: str | None = None
    google_cloud_location: str | None = None
    google_cloud_storage_bucket: str | None = None

    # Optional Artifact Base Path inside the GCS bucket.
    # Remains None if not provided in environment.
    artifact_base_path: str | None = None
    base_path: str | None = None
    google_cloud_storage_bucket_base_path: str | None = None

    # Optional Service Account & Container Configuration
    service_account_email: str | None = None
    docker_repository: str | None = None
    device_type: str | None = None

    # Model Configuration (Single Source of Truth)
    hf_token: str | None = None
    model_id: str = DEFAULT_MODEL_REPO

    # Endpoint & Auth
    auth_token: str | None = None
    endpoint_url: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _load_and_resolve_env(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Loads environment variables dynamically and sets defaults for derived fields."""
        data = dict(values) if isinstance(values, dict) else {}

        # 1. Resolve model_id with priority: explicit arg > MODEL_ID > MODEL > HF_MODEL_REPO > default
        if not data.get("model_id"):
            data["model_id"] = (
                os.environ.get("MODEL_ID")
                or os.environ.get("MODEL")
                or os.environ.get("HF_MODEL_REPO")
                or DEFAULT_MODEL_REPO
            )

        # 2. Resolve GCP Project, Location, and Bucket from env if not explicitly passed
        if "google_cloud_project" not in data:
            data["google_cloud_project"] = (
                os.environ.get("GOOGLE_CLOUD_PROJECT") or None
            )
        if "google_cloud_location" not in data:
            data["google_cloud_location"] = (
                os.environ.get("GOOGLE_CLOUD_LOCATION") or None
            )
        if "google_cloud_storage_bucket" not in data:
            data["google_cloud_storage_bucket"] = (
                os.environ.get("GOOGLE_CLOUD_STORAGE_BUCKET") or None
            )

        # Normalize bucket name if gs:// prefix or trailing slash was included
        if data.get("google_cloud_storage_bucket"):
            bucket_val = str(data["google_cloud_storage_bucket"]).strip()
            if bucket_val.startswith("gs://"):
                bucket_val = bucket_val[len("gs://") :]
            data["google_cloud_storage_bucket"] = bucket_val.strip("/") or None

        # Validate single region if google_cloud_location is set
        if data.get("google_cloud_location"):
            data["google_cloud_location"] = validate_single_region(
                str(data["google_cloud_location"])
            )

        # 3. Resolve artifact_base_path, base_path, and google_cloud_storage_bucket_base_path (stays None if not provided)
        raw_artifact_path = (
            data.get("google_cloud_storage_bucket_base_path")
            or data.get("base_path")
            or data.get("artifact_base_path")
            or os.environ.get("GOOGLE_CLOUD_STORAGE_BUCKET_BASE_PATH")
            or os.environ.get("BASE_PATH")
            or os.environ.get("ARTIFACT_BASE_PATH")
        )
        if raw_artifact_path and str(raw_artifact_path).strip():
            cleaned_base_path = str(raw_artifact_path).strip().strip("/")
            data["artifact_base_path"] = cleaned_base_path
            data["base_path"] = cleaned_base_path
            data["google_cloud_storage_bucket_base_path"] = cleaned_base_path
        else:
            data["artifact_base_path"] = None
            data["base_path"] = None
            data["google_cloud_storage_bucket_base_path"] = None

        # 4. Optional environment variables
        if "service_account_email" not in data:
            data["service_account_email"] = (
                os.environ.get("SERVICE_ACCOUNT_EMAIL") or None
            )
        if "docker_repository" not in data:
            data["docker_repository"] = os.environ.get("DOCKER_REPOSITORY") or None
        if "device_type" not in data:
            data["device_type"] = os.environ.get("DEVICE_TYPE") or None
        if "hf_token" not in data:
            data["hf_token"] = os.environ.get("HF_TOKEN") or None
        if "auth_token" not in data:
            data["auth_token"] = os.environ.get("AUTH_TOKEN") or None
        if "endpoint_url" not in data:
            data["endpoint_url"] = os.environ.get("ENDPOINT_URL") or None

        return data

    @property
    def model(self) -> str:
        """Alias for model_id."""
        return self.model_id

    @property
    def hf_model_repo(self) -> str:
        """Alias for model_id for backwards compatibility."""
        return self.model_id

    @property
    def model_name(self) -> str:
        """The second segment of model_id (e.g. 'gemma-4-31B-it' from 'google/gemma-4-31B-it')."""
        return extract_model_name(self.model_id)

    def validate_deployment_env(
        self,
        override_project: str | None = None,
        override_location: str | None = None,
        override_bucket: str | None = None,
    ) -> dict[str, str]:
        """Validates that compulsory deployment variables are set and valid.

        Compulsory variables:
        - GOOGLE_CLOUD_PROJECT
        - GOOGLE_CLOUD_LOCATION (must be a single region like 'us-central1', not multi-region like 'US')
        - GOOGLE_CLOUD_STORAGE_BUCKET
        """
        project_id = override_project or self.google_cloud_project
        location = override_location or self.google_cloud_location
        bucket = override_bucket or self.google_cloud_storage_bucket

        missing = []
        if not project_id:
            missing.append("GOOGLE_CLOUD_PROJECT")
        if not location:
            missing.append("GOOGLE_CLOUD_LOCATION")
        if not bucket:
            missing.append("GOOGLE_CLOUD_STORAGE_BUCKET")

        if missing or not project_id or not location or not bucket:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Please configure them in your .env file or environment."
            )

        validated_location = validate_single_region(location)

        cleaned_bucket = bucket.strip()
        if cleaned_bucket.startswith("gs://"):
            cleaned_bucket = cleaned_bucket[len("gs://") :]
        cleaned_bucket = cleaned_bucket.strip("/")

        return {
            "project_id": project_id,
            "location": validated_location,
            "bucket": cleaned_bucket,
        }


# Global settings singleton
settings = Settings()
