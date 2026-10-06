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

"""Read/write helpers for pending deploy operations in METADATA_FILE.

When a deployment starts, the long-running operation name and metadata
are persisted as a ``pending_operation`` field inside ``METADATA_FILE``
so that ``deploy --status`` can poll it later.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any

from google.models.cli._project import find_project_root

METADATA_FILE = "deployment_metadata.json"


def _get_metadata_path(project_dir: str | Path | None = None) -> Path:
    """Returns the path to METADATA_FILE at the project root or current working directory."""
    root = Path(project_dir) if project_dir else (find_project_root() or Path.cwd())
    return root / METADATA_FILE


def _read_metadata(project_dir: str | Path | None = None) -> dict[str, Any]:
    """Read METADATA_FILE, tolerating a missing or corrupt file.

    A malformed or zero-byte file (left by an interrupted run, a partial
    write, or a manual edit) is treated as empty so a single bad file can't
    permanently block every subsequent deploy. Returns ``{}`` when the file
    is missing or unreadable.
    """
    metadata_path = _get_metadata_path(project_dir)
    if not metadata_path.is_file():
        return {}
    try:
        with open(metadata_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logging.warning(
            "Ignoring corrupt %s (%s); treating as empty.", metadata_path, e
        )
        return {}
    if not isinstance(data, dict):
        logging.warning(
            "Ignoring %s with unexpected top-level %s; treating as empty.",
            metadata_path,
            type(data).__name__,
        )
        return {}
    return data


def write_operation(
    operation_name: str,
    project: str,
    location: str,
    endpoint: str | None = None,
    deployment_target: str = "geap",
    endpoint_url: str | None = None,
    project_dir: str | Path | None = None,
) -> None:
    """Persist a pending deploy operation to METADATA_FILE."""
    pending: dict[str, Any] = {
        "operation_name": operation_name,
        "project": project,
        "location": location,
        "deployment_target": deployment_target,
        "endpoint": endpoint,
        "endpoint_url": endpoint_url,
        "started_at": datetime.datetime.now(tz=datetime.UTC).isoformat(),
    }

    metadata_path = _get_metadata_path(project_dir)
    data = _read_metadata(project_dir)
    data["pending_operation"] = pending
    if endpoint:
        data["endpoint"] = endpoint
    if endpoint_url:
        data["endpoint_url"] = endpoint_url
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def read_operation(project_dir: str | Path | None = None) -> dict[str, Any] | None:
    """Read a pending operation from METADATA_FILE, or None."""
    return _read_metadata(project_dir).get("pending_operation")


def clear_operation(project_dir: str | Path | None = None) -> None:
    """Remove the pending_operation field from METADATA_FILE while preserving endpoint and endpoint_url."""
    metadata_path = _get_metadata_path(project_dir)
    data = _read_metadata(project_dir)
    if "pending_operation" in data:
        pending = data["pending_operation"]
        if isinstance(pending, dict):
            if pending.get("endpoint") and not data.get("endpoint"):
                data["endpoint"] = pending["endpoint"]
            if pending.get("endpoint_url") and not data.get("endpoint_url"):
                data["endpoint_url"] = pending["endpoint_url"]
        del data["pending_operation"]
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def read_endpoint(project_dir: str | Path | None = None) -> str | None:
    """Read the deployed endpoint resource name or URL from METADATA_FILE."""
    data = _read_metadata(project_dir)
    return (
        data.get("endpoint_url")
        or data.get("endpoint")
        or (data.get("pending_operation") or {}).get("endpoint_url")
        or (data.get("pending_operation") or {}).get("endpoint")
    )


def read_endpoint_url(project_dir: str | Path | None = None) -> str | None:
    """Read the deployed endpoint URL from METADATA_FILE."""
    data = _read_metadata(project_dir)
    return data.get("endpoint_url") or (data.get("pending_operation") or {}).get(
        "endpoint_url"
    )


def write_endpoint(
    endpoint: str,
    endpoint_url: str | None = None,
    project_dir: str | Path | None = None,
) -> None:
    """Record or update the deployed endpoint resource name and URL in METADATA_FILE."""
    metadata_path = _get_metadata_path(project_dir)
    data = _read_metadata(project_dir)
    data["endpoint"] = endpoint
    if endpoint_url:
        data["endpoint_url"] = endpoint_url
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
