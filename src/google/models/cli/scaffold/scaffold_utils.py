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

"""Scaffolding utilities for models-cli project generation."""

import os
import shutil
import subprocess
from pathlib import Path
from google.cloud import aiplatform


def get_base_templates_dir() -> Path:
    """Returns absolute path to base_templates folder."""
    return Path(__file__).parent / "base_templates"


def normalize_project_name(name: str) -> str:
    """Normalizes project name to lowercase with hyphens."""
    normalized = name.strip().lower().replace("_", "-").replace(" ", "-")
    # remove consecutive hyphens
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized


def verify_credentials_and_vertex(project_id: str, location: str) -> tuple[bool, str]:
    """Verifies Google Cloud credentials and Gemini Enterprise Online Prediction initialization.

    Args:
        project_id: Google Cloud Project ID.
        location: Google Cloud Location/Region.

    Returns:
        Tuple of (success_boolean, status_message).
    """
    if not project_id:
        return False, "Project ID is required."

    try:
        from google.models.cli.common.auth import check_adc_validity

        if not check_adc_validity():
            return False, "Google Cloud Application Default Credentials (ADC) are missing or expired."

        # Verify project exists and is accessible via gcloud if available
        if shutil.which("gcloud"):
            res = subprocess.run(
                ["gcloud", "projects", "describe", project_id, "--quiet", "--format=value(projectId)"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if res.returncode != 0:
                err_msg = res.stderr.strip()
                if "not found" in err_msg.lower():
                    return False, f"GCP project '{project_id}' was not found."
                elif "permission" in err_msg.lower() or "denied" in err_msg.lower():
                    return False, f"Permission denied for GCP project '{project_id}'."
                elif err_msg:
                    return False, err_msg.splitlines()[0]

        aiplatform.init(project=project_id, location=location)
        return True, f"Successfully connected to Gemini Enterprise Online Prediction in '{project_id}' ({location})."
    except Exception as e:
        return False, str(e)


def generate_env_from_example(env_example_path: Path, context: dict[str, str]) -> str:
    """Generates .env content directly from .env.example, guaranteeing 100% schema alignment.

    Args:
        env_example_path: Path to the .env.example template file.
        context: Resolved key-value substitutions.

    Returns:
        Formatted .env content string.
    """
    key_to_context_map = {
        "GOOGLE_CLOUD_PROJECT": context.get("project_id", ""),
        "GOOGLE_CLOUD_LOCATION": context.get("region", ""),
        "GOOGLE_CLOUD_STORAGE_BUCKET": context.get("gcs_bucket", ""),
        "GOOGLE_CLOUD_STORAGE_BUCKET_BASE_PATH": context.get("base_path", ""),
        "BASE_PATH": context.get("base_path", ""),
        "ARTIFACT_BASE_PATH": context.get("base_path", ""),
        "SERVICE_ACCOUNT_EMAIL": context.get("service_account_email", ""),
        "MODEL_ID": context.get("model_id", ""),
        "HF_TOKEN": context.get("hf_token", ""),
        "ENDPOINT_URL": context.get("endpoint_url", ""),
    }

    lines = env_example_path.read_text(encoding="utf-8").splitlines()
    output_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            output_lines.append(line)
            continue

        if "=" in line:
            key, _ = line.split("=", 1)
            key = key.strip()
            val = key_to_context_map.get(key, "")
            output_lines.append(f'{key}="{val}"')
        else:
            output_lines.append(line)

    return "\n".join(output_lines) + "\n"


def copy_and_render_templates(
    destination: Path,
    template_name: str,
    context: dict[str, str],
) -> None:
    """Copies base templates and renders placeholder substitutions.

    Creates .env directly derived from .env.example, guaranteeing perfect alignment.

    Args:
        destination: Target directory.
        template_name: Engine template flavor ('vllm' or 'sglang').
        context: Key-value substitutions.
    """
    base_dir = get_base_templates_dir()
    shared_dir = base_dir / "_shared"
    engine_dir = base_dir / template_name

    destination.mkdir(parents=True, exist_ok=True)

    # 1. Copy & render _shared (copies .env.example template)
    if shared_dir.is_dir():
        _copy_and_substitute_tree(shared_dir, destination, context)

    # 2. Copy & render engine specific configs
    if engine_dir.is_dir():
        _copy_and_substitute_tree(engine_dir, destination, context)

    # 3. Derive .env directly from .env.example
    env_example_file = destination / ".env.example"
    if env_example_file.is_file():
        env_content = generate_env_from_example(env_example_file, context)
        (destination / ".env").write_text(env_content, encoding="utf-8")

    # 4. Ensure reports directory exists
    (destination / "reports").mkdir(parents=True, exist_ok=True)


def _copy_and_substitute_tree(source: Path, destination: Path, context: dict[str, str]) -> None:
    """Recursively copies files from source to destination applying context substitutions."""
    for root_str, _, files in os.walk(source):
        root = Path(root_str)
        rel_path = root.relative_to(source)
        target_dir = destination / rel_path
        target_dir.mkdir(parents=True, exist_ok=True)

        for filename in files:
            src_file = root / filename
            dest_file = target_dir / filename

            try:
                content = src_file.read_text(encoding="utf-8")
                # Format string with context
                for key, val in context.items():
                    content = content.replace(f"{{{key}}}", str(val))
                dest_file.write_text(content, encoding="utf-8")
            except UnicodeDecodeError:
                # Binary file fallback
                dest_file.write_bytes(src_file.read_bytes())
