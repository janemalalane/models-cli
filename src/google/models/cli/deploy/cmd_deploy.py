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

"""Deployment command for models-cli."""

import json
from pathlib import Path
from typing import Optional
import typer
import yaml
from rich.console import Console
from rich.panel import Panel

from google.models.cli._gcp_project import resolve_gcp_project
from google.models.cli.common.auth import ensure_authenticated, is_auth_error
from google.models.cli.common.config import settings
from google.models.cli.common.constants import (
    DEFAULT_MODEL_REPO,
    DEFAULT_REGION,
    DEFAULT_SGLANG_CONTAINER_IMAGE,
    DEFAULT_VLLM_CONTAINER_IMAGE,
    InferenceEngine,
)
from google.models.cli.deploy.deploy_utils import (
    configure_sa_permissions,
    copy_hf_model_to_gcs,
    deploy_model_to_geap,
)

console = Console()
deploy_cmd = typer.Typer(
    name="deploy",
    help="Deploy open model container to GEAP / Vertex AI online prediction endpoint",
)


@deploy_cmd.callback(invoke_without_command=True)
def deploy(
    model_id: Optional[str] = typer.Option(
        None,
        "--model-id",
        help="Unique identifier for the deployed model resource.",
    ),
    hf_repo: Optional[str] = typer.Option(
        None,
        "--hf-repo",
        "-m",
        help="Hugging Face model repository (e.g. google/gemma-4-31B-it).",
    ),
    machine_type: Optional[str] = typer.Option(
        None,
        "--machine-type",
        "-M",
        help="Google Cloud Compute machine type (e.g. g4-standard-48, ct6e-standard-4t).",
    ),
    engine: Optional[str] = typer.Option(
        None,
        "--engine",
        "-e",
        help="Serving engine: 'vllm' or 'sglang'.",
    ),
    container_image: Optional[str] = typer.Option(
        None,
        "--container-image",
        "-i",
        help="Custom container image URI. Defaults to Google pre-built container.",
    ),
    bucket: Optional[str] = typer.Option(
        None,
        "--bucket",
        "-b",
        help="GCS bucket name for storing model snapshot.",
    ),
    project: Optional[str] = typer.Option(
        None,
        "--project",
        "-p",
        help="GCP Project ID. Defaults to environment or active gcloud config.",
    ),
    region: Optional[str] = typer.Option(
        None,
        "--region",
        "-r",
        help="GCP Region for endpoint deployment.",
    ),
    copy_weights: bool = typer.Option(
        False,
        "--copy-weights/--no-copy-weights",
        help="Download weights from HF and upload to GCS before deployment.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate parameters and output deployment manifest without executing GCP API calls.",
    ),
) -> None:
    """Deploys an open model container to a GEAP / Vertex AI online prediction endpoint."""
    if not dry_run and not ensure_authenticated(interactive=True):
        raise typer.Exit(1)

    # 1. Load defaults from config/deployment_spec.yaml if present
    spec_data = {}
    spec_path = Path("config/deployment_spec.yaml")
    if spec_path.is_file():
        try:
            with open(spec_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                spec_data = loaded.get("deployment", {})
        except Exception:
            pass

    resolved_machine_type = (
        machine_type
        or spec_data.get("machine_type")
        or "g4-standard-48"
    )
    resolved_engine_str = (
        engine
        or spec_data.get("engine")
        or InferenceEngine.VLLM.value
    )
    engine_enum = InferenceEngine.VLLM if resolved_engine_str.lower() == "vllm" else InferenceEngine.SGLANG

    resolved_image = (
        container_image
        or spec_data.get("container_image_uri")
        or (
            DEFAULT_VLLM_CONTAINER_IMAGE
            if engine_enum == InferenceEngine.VLLM
            else DEFAULT_SGLANG_CONTAINER_IMAGE
        )
    )

    repo = hf_repo or settings.model or DEFAULT_MODEL_REPO
    mid = model_id or repo.split("/")[-1].lower().replace("_", "-").replace(".", "-")
    proj = resolve_gcp_project(override_project=project or settings.google_cloud_project, required=not dry_run) or "YOUR_PROJECT_ID"
    loc = region or settings.google_cloud_location or DEFAULT_REGION
    sa_email = settings.service_account_email or f"models-sa@{proj}.iam.gserviceaccount.com"
    gcs_bucket = bucket or settings.google_cloud_storage_bucket or f"{proj}-models"

    # Model weight source URI
    if copy_weights and not dry_run:
        console.print(f"📦 Transferring '{repo}' to 'gs://{gcs_bucket}/{mid}'...")
        model_uri = copy_hf_model_to_gcs(
            model_id=repo,
            bucket_name=gcs_bucket,
            location=loc,
            hf_token=settings.hf_token,
        )
        configure_sa_permissions(project_id=proj, service_account_email=sa_email)
    elif copy_weights and dry_run:
        model_uri = f"gs://{gcs_bucket}/{mid}"
    else:
        # Direct HF or Model Garden source
        model_uri = repo

    console.print(f"\n[bold cyan]🚀 Deploying Open Model to GEAP / Vertex AI[/bold cyan]")
    console.print(f"   • Project:         [bold]{proj}[/bold]")
    console.print(f"   • Region:          [bold]{loc}[/bold]")
    console.print(f"   • Model ID:        [bold]{mid}[/bold]")
    console.print(f"   • Model Source:    [bold]{repo}[/bold]")
    console.print(f"   • Machine Type:    [bold]{resolved_machine_type}[/bold]")
    console.print(f"   • Serving Engine:  [bold]{engine_enum.value}[/bold]")
    console.print(f"   • Container Image: [bold]{resolved_image}[/bold]")
    console.print(f"   • Service Account: [bold]{sa_email}[/bold]")
    console.print(f"   • Dry Run:         [bold]{dry_run}[/bold]\n")

    result = deploy_model_to_geap(
        project_id=proj,
        location=loc,
        model_id=mid,
        model_uri=model_uri,
        container_image_uri=resolved_image,
        service_account_email=sa_email,
        machine_type=resolved_machine_type,
        engine=engine_enum,
        accelerator_type=spec_data.get("accelerator_type"),
        accelerator_count=spec_data.get("accelerator_count"),
        dry_run=dry_run,
    )

    if dry_run:
        console.print("[bold yellow]📋 Dry Run Deployment Manifest:[/bold yellow]")
        console.print(json.dumps(result["manifest"], indent=2))
        console.print("\n[bold green]✅ Dry run validation succeeded.[/bold green]")
    else:
        console.print(f"[bold green]✅ Success![/bold green] Model successfully deployed to endpoint:")
        console.print(f"   • Endpoint Resource: [cyan]{result.get('endpoint_resource_name')}[/cyan]")
        console.print(f"   • Endpoint URL:      [cyan]{result.get('endpoint_url')}[/cyan]\n")
