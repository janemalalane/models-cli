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
import click

from google.models.cli._gcp_project import resolve_gcp_project
from google.models.cli._project import (
    read_deployment_config,
    read_engine_config,
    write_env,
)
from google.models.cli.common.auth import ensure_authenticated, is_auth_error
from google.models.cli.common.config import settings
from google.models.cli.common.constants import (
    DEFAULT_MODEL_REPO,
    DEFAULT_REGION,
    InferenceEngine,
)
from google.models.cli.deploy._operation import (
    clear_operation,
    read_operation,
    write_operation,
)
from google.models.cli.deploy.deploy_utils import (
    check_model_exists_in_gcs,
    configure_sa_permissions,
    copy_hf_model_to_gcs,
    deploy_model_to_geap,
    get_operation_status,
)


@click.command("deploy")
@click.option(
    "--model-id",
    default=None,
    help="Unique identifier for the deployed model resource.",
)
@click.option(
    "--hf-repo",
    "-m",
    default=None,
    help="Hugging Face model repository (e.g. google/gemma-4-31B-it).",
)
@click.option(
    "--machine-type",
    "-M",
    default=None,
    help="Google Cloud Compute machine type (e.g. g4-standard-48, ct6e-standard-4t).",
)
@click.option(
    "--engine",
    "-e",
    default=None,
    help="Serving engine: 'vllm' or 'sglang'.",
)
@click.option(
    "--bucket",
    "-b",
    default=None,
    help="GCS bucket name for storing model snapshot.",
)
@click.option(
    "--project",
    "-p",
    default=None,
    help="GCP Project ID. Defaults to environment or active gcloud config.",
)
@click.option(
    "--region",
    "-r",
    default=None,
    help="GCP Region for endpoint deployment.",
)
@click.option(
    "--service-account",
    "-s",
    default=None,
    help="GCP Service Account email for endpoint execution. Defaults to Compute Engine default service account if omitted.",
)
@click.option(
    "--copy-weights/--no-copy-weights",
    default=False,
    help="Download weights from HF and upload to GCS before deployment.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate parameters and output deployment manifest without executing GCP API calls.",
)
@click.option(
    "--status",
    is_flag=True,
    default=False,
    help="Check the status of a long-running deployment operation.",
)
@click.option(
    "--operation",
    default=None,
    help="Specific operation resource name to check status for (e.g. projects/.../operations/...).",
)
def deploy(
    model_id: Optional[str] = None,
    hf_repo: Optional[str] = None,
    machine_type: Optional[str] = None,
    engine: Optional[str] = None,
    bucket: Optional[str] = None,
    project: Optional[str] = None,
    region: Optional[str] = None,
    service_account: Optional[str] = None,
    copy_weights: bool = False,
    dry_run: bool = False,
    status: bool = False,
    operation: Optional[str] = None,
) -> None:
    """Deploys an open model container to a GEAP / Vertex AI online prediction endpoint."""
    if not dry_run and not ensure_authenticated(interactive=True):
        raise click.exceptions.Exit(1)

    # 1. Load defaults from project configs via DeploymentConfig & EngineConfig
    deploy_cfg = read_deployment_config()
    engine_cfg = read_engine_config()

    if status:
        pending = read_operation() or {}
        target_op = operation or pending.get("operation_name")
        if not target_op:
            raise click.ClickException(
                "No operation found. Provide --operation <name> or ensure an operation is recorded in deployment_metadata.json."
            )
        click.echo(f"Checking deployment operation status: {target_op}")
        loc = (
            region or pending.get("location") or settings.google_cloud_location or None
        )
        op_info = get_operation_status(target_op, location=loc)

        click.echo(f"   • Status:           {op_info['status']}")
        click.echo(f"   • Done:             {op_info['done']}")
        if op_info.get("deployment_stage"):
            click.echo(f"   • Deployment Stage: {op_info['deployment_stage']}")
        if op_info.get("create_time"):
            click.echo(f"   • Created:          {op_info['create_time']}")
        if op_info.get("update_time"):
            click.echo(f"   • Updated:          {op_info['update_time']}")
        if op_info.get("error"):
            click.echo(f"   • Error Code:       {op_info['error'].get('code')}")
            click.echo(f"   • Error Message:    {op_info['error'].get('message')}")
        if op_info.get("response"):
            click.echo("   • Response Details:")
            click.echo(json.dumps(op_info["response"], indent=4))
        if op_info["done"]:
            clear_operation()
        return

    resolved_machine_type = machine_type or deploy_cfg.machine_type or "g4-standard-48"

    resolved_engine = (
        InferenceEngine(engine.lower().strip())
        if engine
        else (deploy_cfg.engine_enum or engine_cfg.engine_enum)
    )

    repo = hf_repo or settings.model or DEFAULT_MODEL_REPO
    mid = model_id or repo.split("/")[-1].lower().replace("_", "-").replace(".", "-")
    resolved_project = resolve_gcp_project(
        override_project=project or settings.google_cloud_project,
        required=True,
    )
    loc = region or settings.google_cloud_location or DEFAULT_REGION
    sa_email = (
        service_account
        or deploy_cfg.service_account
        or settings.service_account_email
        or None
    )
    gcs_bucket = (
        bucket or settings.google_cloud_storage_bucket or f"{resolved_project}-models"
    )

    artifact_name = repo.split("/")[-1]
    model_uri = f"gs://{gcs_bucket}/{artifact_name}"

    if sa_email:
        configure_sa_permissions(
            project_id=resolved_project, service_account_email=sa_email
        )

    weights_already_in_gcs = check_model_exists_in_gcs(gcs_bucket, repo)

    if weights_already_in_gcs:
        click.echo(
            f"✓ Found existing weights in GCS: {model_uri} (reusing from previous run)"
        )
    elif dry_run:
        click.echo(
            f"ℹ [dry-run] Weights not found in GCS: would transfer '{repo}' to '{model_uri}'"
        )
    else:
        click.echo(f"📦 Transferring '{repo}' to '{model_uri}'...")
        copy_hf_model_to_gcs(
            model_id=repo,
            bucket_name=gcs_bucket,
            location=loc,
            hf_token=settings.hf_token,
        )

    result = deploy_model_to_geap(
        project_id=resolved_project,
        location=loc,
        model_id=mid,
        model_uri=model_uri,
        service_account_email=sa_email,
        machine_type=resolved_machine_type,
        engine=resolved_engine,
        engine_params=engine_cfg.to_engine_params(),
        accelerator_type=deploy_cfg.accelerator_type,
        accelerator_count=deploy_cfg.accelerator_count,
        dry_run=dry_run,
    )

    if dry_run:
        click.echo("📋 Dry Run Deployment Manifest:")
        click.echo(json.dumps(result["manifest"], indent=2))
        click.echo("\n✅ Dry run validation succeeded.")

    else:
        write_env(
            {
                "ENDPOINT_URL": result.get("endpoint_url", ""),
            }
        )
        endpoint_res = result.get("endpoint_resource_name") or result.get("endpoint")
        endpoint_url = result.get("endpoint_url")
        if result.get("operation_name"):
            write_operation(
                operation_name=result["operation_name"],
                project=resolved_project,
                location=loc,
                endpoint=endpoint_res,
                endpoint_url=endpoint_url,
            )
        elif endpoint_res:
            from google.models.cli.deploy._operation import write_endpoint

            write_endpoint(endpoint_res, endpoint_url=endpoint_url)

        click.echo("✅ Success! Model deployment initiated:")
        click.echo(f"   • Endpoint Resource: {result.get('endpoint_resource_name')}")
        click.echo(f"   • Endpoint URL:      {result.get('endpoint_url')}")
        if result.get("operation_name"):
            click.echo(f"   • Backing Operation: {result.get('operation_name')}")
            click.echo("\n💡 Check deployment progress at any time by running:")
            click.echo("   models-cli deploy --status\n")
        else:
            click.echo()


deploy_cmd = deploy
