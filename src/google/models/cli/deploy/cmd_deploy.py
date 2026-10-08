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

import click

from google.models.cli._project import (
    read_deployment_config,
    read_engine_config,
    write_env,
)
from google.models.cli.common.auth import ensure_authenticated
from google.models.cli.common.config import extract_model_name, settings
from google.models.cli.common.constants import (
    DEFAULT_MODEL_REPO,
    InferenceEngine,
)
from google.models.cli.deploy._operation import (
    read_operation,
    write_operation,
)
from google.models.cli.deploy.deploy_utils import (
    build_deployment_status_renderable,
    check_bucket_exists,
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
    help="Hugging Face model path (e.g. google/gemma-4-31B-it).",
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
    "--base-path",
    "--artifact-base-path",
    default=None,
    help="Custom path/prefix within GCS bucket where weights are stored or should be copied.",
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
    "--watch",
    "-w",
    is_flag=True,
    default=False,
    help="Continuously poll deployment status until completion.",
)
@click.option(
    "--operation",
    default=None,
    help="Specific operation resource name to check status for (e.g. projects/.../operations/...).",
)
def deploy(
    model_id: str | None = None,
    bucket: str | None = None,
    project: str | None = None,
    region: str | None = None,
    service_account: str | None = None,
    base_path: str | None = None,
    dry_run: bool = False,
    status: bool = False,
    watch: bool = False,
    operation: str | None = None,
) -> None:
    """Deploys an open model container to a Gemini Enterprise Online Prediction endpoint."""
    if not ensure_authenticated(interactive=True):
        raise click.exceptions.Exit(1)

    # 1. Load defaults from project configs via DeploymentConfig & EngineConfig
    deployment_config = read_deployment_config()
    engine_config = read_engine_config()

    if status:
        pending_operation = read_operation() or {}
        target_operation = operation or pending_operation.get("operation_name")
        if not target_operation:
            raise click.ClickException(
                "No operation found. Provide --operation <name> or ensure an operation is recorded in deployment_metadata.json."
            )
        click.echo(f"Checking deployment operation status: {target_operation}")
        resolved_location = (
            region
            or pending_operation.get("location")
            or settings.google_cloud_location
            or None
        )

        from google.models.cli.common.console import console

        if watch:
            import time

            from rich.live import Live

            click.echo("Watching deployment operation (press Ctrl+C to stop)...\n")
            op_status = get_operation_status(
                target_operation, location=resolved_location
            )
            with Live(
                build_deployment_status_renderable(
                    op_status, pending_operation=pending_operation
                ),
                console=console,
                refresh_per_second=2,
            ) as live:
                try:
                    while not op_status.get("done"):
                        time.sleep(5)
                        op_status = get_operation_status(
                            target_operation, location=resolved_location
                        )
                        live.update(
                            build_deployment_status_renderable(
                                op_status, pending_operation=pending_operation
                            )
                        )
                except KeyboardInterrupt:
                    pass

            if not op_status.get("done"):
                click.echo(
                    "\nStopped watching. Deployment is still running in the background."
                )
                click.echo(
                    "Run 'models-cli deploy --status' at any time to check again.\n"
                )
        else:
            op_status = get_operation_status(
                target_operation, location=resolved_location
            )
            console.print(
                build_deployment_status_renderable(
                    op_status, pending_operation=pending_operation
                )
            )
            if not op_status.get("done"):
                click.echo(
                    "💡 Tip: Run 'models-cli deploy --status --watch' to monitor progress live.\n"
                )
        return

    resolved_machine_type = deployment_config.machine_type
    if not resolved_machine_type:
        raise click.ClickException(
            "Machine type is required. Set 'machine_type' in config/deployment_spec.yaml, or run 'models-cli recommend' "
            "to configure one."
        )

    resolved_engine = InferenceEngine(engine_config.engine_enum)
    if not resolved_engine:
        raise click.ClickException(
            "Inference engine is required. Set 'engine' in config/engine_config.yaml, or run 'models-cli recommend' to "
            "configure one."
        )

    try:
        deployment_env = settings.validate_deployment_env(
            override_project=project,
            override_location=region,
            override_bucket=bucket,
        )
    except ValueError as err:
        raise click.ClickException(str(err)) from err

    project_id = deployment_env["project_id"]
    resolved_location = deployment_env["location"]
    gcs_bucket = deployment_env["bucket"]

    if not check_bucket_exists(gcs_bucket):
        raise click.ClickException(
            f"GCS bucket '{gcs_bucket}' does not exist or you do not have permission to access it."
        )

    model_repo = model_id or settings.model_id or DEFAULT_MODEL_REPO
    model_name = extract_model_name(model_repo)
    model_display_name = model_name.lower()
    service_account_email = (
        service_account
        or deployment_config.service_account
        or settings.service_account_email
        or None
    )

    clean_base_path = (base_path or settings.artifact_base_path or "").strip("/")

    if clean_base_path:
        # 1. Did user point directly to a folder containing .safetensors?
        if check_model_exists_in_gcs(gcs_bucket, clean_base_path):
            target_path = clean_base_path
            weights_already_in_gcs = True
        else:
            # 2. Check if weights exist inside {base_path}/{model_name}
            subfolder_path = (
                clean_base_path
                if clean_base_path.endswith(model_name)
                else f"{clean_base_path}/{model_name}"
            )
            target_path = subfolder_path
            weights_already_in_gcs = check_model_exists_in_gcs(
                gcs_bucket, subfolder_path
            )
    else:
        target_path = model_name
        weights_already_in_gcs = check_model_exists_in_gcs(gcs_bucket, model_name)

    model_uri = f"gs://{gcs_bucket}/{target_path}"

    if weights_already_in_gcs:
        click.echo(
            f"✓ Found existing weights in GCS: {model_uri} (reusing from previous run)"
        )
    elif dry_run:
        click.echo(
            f"ℹ [dry-run] Weights not found in GCS: would transfer '{model_repo}' to '{model_uri}'"
        )
    else:
        click.echo(f"📦 Transferring '{model_repo}' to '{model_uri}'...")
        copy_hf_model_to_gcs(
            repo_id=model_repo,
            bucket_name=gcs_bucket,
            location=resolved_location,
            destination_path=target_path,
            hf_token=settings.hf_token,
        )

    if service_account_email:
        configure_sa_permissions(
            project_id=project_id, service_account_email=service_account_email
        )

    endpoint_display_name = (
        deployment_config.display_name
        if deployment_config.display_name
        and "{project_name}" not in deployment_config.display_name
        else f"{model_display_name}-endpoint"
    )

    result = deploy_model_to_geap(
        project_id=project_id,
        location=resolved_location,
        model_display_name=model_display_name,
        model_id=model_repo,
        endpoint_display_name=endpoint_display_name,
        model_uri=model_uri,
        service_account_email=service_account_email,
        machine_type=resolved_machine_type,
        engine_config=engine_config,
        shared_memory_mb=deployment_config.shared_memory_mb,
        container_image_uri=deployment_config.container_image_uri or None,
        dedicated_endpoint=deployment_config.dedicated_endpoint,
        routes=deployment_config.routes,
        environment_variables=deployment_config.environment_variables,
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
        endpoint_resource_name = result.get("endpoint_resource_name") or result.get(
            "endpoint"
        )
        endpoint_url = result.get("endpoint_url")
        if result.get("operation_name"):
            write_operation(
                operation_name=result["operation_name"],
                project=project_id,
                location=resolved_location,
                endpoint=endpoint_resource_name,
                endpoint_url=endpoint_url,
            )
        elif endpoint_resource_name:
            from google.models.cli.deploy._operation import write_endpoint

            write_endpoint(endpoint_resource_name, endpoint_url=endpoint_url)

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
