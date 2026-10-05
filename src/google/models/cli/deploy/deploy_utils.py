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

"""Deployment utilities for Gemini Enterprise Online Prediction model registration and endpoint provisioning."""

import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union
import click
from google.cloud import aiplatform, aiplatform_v1, resourcemanager_v3, storage
from google.iam.v1 import policy_pb2
from google.longrunning import operations_pb2
from google.protobuf.json_format import MessageToDict

from google.models.cli._gcp_project import get_gcp_project_number
from google.models.cli.common.constants import (
    DEFAULT_CONTAINER_DEPLOYMENT_TIMEOUT,
    DEFAULT_CONTAINER_HOST,
    DEFAULT_CONTAINER_PORT,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_HEALTH_ROUTE,
    DEFAULT_KV_CACHE_DTYPE,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_PREDICTION_ROUTE,
    DEFAULT_SHARED_MEMORY_MB,
    DEFAULT_TENSOR_PARALLEL_SIZE,
    DEFAULT_UPLOAD_REQUEST_TIMEOUT,
    HARDWARE_SPECS,
    AcceleratorFamily,
    InferenceEngine,
)

from google.models.cli._project import (
    EngineConfig,
    VLLMEngineConfig,
    SGLangEngineConfig,
)


def dict_to_cli_args(
    params: dict[str, Any],
    excluded_keys: Optional[set[str]] = None,
) -> list[str]:
    """Converts a dictionary of configuration options into CLI flag strings."""
    args: list[str] = []
    excluded = excluded_keys or set()
    extra_args = params.get("extra_args") or params.get("args") or []

    for key, value in params.items():
        if key in excluded or key in ("extra_args", "args", "raw_config"):
            continue
        if value is None:
            continue

        flag = key if key.startswith("--") else f"--{key.replace('_', '-')}"

        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif isinstance(value, (list, tuple)):
            for item in value:
                args.append(f"{flag}={item}")
        else:
            args.append(f"{flag}={value}")

    if isinstance(extra_args, (list, tuple)):
        for item in extra_args:
            args.append(str(item))

    return args


def build_container_args(
    target: Union[EngineConfig, str],
    engine: Optional[InferenceEngine] = None,
    engine_params: Optional[dict[str, Any]] = None,
    model_uri: Optional[str] = None,
) -> list[str]:
    """Constructs command line arguments for the container entrypoint."""
    if isinstance(target, (VLLMEngineConfig, SGLangEngineConfig)):
        engine_config = target
    else:
        resolved_engine = engine or InferenceEngine.VLLM
        params = engine_params or {}
        if resolved_engine == InferenceEngine.SGLANG:
            engine_config = SGLangEngineConfig.from_dict(params)
        else:
            engine_config = VLLMEngineConfig.from_dict(params)

    params = engine_config.to_engine_params()
    host = params.pop("host", getattr(engine_config, "host", DEFAULT_CONTAINER_HOST))
    port = params.pop("port", getattr(engine_config, "port", DEFAULT_CONTAINER_PORT))

    if "trust_remote_code" not in params:
        params["trust_remote_code"] = True

    if isinstance(engine_config, VLLMEngineConfig):
        args = [
            "python3",
            "-m",
            "vllm.entrypoints.openai.api_server",
            f"--host={host}",
            f"--port={port}",
        ]
        args.extend(dict_to_cli_args(params, excluded_keys={"engine"}))
    elif isinstance(engine_config, SGLangEngineConfig):  # SGLang
        args = [
            f"--host={host}",
            f"--port={port}",
        ]
        args.extend(dict_to_cli_args(params, excluded_keys={"engine"}))
    else:
        args = [f"--host={host}", f"--port={port}"]
        args.extend(dict_to_cli_args(params, excluded_keys={"engine"}))

    return args


def check_bucket_exists(bucket_name: str) -> bool:
    """Checks whether the specified GCS bucket exists and is accessible."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        return bucket.exists()
    except Exception:
        return False


def check_model_exists_in_gcs(bucket_name: str, model_path: str) -> bool:
    """Checks if .safetensors model weights exist in the specified GCS bucket under model_path."""
    clean_prefix = model_path.strip("/") + "/"
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        if not bucket.exists():
            return False
        for blob in bucket.list_blobs(prefix=clean_prefix):
            if blob.name.endswith(".safetensors"):
                return True
        return False
    except Exception:
        return False


def configure_sa_permissions(project_id: str, service_account_email: str) -> bool:
    """Grants roles/storage.objectViewer to the deployment service account."""
    try:
        client = resourcemanager_v3.ProjectsClient()
        project_path = f"projects/{project_id}"
        policy = client.get_iam_policy(resource=project_path)

        role = "roles/storage.objectViewer"
        member = f"serviceAccount:{service_account_email}"

        for binding in policy.bindings:
            if binding.role == role and member in binding.members:
                click.echo(
                    f"✓ Service account '{service_account_email}' already has storage access."
                )
                return True

        new_binding = policy_pb2.Binding(role=role, members=[member])
        policy.bindings.append(new_binding)
        client.set_iam_policy(request={"resource": project_path, "policy": policy})
        click.echo(f"✓ Granted '{role}' to service account '{service_account_email}'.")
        return True
    except Exception:
        return False


def copy_hf_model_to_gcs(
    repo_id: str,
    bucket_name: str,
    location: str,
    destination_path: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> str:
    """Downloads model weights from Hugging Face and uploads to GCS."""
    from huggingface_hub import snapshot_download
    from google.cloud.storage import transfer_manager

    artifact_path = (destination_path or repo_id.split("/")[-1]).strip("/")
    gcs_uri = f"gs://{bucket_name}/{artifact_path}"
    local_dir = Path(f"tmp/{repo_id.replace('/', '--')}")

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        if not bucket.exists():
            raise click.ClickException(
                f"GCS bucket '{bucket_name}' does not exist or you do not have permission to access it."
            )

        click.echo(f"Downloading model snapshot '{repo_id}' from Hugging Face...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            ignore_patterns=["*.bin", "*.pth", "*.gguf", ".gitattributes", ".cache*"],
            token=hf_token,
        )

        # Ensure a clean line after Hugging Face's carriage-return progress bars
        click.echo()

        file_paths = [
            f
            for f in local_dir.rglob("*")
            if f.is_file()
            and not any(
                part == ".cache" or part.startswith(".cache") or part.endswith(".cache")
                for part in f.relative_to(local_dir).parts
            )
        ]
        relative_filenames = sorted(str(f.relative_to(local_dir)) for f in file_paths)

        click.echo(f"Uploading {len(file_paths)} files to {gcs_uri}...")

        # Fast parallel upload using GCS transfer_manager
        results = transfer_manager.upload_many_from_filenames(
            bucket,
            relative_filenames,
            source_directory=str(local_dir),
            blob_name_prefix=f"{artifact_path}/",
            max_workers=8,
        )

        for name, result in zip(relative_filenames, results):
            if isinstance(result, Exception):
                raise click.ClickException(f"Failed to upload {name}: {result}")

        click.echo(f"\n✓ Model files successfully uploaded to {gcs_uri}")
        return gcs_uri
    finally:
        if local_dir.exists():
            shutil.rmtree(local_dir, ignore_errors=True)


def deploy_model_to_geap(
    project_id: str,
    location: str,
    model_display_name: str,
    model_uri: str,
    machine_type: str,
    engine_config: Optional[EngineConfig] = None,
    engine: Optional[InferenceEngine] = None,
    engine_params: Optional[dict[str, Any]] = None,
    service_account_email: Optional[str] = None,
    shared_memory_mb: Optional[int] = None,
    container_image_uri: Optional[str] = None,
    endpoint_display_name: Optional[str] = None,
    dedicated_endpoint: bool = True,
    routes: Optional[dict[str, Any]] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Deploys model to Gemini Enterprise Online Prediction Endpoint.

    Returns deployment metadata dict.
    """
    if engine_config is not None:
        cfg = engine_config
    else:
        resolved = engine or InferenceEngine.VLLM
        params = engine_params or {}
        if resolved == InferenceEngine.SGLANG:
            cfg = SGLangEngineConfig.from_dict(params)
        else:
            cfg = VLLMEngineConfig.from_dict(params)

    resolved_engine = cfg.engine_enum
    resolved_container_image_uri = container_image_uri or resolved_engine.image_uri
    container_args = build_container_args(cfg, model_uri=model_uri)
    env_vars = resolved_engine.env_vars
    resolved_endpoint_display_name = (
        endpoint_display_name or f"{model_display_name}-endpoint"
    )
    resolved_shared_memory_mb = shared_memory_mb or DEFAULT_SHARED_MEMORY_MB

    hardware_spec = HARDWARE_SPECS.get(machine_type, {})

    if not hardware_spec:
        raise click.ClickException(
            f"Unsupported machine type on Gemini Enterprise Online Prediction: {machine_type}"
        )

    raw_accelerator_type = hardware_spec.get("accelerator_type")
    accelerator_type = (
        raw_accelerator_type.strip().upper().replace("-", "_")
        if raw_accelerator_type
        else None
    )
    accelerator_count = hardware_spec.get("accelerator_count")
    if service_account_email:
        resolved_service_account = service_account_email
    else:
        project_number = get_gcp_project_number(project_id)
        resolved_service_account = (
            f"{project_number}-compute@developer.gserviceaccount.com"
            if project_number
            else "(default Compute Engine service account)"
        )

    user_routes = routes or {}
    resolved_routes = {
        "predict": user_routes.get("predict") or DEFAULT_PREDICTION_ROUTE,
        "health": user_routes.get("health") or DEFAULT_HEALTH_ROUTE,
        "port": int(
            user_routes.get("port") or getattr(cfg, "port", DEFAULT_CONTAINER_PORT)
        ),
    }

    manifest: dict[str, Any] = {
        "project": project_id,
        "location": location,
        "model_display_name": model_display_name,
        "endpoint_display_name": resolved_endpoint_display_name,
        "dedicated_endpoint_enabled": dedicated_endpoint,
        "model_uri": model_uri,
        "engine": resolved_engine.value,
        "container_image_uri": resolved_container_image_uri,
        "container_args": container_args,
        "container_env_vars": env_vars,
        "machine_type": machine_type,
    }
    if accelerator_type:
        manifest["accelerator_type"] = accelerator_type
        manifest["accelerator_count"] = accelerator_count
    if service_account_email:
        manifest["service_account"] = service_account_email
    default_routes = {
        "predict": DEFAULT_PREDICTION_ROUTE,
        "health": DEFAULT_HEALTH_ROUTE,
    }
    if routes and routes != default_routes:
        manifest["routes"] = resolved_routes
    if shared_memory_mb and shared_memory_mb != DEFAULT_SHARED_MEMORY_MB:
        manifest["shared_memory_mb"] = shared_memory_mb

    if dry_run:
        return {
            "status": "DRY_RUN",
            "endpoint_resource_name": f"projects/{project_id}/locations/{location}/endpoints/simulated-endpoint-123",
            "endpoint_url": f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/simulated-endpoint-123/invoke/v1",
            "manifest": manifest,
        }

    aiplatform.init(project=project_id, location=location)

    endpoint = None
    model = None

    try:
        from google.api_core.future import polling

        # Override GAPIC default 900s polling timeout for long-running operations so
        # large model uploads/imports do not time out after 15 minutes during Model.upload.
        if hasattr(polling.DEFAULT_POLLING, "_timeout"):
            polling.DEFAULT_POLLING._timeout = max(
                getattr(polling.DEFAULT_POLLING, "_timeout", 900),
                DEFAULT_UPLOAD_REQUEST_TIMEOUT,
            )

        model = aiplatform.Model.upload(
            artifact_uri=model_uri,
            display_name=model_display_name,
            serving_container_image_uri=resolved_container_image_uri,
            serving_container_args=container_args,
            serving_container_environment_variables=env_vars,
            upload_request_timeout=DEFAULT_UPLOAD_REQUEST_TIMEOUT,
            serving_container_deployment_timeout=DEFAULT_CONTAINER_DEPLOYMENT_TIMEOUT,
            serving_container_invoke_route_prefix=resolved_routes["predict"],
            serving_container_health_route=resolved_routes["health"],
            serving_container_ports=[resolved_routes["port"]],
            serving_container_shared_memory_size_mb=resolved_shared_memory_mb,
        )

        if model is None:
            raise click.ClickException(
                "Model upload failed. Please check the logs for details."
            )

        endpoint = aiplatform.Endpoint.create(
            display_name=resolved_endpoint_display_name,
            dedicated_endpoint_enabled=dedicated_endpoint,
        )

        endpoint_service_client = aiplatform_v1.EndpointServiceClient(
            client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
        )

        dedicated_resources = {
            "machine_spec": {
                "machine_type": machine_type,
            },
            "min_replica_count": 1,
            "max_replica_count": 1,
        }
        if (
            accelerator_type
            and accelerator_count
            and hardware_spec.get("family") == AcceleratorFamily.GPU
        ):
            dedicated_resources["machine_spec"]["accelerator_type"] = accelerator_type
            dedicated_resources["machine_spec"]["accelerator_count"] = accelerator_count

        deployed_model = aiplatform_v1.DeployedModel(
            model=model.resource_name,
            display_name=model_display_name,
            dedicated_resources=dedicated_resources,
        )

        if (
            resolved_service_account
            and resolved_service_account != "(default Compute Engine service account)"
        ):
            deployed_model.service_account = resolved_service_account

        click.echo(
            f"Deploying model to endpoint using machine type '{machine_type}'..."
        )
        operation_future = endpoint_service_client.deploy_model(
            endpoint=endpoint.resource_name,
            deployed_model=deployed_model,
            traffic_split={"0": 100},
        )

        operation_name = operation_future.operation.name
        if not operation_name:
            raise click.ClickException(
                "Failed to capture the backing LRO operation name for the deployment."
            )
        click.echo(f"✓ Deployment operation started: {operation_name}")

    except Exception as error:
        click.echo(
            f"\nAn error occurred during the script execution: {error}", err=True
        )
        click.echo("Attempting to clean up created resources...", err=True)
        if endpoint and getattr(endpoint, "resource_name", None):
            try:
                click.echo(f"Deleting endpoint {endpoint.resource_name}...")
                endpoint.delete(force=True, sync=False)
                click.echo("Endpoint deletion initiated.")
            except Exception as cleanup_error:
                click.echo(f"Error deleting endpoint: {cleanup_error}", err=True)
        if model and getattr(model, "resource_name", None):
            try:
                click.echo(f"Deleting model {model.resource_name}...")
                model.delete(sync=False)
                click.echo("Model deletion initiated.")
            except Exception as cleanup_error:
                click.echo(f"Error deleting model: {cleanup_error}", err=True)
        sys.exit(1)

    dedicated_dns = getattr(endpoint, "dedicated_endpoint_dns", None)
    if not dedicated_dns and getattr(endpoint, "gca_resource", None):
        dedicated_dns = getattr(endpoint.gca_resource, "dedicated_endpoint_dns", None)

    if not dedicated_dns:
        is_dedicated = getattr(
            endpoint, "dedicated_endpoint_enabled", False
        ) or getattr(
            getattr(endpoint, "gca_resource", None), "dedicated_endpoint_enabled", False
        )
        if is_dedicated:
            endpoint_match = re.search(
                r"projects/([^/]+)/locations/([^/]+)/endpoints/([^/]+)",
                getattr(endpoint, "resource_name", "") or "",
            )
            if endpoint_match:
                dedicated_dns = (
                    f"{endpoint_match.group(3)}.{endpoint_match.group(2)}-"
                    f"{endpoint_match.group(1)}.prediction.vertexai.goog"
                )

    if dedicated_dns:
        endpoint_url = f"https://{dedicated_dns}/v1/{endpoint.resource_name}/invoke/v1"
    else:
        endpoint_url = f"https://{location}-aiplatform.googleapis.com/v1/{endpoint.resource_name}/invoke/v1"

    return {
        "status": "DEPLOYED",
        "model_resource_name": model.resource_name,
        "endpoint_resource_name": endpoint.resource_name,
        "operation_name": operation_name,
        "endpoint_url": endpoint_url,
        "manifest": manifest,
    }


DEPLOYMENT_STAGES: dict[str, dict[str, Any]] = {
    "STARTING_DEPLOYMENT": {
        "step": 1,
        "total_steps": 7,
        "percent": 15,
        "title": "Environment Setup",
        "description": "Initializing deployment environment and verifying permissions.",
    },
    "PREPARING_MODEL": {
        "step": 2,
        "total_steps": 7,
        "percent": 30,
        "title": "Model Assets",
        "description": "Verifying GCS weights and container configurations.",
    },
    "CREATING_SERVING_CLUSTER": {
        "step": 3,
        "total_steps": 7,
        "percent": 45,
        "title": "Cluster Provisioning",
        "description": "Creating underlying serving cluster and node pool.",
    },
    "ADDING_NODES_TO_CLUSTER": {
        "step": 4,
        "total_steps": 7,
        "percent": 60,
        "title": "Hardware Allocation",
        "description": "Provisioning VM instances and attaching accelerators (GPU/TPU).",
    },
    "GETTING_CONTAINER_IMAGE": {
        "step": 5,
        "total_steps": 7,
        "percent": 75,
        "title": "Container Image",
        "description": "Pulling inference engine container image (vLLM / SGLang).",
    },
    "STARTING_MODEL_SERVER": {
        "step": 6,
        "total_steps": 7,
        "percent": 85,
        "title": "Model Server",
        "description": "Loading weights into accelerator memory and initializing server.",
    },
    "FINISHING_UP": {
        "step": 7,
        "total_steps": 7,
        "percent": 95,
        "title": "Health Checks & Routing",
        "description": "Performing readiness health checks and finalizing endpoint DNS.",
    },
    "SUCCESSFULLY_DEPLOYED": {
        "step": 7,
        "total_steps": 7,
        "percent": 100,
        "title": "Deployed",
        "description": "Endpoint is live and ready to serve prediction requests.",
    },
    "FAILED_TO_DEPLOY": {
        "step": None,
        "total_steps": 7,
        "percent": None,
        "title": "Deployment Failed",
        "description": "The deployment operation encountered an error and failed.",
    },
    "DEPLOYMENT_TERMINATED": {
        "step": None,
        "total_steps": 7,
        "percent": None,
        "title": "Deployment Terminated",
        "description": "The deployment operation was cancelled or terminated.",
    },
}

DEPLOYMENT_PIPELINE_ORDER: list[tuple[str, str]] = [
    ("STARTING_DEPLOYMENT", "Environment Setup"),
    ("PREPARING_MODEL", "Model Assets"),
    ("CREATING_SERVING_CLUSTER", "Cluster Provisioning"),
    ("ADDING_NODES_TO_CLUSTER", "Hardware Allocation"),
    ("GETTING_CONTAINER_IMAGE", "Container Image"),
    ("STARTING_MODEL_SERVER", "Model Server"),
    ("FINISHING_UP", "Health Checks & Routing"),
]


def get_stage_info(stage_name: Optional[str]) -> Optional[dict[str, Any]]:
    """Returns metadata details for a given Gemini Enterprise DeploymentStage name."""
    if not stage_name:
        return None
    return DEPLOYMENT_STAGES.get(stage_name)


def format_elapsed_time(
    create_time_str: Optional[str], end_time_str: Optional[str] = None
) -> Optional[str]:
    """Calculates human-readable elapsed duration from an ISO timestamp string."""
    if not create_time_str:
        return None
    try:
        clean_create = create_time_str.rstrip("Z").split(".")[0]
        created = datetime.fromisoformat(clean_create).replace(tzinfo=timezone.utc)
        if end_time_str:
            clean_end = end_time_str.rstrip("Z").split(".")[0]
            ended = datetime.fromisoformat(clean_end).replace(tzinfo=timezone.utc)
        else:
            ended = datetime.now(timezone.utc)
        total_seconds = max(0, int((ended - created).total_seconds()))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds:02d}s"
        elif minutes > 0:
            return f"{minutes}m {seconds:02d}s"
        else:
            return f"{seconds}s"
    except Exception:
        return None


def render_progress_bar(percent: int, width: int = 30) -> str:
    """Renders a simple text-based progress bar."""
    clamped = max(0, min(100, percent))
    filled = int(width * (clamped / 100.0))
    empty = width - filled
    return f"[{'█' * filled}{'░' * empty}] {clamped}%"


def build_deployment_progress_panel(
    operation_status: dict[str, Any],
    pending_operation: Optional[dict[str, Any]] = None,
) -> Any:
    """Constructs a Rich Panel displaying the deployment progress and pipeline."""
    from rich.panel import Panel

    stage_key = operation_status.get("deployment_stage")
    done = operation_status.get("done", False)
    error = operation_status.get("error")
    create_time = operation_status.get("create_time")
    update_time = operation_status.get("update_time")
    elapsed = format_elapsed_time(create_time, update_time if done else None)

    if done and error:
        stage_info = DEPLOYMENT_STAGES.get("FAILED_TO_DEPLOY")
        failed_stage_meta = DEPLOYMENT_STAGES.get(stage_key) if stage_key else None
        current_step = failed_stage_meta.get("step") if failed_stage_meta else None
    elif done:
        stage_info = DEPLOYMENT_STAGES.get("SUCCESSFULLY_DEPLOYED")
        current_step = stage_info.get("step") if stage_info else None
    else:
        stage_info = DEPLOYMENT_STAGES.get(stage_key) if stage_key else None
        current_step = stage_info.get("step") if stage_info else None
    percent = (
        stage_info.get("percent") if stage_info else (100 if done and not error else 10)
    )
    title_text = (
        stage_info.get("title")
        if stage_info
        else ("In Progress" if not done else "Completed")
    )
    desc_text = (
        stage_info.get("description")
        if stage_info
        else ("Deployment is actively executing on Google Cloud." if not done else "")
    )

    lines: list[str] = []

    if done and error:
        border_style = "red"
        panel_title = "[bold red]❌ Deployment Failed[/bold red]"
        lines.append(
            f"[bold red]Deployment failed:[/bold red] {error.get('message', 'Unknown error')}"
        )
        if stage_key and stage_key in DEPLOYMENT_STAGES:
            failed_stage_title = DEPLOYMENT_STAGES[stage_key].get("title", stage_key)
            lines.append(f"[bold]Failed during phase:[/bold] {failed_stage_title}")
        if elapsed:
            lines.append(f"[bold]Total Runtime:[/bold]      {elapsed}")
        lines.append("")
    elif done:
        border_style = "green"
        panel_title = "[bold green]✅ Deployment Succeeded: 100%[/bold green]"
        lines.append(
            f"[bold green]{render_progress_bar(100)}[/bold green] • [bold green]Ready for Inference[/bold green]\n"
        )
        lines.append(f"[bold]Status:[/bold]       {title_text}")
        lines.append(f"[dim]{desc_text}[/dim]")
        if elapsed:
            lines.append(f"[bold]Total Time:[/bold]   {elapsed}")
        lines.append("")
    else:
        border_style = "blue"
        pct_display = percent if percent is not None else 10
        step_str = f"Step {current_step} of 7" if current_step else "In Progress"
        panel_title = (
            f"[bold blue]Deployment Progress: {step_str} ({pct_display}%)[/bold blue]"
        )
        lines.append(
            f"[bold cyan]{render_progress_bar(pct_display)}[/bold cyan] • [bold]{step_str}[/bold]\n"
        )
        lines.append(f"[bold]Current Phase:[/bold] {title_text}")
        if desc_text:
            lines.append(f"[dim]{desc_text}[/dim]")
        if elapsed:
            lines.append(f"[bold]Elapsed Time:[/bold]  {elapsed}")
        lines.append("")

    # Pipeline checklist
    lines.append("[bold]Deployment Pipeline:[/bold]")
    for idx, (s_name, s_label) in enumerate(DEPLOYMENT_PIPELINE_ORDER, start=1):
        if done and error:
            if current_step and idx < current_step:
                lines.append(f"  [green]✓[/green]  {idx}. {s_label}")
            elif current_step and idx == current_step:
                lines.append(
                    f"  [red bold]✖[/red bold]  {idx}. {s_label} [red](failed)[/red]"
                )
            else:
                lines.append(f"  [dim]○[/dim]  [dim]{idx}. {s_label}[/dim]")
        elif done:
            lines.append(f"  [green]✓[/green]  {idx}. {s_label}")
        else:
            if current_step is not None:
                if idx < current_step:
                    lines.append(f"  [green]✓[/green]  {idx}. {s_label}")
                elif idx == current_step:
                    lines.append(
                        f"  [yellow bold]▶[/yellow bold]  [bold]{idx}. {s_label}[/bold] [yellow](in progress)[/yellow]"
                    )
                else:
                    lines.append(f"  [dim]○[/dim]  [dim]{idx}. {s_label}[/dim]")
            else:
                lines.append(f"  [dim]○[/dim]  [dim]{idx}. {s_label}[/dim]")

    if pending_operation and pending_operation.get("endpoint_url"):
        lines.append("")
        lines.append(
            f"[bold]Endpoint URL:[/bold] [cyan]{pending_operation['endpoint_url']}[/cyan]"
        )

    content = "\n".join(lines)
    return Panel(content, title=panel_title, border_style=border_style, expand=False)


def build_deployment_status_renderable(
    operation_status: dict[str, Any],
    pending_operation: Optional[dict[str, Any]] = None,
) -> Any:
    """Builds a composite Rich renderable combining the progress panel and status details."""
    from rich.console import Group
    from rich.text import Text

    panel = build_deployment_progress_panel(
        operation_status, pending_operation=pending_operation
    )

    bullets = Text()
    bullets.append(f"   • Status:           {operation_status['status']}\n")
    bullets.append(f"   • Done:             {operation_status['done']}\n")
    if operation_status.get("deployment_stage"):
        bullets.append(
            f"   • Deployment Stage: {operation_status['deployment_stage']}\n"
        )
    if operation_status.get("create_time"):
        bullets.append(f"   • Created:          {operation_status['create_time']}\n")
    if operation_status.get("update_time"):
        bullets.append(f"   • Updated:          {operation_status['update_time']}\n")
    if operation_status.get("error"):
        bullets.append(
            f"   • Error Code:       {operation_status['error'].get('code')}\n"
        )
        bullets.append(
            f"   • Error Message:    {operation_status['error'].get('message')}\n"
        )
    if operation_status.get("response"):
        bullets.append("   • Response Details:\n")
        bullets.append(json.dumps(operation_status["response"], indent=4))
        bullets.append("\n")

    return Group(panel, bullets)


def render_deployment_progress(
    operation_status: dict[str, Any],
    pending_operation: Optional[dict[str, Any]] = None,
) -> None:
    """Renders a formatted, human-friendly Rich panel of the deployment progress."""
    from google.models.cli.common.console import console

    panel = build_deployment_progress_panel(
        operation_status, pending_operation=pending_operation
    )
    console.print(panel)


def get_operation_status(
    operation_name: str,
    location: Optional[str] = None,
) -> dict[str, Any]:
    """Gets the latest status of a Long-Running Operation (LRO).

    Uses https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/general/long-running-operations#get-operation

    Args:
        operation_name: Resource name of the operation
            (e.g. projects/.../locations/us-central1/endpoints/.../operations/...)
        location: Optional region override. If omitted, extracted from operation_name.

    Returns:
        Dictionary containing operation status details.
    """
    resolved_location = location
    if not resolved_location:
        location_match = re.search(r"locations/([^/]+)/", operation_name)
        resolved_location = location_match.group(1) if location_match else "us-central1"

    client_options = {"api_endpoint": f"{resolved_location}-aiplatform.googleapis.com"}
    client = aiplatform_v1.EndpointServiceClient(client_options=client_options)
    request = operations_pb2.GetOperationRequest(name=operation_name)
    operation = client.get_operation(request=request)

    operation_dict = MessageToDict(operation)
    metadata = operation_dict.get("metadata", {})
    generic_metadata = metadata.get("genericMetadata", {})

    deployment_stage = metadata.get("deploymentStage")
    if not deployment_stage and operation.HasField("metadata"):
        try:
            deploy_metadata = aiplatform_v1.DeployModelOperationMetadata()
            operation.metadata.Unpack(deploy_metadata._pb)
            if deploy_metadata.deployment_stage:
                stage_name = getattr(
                    deploy_metadata.deployment_stage, "name", None
                ) or str(deploy_metadata.deployment_stage)
                if stage_name and stage_name != "DEPLOYMENT_STAGE_UNSPECIFIED":
                    deployment_stage = stage_name
        except Exception:
            pass

    status_str = "RUNNING"
    if operation.done:
        status_str = "FAILED" if operation.HasField("error") else "SUCCEEDED"
    elif deployment_stage and deployment_stage != "DEPLOYMENT_STAGE_UNSPECIFIED":
        status_str = deployment_stage

    error_detail = None
    if operation.HasField("error"):
        error_detail = {
            "code": operation.error.code,
            "message": operation.error.message,
        }

    stage_info = get_stage_info(deployment_stage)
    if not stage_info and operation.done:
        stage_info = (
            get_stage_info("FAILED_TO_DEPLOY")
            if operation.HasField("error")
            else get_stage_info("SUCCESSFULLY_DEPLOYED")
        )

    return {
        "name": operation.name,
        "done": operation.done,
        "status": status_str,
        "deployment_stage": deployment_stage,
        "stage_info": stage_info,
        "create_time": generic_metadata.get("createTime"),
        "update_time": generic_metadata.get("updateTime"),
        "error": error_detail,
        "response": operation_dict.get("response"),
        "raw": operation_dict,
    }
