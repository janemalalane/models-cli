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

"""Deployment utilities for Vertex AI / GEAP model registration and endpoint provisioning."""

import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Optional
import click
from google.cloud import aiplatform, aiplatform_v1, resourcemanager_v3, storage
from google.iam.v1 import policy_pb2
from google.longrunning import operations_pb2
from google.protobuf.json_format import MessageToDict

from google.models.cli._gcp_project import get_gcp_project_number
from google.models.cli.common.constants import (
    DEFAULT_CONTAINER_HOST,
    DEFAULT_CONTAINER_PORT,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_HEALTH_ROUTE,
    DEFAULT_KV_CACHE_DTYPE,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_PREDICTION_ROUTE,
    DEFAULT_SGLANG_CONTAINER_IMAGE,
    DEFAULT_SHARED_MEMORY_MB,
    DEFAULT_TENSOR_PARALLEL_SIZE,
    DEFAULT_VLLM_CONTAINER_IMAGE,
    HARDWARE_SPECS,
    InferenceEngine,
)


def build_container_args(
    model_uri: str,
    engine: Optional[InferenceEngine] = InferenceEngine.VLLM,
    engine_params: Optional[dict[str, Any]] = None,
) -> list[str]:
    """Constructs command line arguments for the container entrypoint."""
    actual_engine = engine or InferenceEngine.VLLM
    params = engine_params or {}
    host = params.get("host") or DEFAULT_CONTAINER_HOST
    port = params.get("port") or DEFAULT_CONTAINER_PORT
    tp_size = (
        params.get("tensor_parallel_size")
        or params.get("tp_size")
        or DEFAULT_TENSOR_PARALLEL_SIZE
    )
    max_len = (
        params.get("max_model_len")
        or params.get("context_length")
        or DEFAULT_MAX_MODEL_LEN
    )
    kv_cache = params.get("kv_cache_dtype", DEFAULT_KV_CACHE_DTYPE)
    gpu_mem = (
        params.get("gpu_memory_utilization")
        or params.get("mem_fraction_static")
        or DEFAULT_GPU_MEMORY_UTILIZATION
    )

    if actual_engine == InferenceEngine.VLLM:
        args = [
            "python3",
            "-m",
            "vllm.entrypoints.openai.api_server",
            f"--model={model_uri}",
            f"--host={host}",
            f"--port={port}",
            f"--kv-cache-dtype={kv_cache}",
            "--language-model-only",
            f"--max-model-len={max_len}",
            f"--tensor-parallel-size={tp_size}",
            f"--gpu-memory-utilization={gpu_mem}",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
        ]
        return args
    else:  # SGLang
        args = [
            "python3",
            "-m",
            "sglang.launch_server",
            f"--model-path={model_uri}",
            f"--host={host}",
            f"--port={port}",
            f"--tp-size={tp_size}",
            f"--mem-fraction-static={gpu_mem}",
            f"--context-length={max_len}",
        ]
        return args


def check_model_exists_in_gcs(bucket_name: str, model_id: str) -> bool:
    """Checks if model weights already exist in the specified GCS bucket."""
    artifact_path = model_id.split("/")[-1]
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        if not bucket.exists():
            return False
        blobs = list(bucket.list_blobs(prefix=artifact_path + "/", max_results=1))
        return bool(blobs)
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
    model_id: str,
    bucket_name: str,
    location: str,
    hf_token: Optional[str] = None,
) -> str:
    """Downloads model weights from Hugging Face and uploads to GCS."""
    from huggingface_hub import snapshot_download
    from google.cloud.storage import transfer_manager

    artifact_path = model_id.split("/")[-1]
    gcs_uri = f"gs://{bucket_name}/{artifact_path}"
    local_dir = Path(f"tmp/{model_id.replace('/', '--')}")

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        if not bucket.exists():
            click.echo(f"Creating GCS bucket '{bucket_name}' in {location}...")
            bucket = client.create_bucket(bucket_name, location=location)

        # Check if already uploaded
        blobs = list(bucket.list_blobs(prefix=artifact_path + "/", max_results=1))
        if blobs:
            return gcs_uri

        click.echo(f"Downloading model snapshot '{model_id}' from Hugging Face...")
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            ignore_patterns=["*.bin", "*.pth", "*.gguf", ".gitattributes"],
            token=hf_token,
        )

        files = [f for f in local_dir.rglob("*") if f.is_file()]
        click.echo(f"Uploading {len(files)} files to {gcs_uri}...")
        for f in files:
            blob_name = f"{artifact_path}/{f.relative_to(local_dir)}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(f))

        click.echo(f"✓ Model files uploaded to {gcs_uri}")
        return gcs_uri
    finally:
        if local_dir.exists():
            shutil.rmtree(local_dir, ignore_errors=True)


def deploy_model_to_geap(
    project_id: str,
    location: str,
    model_id: str,
    model_uri: str,
    machine_type: str,
    service_account_email: Optional[str] = None,
    engine: Optional[InferenceEngine] = InferenceEngine.VLLM,
    engine_params: Optional[dict[str, Any]] = None,
    accelerator_count: Optional[int] = None,
    accelerator_type: Optional[str] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Deploys model to Vertex AI / GEAP Online Prediction Endpoint.

    Returns deployment metadata dict.
    """
    actual_engine = engine or InferenceEngine.VLLM
    container_image_uri = actual_engine.image_uri
    container_args = build_container_args(model_uri, actual_engine, engine_params)
    env_vars = {"VLLM_LOGGING_LEVEL": "INFO"}

    spec = HARDWARE_SPECS.get(machine_type, {})

    if spec == {}:
        raise click.ClickException(f"Unsupported machine type on GEAP: {machine_type}")

    raw_accel_type = spec.get("accelerator_type") or accelerator_type
    resolved_accel_type = (
        raw_accel_type.strip().upper().replace("-", "_") if raw_accel_type else None
    )
    resolved_accel_count = spec.get("accelerator_count") or accelerator_count
    project_number = get_gcp_project_number(project_id)
    default_sa = (
        f"{project_number}-compute@developer.gserviceaccount.com"
        if project_number
        else "(default Compute Engine service account)"
    )
    resolved_sa = service_account_email or default_sa

    manifest = {
        "project": project_id,
        "location": location,
        "model_id": model_id,
        "model_uri": model_uri,
        "container_image_uri": container_image_uri,
        "container_args": container_args,
        "machine_type": machine_type,
        "accelerator_type": resolved_accel_type,
        "accelerator_count": resolved_accel_count,
        "service_account": resolved_sa,
        "routes": {
            "predict": DEFAULT_PREDICTION_ROUTE,
            "health": DEFAULT_HEALTH_ROUTE,
            "port": DEFAULT_CONTAINER_PORT,
        },
        "shared_memory_mb": DEFAULT_SHARED_MEMORY_MB,
        "dry_run": dry_run,
    }

    if dry_run:
        return {
            "status": "DRY_RUN",
            "endpoint_resource_name": f"projects/{project_id}/locations/{location}/endpoints/simulated-endpoint-123",
            "endpoint_url": f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/simulated-endpoint-123",
            "manifest": manifest,
        }

    aiplatform.init(project=project_id, location=location)

    endpoint = None
    model = None

    try:
        model = aiplatform.Model.upload(
            display_name=model_id,
            serving_container_image_uri=container_image_uri,
            serving_container_args=container_args,
            serving_container_environment_variables=env_vars,
            serving_container_deployment_timeout=3600,
            serving_container_invoke_route_prefix=DEFAULT_PREDICTION_ROUTE,
            serving_container_health_route=DEFAULT_HEALTH_ROUTE,
            serving_container_shared_memory_size_mb=DEFAULT_SHARED_MEMORY_MB,
        )

        if model is None:
            raise click.ClickException(
                "Model upload failed. Please check the logs for details."
            )

        endpoint = aiplatform.Endpoint.create(
            display_name=f"{model_id}-endpoint",
            dedicated_endpoint_enabled=True,
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
        if resolved_accel_type and resolved_accel_count:
            dedicated_resources["machine_spec"][
                "accelerator_type"
            ] = resolved_accel_type
            dedicated_resources["machine_spec"][
                "accelerator_count"
            ] = resolved_accel_count

        deployed_model = aiplatform_v1.DeployedModel(
            model=model.resource_name,
            display_name=model_id,
            dedicated_resources=dedicated_resources,
        )

        if resolved_sa and resolved_sa != "(default Compute Engine service account)":
            deployed_model.service_account = resolved_sa

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

    except Exception as e:
        click.echo(f"\nAn error occurred during the script execution: {e}", err=True)
        click.echo("Attempting to clean up created resources...", err=True)
        if endpoint and getattr(endpoint, "resource_name", None):
            try:
                click.echo(f"Deleting endpoint {endpoint.resource_name}...")
                endpoint.delete(force=True, sync=False)
                click.echo("Endpoint deletion initiated.")
            except Exception as cleanup_e:
                click.echo(f"Error deleting endpoint: {cleanup_e}", err=True)
        if model and getattr(model, "resource_name", None):
            try:
                click.echo(f"Deleting model {model.resource_name}...")
                model.delete(sync=False)
                click.echo("Model deletion initiated.")
            except Exception as cleanup_e:
                click.echo(f"Error deleting model: {cleanup_e}", err=True)
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
            ep_match = re.search(
                r"projects/([^/]+)/locations/([^/]+)/endpoints/([^/]+)",
                getattr(endpoint, "resource_name", "") or "",
            )
            if ep_match:
                dedicated_dns = f"{ep_match.group(3)}.{ep_match.group(2)}-{ep_match.group(1)}.prediction.vertexai.goog"

    if dedicated_dns:
        endpoint_url = f"https://{dedicated_dns}/v1/{endpoint.resource_name}/invoke"
    else:
        endpoint_url = f"https://{location}-aiplatform.googleapis.com/v1/{endpoint.resource_name}/invoke"

    return {
        "status": "DEPLOYED",
        "model_resource_name": model.resource_name,
        "endpoint_resource_name": endpoint.resource_name,
        "operation_name": operation_name,
        "endpoint_url": endpoint_url,
        "manifest": manifest,
    }


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
    loc = location
    if not loc:
        match = re.search(r"locations/([^/]+)/", operation_name)
        loc = match.group(1) if match else "us-central1"

    client_options = {"api_endpoint": f"{loc}-aiplatform.googleapis.com"}
    client = aiplatform_v1.EndpointServiceClient(client_options=client_options)
    request = operations_pb2.GetOperationRequest(name=operation_name)
    op = client.get_operation(request=request)

    op_dict = MessageToDict(op)
    metadata = op_dict.get("metadata", {})
    generic_metadata = metadata.get("genericMetadata", {})

    deployment_stage = metadata.get("deploymentStage")
    if not deployment_stage and op.HasField("metadata"):
        try:
            deploy_meta = aiplatform_v1.DeployModelOperationMetadata()
            op.metadata.Unpack(deploy_meta._pb)
            if deploy_meta.deployment_stage:
                stage_name = getattr(deploy_meta.deployment_stage, "name", None) or str(
                    deploy_meta.deployment_stage
                )
                if stage_name and stage_name != "DEPLOYMENT_STAGE_UNSPECIFIED":
                    deployment_stage = stage_name
        except Exception:
            pass

    status_str = "RUNNING"
    if op.done:
        status_str = "FAILED" if op.HasField("error") else "SUCCEEDED"
    elif deployment_stage and deployment_stage != "DEPLOYMENT_STAGE_UNSPECIFIED":
        status_str = deployment_stage

    error_detail = None
    if op.HasField("error"):
        error_detail = {
            "code": op.error.code,
            "message": op.error.message,
        }

    return {
        "name": op.name,
        "done": op.done,
        "status": status_str,
        "deployment_stage": deployment_stage,
        "create_time": generic_metadata.get("createTime"),
        "update_time": generic_metadata.get("updateTime"),
        "error": error_detail,
        "response": op_dict.get("response"),
        "raw": op_dict,
    }
