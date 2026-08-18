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
import os
import shutil
from pathlib import Path
from typing import Any, Optional
from google.cloud import aiplatform, resourcemanager_v3, storage
from google.iam.v1 import policy_pb2

from google.models.cli.common.constants import (
    DEFAULT_CONTAINER_PORT,
    DEFAULT_HEALTH_ROUTE,
    DEFAULT_PREDICTION_ROUTE,
    DEFAULT_SGLANG_CONTAINER_IMAGE,
    DEFAULT_SHARED_MEMORY_MB,
    DEFAULT_VLLM_CONTAINER_IMAGE,
    HARDWARE_SPECS,
    InferenceEngine,
)


def build_container_args(
    model_uri: str,
    engine: InferenceEngine = InferenceEngine.VLLM,
    engine_params: Optional[dict[str, Any]] = None,
) -> list[str]:
    """Constructs command line arguments for the container entrypoint."""
    params = engine_params or {}
    tp_size = params.get("tensor_parallel_size") or params.get("tp_size") or 1
    max_len = params.get("max_model_len") or params.get("context_length") or 4096
    kv_cache = params.get("kv_cache_dtype", "fp8")
    gpu_mem = params.get("gpu_memory_utilization") or params.get("mem_fraction_static") or 0.90

    if engine == InferenceEngine.VLLM:
        args = [
            "python3",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--host=0.0.0.0",
            f"--port={DEFAULT_CONTAINER_PORT}",
            f"--model={model_uri}",
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
            "--host=0.0.0.0",
            f"--port={DEFAULT_CONTAINER_PORT}",
            f"--model-path={model_uri}",
            f"--tp-size={tp_size}",
            f"--mem-fraction-static={gpu_mem}",
            f"--context-length={max_len}",
        ]
        return args


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
                return True

        new_binding = policy_pb2.Binding(role=role, members=[member])
        policy.bindings.append(new_binding)
        client.set_iam_policy(request={"resource": project_path, "policy": policy})
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
            bucket = client.create_bucket(bucket_name, location=location)

        # Check if already uploaded
        blobs = list(bucket.list_blobs(prefix=artifact_path + "/", max_results=1))
        if blobs:
            return gcs_uri

        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            ignore_patterns=["*.bin", "*.pth", "*.gguf", ".gitattributes"],
            token=hf_token,
        )

        files = [f for f in local_dir.rglob("*") if f.is_file()]
        for f in files:
            blob_name = f"{artifact_path}/{f.relative_to(local_dir)}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(f))

        return gcs_uri
    finally:
        if local_dir.exists():
            shutil.rmtree(local_dir, ignore_errors=True)


def deploy_model_to_geap(
    project_id: str,
    location: str,
    model_id: str,
    model_uri: str,
    container_image_uri: str,
    service_account_email: str,
    machine_type: str,
    engine: InferenceEngine = InferenceEngine.VLLM,
    engine_params: Optional[dict[str, Any]] = None,
    accelerator_count: Optional[int] = None,
    accelerator_type: Optional[str] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Deploys model to Vertex AI / GEAP Online Prediction Endpoint.

    Returns deployment metadata dict.
    """
    container_args = build_container_args(model_uri, engine, engine_params)
    env_vars = {"VLLM_LOGGING_LEVEL": "INFO"}

    spec = HARDWARE_SPECS.get(machine_type, {})
    resolved_accel_type = accelerator_type or spec.get("accelerator_type")
    resolved_accel_count = accelerator_count or spec.get("accelerator_count")

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
        "service_account": service_account_email,
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
            "endpoint_name": f"projects/{project_id}/locations/{location}/endpoints/simulated-endpoint-123",
            "endpoint_url": f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/simulated-endpoint-123",
            "manifest": manifest,
        }

    aiplatform.init(project=project_id, location=location)

    uploaded_model = aiplatform.Model.upload(
        display_name=model_id,
        serving_container_image_uri=container_image_uri,
        serving_container_args=container_args,
        serving_container_environment_variables=env_vars,
        serving_container_deployment_timeout=3600,
        serving_container_invoke_route_prefix=DEFAULT_PREDICTION_ROUTE,
        serving_container_health_route=DEFAULT_HEALTH_ROUTE,
        serving_container_shared_memory_size_mb=DEFAULT_SHARED_MEMORY_MB,
    )

    endpoint = aiplatform.Endpoint.create(
        display_name=f"{model_id}-endpoint",
        dedicated_endpoint_enabled=True,
    )

    uploaded_model.deploy(
        endpoint=endpoint,
        machine_type=machine_type,
        service_account=service_account_email,
        accelerator_count=resolved_accel_count,
        accelerator_type=resolved_accel_type,
    )

    return {
        "status": "DEPLOYED",
        "model_resource_name": uploaded_model.resource_name,
        "endpoint_resource_name": endpoint.resource_name,
        "endpoint_url": f"https://{location}-aiplatform.googleapis.com/v1/{endpoint.resource_name}",
        "manifest": manifest,
    }
