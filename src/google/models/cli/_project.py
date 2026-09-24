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

"""Project configuration reader and discovery utilities for models-cli."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Optional
import click
from dotenv import set_key
import yaml

from google.models.cli.common.constants import (
    DEFAULT_CONTAINER_HOST,
    DEFAULT_CONTAINER_PORT,
    DEFAULT_ENGINE,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_HEALTH_ROUTE,
    DEFAULT_KV_CACHE_DTYPE,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_PREDICTION_ROUTE,
    DEFAULT_SHARED_MEMORY_MB,
    DEFAULT_TENSOR_PARALLEL_SIZE,
    InferenceEngine,
)


def _default_routes() -> dict[str, str]:
    return {"predict": DEFAULT_PREDICTION_ROUTE, "health": DEFAULT_HEALTH_ROUTE}


@dataclass
class DeploymentConfig:
    """Configuration derived from deployment_spec.yaml."""

    display_name: str = ""
    engine: str = DEFAULT_ENGINE
    machine_type: Optional[str] = None
    accelerator_type: Optional[str] = None
    accelerator_count: Optional[int] = None
    container_image_uri: Optional[str] = None
    service_account: Optional[str] = None
    dedicated_endpoint: bool = True
    shared_memory_mb: int = DEFAULT_SHARED_MEMORY_MB
    routes: dict[str, str] = field(default_factory=_default_routes)

    @property
    def engine_enum(self) -> Optional[InferenceEngine]:
        """Resolves engine string to an InferenceEngine enum if valid."""
        if not self.engine:
            return None
        try:
            return InferenceEngine(self.engine.lower().strip())
        except ValueError:
            return None

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | Any,
        filename: str = "deployment_spec.yaml",
    ) -> DeploymentConfig:
        """Create a DeploymentConfig from a raw dictionary."""
        if not isinstance(data, Mapping):
            raise click.ClickException(f"malformed {filename}")

        cfg = cls()
        cfg.display_name = data.get("display_name", cfg.display_name)
        cfg.engine = data.get("engine", cfg.engine)
        cfg.machine_type = data.get("machine_type", cfg.machine_type)
        cfg.accelerator_type = data.get("accelerator_type", cfg.accelerator_type)
        cfg.accelerator_count = data.get("accelerator_count", cfg.accelerator_count)
        cfg.container_image_uri = data.get(
            "container_image_uri", cfg.container_image_uri
        )
        cfg.service_account = data.get("service_account", cfg.service_account)
        cfg.dedicated_endpoint = data.get("dedicated_endpoint", cfg.dedicated_endpoint)
        cfg.shared_memory_mb = data.get("shared_memory_mb", cfg.shared_memory_mb)
        cfg.routes = data.get("routes") or cfg.routes

        return cfg


@dataclass
class VLLMEngineConfig:
    """Configuration derived from engine_config.yaml for vLLM."""

    engine: str = "vllm"
    tensor_parallel_size: int = DEFAULT_TENSOR_PARALLEL_SIZE
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    kv_cache_dtype: str = DEFAULT_KV_CACHE_DTYPE
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 2048
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    host: str = DEFAULT_CONTAINER_HOST
    port: int = DEFAULT_CONTAINER_PORT

    @property
    def engine_enum(self) -> InferenceEngine:
        return InferenceEngine.VLLM

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | Any,
        filename: str = "engine_config.yaml",
    ) -> VLLMEngineConfig:
        """Create a VLLMEngineConfig from a raw engine dictionary."""
        if not isinstance(data, Mapping):
            raise click.ClickException(f"malformed {filename}")

        cfg = cls()
        cfg.engine = data.get("engine", cfg.engine)
        cfg.tensor_parallel_size = data.get(
            "tensor_parallel_size", cfg.tensor_parallel_size
        )
        cfg.pipeline_parallel_size = data.get(
            "pipeline_parallel_size", cfg.pipeline_parallel_size
        )
        cfg.gpu_memory_utilization = data.get(
            "gpu_memory_utilization", cfg.gpu_memory_utilization
        )
        cfg.kv_cache_dtype = data.get("kv_cache_dtype", cfg.kv_cache_dtype)
        cfg.max_model_len = data.get("max_model_len", cfg.max_model_len)
        cfg.max_num_seqs = data.get("max_num_seqs", cfg.max_num_seqs)
        cfg.max_num_batched_tokens = data.get(
            "max_num_batched_tokens", cfg.max_num_batched_tokens
        )
        cfg.enable_prefix_caching = data.get(
            "enable_prefix_caching", cfg.enable_prefix_caching
        )
        cfg.enable_chunked_prefill = data.get(
            "enable_chunked_prefill", cfg.enable_chunked_prefill
        )
        cfg.host = data.get("host", cfg.host)
        cfg.port = data.get("port", cfg.port)
        return cfg

    def to_engine_params(self) -> dict[str, Any]:
        """Returns engine parameters dictionary for container entrypoints."""
        return {
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "kv_cache_dtype": self.kv_cache_dtype,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "host": self.host,
            "port": self.port,
        }


@dataclass
class SGLangEngineConfig:
    """Configuration derived from engine_config.yaml for SGLang."""

    engine: str = "sglang"
    tp_size: int = DEFAULT_TENSOR_PARALLEL_SIZE
    mem_fraction_static: float = 0.88
    context_length: int = DEFAULT_MAX_MODEL_LEN
    host: str = DEFAULT_CONTAINER_HOST
    port: int = DEFAULT_CONTAINER_PORT

    @property
    def engine_enum(self) -> InferenceEngine:
        return InferenceEngine.SGLANG

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | Any,
        filename: str = "engine_config.yaml",
    ) -> SGLangEngineConfig:
        """Create an SGLangEngineConfig from a raw engine dictionary."""
        if not isinstance(data, Mapping):
            raise click.ClickException(f"malformed {filename}")

        cfg = cls()
        cfg.engine = data.get("engine", cfg.engine)
        cfg.tp_size = data.get("tp_size", cfg.tp_size)
        cfg.mem_fraction_static = data.get(
            "mem_fraction_static", cfg.mem_fraction_static
        )
        cfg.context_length = data.get("context_length", cfg.context_length)
        cfg.host = data.get("host", cfg.host)
        cfg.port = data.get("port", cfg.port)
        return cfg

    def to_engine_params(self) -> dict[str, Any]:
        """Returns engine parameters dictionary for container entrypoints."""
        return {
            "tp_size": self.tp_size,
            "mem_fraction_static": self.mem_fraction_static,
            "context_length": self.context_length,
            "host": self.host,
            "port": self.port,
        }


EngineConfig = VLLMEngineConfig | SGLangEngineConfig


# ---------------------------------------------------------------------------
# Project Discovery & Readers
# ---------------------------------------------------------------------------


def find_project_root(dir: Path | None = None) -> Path | None:
    """Find the project root by walking up looking for config/."""
    if dir is None:
        dir = Path.cwd()

    # If currently inside a 'config' subdirectory, check parent first
    start = dir.parent if dir.name == "config" else dir

    for parent in [start, *start.parents]:
        if (parent / "config" / "deployment_spec.yaml").is_file():
            return parent
        if (parent / "config" / "engine_config.yaml").is_file():
            return parent
    return None


def _read_yaml(file_path: Path) -> dict[str, Any]:
    """Read a YAML file safely into a dictionary."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_deployment_config_from_manifest(manifest_path: Path) -> DeploymentConfig:
    """Read deployment configuration from a deployment_spec.yaml file."""
    data = _read_yaml(manifest_path)
    return DeploymentConfig.from_dict(data, filename=manifest_path.name)


def read_deployment_config(project_dir: str | Path | None = None) -> DeploymentConfig:
    """Read deployment metadata from config/deployment_spec.yaml.

    Falls back to sensible defaults if the file doesn't exist.
    """
    root = Path(project_dir) if project_dir else (find_project_root() or Path.cwd())
    spec_path = root / "config" / "deployment_spec.yaml"
    if spec_path.is_file():
        return _read_deployment_config_from_manifest(spec_path)

    return DeploymentConfig()


def _read_engine_config_from_manifest(manifest_path: Path) -> EngineConfig:
    """Read engine configuration from an engine_config.yaml file."""
    data = _read_yaml(manifest_path)
    engine_name = str(data.get("engine") or DEFAULT_ENGINE).lower().strip()

    if engine_name == "sglang":
        return SGLangEngineConfig.from_dict(data, filename=manifest_path.name)
    return VLLMEngineConfig.from_dict(data, filename=manifest_path.name)


def read_engine_config(project_dir: str | Path | None = None) -> EngineConfig:
    """Read engine runtime metadata from config/engine_config.yaml.

    Falls back to sensible defaults if the file doesn't exist.
    """
    root = Path(project_dir) if project_dir else (find_project_root() or Path.cwd())
    engine_path = root / "config" / "engine_config.yaml"
    if engine_path.is_file():
        return _read_engine_config_from_manifest(engine_path)

    return VLLMEngineConfig()


def write_env(
    env_vars: dict[str, str],
    project_dir: str | Path | None = None,
) -> None:
    """Updates or adds environment variables in .env in-place."""
    root = Path(project_dir) if project_dir else (find_project_root() or Path.cwd())
    env_path = root / ".env"
    env_path.touch(exist_ok=True)
    for key, value in env_vars.items():
        set_key(env_path, key, value, quote_mode="always")
