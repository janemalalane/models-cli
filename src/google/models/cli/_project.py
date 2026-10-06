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
from typing import Any, Literal
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
    DEFAULT_MAX_NUM_SEQS,
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
    machine_type: str | None = None
    container_image_uri: str = ""
    service_account: str | None = None
    dedicated_endpoint: bool = True
    shared_memory_mb: int = DEFAULT_SHARED_MEMORY_MB
    routes: dict[str, str] = field(default_factory=_default_routes)
    raw_config: dict[str, Any] = field(default_factory=dict)

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
        cfg.raw_config = dict(data)
        cfg.display_name = data.get("display_name", cfg.display_name)
        cfg.machine_type = data.get("machine_type", cfg.machine_type)
        cfg.container_image_uri = data.get(
            "container_image_uri", cfg.container_image_uri
        )
        cfg.service_account = data.get("service_account", cfg.service_account)
        cfg.shared_memory_mb = data.get("shared_memory_mb", cfg.shared_memory_mb)
        if "dedicated_endpoint" in data:
            cfg.dedicated_endpoint = bool(data["dedicated_endpoint"])
        if "routes" in data and isinstance(data["routes"], Mapping):
            cfg.routes = {**cfg.routes, **data["routes"]}

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
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS
    max_num_batched_tokens: int = 2048
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    host: str = DEFAULT_CONTAINER_HOST
    port: int = DEFAULT_CONTAINER_PORT
    raw_config: dict[str, Any] = field(default_factory=dict)

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
        cfg.raw_config = dict(data)
        for key, val in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)
        if "port" in data:
            cfg.port = int(data["port"])
        return cfg

    def to_engine_params(self) -> dict[str, Any]:
        """Returns engine parameters dictionary for container entrypoints."""
        if self.raw_config:
            params = {}
            for k, v in self.raw_config.items():
                if k not in ("engine", "raw_config") and v is not None:
                    params[k] = v
            return params

        params = {"tensor_parallel_size": self.tensor_parallel_size}
        if self.pipeline_parallel_size != 1:
            params["pipeline_parallel_size"] = self.pipeline_parallel_size
        return params

    def __getattr__(self, name: str) -> Any:
        if name != "raw_config" and "raw_config" in self.__dict__ and name in self.raw_config:
            return self.raw_config[name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")


@dataclass
class SGLangEngineConfig:
    """Configuration derived from engine_config.yaml for SGLang."""

    engine: str = "sglang"
    tp_size: int = DEFAULT_TENSOR_PARALLEL_SIZE
    mem_fraction_static: float = 0.88
    context_length: int = DEFAULT_MAX_MODEL_LEN
    host: str = DEFAULT_CONTAINER_HOST
    port: int = DEFAULT_CONTAINER_PORT
    raw_config: dict[str, Any] = field(default_factory=dict)

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
        cfg.raw_config = dict(data)
        for key, val in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)
        if "port" in data:
            cfg.port = int(data["port"])
        return cfg

    def to_engine_params(self) -> dict[str, Any]:
        """Returns engine parameters dictionary for container entrypoints."""
        if self.raw_config:
            params = {}
            for k, v in self.raw_config.items():
                if k not in ("engine", "raw_config") and v is not None:
                    params[k] = v
            return params

        return {"tp_size": self.tp_size}

    def __getattr__(self, name: str) -> Any:
        if name != "raw_config" and "raw_config" in self.__dict__ and name in self.raw_config:
            return self.raw_config[name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")


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
