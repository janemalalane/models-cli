# Copyright 2026 Google LLC
from pathlib import Path
import tempfile
import click
import pytest
import yaml

from google.models.cli._project import (
    DeploymentConfig,
    EngineConfig,
    SGLangEngineConfig,
    VLLMEngineConfig,
    find_project_root,
    read_deployment_config,
    read_engine_config,
)
from google.models.cli.common.constants import InferenceEngine


def test_deployment_config_defaults():
    cfg = DeploymentConfig()
    assert cfg.machine_type is None
    assert cfg.service_account is None
    assert cfg.dedicated_endpoint is True
    assert cfg.routes == {"predict": "/*", "health": "/health"}
    assert cfg.shared_memory_mb == 131072


def test_deployment_config_from_dict_flat():
    data = {
        "display_name": "my-endpoint",
        "machine_type": "ct6e-standard-4t",
        "container_image_uri": "us-docker.pkg.dev/my-proj/custom-vllm:latest",
        "service_account": "custom-sa@my-proj.iam.gserviceaccount.com",
        "dedicated_endpoint": False,
        "shared_memory_mb": 65536,
        "routes": {"predict": "/v1/chat/completions", "health": "/ready", "port": 8000},
    }
    cfg = DeploymentConfig.from_dict(data)
    assert cfg.display_name == "my-endpoint"
    assert cfg.machine_type == "ct6e-standard-4t"
    assert cfg.container_image_uri == "us-docker.pkg.dev/my-proj/custom-vllm:latest"
    assert cfg.service_account == "custom-sa@my-proj.iam.gserviceaccount.com"
    assert cfg.dedicated_endpoint is False
    assert cfg.shared_memory_mb == 65536
    assert cfg.routes == {"predict": "/v1/chat/completions", "health": "/ready", "port": 8000}


def test_deployment_config_malformed_raises_click_exception():
    with pytest.raises(click.ClickException):
        DeploymentConfig.from_dict("not-a-dict")


def test_vllm_engine_config_defaults():
    cfg = VLLMEngineConfig()
    assert cfg.engine == "vllm"
    assert cfg.engine_enum == InferenceEngine.VLLM
    assert cfg.tensor_parallel_size == 1
    assert cfg.gpu_memory_utilization == 0.90
    params = cfg.to_engine_params()
    assert params["tensor_parallel_size"] == 1
    assert "extra_params" not in params


def test_sglang_engine_config_defaults_and_from_dict():
    cfg = SGLangEngineConfig()
    assert cfg.engine == "sglang"
    assert cfg.engine_enum == InferenceEngine.SGLANG
    assert cfg.tp_size == 1
    assert cfg.mem_fraction_static == 0.88
    assert cfg.context_length == 4096

    data = {
        "engine": "sglang",
        "tp_size": 4,
        "context_length": 8192,
        "mem_fraction_static": 0.85,
    }
    loaded = SGLangEngineConfig.from_dict(data)
    assert loaded.tp_size == 4
    assert loaded.context_length == 8192
    assert loaded.mem_fraction_static == 0.85
    params = loaded.to_engine_params()
    assert params["tp_size"] == 4
    assert params["context_length"] == 8192


def test_read_deployment_and_engine_config_from_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True)

        (config_dir / "deployment_spec.yaml").write_text(
            yaml.dump(
                {
                    "display_name": "test-project-endpoint",
                    "machine_type": "g4-standard-48",
                }
            )
        )

        (config_dir / "engine_config.yaml").write_text(
            yaml.dump(
                {
                    "engine": "sglang",
                    "tp_size": 2,
                    "context_length": 16384,
                }
            )
        )

        # Test project root discovery
        assert find_project_root(config_dir) == tmp_path

        # Test readers
        deploy_cfg = read_deployment_config(tmp_path)
        assert deploy_cfg.display_name == "test-project-endpoint"
        assert deploy_cfg.machine_type == "g4-standard-48"

        engine_cfg = read_engine_config(tmp_path)
        assert engine_cfg.engine == "sglang"
        assert engine_cfg.engine_enum == InferenceEngine.SGLANG
        assert engine_cfg.tp_size == 2
        assert engine_cfg.context_length == 16384
