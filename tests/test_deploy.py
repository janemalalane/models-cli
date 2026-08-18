# Copyright 2026 Google LLC
from typer.testing import CliRunner
from google.models.cli.common.constants import InferenceEngine
from google.models.cli.deploy.deploy_utils import build_container_args, deploy_model_to_geap
from google.models.cli.main import app

runner = CliRunner()


def test_build_container_args_vllm():
    args = build_container_args("gs://my-bucket/gemma-31b", engine=InferenceEngine.VLLM)
    assert "vllm.entrypoints.openai.api_server" in args
    assert "--model=gs://my-bucket/gemma-31b" in args


def test_build_container_args_sglang():
    args = build_container_args("gs://my-bucket/gemma-31b", engine=InferenceEngine.SGLANG)
    assert "sglang.launch_server" in args
    assert "--model-path=gs://my-bucket/gemma-31b" in args


def test_deploy_model_dry_run():
    result = deploy_model_to_geap(
        project_id="test-proj",
        location="us-central1",
        model_id="gemma-31b",
        model_uri="google/gemma-4-31B-it",
        container_image_uri="us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:latest",
        service_account_email="sa@test-proj.iam.gserviceaccount.com",
        machine_type="ct6e-standard-4t",
        dry_run=True,
    )
    assert result["status"] == "DRY_RUN"
    assert "manifest" in result
    assert result["manifest"]["machine_type"] == "ct6e-standard-4t"


def test_deploy_command_dry_run():
    result = runner.invoke(
        app,
        [
            "deploy",
            "--model-id",
            "gemma-31b",
            "--hf-repo",
            "google/gemma-4-31B-it",
            "--machine-type",
            "g4-standard-48",
            "--dry-run",
            "--project",
            "test-project-123",
        ],
    )
    assert result.exit_code == 0
    assert "Dry Run Deployment Manifest" in result.stdout
