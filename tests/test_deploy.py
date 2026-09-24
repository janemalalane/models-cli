# Copyright 2026 Google LLC
from click.testing import CliRunner
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


def test_build_container_args_none_defaults_to_vllm():
    args = build_container_args("gs://my-bucket/gemma-31b", engine=None)
    assert "vllm.entrypoints.openai.api_server" in args
    assert "--model=gs://my-bucket/gemma-31b" in args
    assert "--host=0.0.0.0" in args
    assert "--port=8080" in args


def test_build_container_args_custom_host_and_port():
    args = build_container_args(
        "gs://my-bucket/gemma-31b",
        engine=InferenceEngine.VLLM,
        engine_params={"host": "127.0.0.1", "port": 9000},
    )
    assert "--host=127.0.0.1" in args
    assert "--port=9000" in args


def test_deploy_model_dry_run():
    result = deploy_model_to_geap(
        project_id="test-proj",
        location="us-central1",
        model_id="gemma-31b",
        model_uri="google/gemma-4-31B-it",
        service_account_email="sa@test-proj.iam.gserviceaccount.com",
        machine_type="ct6e-standard-4t",
        engine=InferenceEngine.VLLM,
        dry_run=True,
    )
    assert result["status"] == "DRY_RUN"
    assert "manifest" in result
    assert result["manifest"]["machine_type"] == "ct6e-standard-4t"


def test_deploy_model_with_none_engine_defaults_to_vllm():
    result = deploy_model_to_geap(
        project_id="test-proj",
        location="us-central1",
        model_id="gemma-31b",
        model_uri="google/gemma-4-31B-it",
        service_account_email="sa@test-proj.iam.gserviceaccount.com",
        machine_type="ct6e-standard-4t",
        engine=None,
        dry_run=True,
    )
    assert "vllm.entrypoints.openai.api_server" in result["manifest"]["container_args"]
    assert "pytorch-vllm-serve" in result["manifest"]["container_image_uri"]


def test_deploy_model_with_sglang_defaults_to_sglang_image():
    result = deploy_model_to_geap(
        project_id="test-proj",
        location="us-central1",
        model_id="gemma-31b",
        model_uri="google/gemma-4-31B-it",
        service_account_email="sa@test-proj.iam.gserviceaccount.com",
        machine_type="ct6e-standard-4t",
        engine=InferenceEngine.SGLANG,
        dry_run=True,
    )
    assert "sglang.launch_server" in result["manifest"]["container_args"]
    assert "pytorch-sglang-serve" in result["manifest"]["container_image_uri"]


def test_deploy_command_no_container_image_option():
    # Verify that --container-image was removed from options
    result = runner.invoke(app, ["deploy", "--help"])
    assert result.exit_code == 0
    assert "--container-image" not in result.stdout

    # Passing --container-image should error as an unrecognized option
    error_result = runner.invoke(
        app,
        ["deploy", "--container-image", "my-image:latest", "--dry-run"],
    )
    assert error_result.exit_code != 0
    assert "No such option" in error_result.output
    assert "--container-image" in error_result.output


def test_deploy_command_reuses_existing_gcs_weights(monkeypatch):
    import google.models.cli.deploy.cmd_deploy as cmd_deploy_mod

    monkeypatch.setattr(cmd_deploy_mod, "ensure_authenticated", lambda interactive: True)
    monkeypatch.setattr(
        cmd_deploy_mod,
        "check_model_exists_in_gcs",
        lambda bucket, repo: True,
    )
    deploy_mock_called = {}

    def fake_deploy_to_geap(**kwargs):
        deploy_mock_called.update(kwargs)
        return {"status": "SUCCESS", "endpoint": "projects/123/locations/us-central1/endpoints/456"}

    monkeypatch.setattr(cmd_deploy_mod, "deploy_model_to_geap", fake_deploy_to_geap)

    result = runner.invoke(
        app,
        [
            "deploy",
            "--model-id",
            "gemma-31b",
            "--hf-repo",
            "google/gemma-4-31B-it",
            "--bucket",
            "my-custom-bucket",
            "--project",
            "test-project-123",
        ],
    )
    assert result.exit_code == 0
    assert "Found existing weights in GCS: gs://my-custom-bucket/gemma-4-31B-it" in result.stdout
    assert deploy_mock_called["model_uri"] == "gs://my-custom-bucket/gemma-4-31B-it"
    assert deploy_mock_called["service_account_email"] is None


def test_deploy_command_with_explicit_service_account(monkeypatch):
    import google.models.cli.deploy.cmd_deploy as cmd_deploy_mod

    monkeypatch.setattr(cmd_deploy_mod, "ensure_authenticated", lambda interactive: True)
    monkeypatch.setattr(
        cmd_deploy_mod,
        "check_model_exists_in_gcs",
        lambda bucket, repo: True,
    )
    deploy_mock_called = {}

    def fake_deploy_to_geap(**kwargs):
        deploy_mock_called.update(kwargs)
        return {"status": "SUCCESS"}

    monkeypatch.setattr(cmd_deploy_mod, "deploy_model_to_geap", fake_deploy_to_geap)

    result = runner.invoke(
        app,
        [
            "deploy",
            "--model-id",
            "gemma-31b",
            "--hf-repo",
            "google/gemma-4-31B-it",
            "--project",
            "test-project-123",
            "--service-account",
            "my-custom-sa@test-project-123.iam.gserviceaccount.com",
        ],
    )
    assert result.exit_code == 0
    assert deploy_mock_called["service_account_email"] == "my-custom-sa@test-project-123.iam.gserviceaccount.com"


def test_deploy_model_cleanup_on_failure(monkeypatch):
    import pytest
    from unittest.mock import MagicMock
    from google.cloud import aiplatform, aiplatform_v1

    fake_model = MagicMock()
    fake_model.resource_name = "projects/123/locations/us-central1/models/456"
    fake_endpoint = MagicMock()
    fake_endpoint.resource_name = "projects/123/locations/us-central1/endpoints/789"

    mock_client = MagicMock()
    mock_client.deploy_model.side_effect = RuntimeError("Quota exceeded or deployment failure")

    monkeypatch.setattr(aiplatform, "init", lambda **kwargs: None)
    monkeypatch.setattr(aiplatform.Model, "upload", lambda **kwargs: fake_model)
    monkeypatch.setattr(aiplatform.Endpoint, "create", lambda **kwargs: fake_endpoint)
    monkeypatch.setattr(
        aiplatform_v1, "EndpointServiceClient", lambda client_options=None: mock_client
    )

    with pytest.raises(SystemExit) as exc_info:
        deploy_model_to_geap(
            project_id="test-proj",
            location="us-central1",
            model_id="gemma-31b",
            model_uri="gs://test-proj-models/gemma-31b",
            machine_type="g4-standard-48",
            dry_run=False,
        )

    assert exc_info.value.code == 1
    # Verify cleanup operations were invoked
    fake_endpoint.delete.assert_called_once_with(force=True, sync=False)
    fake_model.delete.assert_called_once_with(sync=False)


def test_get_operation_status(monkeypatch):
    from unittest.mock import MagicMock
    from google.cloud import aiplatform_v1
    from google.longrunning import operations_pb2
    from google.protobuf import any_pb2
    from google.models.cli.deploy.deploy_utils import get_operation_status

    mock_client = MagicMock()
    mock_op = operations_pb2.Operation(
        name="projects/123/locations/us-central1/endpoints/456/operations/789",
        done=False,
    )
    mock_client.get_operation.return_value = mock_op

    monkeypatch.setattr(
        aiplatform_v1, "EndpointServiceClient", lambda client_options=None: mock_client
    )

    status_result = get_operation_status("projects/123/locations/us-central1/endpoints/456/operations/789")
    assert status_result["name"] == "projects/123/locations/us-central1/endpoints/456/operations/789"
    assert status_result["done"] is False
    assert status_result["status"] == "RUNNING"

    # Now test with DeploymentStage metadata
    meta = aiplatform_v1.types.endpoint_service.DeployModelOperationMetadata()
    meta.deployment_stage = aiplatform_v1.types.DeploymentStage.PREPARING_MODEL
    any_val = any_pb2.Any()
    any_val.Pack(meta._pb)
    mock_op_with_stage = operations_pb2.Operation(
        name="projects/123/locations/us-central1/endpoints/456/operations/789",
        done=False,
        metadata=any_val,
    )
    mock_client.get_operation.return_value = mock_op_with_stage
    status_with_stage = get_operation_status("projects/123/locations/us-central1/endpoints/456/operations/789")
    assert status_with_stage["status"] == "PREPARING_MODEL"
    assert status_with_stage["deployment_stage"] == "PREPARING_MODEL"


def test_deploy_status_cli(monkeypatch):
    import google.models.cli.deploy.cmd_deploy as cmd_deploy_mod

    monkeypatch.setattr(cmd_deploy_mod, "ensure_authenticated", lambda interactive: True)
    monkeypatch.setattr(
        cmd_deploy_mod,
        "get_operation_status",
        lambda op_name, location=None: {
            "name": op_name,
            "status": "RUNNING",
            "done": False,
            "deployment_stage": "ADDING_NODES_TO_CLUSTER",
            "create_time": "2026-09-24T19:00:00Z",
            "update_time": "2026-09-24T19:05:00Z",
            "error": None,
            "response": None,
        },
    )

    result = runner.invoke(
        app,
        [
            "deploy",
            "--status",
            "--operation",
            "projects/123/locations/us-central1/endpoints/456/operations/789",
        ],
    )
    assert result.exit_code == 0
    assert "Status:           RUNNING" in result.stdout
    assert "Deployment Stage: ADDING_NODES_TO_CLUSTER" in result.stdout


def test_deploy_status_from_metadata_file(tmp_path, monkeypatch):
    import google.models.cli.deploy.cmd_deploy as cmd_deploy_mod
    from google.models.cli.deploy._operation import write_operation, read_operation, clear_operation

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").touch()
    monkeypatch.setattr(cmd_deploy_mod, "ensure_authenticated", lambda interactive: True)
    monkeypatch.setattr(
        cmd_deploy_mod,
        "get_operation_status",
        lambda op_name, location=None: {
            "name": op_name,
            "status": "RUNNING",
            "done": False,
            "deployment_stage": "CREATING_SERVING_CLUSTER",
            "create_time": "2026-09-24T19:00:00Z",
            "update_time": "2026-09-24T19:05:00Z",
            "error": None,
            "response": None,
        },
    )

    write_operation(
        operation_name="projects/123/locations/us-central1/endpoints/456/operations/999",
        project="test-proj",
        location="us-central1",
        endpoint="projects/123/locations/us-central1/endpoints/456",
        project_dir=tmp_path,
    )

    op_data = read_operation(project_dir=tmp_path)
    assert op_data["operation_name"] == "projects/123/locations/us-central1/endpoints/456/operations/999"

    # Invoke deploy --status without --operation flag, should read from deployment_metadata.json
    result = runner.invoke(app, ["deploy", "--status"])
    assert result.exit_code == 0
    assert "Checking deployment operation status: projects/123/locations/us-central1/endpoints/456/operations/999" in result.stdout
    assert "Deployment Stage: CREATING_SERVING_CLUSTER" in result.stdout



