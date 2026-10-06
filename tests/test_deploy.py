import click
from click.testing import CliRunner
from google.models.cli.common.constants import (
    DEFAULT_CONTAINER_DEPLOYMENT_TIMEOUT,
    DEFAULT_UPLOAD_REQUEST_TIMEOUT,
    InferenceEngine,
)
from google.models.cli._project import DeploymentConfig
from google.models.cli.deploy.deploy_utils import (
    build_container_args,
    deploy_model_to_geap,
)
from google.models.cli.main import app

runner = CliRunner()


def test_build_container_args_vllm():
    args = build_container_args("gs://my-bucket/gemma-31b", engine=InferenceEngine.VLLM)
    assert args[:3] == ["python3", "-m", "vllm.entrypoints.openai.api_server"]
    assert "--host=0.0.0.0" in args
    assert "--port=8080" in args


def test_build_container_args_sglang():
    args = build_container_args(
        "gs://my-bucket/gemma-31b", engine=InferenceEngine.SGLANG
    )
    assert not any(arg.startswith("--model-path") for arg in args)
    assert "--host=0.0.0.0" in args
    assert "--port=8080" in args


def test_build_container_args_none_defaults_to_vllm():
    args = build_container_args("gs://my-bucket/gemma-31b", engine=None)
    assert args[:3] == ["python3", "-m", "vllm.entrypoints.openai.api_server"]
    assert "--host=0.0.0.0" in args
    assert "--port=8080" in args
    assert "--tensor-parallel-size=1" in args


def test_build_container_args_dynamic_passthrough():
    from google.models.cli._project import VLLMEngineConfig
    cfg = VLLMEngineConfig.from_dict({
        "engine": "vllm",
        "tensor_parallel_size": 2,
        "enable_reasoning": True,
        "reasoning_parser": "deepseek_r1",
        "enable_chunked_prefill": False,
        "extra_args": ["--speculative-draft-model=my-draft"],
    })
    args = build_container_args(cfg)
    assert args[:3] == ["python3", "-m", "vllm.entrypoints.openai.api_server"]
    assert "--enable-reasoning" in args
    assert "--reasoning-parser=deepseek_r1" in args
    assert "--tensor-parallel-size=2" in args
    assert "--speculative-draft-model=my-draft" in args
    # Boolean False should not be passed as positive flag
    assert "--enable-chunked-prefill" not in args


def test_deploy_model_dry_run():
    result = deploy_model_to_geap(
        project_id="test-proj",
        location="us-central1",
        model_display_name="gemma-31b",
        model_uri="gs://test-proj-models/gemma-4-31B-it",
        service_account_email="sa@test-proj.iam.gserviceaccount.com",
        machine_type="ct6e-standard-4t",
        engine=InferenceEngine.VLLM,
        dry_run=True,
    )
    assert result["status"] == "DRY_RUN"
    manifest = result["manifest"]
    assert manifest["machine_type"] == "ct6e-standard-4t"
    assert manifest["endpoint_display_name"] == "gemma-31b-endpoint"
    assert manifest["dedicated_endpoint_enabled"] is True
    assert manifest["engine"] == "vllm"
    assert manifest["container_env_vars"] == {
        "VLLM_LOGGING_LEVEL": "INFO",
        "HF_HOME": "/dev/shm/hf",
        "TMPDIR": "/dev/shm",
        "LOCAL_DOWNLOAD_DIR": "/dev/shm",
        "LOCAL_MODEL_DIR": "/dev/shm/model_dir",
        "CLOUDSDK_STORAGE_MAX_RETRIES": "3",
    }


def test_deploy_model_with_none_engine_defaults_to_vllm():
    result = deploy_model_to_geap(
        project_id="test-proj",
        location="us-central1",
        model_display_name="gemma-31b",
        model_uri="gs://test-proj-models/gemma-4-31B-it",
        service_account_email="sa@test-proj.iam.gserviceaccount.com",
        machine_type="ct6e-standard-4t",
        engine=None,
        dry_run=True,
    )
    assert result["manifest"]["engine"] == "vllm"
    assert result["manifest"]["container_args"][:3] == [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
    ]
    assert "pytorch-vllm-serve" in result["manifest"]["container_image_uri"]


def test_deploy_model_with_sglang_defaults_to_sglang_image():
    result = deploy_model_to_geap(
        project_id="test-proj",
        location="us-central1",
        model_display_name="gemma-31b",
        model_uri="gs://test-proj-models/gemma-4-31B-it",
        service_account_email="sa@test-proj.iam.gserviceaccount.com",
        machine_type="ct6e-standard-4t",
        engine=InferenceEngine.SGLANG,
        dry_run=True,
    )
    assert result["manifest"]["engine"] == "sglang"
    assert not any(arg.startswith("--model-path") for arg in result["manifest"]["container_args"])
    assert "pytorch-sglang-serve" in result["manifest"]["container_image_uri"]
    assert result["manifest"]["container_env_vars"] == {
        "SGLANG_LOGGING_LEVEL": "INFO",
        "HF_HOME": "/dev/shm/hf",
        "TMPDIR": "/dev/shm",
        "LOCAL_DOWNLOAD_DIR": "/dev/shm",
        "LOCAL_MODEL_DIR": "/dev/shm/model_dir",
        "CLOUDSDK_STORAGE_MAX_RETRIES": "3",
    }


def test_deploy_model_with_custom_routes_and_options():
    result = deploy_model_to_geap(
        project_id="test-proj",
        location="us-central1",
        model_display_name="gemma-31b",
        model_uri="gs://test-proj-models/gemma-4-31B-it",
        service_account_email="sa@test-proj.iam.gserviceaccount.com",
        machine_type="ct6e-standard-4t",
        container_image_uri="us-docker.pkg.dev/custom/vllm:v1",
        endpoint_display_name="my-custom-endpoint",
        dedicated_endpoint=False,
        routes={"predict": "/v1/chat/completions", "health": "/ready", "port": 8000},
        dry_run=True,
    )
    assert result["status"] == "DRY_RUN"
    manifest = result["manifest"]
    assert manifest["endpoint_display_name"] == "my-custom-endpoint"
    assert manifest["dedicated_endpoint_enabled"] is False
    assert manifest["container_image_uri"] == "us-docker.pkg.dev/custom/vllm:v1"
    assert manifest["routes"] == {
        "predict": "/v1/chat/completions",
        "health": "/ready",
        "port": 8000,
    }


def test_deploy_model_upload_and_create_use_custom_spec(monkeypatch):
    from unittest.mock import MagicMock
    from google.cloud import aiplatform, aiplatform_v1

    fake_model = MagicMock()
    fake_model.resource_name = "projects/123/locations/us-central1/models/456"
    fake_endpoint = MagicMock()
    fake_endpoint.resource_name = "projects/123/locations/us-central1/endpoints/789"
    fake_endpoint.dedicated_endpoint_enabled = False
    fake_endpoint.dedicated_endpoint_dns = None
    fake_endpoint.gca_resource = None

    mock_client = MagicMock()
    mock_future = MagicMock()
    mock_future.operation.name = "projects/123/locations/us-central1/endpoints/789/operations/999"
    mock_client.deploy_model.return_value = mock_future

    uploaded_kwargs = {}
    def mock_upload(**kwargs):
        uploaded_kwargs.update(kwargs)
        return fake_model

    created_endpoint_kwargs = {}
    def mock_endpoint_create(**kwargs):
        created_endpoint_kwargs.update(kwargs)
        return fake_endpoint

    monkeypatch.setattr(aiplatform, "init", lambda **kwargs: None)
    monkeypatch.setattr(aiplatform.Model, "upload", mock_upload)
    monkeypatch.setattr(aiplatform.Endpoint, "create", mock_endpoint_create)
    monkeypatch.setattr(
        aiplatform_v1, "EndpointServiceClient", lambda client_options=None: mock_client
    )

    result = deploy_model_to_geap(
        project_id="test-proj",
        location="us-central1",
        model_display_name="gemma-31b",
        model_uri="gs://test-proj-models/gemma-31b",
        machine_type="g4-standard-48",
        container_image_uri="us-docker.pkg.dev/custom/vllm:v1",
        endpoint_display_name="custom-endpoint-display",
        dedicated_endpoint=False,
        routes={"predict": "/v1/chat/completions", "health": "/ready", "port": 8000},
        dry_run=False,
    )

    assert result["status"] == "DEPLOYED"
    assert uploaded_kwargs["serving_container_image_uri"] == "us-docker.pkg.dev/custom/vllm:v1"
    assert uploaded_kwargs["serving_container_invoke_route_prefix"] == "/v1/chat/completions"
    assert uploaded_kwargs["serving_container_health_route"] == "/ready"
    assert uploaded_kwargs["serving_container_ports"] == [8000]
    assert uploaded_kwargs["upload_request_timeout"] == DEFAULT_UPLOAD_REQUEST_TIMEOUT
    assert (
        uploaded_kwargs["serving_container_deployment_timeout"]
        == DEFAULT_CONTAINER_DEPLOYMENT_TIMEOUT
    )
    from google.api_core.future import polling

    assert polling.DEFAULT_POLLING._timeout == DEFAULT_UPLOAD_REQUEST_TIMEOUT
    assert created_endpoint_kwargs["display_name"] == "custom-endpoint-display"
    assert created_endpoint_kwargs["dedicated_endpoint_enabled"] is False
    assert result["endpoint_url"] == "https://us-central1-aiplatform.googleapis.com/v1/projects/123/locations/us-central1/endpoints/789/invoke/v1"


def test_deploy_model_endpoint_url_dedicated(monkeypatch):
    from unittest.mock import MagicMock
    from google.cloud import aiplatform, aiplatform_v1

    fake_model = MagicMock()
    fake_model.resource_name = "projects/123456789012/locations/us-central1/models/456"
    fake_endpoint = MagicMock()
    fake_endpoint.resource_name = "projects/123456789012/locations/us-central1/endpoints/987654321098"
    fake_endpoint.dedicated_endpoint_enabled = True
    fake_endpoint.dedicated_endpoint_dns = "987654321098.us-central1-123456789012.prediction.vertexai.goog"

    mock_client = MagicMock()
    mock_future = MagicMock()
    mock_future.operation.name = "projects/123456789012/locations/us-central1/endpoints/987654321098/operations/111222333444"
    mock_client.deploy_model.return_value = mock_future

    monkeypatch.setattr(aiplatform, "init", lambda **kwargs: None)
    monkeypatch.setattr(aiplatform.Model, "upload", lambda **kwargs: fake_model)
    monkeypatch.setattr(aiplatform.Endpoint, "create", lambda **kwargs: fake_endpoint)
    monkeypatch.setattr(
        aiplatform_v1, "EndpointServiceClient", lambda client_options=None: mock_client
    )

    result = deploy_model_to_geap(
        project_id="123456789012",
        location="us-central1",
        model_display_name="gemma-31b",
        model_uri="gs://test-proj-models/gemma-31b",
        machine_type="g4-standard-48",
        dedicated_endpoint=True,
        dry_run=False,
    )

    assert result["endpoint_url"] == (
        "https://987654321098.us-central1-123456789012.prediction.vertexai.goog"
        "/v1/projects/123456789012/locations/us-central1/endpoints/987654321098/invoke/v1"
    )


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


def test_deploy_command_no_machine_type_option():
    # Verify that --machine-type was removed from options
    result = runner.invoke(app, ["deploy", "--help"])
    assert result.exit_code == 0
    assert "--machine-type" not in result.stdout

    # Passing --machine-type should error as an unrecognized option
    error_result = runner.invoke(
        app,
        ["deploy", "--machine-type", "g4-standard-48", "--dry-run"],
    )
    assert error_result.exit_code != 0
    assert "No such option" in error_result.output
    assert "--machine-type" in error_result.output


def test_deploy_command_missing_machine_type_in_spec_raises_error(monkeypatch):
    import google.models.cli.deploy.cmd_deploy as cmd_deploy_mod

    monkeypatch.setattr(
        cmd_deploy_mod, "ensure_authenticated", lambda interactive: True
    )
    monkeypatch.setattr(
        cmd_deploy_mod,
        "read_deployment_config",
        lambda project_dir=None: DeploymentConfig(machine_type=None),
    )

    result = runner.invoke(app, ["deploy", "--dry-run"])
    assert result.exit_code != 0
    assert "Machine type is required. Set 'machine_type' in config/deployment_spec.yaml" in result.output


def test_deploy_command_reuses_existing_gcs_weights(monkeypatch):
    import google.models.cli.deploy.cmd_deploy as cmd_deploy_mod

    monkeypatch.setattr(
        cmd_deploy_mod, "ensure_authenticated", lambda interactive: True
    )
    monkeypatch.setattr(
        cmd_deploy_mod,
        "read_deployment_config",
        lambda project_dir=None: DeploymentConfig(machine_type="g4-standard-48"),
    )
    monkeypatch.setattr(
        cmd_deploy_mod, "check_bucket_exists", lambda bucket: True
    )
    monkeypatch.setattr(
        cmd_deploy_mod,
        "check_model_exists_in_gcs",
        lambda bucket, repo: True,
    )
    monkeypatch.setattr(cmd_deploy_mod, "write_env", lambda env_vars, **kwargs: None)
    monkeypatch.setattr(cmd_deploy_mod, "write_operation", lambda **kwargs: None)
    monkeypatch.setattr(
        cmd_deploy_mod.settings, "artifact_base_path", None, raising=False
    )
    monkeypatch.setattr(
        cmd_deploy_mod.settings, "service_account_email", None, raising=False
    )
    deploy_mock_called = {}

    def fake_deploy_to_geap(**kwargs):
        deploy_mock_called.update(kwargs)
        return {
            "status": "SUCCESS",
            "endpoint": "projects/123/locations/us-central1/endpoints/456",
        }

    monkeypatch.setattr(cmd_deploy_mod, "deploy_model_to_geap", fake_deploy_to_geap)

    result = runner.invoke(
        app,
        [
            "deploy",
            "--model-id",
            "google/gemma-4-31B-it",
            "--bucket",
            "my-custom-bucket",
            "--project",
            "test-project-123",
            "--region",
            "us-central1",
        ],
    )
    assert result.exit_code == 0
    assert (
        "Found existing weights in GCS: gs://my-custom-bucket/gemma-4-31B-it"
        in result.stdout
    )
    assert deploy_mock_called["model_uri"] == "gs://my-custom-bucket/gemma-4-31B-it"
    assert deploy_mock_called["model_display_name"] == "gemma-4-31b-it"
    assert deploy_mock_called["service_account_email"] is None


def test_deploy_command_with_explicit_service_account(monkeypatch):
    import google.models.cli.deploy.cmd_deploy as cmd_deploy_mod

    monkeypatch.setattr(
        cmd_deploy_mod, "ensure_authenticated", lambda interactive: True
    )
    monkeypatch.setattr(
        cmd_deploy_mod,
        "read_deployment_config",
        lambda project_dir=None: DeploymentConfig(machine_type="g4-standard-48"),
    )
    monkeypatch.setattr(
        cmd_deploy_mod, "check_bucket_exists", lambda bucket: True
    )
    monkeypatch.setattr(
        cmd_deploy_mod,
        "check_model_exists_in_gcs",
        lambda bucket, repo: True,
    )
    monkeypatch.setattr(
        cmd_deploy_mod, "configure_sa_permissions", lambda **kwargs: True
    )
    monkeypatch.setattr(cmd_deploy_mod, "write_env", lambda env_vars, **kwargs: None)
    monkeypatch.setattr(
        cmd_deploy_mod.settings, "artifact_base_path", None, raising=False
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
            "google/gemma-4-31B-it",
            "--bucket",
            "my-custom-bucket",
            "--project",
            "test-project-123",
            "--region",
            "us-central1",
            "--service-account",
            "my-custom-sa@test-project-123.iam.gserviceaccount.com",
        ],
    )
    assert result.exit_code == 0
    assert (
        deploy_mock_called["service_account_email"]
        == "my-custom-sa@test-project-123.iam.gserviceaccount.com"
    )


def test_deploy_command_forwards_spec_yaml_options(monkeypatch):
    import google.models.cli.deploy.cmd_deploy as cmd_deploy_mod

    monkeypatch.setattr(
        cmd_deploy_mod, "ensure_authenticated", lambda interactive: True
    )
    custom_spec = DeploymentConfig(
        machine_type="g4-standard-48",
        display_name="my-custom-endpoint-display",
        container_image_uri="us-docker.pkg.dev/custom/vllm:v2",
        dedicated_endpoint=False,
        shared_memory_mb=65536,
        routes={"predict": "/v1/chat/completions", "health": "/ready", "port": 8000},
    )
    monkeypatch.setattr(
        cmd_deploy_mod,
        "read_deployment_config",
        lambda project_dir=None: custom_spec,
    )
    monkeypatch.setattr(
        cmd_deploy_mod, "check_bucket_exists", lambda bucket: True
    )
    monkeypatch.setattr(
        cmd_deploy_mod,
        "check_model_exists_in_gcs",
        lambda bucket, repo: True,
    )
    monkeypatch.setattr(cmd_deploy_mod, "write_env", lambda env_vars, **kwargs: None)
    monkeypatch.setattr(
        cmd_deploy_mod.settings, "artifact_base_path", None, raising=False
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
            "google/gemma-4-31B-it",
            "--bucket",
            "my-custom-bucket",
            "--project",
            "test-project-123",
            "--region",
            "us-central1",
        ],
    )
    assert result.exit_code == 0
    assert deploy_mock_called["endpoint_display_name"] == "my-custom-endpoint-display"
    assert deploy_mock_called["container_image_uri"] == "us-docker.pkg.dev/custom/vllm:v2"
    assert deploy_mock_called["dedicated_endpoint"] is False
    assert deploy_mock_called["shared_memory_mb"] == 65536
    assert deploy_mock_called["routes"] == {
        "predict": "/v1/chat/completions",
        "health": "/ready",
        "port": 8000,
    }


def test_deploy_model_cleanup_on_failure(monkeypatch):
    import pytest
    from unittest.mock import MagicMock
    from google.cloud import aiplatform, aiplatform_v1

    fake_model = MagicMock()
    fake_model.resource_name = "projects/123/locations/us-central1/models/456"
    fake_endpoint = MagicMock()
    fake_endpoint.resource_name = "projects/123/locations/us-central1/endpoints/789"

    mock_client = MagicMock()
    mock_client.deploy_model.side_effect = RuntimeError(
        "Quota exceeded or deployment failure"
    )

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
            model_display_name="gemma-31b",
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

    status_result = get_operation_status(
        "projects/123/locations/us-central1/endpoints/456/operations/789"
    )
    assert (
        status_result["name"]
        == "projects/123/locations/us-central1/endpoints/456/operations/789"
    )
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
    status_with_stage = get_operation_status(
        "projects/123/locations/us-central1/endpoints/456/operations/789"
    )
    assert status_with_stage["status"] == "PREPARING_MODEL"
    assert status_with_stage["deployment_stage"] == "PREPARING_MODEL"


def test_deploy_status_cli(monkeypatch):
    import google.models.cli.deploy.cmd_deploy as cmd_deploy_mod

    monkeypatch.setattr(
        cmd_deploy_mod, "ensure_authenticated", lambda interactive: True
    )
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
    from google.models.cli.deploy._operation import (
        read_operation,
        write_operation,
    )

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").touch()
    monkeypatch.setattr(
        cmd_deploy_mod, "ensure_authenticated", lambda interactive: True
    )
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
    assert (
        op_data["operation_name"]
        == "projects/123/locations/us-central1/endpoints/456/operations/999"
    )

    # Invoke deploy --status without --operation flag, should read from deployment_metadata.json
    result = runner.invoke(app, ["deploy", "--status"])
    assert result.exit_code == 0
    assert (
        "Checking deployment operation status: projects/123/locations/us-central1/endpoints/456/operations/999"
        in result.stdout
    )
    assert "Deployment Stage: CREATING_SERVING_CLUSTER" in result.stdout
    assert "Deployment Progress: Step 3 of 7" in result.stdout
    assert "Cluster Provisioning" in result.stdout


def test_deploy_status_preserves_operation_when_done(tmp_path, monkeypatch):
    import google.models.cli.deploy.cmd_deploy as cmd_deploy_mod
    from google.models.cli.deploy._operation import (
        write_operation,
        read_operation,
    )

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").touch()
    monkeypatch.setattr(
        cmd_deploy_mod, "ensure_authenticated", lambda interactive: True
    )
    monkeypatch.setattr(
        cmd_deploy_mod,
        "get_operation_status",
        lambda op_name, location=None: {
            "name": op_name,
            "status": "SUCCEEDED",
            "done": True,
            "deployment_stage": "SUCCESSFULLY_DEPLOYED",
            "create_time": "2026-09-24T19:00:00Z",
            "update_time": "2026-09-24T19:15:00Z",
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

    result = runner.invoke(app, ["deploy", "--status"])
    assert result.exit_code == 0
    assert "Deployment Succeeded" in result.stdout

    # Verify operation was not cleared from metadata
    op_data = read_operation(project_dir=tmp_path)
    assert op_data is not None
    assert (
        op_data["operation_name"]
        == "projects/123/locations/us-central1/endpoints/456/operations/999"
    )

    # Calling a second time should still succeed without 'No operation found' error
    result2 = runner.invoke(app, ["deploy", "--status"])
    assert result2.exit_code == 0
    assert "Deployment Succeeded" in result2.stdout


def test_format_elapsed_time():
    from google.models.cli.deploy.deploy_utils import format_elapsed_time

    assert format_elapsed_time(None) is None
    assert format_elapsed_time("invalid-time") is None
    # 5 seconds
    assert (
        format_elapsed_time("2026-09-24T19:00:00Z", "2026-09-24T19:00:05Z")
        == "5s"
    )
    # 5 minutes 23 seconds
    assert (
        format_elapsed_time("2026-09-24T19:00:00Z", "2026-09-24T19:05:23Z")
        == "5m 23s"
    )
    # 2 hours 5 minutes 23 seconds
    assert (
        format_elapsed_time("2026-09-24T19:00:00Z", "2026-09-24T21:05:23Z")
        == "2h 5m 23s"
    )


def test_get_stage_info():
    from google.models.cli.deploy.deploy_utils import get_stage_info

    info = get_stage_info("ADDING_NODES_TO_CLUSTER")
    assert info is not None
    assert info["step"] == 4
    assert info["total_steps"] == 7
    assert info["percent"] == 60
    assert info["title"] == "Hardware Allocation"

    assert get_stage_info(None) is None
    assert get_stage_info("UNKNOWN_STAGE") is None


def test_deploy_status_watch_mode(monkeypatch):
    import google.models.cli.deploy.cmd_deploy as cmd_deploy_mod

    monkeypatch.setattr(
        cmd_deploy_mod, "ensure_authenticated", lambda interactive: True
    )

    calls = []

    def mock_status(op_name, location=None):
        call_num = len(calls)
        calls.append(call_num)
        if call_num == 0:
            return {
                "name": op_name,
                "status": "RUNNING",
                "done": False,
                "deployment_stage": "ADDING_NODES_TO_CLUSTER",
                "create_time": "2026-09-24T19:00:00Z",
                "update_time": "2026-09-24T19:05:00Z",
                "error": None,
                "response": None,
            }
        else:
            return {
                "name": op_name,
                "status": "SUCCEEDED",
                "done": True,
                "deployment_stage": "SUCCESSFULLY_DEPLOYED",
                "create_time": "2026-09-24T19:00:00Z",
                "update_time": "2026-09-24T19:15:00Z",
                "error": None,
                "response": None,
            }

    monkeypatch.setattr(cmd_deploy_mod, "get_operation_status", mock_status)
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)

    result = runner.invoke(
        app,
        [
            "deploy",
            "--status",
            "--watch",
            "--operation",
            "projects/123/locations/us-central1/endpoints/456/operations/789",
        ],
    )
    assert result.exit_code == 0
    assert len(calls) == 2
    assert "Watching deployment operation" in result.stdout
    assert "Deployment Succeeded: 100%" in result.stdout


def test_deploy_status_watch_mode_interrupted(monkeypatch):
    import google.models.cli.deploy.cmd_deploy as cmd_deploy_mod

    monkeypatch.setattr(
        cmd_deploy_mod, "ensure_authenticated", lambda interactive: True
    )

    calls = []

    def mock_status(op_name, location=None):
        calls.append(len(calls))
        return {
            "name": op_name,
            "status": "RUNNING",
            "done": False,
            "deployment_stage": "ADDING_NODES_TO_CLUSTER",
            "create_time": "2026-09-24T19:00:00Z",
            "update_time": "2026-09-24T19:05:00Z",
            "error": None,
            "response": None,
        }

    monkeypatch.setattr(cmd_deploy_mod, "get_operation_status", mock_status)
    import time

    def mock_sleep(s):
        raise KeyboardInterrupt()

    monkeypatch.setattr(time, "sleep", mock_sleep)

    result = runner.invoke(
        app,
        [
            "deploy",
            "--status",
            "--watch",
            "--operation",
            "projects/123/locations/us-central1/endpoints/456/operations/789",
        ],
    )
    assert result.exit_code == 0
    assert (
        "Stopped watching. Deployment is still running in the background."
        in result.stdout
    )


def test_deploy_status_failure_display(monkeypatch):
    import google.models.cli.deploy.cmd_deploy as cmd_deploy_mod

    monkeypatch.setattr(
        cmd_deploy_mod, "ensure_authenticated", lambda interactive: True
    )
    monkeypatch.setattr(
        cmd_deploy_mod,
        "get_operation_status",
        lambda op_name, location=None: {
            "name": op_name,
            "status": "FAILED",
            "done": True,
            "deployment_stage": "ADDING_NODES_TO_CLUSTER",
            "create_time": "2026-09-24T19:00:00Z",
            "update_time": "2026-09-24T19:05:00Z",
            "error": {"code": 8, "message": "Resource quota exceeded"},
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
    assert "Deployment Failed" in result.stdout
    assert "Resource quota exceeded" in result.stdout
    assert "Hardware Allocation (failed)" in result.stdout


def test_copy_hf_model_to_gcs_excludes_cache_files(monkeypatch):
    from pathlib import Path
    from unittest.mock import MagicMock

    from google.cloud import storage
    from google.cloud.storage import transfer_manager
    from google.models.cli.deploy.deploy_utils import copy_hf_model_to_gcs
    import huggingface_hub

    mock_bucket = MagicMock()
    mock_bucket.exists.return_value = True
    mock_bucket.list_blobs.return_value = []

    mock_storage_client = MagicMock()
    mock_storage_client.bucket.return_value = mock_bucket
    monkeypatch.setattr(storage, "Client", lambda: mock_storage_client)

    uploaded_files = []

    def fake_upload_many(
        bucket, filenames, source_directory, blob_name_prefix, max_workers
    ):
        uploaded_files.extend(filenames)
        return [None] * len(filenames)

    monkeypatch.setattr(
        transfer_manager, "upload_many_from_filenames", fake_upload_many
    )

    def fake_snapshot_download(repo_id, local_dir, ignore_patterns, token):
        p = Path(local_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "config.json").write_text("{}")
        (p / "model.safetensors").write_text("data")
        cache_dir = p / ".cache" / "huggingface" / "download"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "model.safetensors.lock").write_text("lock")
        (cache_dir / "model.safetensors.metadata").write_text("meta")
        (p / "temp.cache").write_text("cache")

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", fake_snapshot_download
    )

    gcs_uri = copy_hf_model_to_gcs(
        repo_id="test-org/test-model",
        bucket_name="test-bucket",
        location="us-central1",
    )

    assert gcs_uri == "gs://test-bucket/test-model"
    assert "config.json" in uploaded_files
    assert "model.safetensors" in uploaded_files
    assert not any(".cache" in f for f in uploaded_files)
    assert not any(f.endswith(".cache") for f in uploaded_files)


def test_check_model_exists_in_gcs_safetensors(monkeypatch):
    from unittest.mock import MagicMock
    from google.cloud import storage
    from google.models.cli.deploy.deploy_utils import check_model_exists_in_gcs

    mock_bucket = MagicMock()
    mock_bucket.exists.return_value = True

    blob_json = MagicMock()
    blob_json.name = "my-model/config.json"

    blob_st = MagicMock()
    blob_st.name = "my-model/model.safetensors"

    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    monkeypatch.setattr(storage, "Client", lambda: mock_client)

    # When no .safetensors file exists
    mock_bucket.list_blobs.return_value = [blob_json]
    assert not check_model_exists_in_gcs("test-bucket", "my-model")

    # When a .safetensors file exists
    mock_bucket.list_blobs.return_value = [blob_json, blob_st]
    assert check_model_exists_in_gcs("test-bucket", "my-model")


def test_copy_hf_model_to_gcs_bucket_not_found(monkeypatch):
    from unittest.mock import MagicMock
    from google.cloud import storage
    from google.models.cli.deploy.deploy_utils import copy_hf_model_to_gcs
    import pytest

    mock_bucket = MagicMock()
    mock_bucket.exists.return_value = False

    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    monkeypatch.setattr(storage, "Client", lambda: mock_client)

    with pytest.raises(click.ClickException, match="does not exist or you do not have permission"):
        copy_hf_model_to_gcs(
            repo_id="test/model",
            bucket_name="nonexistent-bucket",
            location="us-central1",
        )


def test_deploy_command_with_base_path_direct_weights(monkeypatch):
    import google.models.cli.deploy.cmd_deploy as cmd_deploy_mod

    monkeypatch.setattr(
        cmd_deploy_mod, "ensure_authenticated", lambda interactive: True
    )
    monkeypatch.setattr(
        cmd_deploy_mod,
        "read_deployment_config",
        lambda project_dir=None: DeploymentConfig(machine_type="g4-standard-48"),
    )
    # Return True for direct folder check
    monkeypatch.setattr(
        cmd_deploy_mod, "check_bucket_exists", lambda bucket: True
    )
    monkeypatch.setattr(
        cmd_deploy_mod,
        "check_model_exists_in_gcs",
        lambda bucket, model_path: model_path == "team-share/custom-weights",
    )
    monkeypatch.setattr(cmd_deploy_mod, "write_env", lambda env_vars, **kwargs: None)
    monkeypatch.setattr(cmd_deploy_mod, "write_operation", lambda **kwargs: None)
    monkeypatch.setattr(
        cmd_deploy_mod.settings, "artifact_base_path", None, raising=False
    )
    monkeypatch.setattr(
        cmd_deploy_mod.settings, "service_account_email", None, raising=False
    )
    deploy_mock_called = {}

    def fake_deploy_to_geap(**kwargs):
        deploy_mock_called.update(kwargs)
        return {"status": "SUCCESS", "endpoint": "ep-1"}

    monkeypatch.setattr(cmd_deploy_mod, "deploy_model_to_geap", fake_deploy_to_geap)

    result = runner.invoke(
        app,
        [
            "deploy",
            "--model-id",
            "google/gemma-4-31B-it",
            "--bucket",
            "my-custom-bucket",
            "--project",
            "test-project-123",
            "--region",
            "us-central1",
            "--base-path",
            "team-share/custom-weights",
        ],
    )
    assert result.exit_code == 0
    assert (
        "Found existing weights in GCS: gs://my-custom-bucket/team-share/custom-weights"
        in result.stdout
    )
    assert (
        deploy_mock_called["model_uri"]
        == "gs://my-custom-bucket/team-share/custom-weights"
    )


def test_deploy_command_with_base_path_subfolder_transfer(monkeypatch):
    import google.models.cli.deploy.cmd_deploy as cmd_deploy_mod

    monkeypatch.setattr(
        cmd_deploy_mod, "ensure_authenticated", lambda interactive: True
    )
    monkeypatch.setattr(
        cmd_deploy_mod,
        "read_deployment_config",
        lambda project_dir=None: DeploymentConfig(machine_type="g4-standard-48"),
    )
    # Return False for all checks so transfer is triggered
    monkeypatch.setattr(
        cmd_deploy_mod, "check_bucket_exists", lambda bucket: True
    )
    monkeypatch.setattr(
        cmd_deploy_mod,
        "check_model_exists_in_gcs",
        lambda bucket, model_path: False,
    )
    monkeypatch.setattr(cmd_deploy_mod, "write_env", lambda env_vars, **kwargs: None)
    monkeypatch.setattr(cmd_deploy_mod, "write_operation", lambda **kwargs: None)
    monkeypatch.setattr(
        cmd_deploy_mod.settings, "artifact_base_path", None, raising=False
    )
    monkeypatch.setattr(
        cmd_deploy_mod.settings, "service_account_email", None, raising=False
    )

    transfer_called = {}

    def fake_copy(repo_id, bucket_name, location, destination_path=None, hf_token=None):
        transfer_called["destination_path"] = destination_path
        transfer_called["repo_id"] = repo_id
        return f"gs://{bucket_name}/{destination_path}"

    monkeypatch.setattr(cmd_deploy_mod, "copy_hf_model_to_gcs", fake_copy)

    deploy_mock_called = {}

    def fake_deploy_to_geap(**kwargs):
        deploy_mock_called.update(kwargs)
        return {"status": "SUCCESS", "endpoint": "ep-1"}

    monkeypatch.setattr(cmd_deploy_mod, "deploy_model_to_geap", fake_deploy_to_geap)

    result = runner.invoke(
        app,
        [
            "deploy",
            "--model-id",
            "google/gemma-4-31B-it",
            "--bucket",
            "my-custom-bucket",
            "--project",
            "test-project-123",
            "--region",
            "us-central1",
            "--base-path",
            "team-share",
        ],
    )
    assert result.exit_code == 0
    assert transfer_called["destination_path"] == "team-share/gemma-4-31B-it"
    assert transfer_called["repo_id"] == "google/gemma-4-31B-it"
    assert (
        deploy_mock_called["model_uri"]
        == "gs://my-custom-bucket/team-share/gemma-4-31B-it"
    )


def test_deploy_command_bucket_not_found_fails_early_even_on_dry_run(monkeypatch):
    import google.models.cli.deploy.cmd_deploy as cmd_deploy_mod

    monkeypatch.setattr(
        cmd_deploy_mod, "ensure_authenticated", lambda interactive: True
    )
    monkeypatch.setattr(
        cmd_deploy_mod,
        "read_deployment_config",
        lambda project_dir=None: DeploymentConfig(machine_type="g4-standard-48"),
    )
    monkeypatch.setattr(
        cmd_deploy_mod, "check_bucket_exists", lambda bucket: False
    )

    result = runner.invoke(
        app,
        [
            "deploy",
            "--model-id",
            "google/gemma-4-31B-it",
            "--bucket",
            "missing-bucket",
            "--project",
            "test-project-123",
            "--region",
            "us-central1",
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    assert "GCS bucket 'missing-bucket' does not exist or you do not have permission" in result.output


def test_deploy_command_dry_run_requires_authentication(monkeypatch):
    import google.models.cli.deploy.cmd_deploy as cmd_deploy_mod

    auth_checked = []

    def fake_ensure_authenticated(interactive=True):
        auth_checked.append(interactive)
        return False

    monkeypatch.setattr(
        cmd_deploy_mod, "ensure_authenticated", fake_ensure_authenticated
    )

    result = runner.invoke(
        app,
        [
            "deploy",
            "--model-id",
            "google/gemma-4-31B-it",
            "--bucket",
            "my-bucket",
            "--project",
            "test-project-123",
            "--region",
            "us-central1",
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    assert auth_checked == [True]


def test_manifest_omits_unconfigured_options():
    """Verify that unconfigured optional fields (service_account, routes, shared_memory_mb) are omitted from manifest."""
    result = deploy_model_to_geap(
        project_id="test-proj",
        location="us-central1",
        model_display_name="gemma-31b",
        model_uri="gs://test-proj-models/gemma-4-31B-it",
        machine_type="ct6e-standard-4t",
        engine=InferenceEngine.VLLM,
        dry_run=True,
    )
    manifest = result["manifest"]
    assert "service_account" not in manifest
    assert "routes" not in manifest
    assert "shared_memory_mb" not in manifest
    assert "dry_run" not in manifest
    assert manifest["machine_type"] == "ct6e-standard-4t"
    assert manifest["dedicated_endpoint_enabled"] is True


def test_container_args_omits_unconfigured_engine_options():
    """Verify that container_args only includes explicitly configured options and required server flags."""
    from google.models.cli._project import VLLMEngineConfig
    cfg = VLLMEngineConfig.from_dict({
        "engine": "vllm",
        "tensor_parallel_size": 2,
    })
    args = build_container_args(cfg)
    assert "--tensor-parallel-size=2" in args
    assert "--host=0.0.0.0" in args
    assert "--port=8080" in args
    assert "--trust-remote-code" in args
    # Verify unconfigured parameters are NOT injected
    assert not any(arg.startswith("--max-model-len") for arg in args)
    assert not any(arg.startswith("--kv-cache-dtype") for arg in args)
    assert not any(arg.startswith("--gpu-memory-utilization") for arg in args)
    assert not any(arg.startswith("--max-num-batched-tokens") for arg in args)
    assert not any(arg.startswith("--pipeline-parallel-size") for arg in args)
    assert "--enable-prefix-caching" not in args
    assert "--enable-chunked-prefill" not in args


