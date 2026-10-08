import getpass
import sys
from pathlib import Path
import pytest
import yaml
from click.testing import CliRunner
from google.models.cli.main import app
from google.models.cli.scaffold.scaffold_utils import (
    copy_and_render_templates,
    generate_env_from_example,
    normalize_project_name,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def mock_getpass_stdin(monkeypatch):
    monkeypatch.setattr(
        getpass,
        "getpass",
        lambda prompt="", stream=None: sys.stdin.readline().rstrip("\n"),
    )


def test_normalize_project_name():
    assert normalize_project_name("My_Model_Project") == "my-model-project"
    assert normalize_project_name("Test Project  ") == "test-project"


def test_copy_and_render_templates(tmp_path):
    dest = tmp_path / "test-model-service"
    context = {
        "project_name": "test-model-service",
        "model_id": "google/gemma-4-31B-it",
        "engine": "vllm",
        "machine_type": "",
        "region": "us-central1",
        "project_id": "test-gcp-proj",
        "gcs_bucket": "test-gcp-proj-models",
        "service_account_email": "sa@test-gcp-proj.iam.gserviceaccount.com",
        "hf_token": "",
        "endpoint_url": "",
    }
    copy_and_render_templates(dest, template_name="vllm", context=context)

    # Both .env and .env.example should be created
    assert (dest / ".env").is_file()
    assert (dest / ".env.example").is_file()
    assert (dest / "README.md").is_file()
    assert (dest / "pyproject.toml").is_file()
    assert (dest / "config" / "engine_config.yaml").is_file()
    assert (dest / "config" / "deployment_spec.yaml").is_file()
    assert (dest / "tests" / "eval" / "golden_dataset.jsonl").is_file()
    assert (dest / "tests" / "benchmark" / "config.yaml").is_file()
    assert (dest / "reports").is_dir()

    # Check benchmark config server.model_name is null and tokenizer is populated with model_id
    bench_data = yaml.safe_load((dest / "tests" / "benchmark" / "config.yaml").read_text(encoding="utf-8"))
    assert bench_data["server"]["model_name"] is None
    assert bench_data["tokenizer"]["pretrained_model_name_or_path"] == "google/gemma-4-31B-it"

    # Check content substitution in .env (populated with actual values)
    env_content = (dest / ".env").read_text(encoding="utf-8")
    assert 'MODEL_ID="google/gemma-4-31B-it"' in env_content
    assert 'GOOGLE_CLOUD_PROJECT="test-gcp-proj"' in env_content
    assert "AUTH_TOKEN" not in env_content

    # Check .env.example has template placeholders
    env_example_content = (dest / ".env.example").read_text(encoding="utf-8")
    assert "GOOGLE_CLOUD_PROJECT=your-gcp-project-id" in env_example_content
    assert "MODEL_ID=your-model-id" in env_example_content
    assert "AUTH_TOKEN" not in env_example_content

    # Check content substitution in README
    readme_content = (dest / "README.md").read_text(encoding="utf-8")
    assert "google/gemma-4-31B-it" in readme_content
    assert "test-model-service" in readme_content


def test_env_and_env_example_alignment(tmp_path):
    dest = tmp_path / "alignment-test"
    context = {
        "project_id": "proj-1",
        "region": "us-central1",
        "gcs_bucket": "bkt-1",
        "service_account_email": "sa@proj-1.iam.gserviceaccount.com",
        "model_id": "google/gemma-4-31B-it",
    }
    copy_and_render_templates(dest, template_name="vllm", context=context)

    # Extract keys from .env.example
    example_keys = [
        line.split("=")[0].strip()
        for line in (dest / ".env.example").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#") and "=" in line
    ]

    # Extract keys from .env
    env_keys = [
        line.split("=")[0].strip()
        for line in (dest / ".env").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#") and "=" in line
    ]

    # Key order and names must be 100% identical and exclude AUTH_TOKEN
    assert example_keys == env_keys
    assert "AUTH_TOKEN" not in env_keys


def test_create_command(tmp_path):
    target_name = "test-created-app"
    result = runner.invoke(
        app,
        [
            "create",
            target_name,
            "--output-dir",
            str(tmp_path),
            "--auto-approve",
            "--skip-checks",
            "--model",
            "google/gemma-4-31B-it",
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / target_name / ".env").is_file()
    assert (tmp_path / target_name / ".env.example").is_file()
    assert (tmp_path / target_name / "pyproject.toml").is_file()
    assert (tmp_path / target_name / "config" / "engine_config.yaml").is_file()


def test_interactive_create_custom_project(tmp_path, monkeypatch):
    target_name = "interactive-app-custom"
    monkeypatch.setattr(
        "google.models.cli.scaffold.cmd_scaffold.resolve_gcp_project",
        lambda override_project=None: "active-gcp-proj",
    )

    # Inputs: model, engine, region, custom project ID, bucket, base_path, service_account, hf_token, endpoint_url
    user_inputs = (
        "\n".join(
            [
                "google/gemma-4-31B-it",  # model
                "vllm",  # engine
                "us-central1",  # region
                "my-desired-project",  # type desired project
                "my-custom-bucket",  # bucket
                "models/weights",  # base_path
                "custom-sa@my-desired-project.iam.gserviceaccount.com",  # service_account
                "hf_my_secret_token",  # hf_token
                "https://my-endpoint.domain.com",  # endpoint_url
            ]
        )
        + "\n"
    )

    result = runner.invoke(
        app,
        [
            "create",
            target_name,
            "-i",
            "--output-dir",
            str(tmp_path),
        ],
        input=user_inputs,
    )
    assert result.exit_code == 0
    assert "What is your GCP project ID?" in result.output
    assert "What is your GCS bucket for model artifacts?" in result.output
    assert "What is your base path in GCS bucket" in result.output
    assert "What is your service account email" in result.output
    assert "What is your Hugging Face token" in result.output
    assert "What is your deployed endpoint URL" in result.output
    assert "hf_my_secret_token" not in result.output
    env_content = (tmp_path / target_name / ".env").read_text()
    assert 'GOOGLE_CLOUD_PROJECT="my-desired-project"' in env_content
    assert 'GOOGLE_CLOUD_LOCATION="us-central1"' in env_content
    assert 'GOOGLE_CLOUD_STORAGE_BUCKET="my-custom-bucket"' in env_content
    assert 'GOOGLE_CLOUD_STORAGE_BUCKET_BASE_PATH="models/weights"' in env_content
    assert (
        'SERVICE_ACCOUNT_EMAIL="custom-sa@my-desired-project.iam.gserviceaccount.com"'
        in env_content
    )
    assert 'MODEL_ID="google/gemma-4-31B-it"' in env_content
    assert 'HF_TOKEN="hf_my_secret_token"' in env_content
    assert 'ENDPOINT_URL="https://my-endpoint.domain.com"' in env_content


def test_interactive_create_accept_default_project(tmp_path, monkeypatch):
    target_name = "interactive-app-default"
    monkeypatch.setattr(
        "google.models.cli.scaffold.cmd_scaffold.resolve_gcp_project",
        lambda override_project=None: "active-gcloud-config-proj",
    )

    # Inputs: model, engine, region, default project, bucket (required, no default), default base_path, default sa, default hf_token, default endpoint_url
    user_inputs = (
        "\n".join(
            [
                "google/gemma-4-31B-it",  # model
                "vllm",  # engine
                "us-central1",  # region
                "",  # empty input -> accepts default active-gcloud-config-proj
                "active-custom-bucket",  # compulsory bucket entered by user
                "",  # empty input -> accepts default "" base_path
                "",  # empty input -> accepts default "" sa
                "",  # empty input -> accepts default "" hf_token
                "",  # empty input -> accepts default "" endpoint_url
            ]
        )
        + "\n"
    )

    result = runner.invoke(
        app,
        [
            "create",
            target_name,
            "-i",
            "--output-dir",
            str(tmp_path),
        ],
        input=user_inputs,
    )
    assert result.exit_code == 0
    assert "What is your GCP project ID?" in result.output
    assert "What is your GCS bucket for model artifacts?" in result.output
    assert "active-gcloud-config-proj-models" not in result.output
    assert "What is your base path in GCS bucket" in result.output
    assert "What is your service account email" in result.output
    assert "What is your Hugging Face token" in result.output
    assert "What is your deployed endpoint URL" in result.output
    env_content = (tmp_path / target_name / ".env").read_text()
    assert 'GOOGLE_CLOUD_PROJECT="active-gcloud-config-proj"' in env_content
    assert 'GOOGLE_CLOUD_STORAGE_BUCKET="active-custom-bucket"' in env_content
    assert 'GOOGLE_CLOUD_STORAGE_BUCKET_BASE_PATH=""' in env_content
    assert 'SERVICE_ACCOUNT_EMAIL=""' in env_content
    assert 'HF_TOKEN=""' in env_content
    assert 'ENDPOINT_URL=""' in env_content


def test_interactive_create_when_gcloud_config_empty_requires_project(
    tmp_path, monkeypatch
):
    target_name = "empty-gcloud-required"
    monkeypatch.setattr(
        "google.models.cli.scaffold.cmd_scaffold.resolve_gcp_project",
        lambda override_project=None: None,
    )

    # Inputs: model, engine, region, empty project (rejected), typed project, compulsory bucket, default base_path, default sa, default hf_token, default endpoint
    user_inputs = (
        "\n".join(
            [
                "google/gemma-4-31B-it",  # model
                "vllm",  # engine
                "us-central1",  # region
                "",  # empty input -> rejected!
                "my-entered-project",  # user provides real project
                "placeholder-bucket",  # compulsory bucket
                "",  # empty input -> default "" base_path
                "",  # empty input -> default "" sa
                "",  # empty input -> default "" hf_token
                "",  # empty input -> default "" endpoint
            ]
        )
        + "\n"
    )

    result = runner.invoke(
        app,
        [
            "create",
            target_name,
            "-i",
            "--output-dir",
            str(tmp_path),
        ],
        input=user_inputs,
    )
    assert result.exit_code == 0
    assert "GCP project ID is required" in result.output
    env_content = (tmp_path / target_name / ".env").read_text()
    assert 'GOOGLE_CLOUD_PROJECT="my-entered-project"' in env_content
    assert 'GOOGLE_CLOUD_STORAGE_BUCKET="placeholder-bucket"' in env_content
    assert 'GOOGLE_CLOUD_STORAGE_BUCKET_BASE_PATH=""' in env_content
    assert 'HF_TOKEN=""' in env_content
    assert 'ENDPOINT_URL=""' in env_content


def test_interactive_create_when_gcloud_config_empty_user_types_project(
    tmp_path, monkeypatch
):
    target_name = "empty-gcloud-custom"
    monkeypatch.setattr(
        "google.models.cli.scaffold.cmd_scaffold.resolve_gcp_project",
        lambda override_project=None: None,
    )

    # Inputs: model, engine, region, type custom project, compulsory bucket, default base_path, default sa, default hf_token, default endpoint
    user_inputs = (
        "\n".join(
            [
                "google/gemma-4-31B-it",  # model
                "vllm",  # engine
                "us-central1",  # region
                "typed-custom-project",  # project
                "typed-custom-models-bucket",  # compulsory bucket
                "",  # default base_path
                "",  # default sa
                "",  # default hf_token
                "",  # default endpoint
            ]
        )
        + "\n"
    )

    result = runner.invoke(
        app,
        [
            "create",
            target_name,
            "-i",
            "--output-dir",
            str(tmp_path),
        ],
        input=user_inputs,
    )
    assert result.exit_code == 0
    assert "typed-custom-project-models" not in result.output
    env_content = (tmp_path / target_name / ".env").read_text()
    assert 'GOOGLE_CLOUD_PROJECT="typed-custom-project"' in env_content
    assert 'GOOGLE_CLOUD_STORAGE_BUCKET="typed-custom-models-bucket"' in env_content
    assert 'GOOGLE_CLOUD_STORAGE_BUCKET_BASE_PATH=""' in env_content
    assert 'HF_TOKEN=""' in env_content
    assert 'ENDPOINT_URL=""' in env_content


def test_interactive_create_bucket_compulsory_reprompts(tmp_path, monkeypatch):
    target_name = "interactive-bucket-compulsory"
    monkeypatch.setattr(
        "google.models.cli.scaffold.cmd_scaffold.resolve_gcp_project",
        lambda override_project=None: "my-test-gcp-proj",
    )

    # Inputs:
    # model, engine, region, project,
    # empty bucket (should be rejected), empty bucket (rejected again), valid bucket gs://actual-bucket/
    # base_path, sa, hf_token, endpoint
    user_inputs = (
        "\n".join(
            [
                "google/gemma-4-31B-it",  # model
                "vllm",  # engine
                "us-central1",  # region
                "",  # default project
                "",  # empty bucket -> rejected!
                "   ",  # whitespace bucket -> rejected!
                "gs://actual-bucket/",  # valid bucket with gs:// prefix & trailing slash
                "",  # default base_path
                "",  # default sa
                "",  # default hf_token
                "",  # default endpoint
            ]
        )
        + "\n"
    )

    result = runner.invoke(
        app,
        [
            "create",
            target_name,
            "-i",
            "--output-dir",
            str(tmp_path),
        ],
        input=user_inputs,
    )
    assert result.exit_code == 0
    assert "GCS bucket name is required." in result.output
    assert "my-test-gcp-proj-models" not in result.output
    env_content = (tmp_path / target_name / ".env").read_text()
    assert 'GOOGLE_CLOUD_STORAGE_BUCKET="actual-bucket"' in env_content


def test_create_with_all_env_options_flags(tmp_path):
    target_name = "test-flags-app"
    result = runner.invoke(
        app,
        [
            "create",
            target_name,
            "--output-dir",
            str(tmp_path),
            "--auto-approve",
            "--skip-checks",
            "--model",
            "google/gemma-4-31B-it",
            "--project",
            "flag-project-123",
            "--region",
            "us-east4",
            "--bucket",
            "flag-bucket",
            "--base-path",
            "weights/gemma",
            "--service-account",
            "sa@flag-project-123.iam.gserviceaccount.com",
            "--hf-token",
            "hf_flag_token",
            "--endpoint-url",
            "https://flag-endpoint.internal",
        ],
    )
    assert result.exit_code == 0
    env_content = (tmp_path / target_name / ".env").read_text()
    assert 'GOOGLE_CLOUD_PROJECT="flag-project-123"' in env_content
    assert 'GOOGLE_CLOUD_LOCATION="us-east4"' in env_content
    assert 'GOOGLE_CLOUD_STORAGE_BUCKET="flag-bucket"' in env_content
    assert 'GOOGLE_CLOUD_STORAGE_BUCKET_BASE_PATH="weights/gemma"' in env_content
    assert (
        'SERVICE_ACCOUNT_EMAIL="sa@flag-project-123.iam.gserviceaccount.com"'
        in env_content
    )
    assert 'MODEL_ID="google/gemma-4-31B-it"' in env_content
    assert 'HF_TOKEN="hf_flag_token"' in env_content
    assert 'ENDPOINT_URL="https://flag-endpoint.internal"' in env_content


def test_interactive_create_existing_directory_reprompts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "google.models.cli.scaffold.cmd_scaffold.resolve_gcp_project",
        lambda override_project=None: "my-valid-proj",
    )

    # Pre-create a colliding non-empty directory
    colliding_dir = tmp_path / "colliding-app"
    colliding_dir.mkdir(parents=True)
    (colliding_dir / "existing_file.txt").write_text("already here")

    # Inputs:
    # 1. colliding name (rejected)
    # 2. empty name (rejected)
    # 3. fresh valid name (accepted)
    # 4. model, engine, region, project, bucket, base_path, sa, hf_token, endpoint
    user_inputs = (
        "\n".join(
            [
                "colliding-app",
                "fresh-valid-app",
                "google/gemma-4-31B-it",
                "vllm",
                "us-central1",
                "",  # default project
                "my-test-bucket",
                "",  # default base_path
                "",  # default sa
                "",  # default hf_token
                "",  # default endpoint
            ]
        )
        + "\n"
    )

    result = runner.invoke(
        app,
        [
            "create",
            "-i",
            "--skip-checks",
            "--output-dir",
            str(tmp_path),
        ],
        input=user_inputs,
    )
    assert result.exit_code == 0
    assert "already exists and is not empty" in result.output
    assert (tmp_path / "fresh-valid-app" / ".env").is_file()


def test_interactive_create_skips_prompts_for_cli_provided_flags(tmp_path):
    # Pass project_name, project_id (-p), bucket, model, region via CLI flags with -i.
    # The prompts for project_name, model, engine, region, project_id, and bucket should NOT be shown or ask for input.
    # Only optional unprovided params (base_path, sa, hf_token, endpoint) should prompt.
    user_inputs = (
        "\n".join(
            [
                "vllm",  # engine (was not passed via flag)
                "custom-base-path",  # base_path
                "custom-sa@my-proj.iam.gserviceaccount.com",  # sa
                "my-hf-token",  # hf_token
                "https://my-endpoint.internal",  # endpoint
            ]
        )
        + "\n"
    )

    result = runner.invoke(
        app,
        [
            "create",
            "cli-provided-app",
            "-i",
            "--skip-checks",
            "--project",
            "cli-flag-project",
            "--bucket",
            "cli-flag-bucket",
            "--model",
            "google/gemma-4-26B-it",
            "--region",
            "europe-west1",
            "--output-dir",
            str(tmp_path),
        ],
        input=user_inputs,
    )

    assert result.exit_code == 0
    # Prompts for CLI-provided options must not appear
    assert "What is your project name?" not in result.output
    assert "What is your model ID" not in result.output
    assert "What is your GCP region?" not in result.output
    assert "What is your GCP project ID?" not in result.output
    assert "What is your GCS bucket" not in result.output

    # Prompts for non-provided options should appear
    assert "What is your inference engine" in result.output
    assert "What is your base path in GCS bucket" in result.output
    assert "What is your service account email" in result.output
    assert "What is your Hugging Face token" in result.output
    assert "What is your deployed endpoint URL" in result.output

    env_content = (tmp_path / "cli-provided-app" / ".env").read_text()
    assert 'GOOGLE_CLOUD_PROJECT="cli-flag-project"' in env_content
    assert 'GOOGLE_CLOUD_STORAGE_BUCKET="cli-flag-bucket"' in env_content
    assert 'GOOGLE_CLOUD_LOCATION="europe-west1"' in env_content
    assert 'MODEL_ID="google/gemma-4-26B-it"' in env_content
    assert 'GOOGLE_CLOUD_STORAGE_BUCKET_BASE_PATH="custom-base-path"' in env_content
    assert 'SERVICE_ACCOUNT_EMAIL="custom-sa@my-proj.iam.gserviceaccount.com"' in env_content
    assert 'HF_TOKEN="my-hf-token"' in env_content
    assert 'ENDPOINT_URL="https://my-endpoint.internal"' in env_content

