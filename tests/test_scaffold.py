# Copyright 2026 Google LLC
from pathlib import Path
from typer.testing import CliRunner
from google.models.cli.main import app
from google.models.cli.scaffold.scaffold_utils import (
    copy_and_render_templates,
    generate_env_from_example,
    normalize_project_name,
)

runner = CliRunner()


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
