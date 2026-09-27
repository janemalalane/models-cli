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

"""Scaffold / create command for models-cli."""

from pathlib import Path
from typing import Optional
import click
from rich.console import Console
from rich.prompt import Prompt

from google.models.cli._gcp_project import resolve_gcp_project
from google.models.cli.common.constants import (
    DEFAULT_ENGINE,
    DEFAULT_MODEL_REPO,
    DEFAULT_REGION,
    InferenceEngine,
)
from google.models.cli.scaffold.scaffold_utils import (
    copy_and_render_templates,
    normalize_project_name,
)

console = Console()


@click.command("create")
@click.argument("project_name", required=False, default=None)
@click.option(
    "--template",
    "-t",
    default=DEFAULT_ENGINE,
    help="Serving engine template: 'vllm' or 'sglang'.",
)
@click.option(
    "--model",
    "-m",
    default=DEFAULT_MODEL_REPO,
    help="Target Hugging Face model repository.",
)
@click.option(
    "--region",
    "-r",
    default=DEFAULT_REGION,
    help="GCP region for deployment (e.g. us-central1, us-east1).",
)
@click.option(
    "--project",
    "-p",
    default=None,
    help="GCP project ID (GOOGLE_CLOUD_PROJECT). Defaults to active gcloud / ADC project.",
)
@click.option(
    "--bucket",
    "-b",
    default=None,
    help="GCS bucket name for model artifacts (GOOGLE_CLOUD_STORAGE_BUCKET).",
)
@click.option(
    "--base-path",
    "--artifact-base-path",
    default=None,
    help="Base path or prefix inside GCS bucket for model artifacts (GOOGLE_CLOUD_STORAGE_BUCKET_BASE_PATH).",
)
@click.option(
    "--service-account",
    "--service-account-email",
    default=None,
    help="GCP Service Account email (SERVICE_ACCOUNT_EMAIL).",
)
@click.option(
    "--hf-token",
    default=None,
    help="Hugging Face access token (HF_TOKEN).",
)
@click.option(
    "--endpoint-url",
    default=None,
    help="Deployed endpoint URL (ENDPOINT_URL).",
)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    default=False,
    help="Enable interactive prompts.",
)
@click.option(
    "--auto-approve",
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip interactive prompts and use defaults.",
)
@click.option(
    "--skip-checks",
    "-s",
    is_flag=True,
    default=False,
    help="Skip GCP and Gemini Enterprise verification checks.",
)
@click.option(
    "--output-dir",
    "-o",
    default=None,
    help="Parent directory where the project directory will be created.",
)
def create_project(
    project_name: Optional[str] = None,
    template: str = DEFAULT_ENGINE,
    model: str = DEFAULT_MODEL_REPO,
    region: str = DEFAULT_REGION,
    project: Optional[str] = None,
    bucket: Optional[str] = None,
    base_path: Optional[str] = None,
    service_account: Optional[str] = None,
    hf_token: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    interactive: bool = False,
    auto_approve: bool = False,
    skip_checks: bool = False,
    output_dir: Optional[str] = None,
) -> None:
    """Creates an open model serving, deployment, and benchmarking project workspace."""
    is_interactive = interactive or (not auto_approve and project_name is None)

    # 1. Project Name
    if not project_name:
        if is_interactive:
            project_name = Prompt.ask(
                "\n> What is your project name?",
                default="my-model-service",
            )
        else:
            project_name = "my-model-service"

    project_name = normalize_project_name(project_name)
    target_parent = Path(output_dir).resolve() if output_dir else Path.cwd()
    project_dir = target_parent / project_name

    if project_dir.exists() and any(project_dir.iterdir()):
        console.print(
            f"[bold red]❌ Error:[/bold red] Target directory '{project_dir}' already exists and is not empty."
        )
        raise click.exceptions.Exit(1)

    # 2. Model Repo
    if is_interactive and not auto_approve:
        model = Prompt.ask(
            "> 📦 What is your model ID (Hugging Face repository)?",
            default=model or DEFAULT_MODEL_REPO,
        )

    # 3. Serving Engine
    if is_interactive and not auto_approve:
        template = Prompt.ask(
            "> ⚡ What is your inference engine [vllm / sglang]?",
            default=template or DEFAULT_ENGINE,
        )
    engine_val = template.lower().strip()
    if engine_val not in ("vllm", "sglang"):
        console.print(
            f"[bold yellow]⚠️ Unknown engine '{template}', falling back to 'vllm'[/bold yellow]"
        )
        engine_val = "vllm"

    # 4. GCP Region
    if is_interactive and not auto_approve:
        default_region = region or DEFAULT_REGION
        region = ""
        while not region:
            region = Prompt.ask(
                "> 🌍 What is your GCP region?",
                default=default_region,
            ).strip()
            if not region:
                console.print("[bold red]❌ GCP region is required.[/bold red]")

    # 5. GCP Project ID Selection
    default_project = (
        project or resolve_gcp_project(override_project=project) or "YOUR_GCP_PROJECT_ID"
    )
    if is_interactive and not auto_approve:
        resolved_project = ""
        while not resolved_project:
            resolved_project = Prompt.ask(
                "> 📁 What is your GCP project ID?",
                default=default_project,
            ).strip()
            if not resolved_project:
                console.print("[bold red]❌ GCP project ID is required.[/bold red]")
    else:
        resolved_project = default_project

    # 6. GCS Bucket Name
    if is_interactive and not auto_approve:
        gcs_bucket = (bucket or "").strip()
        if gcs_bucket.startswith("gs://"):
            gcs_bucket = gcs_bucket[len("gs://") :]
        gcs_bucket = gcs_bucket.strip("/")

        while not gcs_bucket:
            if bucket:
                raw_bucket = Prompt.ask(
                    "> 🪣 What is your GCS bucket for model artifacts?",
                    default=bucket,
                )
            else:
                raw_bucket = Prompt.ask(
                    "> 🪣 What is your GCS bucket for model artifacts?",
                )
            raw_bucket = raw_bucket.strip()
            if raw_bucket.startswith("gs://"):
                raw_bucket = raw_bucket[len("gs://") :]
            gcs_bucket = raw_bucket.strip("/")
            if not gcs_bucket:
                console.print(
                    "[bold red]❌ GCS bucket name is required.[/bold red]"
                )
    else:
        raw_bucket = (bucket or "").strip()
        if raw_bucket.startswith("gs://"):
            raw_bucket = raw_bucket[len("gs://") :]
        gcs_bucket = raw_bucket.strip("/")

    # 7. Artifact Base Path
    default_base_path = base_path or ""
    if is_interactive and not auto_approve:
        resolved_base_path = Prompt.ask(
            "> 📂 What is your base path in GCS bucket (optional)?",
            default=default_base_path,
        ).strip()
    else:
        resolved_base_path = default_base_path
    resolved_base_path = (
        resolved_base_path.strip().strip("/") if resolved_base_path else ""
    )

    # 8. Service Account Email
    default_sa = service_account or ""
    if is_interactive and not auto_approve:
        sa_email = Prompt.ask(
            "> 👤 What is your service account email (optional)?",
            default=default_sa,
        ).strip()
    else:
        sa_email = default_sa

    # 9. Hugging Face Token
    default_hf = hf_token or ""
    if is_interactive and not auto_approve:
        resolved_hf_token = Prompt.ask(
            "> 🔑 What is your Hugging Face token (optional)?",
            default=default_hf,
            password=True,
        ).strip()
    else:
        resolved_hf_token = default_hf

    # 10. Deployed Endpoint URL
    default_endpoint = endpoint_url or ""
    if is_interactive and not auto_approve:
        resolved_endpoint_url = Prompt.ask(
            "> 🌐 What is your deployed endpoint URL (optional)?",
            default=default_endpoint,
        ).strip()
    else:
        resolved_endpoint_url = default_endpoint

    context = {
        "project_name": project_name,
        "model_id": model,
        "engine": engine_val,
        "machine_type": "",
        "region": region,
        "project_id": resolved_project,
        "gcs_bucket": gcs_bucket,
        "base_path": resolved_base_path,
        "service_account_email": sa_email,
        "hf_token": resolved_hf_token,
        "endpoint_url": resolved_endpoint_url,
        "auth_token": "",
    }

    copy_and_render_templates(
        destination=project_dir,
        template_name=engine_val,
        context=context,
    )

    # 7. Success Banner
    console.print(
        f"\n[bold green]✅ Success![/bold green] Your open model project [bold cyan]'{project_name}'[/bold cyan] is ready.\n"
    )
    console.print("[bold]📖 Documentation[/bold]")
    console.print(f"   README:    cat {project_name}/README.md\n")
    console.print("[bold]🚀 Get Started[/bold]")
    console.print(f"   cd {project_name}")
    console.print(f"   models-cli recommend --model {model}")
    console.print("   models-cli deploy --dry-run\n")


scaffold_cmd = create_project
