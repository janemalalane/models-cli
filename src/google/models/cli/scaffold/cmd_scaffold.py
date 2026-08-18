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
import typer
from rich.console import Console
from rich.prompt import Prompt

from google.models.cli._gcp_project import get_active_gcp_account, resolve_gcp_project
from google.models.cli.common.constants import (
    DEFAULT_ENGINE,
    DEFAULT_MODEL_REPO,
    DEFAULT_REGION,
    InferenceEngine,
)
from google.models.cli.scaffold.scaffold_utils import (
    copy_and_render_templates,
    normalize_project_name,
    verify_credentials_and_vertex,
)

console = Console()
scaffold_cmd = typer.Typer(
    name="create",
    help="Scaffold a new open-weights model serving & benchmarking workspace",
)


@scaffold_cmd.callback(invoke_without_command=True)
def create_project(
    project_name: Optional[str] = typer.Argument(
        None,
        help="Name of the project directory to create.",
    ),
    template: str = typer.Option(
        DEFAULT_ENGINE.value,
        "--template",
        "-t",
        help="Serving engine template: 'vllm' or 'sglang'.",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL_REPO,
        "--model",
        "-m",
        help="Target Hugging Face model repository.",
    ),
    region: str = typer.Option(
        DEFAULT_REGION,
        "--region",
        "-r",
        help="GCP region for deployment (e.g. us-central1, us-east1).",
    ),
    project: Optional[str] = typer.Option(
        None,
        "--project",
        "-p",
        help="GCP project ID. Defaults to active gcloud / ADC project.",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Enable interactive prompts.",
    ),
    auto_approve: bool = typer.Option(
        False,
        "--auto-approve",
        "--yes",
        "-y",
        help="Skip interactive prompts and use defaults.",
    ),
    skip_checks: bool = typer.Option(
        False,
        "--skip-checks",
        "-s",
        help="Skip GCP and Vertex AI verification checks.",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Parent directory where the project directory will be created.",
    ),
) -> None:
    """Creates an open model serving, deployment, and benchmarking project workspace."""
    is_interactive = interactive or (not auto_approve and project_name is None)

    # 1. Project Name
    if not project_name:
        if is_interactive:
            project_name = Prompt.ask(
                "\n> Enter project name",
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
        raise typer.Exit(1)

    # 2. Model Repo
    if is_interactive and not auto_approve:
        model = Prompt.ask(
            "> 📦 Enter Hugging Face model repository",
            default=model or DEFAULT_MODEL_REPO,
        )

    # 3. Serving Engine
    if is_interactive and not auto_approve:
        template = Prompt.ask(
            "> ⚡ Select inference engine [vllm / sglang]",
            default=template or DEFAULT_ENGINE.value,
        )
    engine_val = template.lower().strip()
    if engine_val not in ("vllm", "sglang"):
        console.print(f"[bold yellow]⚠️ Unknown engine '{template}', falling back to 'vllm'[/bold yellow]")
        engine_val = "vllm"

    # 4. GCP Region
    if is_interactive and not auto_approve:
        region = Prompt.ask(
            "> 🌍 Enter GCP region",
            default=region or DEFAULT_REGION,
        )

    # 5. GCP Project & Credentials Check
    gcp_account = get_active_gcp_account() or "default"
    resolved_project = resolve_gcp_project(override_project=project) or "YOUR_GCP_PROJECT_ID"

    if is_interactive and not auto_approve and not skip_checks:
        console.print("\n> Verifying GCP credentials...")
        if gcp_account:
            console.print(f"> You are logged in with account: [cyan]'{gcp_account}'[/cyan]")
        console.print(f"> You are using project: [cyan]'{resolved_project}'[/cyan]")

        choice = Prompt.ask(
            "> Do you want to continue? (The CLI will check if Vertex AI is enabled in this project) [y/skip/edit]",
            default="y",
        ).lower().strip()

        if choice == "edit":
            resolved_project = Prompt.ask("> Enter GCP project ID", default=resolved_project)
        elif choice == "skip":
            skip_checks = True

    if not skip_checks and resolved_project and resolved_project != "YOUR_GCP_PROJECT_ID":
        console.print("> Testing Vertex AI connection...")
        valid, msg = verify_credentials_and_vertex(project_id=resolved_project, location=region)
        if not valid:
            console.print("[bold yellow]⚠️  Looks like you are not authenticated with Google Cloud.[/bold yellow]")
            console.print("Please run: [cyan]gcloud auth login --update-adc[/cyan]")
            console.print(f"Then set your project: [cyan]gcloud config set project {resolved_project}[/cyan]")
            console.print(f"Details: {msg}")
            console.print("> Continuing with template processing...\n")
        else:
            console.print(f"[bold green]✅ Vertex AI API is ready in project '{resolved_project}'.[/bold green]\n")

    # 6. Context variables & Template rendering
    gcs_bucket = f"{resolved_project}-models" if resolved_project != "YOUR_GCP_PROJECT_ID" else "your-gcs-bucket"
    sa_email = f"models-sa@{resolved_project}.iam.gserviceaccount.com" if resolved_project != "YOUR_GCP_PROJECT_ID" else "your-sa@project.iam.gserviceaccount.com"

    context = {
        "project_name": project_name,
        "model_id": model,
        "engine": engine_val,
        "machine_type": "",
        "region": region,
        "project_id": resolved_project,
        "gcs_bucket": gcs_bucket,
        "service_account_email": sa_email,
        "hf_token": "",
        "endpoint_url": "",
        "auth_token": "",
    }

    copy_and_render_templates(
        destination=project_dir,
        template_name=engine_val,
        context=context,
    )

    # 7. Success Banner
    console.print(f"\n[bold green]✅ Success![/bold green] Your open model project [bold cyan]'{project_name}'[/bold cyan] is ready.\n")
    console.print("[bold]📖 Documentation[/bold]")
    console.print(f"   README:    cat {project_name}/README.md\n")
    console.print("[bold]🚀 Get Started[/bold]")
    console.print(f"   cd {project_name}")
    console.print(f"   models-cli recommend --model {model}")
    console.print("   models-cli deploy --dry-run\n")
