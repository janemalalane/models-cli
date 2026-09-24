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
    help="GCP project ID. Defaults to active gcloud / ADC project.",
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
    help="Skip GCP and Vertex AI verification checks.",
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
        raise click.exceptions.Exit(1)

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
            default=template or DEFAULT_ENGINE,
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

    # 5. GCP Project ID Selection
    default_project = resolve_gcp_project(override_project=project) or "YOUR_GCP_PROJECT_ID"
    if is_interactive and not auto_approve:
        resolved_project = (
            Prompt.ask(
                "> 📁 Enter GCP project ID",
                default=default_project,
            ).strip()
            or default_project
        )
    else:
        resolved_project = default_project
    # 6. Context variables & Template rendering
    gcs_bucket = f"{resolved_project}-models" if resolved_project != "YOUR_GCP_PROJECT_ID" else "your-gcs-bucket"
    sa_email = ""

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


scaffold_cmd = create_project
