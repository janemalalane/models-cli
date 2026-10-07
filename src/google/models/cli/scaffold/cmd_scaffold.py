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

import click
from click.core import ParameterSource
from rich.console import Console
from rich.prompt import Prompt

from google.models.cli._gcp_project import get_active_gcp_account, resolve_gcp_project
from google.models.cli.common.auth import is_authenticated
from google.models.cli.common.banner import display_banner
from google.models.cli.common.constants import (
    DEFAULT_ENGINE,
    DEFAULT_MODEL_REPO,
    DEFAULT_REGION,
)
from google.models.cli.scaffold.scaffold_utils import (
    copy_and_render_templates,
    normalize_project_name,
    verify_credentials_and_vertex,
)

console = Console()


def _print_section(number: int, title: str, subtitle: str | None = None) -> None:
    """Print a numbered section header with underline matching title length."""
    header = f" {number}. {title}"
    console.print()
    console.print(f"[bold]{header}[/bold]")
    console.print(f" {'─' * (len(header) - 1)}")
    if subtitle:
        console.print(f"  [dim]{subtitle}[/dim]\n")


def _display_intro(c: Console) -> None:
    """Brief, oriented welcome shown right after the banner."""
    c.print("  [bold]Setting up open-weights model workspace.[/]")
    c.print(
        "  [dim]This scaffolds serving configs, deployment specs, and evaluation datasets.[/]"
    )
    c.print()
    c.print("  [dim]Here's what we'll do together:[/]")
    c.print("    [dim]1.[/] Name your workspace directory")
    c.print("    [dim]2.[/] Select target Hugging Face model & inference serving engine")
    c.print("    [dim]3.[/] Verify Google Cloud environment & configure artifact storage (GCS)")
    c.print("    [dim]4.[/] Configure service account & optional deployment settings")
    c.print()


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
    "project_id",
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
@click.pass_context
def create_project(
    ctx: click.Context,
    project_name: str | None,
    *,
    template: str,
    model: str,
    region: str,
    project_id: str | None,
    bucket: str | None,
    base_path: str | None,
    service_account: str | None,
    hf_token: str | None,
    endpoint_url: str | None,
    interactive: bool,
    auto_approve: bool,
    skip_checks: bool,
    output_dir: str | None,
) -> None:
    """Creates an open model serving, deployment, and benchmarking project workspace."""
    is_interactive = interactive or (not auto_approve and project_name is None)

    if is_interactive and not auto_approve:
        console.print("Setting up...")
        console.print()
        display_banner(console)
        _display_intro(console)

    target_parent = Path(output_dir).resolve() if output_dir else Path.cwd()

    # 1. Project Name
    if not project_name:
        if is_interactive:
            _print_section(
                1,
                "Project Workspace Setup",
                "Choose a workspace directory name for your model service.",
            )
            while True:
                candidate_name = Prompt.ask(
                    "> What is your project name?",
                    default="my-model-service",
                ).strip()
                if not candidate_name:
                    console.print(
                        "[bold red]❌ Project name cannot be empty.[/bold red]"
                    )
                    continue

                normalized_name = normalize_project_name(candidate_name)
                candidate_dir = target_parent / normalized_name

                if candidate_dir.exists() and any(candidate_dir.iterdir()):
                    console.print(
                        f"[bold red]❌ Error:[/bold red] Directory '{candidate_dir}' already exists and is not empty. Please choose a different name."
                    )
                    continue

                project_name = normalized_name
                break
        else:
            project_name = "my-model-service"

    project_name = normalize_project_name(project_name)
    project_dir = target_parent / project_name

    if project_dir.exists() and any(project_dir.iterdir()):
        console.print(
            f"[bold red]❌ Error:[/bold red] Target directory '{project_dir}' already exists and is not empty."
        )
        raise click.exceptions.Exit(1)

    # 2. Model Repo & Engine
    step_2_needed = is_interactive and not auto_approve and (
        ctx.get_parameter_source("model") != ParameterSource.COMMANDLINE
        or ctx.get_parameter_source("template") != ParameterSource.COMMANDLINE
    )
    if step_2_needed:
        _print_section(
            2,
            "Target Model & Serving Engine",
            "Specify the Hugging Face model repository and inference serving engine.",
        )

    if (
        is_interactive
        and not auto_approve
        and ctx.get_parameter_source("model") != ParameterSource.COMMANDLINE
    ):
        model = Prompt.ask(
            "> 📦 What is your model ID (Hugging Face repository)?",
            default=model,
        )

    # 3. Serving Engine
    if (
        is_interactive
        and not auto_approve
        and ctx.get_parameter_source("template") != ParameterSource.COMMANDLINE
    ):
        template = Prompt.ask(
            "> ⚡ What is your inference engine [vllm / sglang]?",
            default=template,
        )
    engine_val = template.lower().strip()
    if engine_val not in ("vllm", "sglang"):
        console.print(
            f"[bold yellow]⚠️ Unknown engine '{template}', falling back to 'vllm'[/bold yellow]"
        )
        engine_val = "vllm"

    # 4. GCP Region
    if is_interactive and not auto_approve:
        _print_section(
            3,
            "Google Cloud Environment & Storage",
            "Verify GCP project, credentials, and configure model artifact storage.",
        )

    if (
        is_interactive
        and not auto_approve
        and ctx.get_parameter_source("region") != ParameterSource.COMMANDLINE
    ):
        default_region = region
        region = ""
        while not region:
            region = Prompt.ask(
                "> 🌍 What is your GCP region?",
                default=default_region,
            ).strip()
            if not region:
                console.print("[bold red]❌ GCP region is required.[/bold red]")

    # 5. GCP Project ID Selection
    if project_id:
        resolved_project = project_id
    elif is_interactive and not auto_approve:
        default_project = resolve_gcp_project() or ""
        resolved_project = ""
        while not resolved_project:
            if default_project:
                resolved_project = Prompt.ask(
                    "> 📁 What is your GCP project ID?",
                    default=default_project,
                ).strip()
            else:
                resolved_project = Prompt.ask(
                    "> 📁 What is your GCP project ID?",
                ).strip()
            if not resolved_project:
                console.print("[bold red]❌ GCP project ID is required.[/bold red]")
    else:
        default_project = resolve_gcp_project() or ""
        if not default_project:
            console.print(
                "[bold red]❌ Error:[/bold red] GCP project ID is required. Please pass --project <PROJECT_ID> or set gcloud config."
            )
            raise click.exceptions.Exit(1)
        resolved_project = default_project

    # Environment status readout & verification checks
    if is_interactive and not auto_approve:
        console.print()
        console.print(f"  [green]>[/] GOOGLE_CLOUD_PROJECT = [cyan]{resolved_project}[/cyan]")
        console.print(f"  [green]>[/] GOOGLE_CLOUD_LOCATION = [cyan]{region}[/cyan]")
        active_account = get_active_gcp_account()
        if active_account:
            console.print(f"  [green]>[/] Active GCP Account = [cyan]{active_account}[/cyan]")
        authed, _ = is_authenticated()
        if authed:
            console.print("  [green]>[/] Application Default Credentials present")
        else:
            console.print("  [yellow]![/] Application Default Credentials not detected")

    if not skip_checks:
        is_ok, msg = verify_credentials_and_vertex(resolved_project, region)
        if is_ok:
            if is_interactive and not auto_approve:
                console.print(f"  [green]>[/] {msg}")
            else:
                console.print(f"[dim]✓ {msg}[/dim]")
        else:
            console.print(
                f"[bold yellow]⚠️  GCP verification warning:[/bold yellow] {msg}\n"
                "[dim]Continuing with local project scaffolding...[/dim]"
            )

    # 6. GCS Bucket Name
    gcs_bucket = (bucket or "").strip().removeprefix("gs://").strip("/")
    if not gcs_bucket and is_interactive and not auto_approve:
        while not gcs_bucket:
            raw_bucket = Prompt.ask(
                "> 🪣 What is your GCS bucket for model artifacts?",
            )
            gcs_bucket = raw_bucket.strip().removeprefix("gs://").strip("/")
            if not gcs_bucket:
                console.print("[bold red]❌ GCS bucket name is required.[/bold red]")

    # 7. Artifact Base Path
    if base_path is not None:
        resolved_base_path = base_path
    elif is_interactive and not auto_approve:
        resolved_base_path = Prompt.ask(
            "> 📂 What is your base path in GCS bucket (optional)?",
            default="",
        ).strip()
    else:
        resolved_base_path = ""
    resolved_base_path = (
        resolved_base_path.strip().strip("/") if resolved_base_path else ""
    )

    # 8. Service Account Email & Credentials
    step_4_needed = is_interactive and not auto_approve and (
        service_account is None or hf_token is None or endpoint_url is None
    )
    if step_4_needed:
        _print_section(
            4,
            "Service Account & Deployment Target",
            "Configure identity, Hugging Face access token, and optional endpoint URL.",
        )

    if service_account is not None:
        sa_email = service_account
    elif is_interactive and not auto_approve:
        sa_email = Prompt.ask(
            "> 👤 What is your service account email (optional)?",
            default="",
        ).strip()
    else:
        sa_email = ""

    # 9. Hugging Face Token
    if hf_token is not None:
        resolved_hf_token = hf_token
    elif is_interactive and not auto_approve:
        resolved_hf_token = Prompt.ask(
            "> 🔑 What is your Hugging Face token (optional)?",
            default="",
            password=True,
        ).strip()
    else:
        resolved_hf_token = ""

    # 10. Deployed Endpoint URL
    if endpoint_url is not None:
        resolved_endpoint_url = endpoint_url
    elif is_interactive and not auto_approve:
        resolved_endpoint_url = Prompt.ask(
            "> 🌐 What is your deployed endpoint URL (optional)?",
            default="",
        ).strip()
    else:
        resolved_endpoint_url = ""

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

    # Success readout & next steps
    console.print()
    console.print(
        f"[bold green]✓[/bold green] [bold]Workspace ready:[/] [cyan]{project_name}[/cyan]"
    )
    console.print(
        f"  [dim]Engine:[/] [cyan]{engine_val}[/cyan]  •  "
        f"[dim]Model:[/] [cyan]{model}[/cyan]  •  "
        f"[dim]Region:[/] [cyan]{region}[/cyan]"
    )
    console.print()
    console.print("  [bold]Get started:[/bold]")
    console.print(f"    [dim]1.[/] cd {project_name}")
    console.print(f"    [dim]2.[/] models-cli recommend --model {model}")
    console.print("    [dim]3.[/] models-cli deploy --dry-run")
    console.print()


scaffold_cmd = create_project
