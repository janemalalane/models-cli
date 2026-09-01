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

"""Main entry point for models-cli."""

import sys
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from google.models.cli import __version__
from google.models.cli._gcp_project import get_active_gcp_account, resolve_gcp_project
from google.models.cli.benchmark.cmd_benchmark import benchmark
from google.models.cli.common.auth import is_authenticated
from google.models.cli.common.constants import DEFAULT_MODEL_REPO, DEFAULT_REGION
from google.models.cli.deploy.cmd_deploy import deploy
from google.models.cli.eval.cmd_eval import eval as eval_fn
from google.models.cli.recommend.cmd_recommend import recommend
from google.models.cli.scaffold.cmd_scaffold import create_project

console = Console()

app = typer.Typer(
    name="models-cli",
    help="Lifecycle CLI for evaluating, optimizing, deploying, and benchmarking open-weights models on Google Cloud (GEAP / Vertex AI).",
    no_args_is_help=True,
    add_completion=False,
)

# Register subcommands directly as Typer commands
app.command(
    name="create",
    help="Scaffold a new open-weights model serving & benchmarking workspace",
)(create_project)
app.command(name="init", hidden=True)(create_project)
app.command(name="scaffold", hidden=True)(create_project)

app.command(
    name="eval", help="Evaluate model response accuracy and quality on golden datasets"
)(eval_fn)
app.command(name="evaluate", hidden=True)(eval_fn)
app.command(
    name="recommend",
    help="Hardware & engine parameter recommendations from Google Cloud GKE Recommender",
)(recommend)
app.command(
    name="deploy",
    help="Deploy open model container to GEAP / Vertex AI online prediction endpoint",
)(deploy)
app.command(
    name="benchmark",
    help="Run performance & load testing with inference-perf and generate analysis reports",
)(benchmark)


def version_callback(value: bool) -> None:
    if value:
        console.print(
            f"[bold cyan]models-cli[/bold cyan] version [bold green]{__version__}[/bold green]"
        )
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show models-cli version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """models-cli: Turn any developer into an expert at open model inference and deployment on Google Cloud."""
    pass


@app.command(name="info")
def info() -> None:
    """Displays current environment diagnostics and active GCP configuration."""
    from google.models.cli.common.config import settings

    project_id = resolve_gcp_project() or "(not set)"
    account = get_active_gcp_account() or "(not set)"

    table = Table(
        title="models-cli Environment Information",
        border_style="dim",
        header_style="bold magenta",
    )
    table.add_column("Property", style="bold cyan")
    table.add_column("Value", style="green")

    table.add_row("models-cli Version", __version__)
    table.add_row("Active GCP Account", account)
    table.add_row("Resolved GCP Project", project_id)
    authed, auth_display = is_authenticated()
    table.add_row(
        "Authentication Status",
        f"[green]✓ {auth_display}[/green]"
        if authed
        else "[bold yellow]❌ Not authenticated (run 'gcloud auth application-default login')[/bold yellow]",
    )
    table.add_row(
        "Default Location/Region", settings.google_cloud_location or DEFAULT_REGION
    )
    table.add_row("Default Model", settings.hf_model_repo or DEFAULT_MODEL_REPO)
    table.add_row("Service Account", settings.service_account_email or "(not set)")
    table.add_row(
        "GCS Storage Bucket", settings.google_cloud_storage_bucket or "(not set)"
    )

    console.print(table)


if __name__ == "__main__":
    app()
