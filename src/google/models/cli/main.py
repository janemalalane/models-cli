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

import io
import sys
import traceback

import click
from rich.console import Console
from rich.table import Table

from google.models.cli import __version__
from google.models.cli._click import LazyGroup, patch_source_in_help
from google.models.cli._gcp_project import get_active_gcp_account, resolve_gcp_project
from google.models.cli.common.auth import is_authenticated
from google.models.cli.common.constants import DEFAULT_MODEL_REPO, DEFAULT_REGION

# Force utf-8 encoding and non-exception fallback for printing
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

console = Console()


class _MainGroup(LazyGroup):
    """Click group with lazy command loading and full-traceback exception handling."""

    def invoke(self, ctx: click.Context) -> None:
        try:
            super().invoke(ctx)
        except click.exceptions.Exit:
            raise
        except click.ClickException:
            click.echo(f"models-cli v{__version__}", err=True)
            raise
        except KeyboardInterrupt:
            Console().print(f"\nmodels-cli v{__version__}", style="dim")
            Console().print("Operation cancelled by user", style="yellow")
            ctx.exit(130)
        except Exception:  # noqa: BLE001
            click.echo(f"models-cli v{__version__}", err=True)
            traceback.print_exc()
            ctx.exit(1)


@click.group(cls=_MainGroup, no_args_is_help=True)
@click.version_option(version=__version__, prog_name="models-cli")
def main() -> None:
    """Lifecycle CLI for evaluating, optimizing, deploying, and benchmarking open-weights models on Google Cloud (GEAP / Vertex AI)."""


@main.command(name="info")
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


# Subcommands registered lazily
main.add_lazy_command(
    "create",
    "google.models.cli.scaffold.cmd_scaffold:create_project",
    "Scaffold a new open-weights model serving & benchmarking workspace",
)
main.add_lazy_command(
    "init",
    "google.models.cli.scaffold.cmd_scaffold:create_project",
    "Scaffold a new open-weights model serving & benchmarking workspace",
    hidden=True,
)
main.add_lazy_command(
    "scaffold",
    "google.models.cli.scaffold.cmd_scaffold:create_project",
    "Scaffold a new open-weights model serving & benchmarking workspace",
    hidden=True,
)
main.add_lazy_command(
    "eval",
    "google.models.cli.eval.cmd_eval:eval",
    "Evaluate model response accuracy and quality on golden datasets",
)
main.add_lazy_command(
    "evaluate",
    "google.models.cli.eval.cmd_eval:eval",
    "Evaluate model response accuracy and quality on golden datasets",
    hidden=True,
)
main.add_lazy_command(
    "recommend",
    "google.models.cli.recommend.cmd_recommend:recommend",
    "Hardware & engine parameter recommendations from Google Cloud GKE Recommender",
)
main.add_lazy_command(
    "deploy",
    "google.models.cli.deploy.cmd_deploy:deploy",
    "Deploy open model container to GEAP / Vertex AI online prediction endpoint",
)
main.add_lazy_command(
    "benchmark",
    "google.models.cli.benchmark.cmd_benchmark:benchmark",
    "Run performance & load testing with inference-perf and generate analysis reports",
)
main.add_lazy_command(
    "playground",
    "google.models.cli.playground.cmd_playground:playground",
    "Interactive chat playground for deployed endpoints using Chat Completions API",
)

patch_source_in_help(main)

# Alias app for backwards-compatibility
app = main

if __name__ == "__main__":
    main()
