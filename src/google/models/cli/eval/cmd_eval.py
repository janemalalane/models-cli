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

"""Evaluation command for models-cli."""

from pathlib import Path
import click
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from google.models.cli.common.auth import ensure_authenticated
from google.models.cli.common.config import Settings
from google.models.cli.eval.eval_utils import (
    evaluate_candidate_models,
    load_golden_dataset,
    update_project_with_winner,
)

console = Console()
settings = Settings()


@click.command("eval")
@click.option(
    "--models",
    "-m",
    default=str(settings.model_id),
    help="Comma-separated list of candidate model names or endpoint URLs to evaluate.",
)
@click.option(
    "--dataset",
    "-d",
    default="tests/eval/golden_dataset.jsonl",
    help="Path to golden evaluation JSONL dataset file.",
)
@click.option(
    "--metrics",
    "-M",
    default="accuracy,quality",
    help="Comma-separated list of metrics to evaluate (default: accuracy,quality).",
)
@click.option(
    "--set-winner",
    is_flag=True,
    default=False,
    help="Automatically update project .env and engine configs with the winning model.",
)
@click.option(
    "--mock",
    is_flag=True,
    default=False,
    help="Simulate model evaluation responses without making live model API calls.",
)
def eval(
    *,
    models: str,
    dataset: str,
    metrics: str,
    set_winner: bool,
    mock: bool,
) -> None:
    """Evaluates candidate open models on a golden benchmark dataset and ranks the winner."""
    if not mock and not ensure_authenticated(interactive=True):
        raise click.exceptions.Exit(1)

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not model_list:
        console.print(
            "[bold red]❌ Error: No model names provided for evaluation.[/bold red]"
        )
        raise click.exceptions.Exit(1)

    metrics_list = [m.strip() for m in metrics.split(",") if m.strip()]
    dataset_path = Path(dataset)
    dataset_records = load_golden_dataset(dataset_path)

    console.print(f"\n[bold cyan]🔍 Running Model Response Evaluation[/bold cyan]")
    console.print(f"   • Candidate Models: [bold]{', '.join(model_list)}[/bold]")
    console.print(
        f"   • Dataset:          [bold]{dataset_path}[/bold] ({len(dataset_records)} samples)"
    )
    console.print(f"   • Metrics:          [bold]{', '.join(metrics_list)}[/bold]")
    console.print(f"   • Mock Mode:        [bold]{mock}[/bold]\n")

    results = evaluate_candidate_models(
        models=model_list,
        dataset=dataset_records,
        metrics_list=metrics_list,
        mock=mock,
    )

    table = Table(
        title="Model Evaluation Scorecard",
        border_style="dim",
        header_style="bold magenta",
    )
    table.add_column("Rank", style="bold yellow", justify="center")
    table.add_column("Candidate Model", style="bold cyan")
    table.add_column("Accuracy", justify="right", style="green")
    table.add_column("Quality", justify="right", style="green")
    table.add_column("Composite Score", justify="right", style="bold yellow")
    table.add_column("Avg Latency", justify="right")

    for idx, r in enumerate(results, start=1):
        rank_str = f"🏆 #1" if idx == 1 else f"#{idx}"
        table.add_row(
            rank_str,
            r["model"],
            f"{r['accuracy']}%",
            f"{r['quality']}%",
            f"{r['composite_score']:.3f}",
            f"{r['avg_latency_ms']:.1f} ms",
        )

    console.print(table)

    winner = results[0]["model"]
    console.print(
        f"\n[bold green]🏆 Winner:[/bold green] [bold cyan]{winner}[/bold cyan] (Score: {results[0]['composite_score']}, Latency: {results[0]['avg_latency_ms']} ms)"
    )

    # Offer to update project configs if in a project directory
    current_dir = Path.cwd()
    if set_winner or (Path(".env").is_file() and not mock):
        if Path(".env").is_file() and settings.model_id == winner:
            return
        should_update = set_winner or Confirm.ask(
            f"Do you want to set '{winner}' as the primary model for this project?",
            default=True,
        )
        if should_update:
            updated = update_project_with_winner(winner, current_dir)
            if updated:
                console.print(
                    f"[bold green]✅ Updated project configuration with winning model '{winner}'.[/bold green]\n"
                )


eval_cmd = eval
