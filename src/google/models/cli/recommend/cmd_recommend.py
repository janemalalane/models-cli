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

"""Recommendation command for models-cli."""

import json
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from google.models.cli.common.config import settings
from google.models.cli.common.constants import (
    DEFAULT_MODEL_REPO,
    AcceleratorFamily,
    InferenceEngine,
    OptimizationMetric,
)
from google.models.cli.recommend.recommend_utils import (
    apply_recommendation,
    generate_recommendations,
)

console = Console()
recommend_cmd = typer.Typer(
    name="recommend",
    help="Zero-shot hardware & engine parameter recommendation for open-weights models",
)


@recommend_cmd.callback(invoke_without_command=True)
def recommend(
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Hugging Face model repository identifier (e.g. google/gemma-4-31B-it).",
    ),
    objective: str = typer.Option(
        OptimizationMetric.COST.value,
        "--objective",
        "-O",
        help="Optimization objective: 'cost', 'ttft', 'tpot', or 'throughput'.",
    ),
    family: str = typer.Option(
        AcceleratorFamily.ANY.value,
        "--family",
        "-f",
        help="Accelerator family: 'gpu', 'tpu', or 'any'.",
    ),
    budget: Optional[float] = typer.Option(
        None,
        "--budget",
        "-b",
        help="Maximum hourly hardware budget in USD.",
    ),
    engine: str = typer.Option(
        InferenceEngine.VLLM.value,
        "--engine",
        "-e",
        help="Serving engine target: 'vllm' or 'sglang'.",
    ),
    format_type: str = typer.Option(
        "table",
        "--format",
        "-F",
        help="Output format: 'table' or 'json'.",
    ),
    apply: bool = typer.Option(
        False,
        "--apply",
        "-a",
        help="Write the top recommended machine type and engine tuning to config/deployment_spec.yaml.",
    ),
) -> None:
    """Recommends GEAP hardware configurations and tuned engine parameters for a given model."""
    target_model = model or settings.model_id or DEFAULT_MODEL_REPO

    # Parse objective
    try:
        obj_enum = OptimizationMetric(objective.lower())
    except ValueError:
        obj_enum = OptimizationMetric.COST

    # Parse family
    try:
        fam_enum = AcceleratorFamily(family.lower())
    except ValueError:
        fam_enum = AcceleratorFamily.ANY

    # Parse engine
    try:
        eng_enum = InferenceEngine(engine.lower())
    except ValueError:
        eng_enum = InferenceEngine.VLLM

    recs = generate_recommendations(
        model_id=target_model,
        objective=obj_enum,
        family=fam_enum,
        max_budget_hourly=budget,
        engine=eng_enum,
    )

    if not recs:
        console.print(f"[bold red]❌ No hardware configurations found matching the criteria for '{target_model}'.[/bold red]")
        raise typer.Exit(1)

    if format_type.lower() == "json":
        print(json.dumps({"model": target_model, "objective": obj_enum.value, "recommendations": recs}, indent=2))
        return

    # Rich Table Display
    console.print(f"\n[bold cyan]🚀 Hardware & Engine Recommendations for[/bold cyan] [bold yellow]{target_model}[/bold yellow]")
    console.print(f"🎯 Objective: [bold green]{obj_enum.value.upper()}[/bold green] | Family: [bold blue]{fam_enum.value.upper()}[/bold blue]\n")

    table = Table(border_style="dim", header_style="bold magenta")
    table.add_column("Rank", style="bold yellow", justify="center")
    table.add_column("Machine Type", style="bold cyan")
    table.add_column("Accelerator", style="green")
    table.add_column("VRAM", justify="right")
    table.add_column("Est. Cost/Hr", justify="right", style="bold yellow")
    table.add_column("Est. TTFT", justify="right")
    table.add_column("Est. TPOT", justify="right")
    table.add_column("Est. Tokens/s", justify="right")
    table.add_column("TP Size", justify="center")

    for idx, r in enumerate(recs, start=1):
        rank_str = f"#{idx}" if idx > 1 else "⭐ #1"
        table.add_row(
            rank_str,
            r["machine_type"],
            r["chip_name"],
            f"{r['vram_gb']} GB",
            f"${r['hourly_cost_usd']:.2f}",
            f"~{r['est_ttft_ms']} ms",
            f"~{r['est_tpot_ms']} ms",
            f"~{r['est_throughput_tokens_sec']}",
            str(r["engine_params"]["tensor_parallel_size"]),
        )

    console.print(table)

    top = recs[0]
    console.print(f"\n[bold green]💡 Recommended Machine Type:[/bold green] [bold cyan]{top['machine_type']}[/bold cyan] ({top['chip_name']}) at [yellow]${top['hourly_cost_usd']:.2f}/hr[/yellow]")

    # Check if inside a project directory with config/deployment_spec.yaml
    deploy_spec = Path("config/deployment_spec.yaml")
    if deploy_spec.is_file():
        should_apply = apply
        if not apply and format_type == "table":
            should_apply = Confirm.ask(f"\nApply '{top['machine_type']}' to config/deployment_spec.yaml?", default=True)
        if should_apply:
            updated = apply_recommendation(top, Path.cwd())
            if updated:
                console.print(f"[bold green]✅ Updated config/deployment_spec.yaml with machine type '{top['machine_type']}'.[/bold green]\n")
    else:
        console.print(f"[dim]Run inside your project workspace or pass --apply to update config/deployment_spec.yaml[/dim]\n")
