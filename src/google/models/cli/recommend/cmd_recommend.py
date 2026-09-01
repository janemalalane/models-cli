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

"""Recommendation command for models-cli powered by Google Cloud GKE Recommender."""

import json
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table
import google.auth.exceptions
import google.api_core.exceptions

from google.models.cli.common.auth import ensure_authenticated, is_auth_error
from google.models.cli.common.config import settings
from google.models.cli.common.constants import (
    DEFAULT_MODEL_REPO,
    DEFAULT_PRICING_MODEL,
    AcceleratorFamily,
)
from google.models.cli.recommend.recommend_utils import (
    ModelNotSupportedError,
    apply_recommendation,
    fetch_supported_models,
    get_recommendations,
)

console = Console()
recommend_cmd = typer.Typer(
    name="recommend",
    help="Hardware & engine parameter recommendations from Google Cloud GKE Recommender.",
)


@recommend_cmd.callback(invoke_without_command=True)
def recommend(
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Hugging Face model repository identifier (e.g. google/gemma-3-4b-it).",
    ),
    target_cost_per_million_input_tokens: Optional[float] = typer.Option(
        None,
        "--target-cost-per-million-input-tokens",
        help="Target maximum cost per 1M input tokens in USD.",
    ),
    target_cost_per_million_output_tokens: Optional[float] = typer.Option(
        None,
        "--target-cost-per-million-output-tokens",
        help="Target maximum cost per 1M output tokens in USD.",
    ),
    output_input_cost_ratio: Optional[float] = typer.Option(
        None,
        "--output-input-cost-ratio",
        "-r",
        help="Pricing conversion ratio between output and input tokens (e.g. 4.0).",
    ),
    target_ttft_milliseconds: Optional[int] = typer.Option(
        None,
        "--target-ttft-milliseconds",
        help="Maximum Time to First Token in milliseconds.",
    ),
    target_ntpot_milliseconds: Optional[int] = typer.Option(
        None,
        "--target-ntpot-milliseconds",
        help="Maximum Normalized Time per Output Token in milliseconds.",
    ),
    use_case: Optional[str] = typer.Option(
        None,
        "--use-case",
        "-u",
        help="Workload traffic pattern filter (e.g. chatbot, summarization, code-completion, text-generation, deep-research).",
    ),
    pricing_model: str = typer.Option(
        DEFAULT_PRICING_MODEL,
        "--pricing-model",
        help="Pricing model tier ('on-demand', 'spot', '1-year-cud', '3-years-cud').",
    ),
    model_server: Optional[str] = typer.Option(
        None,
        "--model-server",
        "--engine",
        "-e",
        help="Serving engine target (defaults to server matching model profile, e.g. 'vllm').",
    ),
    model_server_version: Optional[str] = typer.Option(
        None,
        "--model-server-version",
        help="Optional model server version string.",
    ),
    family: str = typer.Option(
        AcceleratorFamily.ANY.value,
        "--family",
        "-f",
        help="Accelerator family: 'gpu', 'tpu', or 'any'.",
    ),
    sort_by: str = typer.Option(
        "cost",
        "--sort-by",
        "-s",
        help="Sort recommendations by: 'cost' (default), 'throughput', 'ttft', or 'ntpot'.",
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
    list_models: bool = typer.Option(
        False,
        "--list-models",
        help="List all supported models in GKE Recommender and exit.",
    ),
) -> None:
    """Recommends hardware configurations and engine parameters using GKE Recommender."""
    if not ensure_authenticated(interactive=True):
        raise typer.Exit(1)

    if list_models:
        try:
            supported = fetch_supported_models()
        except Exception as err:
            if is_auth_error(err):
                if ensure_authenticated(interactive=True):
                    try:
                        supported = fetch_supported_models()
                    except Exception as retry_err:
                        console.print(f"[bold red]❌ Failed to fetch supported models: {retry_err}[/bold red]")
                        raise typer.Exit(1)
                else:
                    raise typer.Exit(1)
            else:
                console.print(f"[bold red]❌ Failed to fetch supported models: {err}[/bold red]")
                raise typer.Exit(1)

        console.print(
            f"\n[bold cyan]Supported models in GKE Recommender ({len(supported)} models):[/bold cyan]\n"
        )
        for m in supported:
            console.print(f"  • [green]{m}[/green]")
        console.print(
            "\n[dim]Run 'models-cli recommend --model <MODEL>' to see hardware recommendations.[/dim]\n"
        )
        return

    target_model = model
    if not target_model:
        deploy_spec = Path("config/deployment_spec.yaml")
        if deploy_spec.is_file():
            try:
                import yaml

                with open(deploy_spec, "r", encoding="utf-8") as f:
                    spec_yaml = yaml.safe_load(f) or {}
                target_model = spec_yaml.get("model", {}).get("hf_repo")
            except Exception:
                pass
    target_model = target_model or settings.model_id or DEFAULT_MODEL_REPO

    # Parse family
    try:
        fam_enum = AcceleratorFamily(family.lower())
    except ValueError:
        fam_enum = AcceleratorFamily.ANY

    try:
        recs = get_recommendations(
            model_id=target_model,
            model_server=model_server,
            model_server_version=model_server_version,
            target_cost_per_million_input_tokens=target_cost_per_million_input_tokens,
            target_cost_per_million_output_tokens=target_cost_per_million_output_tokens,
            output_input_cost_ratio=output_input_cost_ratio,
            pricing_model=pricing_model,
            target_ttft_milliseconds=target_ttft_milliseconds,
            target_ntpot_milliseconds=target_ntpot_milliseconds,
            use_case=use_case,
            family=fam_enum,
            sort_by=sort_by,
        )
    except google.api_core.exceptions.PermissionDenied as err:
        console.print(
            f"\n[bold red]❌ Permission denied accessing GKE Recommender service: {err}[/bold red]\n"
        )
        raise typer.Exit(1)
    except ModelNotSupportedError as err:
        console.print(
            f"\n[bold red]❌ Model '{err.model_id}' is not supported by the Google Cloud GKE Recommender service.[/bold red]\n"
        )
        if err.supported_models:
            console.print("[bold yellow]Supported models include:[/bold yellow]")
            for m in err.supported_models[:8]:
                console.print(f"  • [green]{m}[/green]")
            if len(err.supported_models) > 8:
                console.print(
                    f"  [dim]... and {len(err.supported_models) - 8} more. Run 'models-cli recommend --list-models' to view all.[/dim]"
                )
        console.print()
        raise typer.Exit(1)
    except Exception as err:
        if is_auth_error(err):
            if ensure_authenticated(interactive=True):
                try:
                    recs = get_recommendations(
                        model_id=target_model,
                        model_server=model_server,
                        model_server_version=model_server_version,
                        target_cost_per_million_input_tokens=target_cost_per_million_input_tokens,
                        target_cost_per_million_output_tokens=target_cost_per_million_output_tokens,
                        output_input_cost_ratio=output_input_cost_ratio,
                        pricing_model=pricing_model,
                        target_ttft_milliseconds=target_ttft_milliseconds,
                        target_ntpot_milliseconds=target_ntpot_milliseconds,
                        use_case=use_case,
                        family=fam_enum,
                        sort_by=sort_by,
                    )
                except Exception as retry_err:
                    console.print(f"[bold red]❌ Error retrieving recommendations: {retry_err}[/bold red]")
                    raise typer.Exit(1)
            else:
                raise typer.Exit(1)
        else:
            console.print(f"[bold red]❌ Error retrieving recommendations: {err}[/bold red]")
            raise typer.Exit(1)

    if not recs:
        console.print(
            f"[bold red]❌ No hardware configurations found matching the criteria for '{target_model}'.[/bold red]"
        )
        raise typer.Exit(1)

    if format_type.lower() == "json":
        output_payload = {
            "model": target_model,
            "sort_by": sort_by.lower(),
            "use_case": use_case,
            "output_input_cost_ratio": output_input_cost_ratio,
            "recommendations": recs,
        }
        print(json.dumps(output_payload, indent=2))
        return

    # Rich Table Display
    console.print(
        f"\n[bold cyan]🚀 Hardware & Engine Recommendations for[/bold cyan] [bold yellow]{target_model}[/bold yellow]"
    )
    meta_parts = [f"🎯 Sort By: [bold green]{sort_by.upper()}[/bold green]"]
    active_pricing = pricing_model or (recs[0].get("pricing_model") if recs else None) or "spot"
    meta_parts.append(f"Pricing: [bold cyan]{active_pricing.upper()}[/bold cyan]")
    if use_case:
        meta_parts.append(f"Workload: [bold magenta]{use_case}[/bold magenta]")
    if output_input_cost_ratio is not None:
        meta_parts.append(f"Ratio: [bold blue]{output_input_cost_ratio}:1[/bold blue]")
    if target_cost_per_million_input_tokens is not None:
        meta_parts.append(
            f"In Budget: [bold yellow]≤ ${target_cost_per_million_input_tokens:.3f}/M[/bold yellow]"
        )
    console.print(" | ".join(meta_parts) + "\n")

    table = Table(border_style="dim", header_style="bold magenta")
    table.add_column("Rank", style="bold yellow", justify="center")
    table.add_column("Machine Type", style="bold cyan")
    table.add_column("Accelerator", style="green")
    table.add_column("Cost/M In", justify="right", style="bold yellow")
    table.add_column("Cost/M Out", justify="right", style="bold yellow")
    table.add_column("TTFT", justify="right")
    table.add_column("NTPOT", justify="right")
    table.add_column("Output Tok/s", justify="right")
    table.add_column("Server (TP)", justify="center")
    table.add_column("Use Case", style="dim")

    for idx, r in enumerate(recs, start=1):
        rank_str = f"#{idx}" if idx > 1 else "⭐ #1"
        cost_in_str = (
            f"${r['input_cost_per_m']:.3f}" if r["input_cost_per_m"] is not None else "-"
        )
        cost_out_str = (
            f"${r['output_cost_per_m']:.3f}" if r["output_cost_per_m"] is not None else "-"
        )
        ttft_str = f"{r['ttft_ms']} ms" if r["ttft_ms"] is not None else "-"
        ntpot_str = f"{r['ntpot_ms']} ms" if r["ntpot_ms"] is not None else "-"
        tok_s_str = (
            f"{r['output_tokens_per_sec']:,}"
            if r["output_tokens_per_sec"] is not None
            else "-"
        )
        server_tp = (
            f"{r['model_server']} (TP={r['engine_params']['tensor_parallel_size']})"
        )
        use_case_str = r.get("use_case") or "-"

        table.add_row(
            rank_str,
            r["machine_type"],
            r["chip_name"],
            cost_in_str,
            cost_out_str,
            ttft_str,
            ntpot_str,
            tok_s_str,
            server_tp,
            use_case_str,
        )

    console.print(table)

    top = recs[0]
    top_cost_in = (
        f"${top['input_cost_per_m']:.3f}/M in" if top.get("input_cost_per_m") else ""
    )
    top_cost_out = (
        f"(${top['output_cost_per_m']:.3f}/M out)" if top.get("output_cost_per_m") else ""
    )
    console.print(
        f"\n[bold green]💡 Recommended Machine Type:[/bold green] [bold cyan]{top['machine_type']}[/bold cyan] ({top['chip_name']}) {top_cost_in} {top_cost_out}"
    )
    console.print(
        f"   [dim]Serving Engine: {top['model_server']} {top.get('model_server_version', '')} with Tensor Parallelism = {top['engine_params']['tensor_parallel_size']}[/dim]"
    )

    # Check if inside a project directory with config/deployment_spec.yaml
    deploy_spec = Path("config/deployment_spec.yaml")
    if deploy_spec.is_file():
        should_apply = apply
        if not apply and format_type == "table":
            should_apply = Confirm.ask(
                f"\nApply '{top['machine_type']}' to config/deployment_spec.yaml?",
                default=True,
            )
        if should_apply:
            updated = apply_recommendation(top, Path.cwd())
            if updated:
                console.print(
                    f"[bold green]✅ Updated config/deployment_spec.yaml with machine type '{top['machine_type']}'.[/bold green]\n"
                )
    else:
        console.print(
            "[dim]Run inside your project workspace or pass --apply to update config/deployment_spec.yaml[/dim]\n"
        )
