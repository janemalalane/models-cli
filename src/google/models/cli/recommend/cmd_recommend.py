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
import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
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


def _format_distribution_ratio(input_tokens: int, output_tokens: int) -> str:
    """Formats the custom distribution ratio without decimal places (e.g. 8:1 or 1:4)."""
    if input_tokens >= output_tokens:
        ratio_int = int(round(input_tokens / max(output_tokens, 1)))
        return f"{ratio_int}:1"
    else:
        ratio_int = int(round(output_tokens / max(input_tokens, 1)))
        return f"1:{ratio_int}"


console = Console()


@click.command("recommend")
@click.option(
    "--model",
    "-m",
    default=None,
    help="Hugging Face model repository identifier (e.g. google/gemma-3-4b-it).",
)
@click.option(
    "--target-cost-per-million-input-tokens",
    type=float,
    default=None,
    help="Target maximum cost per 1M input tokens in USD.",
)
@click.option(
    "--target-cost-per-million-output-tokens",
    type=float,
    default=None,
    help="Target maximum cost per 1M output tokens in USD.",
)
@click.option(
    "--output-input-cost-ratio",
    "-r",
    type=float,
    default=None,
    help="Pricing conversion ratio between output and input tokens (e.g. 4.0).",
)
@click.option(
    "--target-ttft-milliseconds",
    type=int,
    default=None,
    help="Maximum Time to First Token in milliseconds.",
)
@click.option(
    "--target-ntpot-milliseconds",
    type=int,
    default=None,
    help="Maximum Normalized Time per Output Token in milliseconds.",
)
@click.option(
    "--use-case",
    "-u",
    default=None,
    help="Workload traffic pattern filter (e.g. chatbot (32:1), summarization (8:1), code-completion (16:1), text-generation (1:4), deep-research (1:16)).",
)
@click.option(
    "--input-tokens",
    "-I",
    type=int,
    default=None,
    help="Target average prompt tokens for custom workload distribution (interpolates profile).",
)
@click.option(
    "--output-tokens",
    "-O",
    type=int,
    default=None,
    help="Target average generation tokens for custom workload distribution (interpolates profile).",
)
@click.option(
    "--pricing-model",
    default=DEFAULT_PRICING_MODEL,
    help="Pricing model tier ('on-demand', 'spot', '1-year-cud', '3-years-cud').",
)
@click.option(
    "--model-server",
    "--engine",
    "-e",
    default=None,
    help="Serving engine target (defaults to server matching model profile, e.g. 'vllm').",
)
@click.option(
    "--model-server-version",
    default=None,
    help="Optional model server version string.",
)
@click.option(
    "--family",
    "-f",
    default=AcceleratorFamily.ANY.value,
    help="Accelerator family: 'gpu', 'tpu', or 'any'.",
)
@click.option(
    "--sort-by",
    "-s",
    default="cost",
    help="Sort recommendations by: 'cost' (default), 'throughput', 'ttft', or 'ntpot'.",
)
@click.option(
    "--format",
    "-F",
    "format_type",
    default="table",
    help="Output format: 'table' or 'json'.",
)
@click.option(
    "--apply",
    "-a",
    is_flag=True,
    default=False,
    help="Write the top recommended machine type and engine tuning to config/deployment_spec.yaml.",
)
@click.option(
    "--list-models",
    is_flag=True,
    default=False,
    help="List all supported models in GKE Recommender and exit.",
)
def recommend(
    *,
    model: str | None,
    target_cost_per_million_input_tokens: float | None,
    target_cost_per_million_output_tokens: float | None,
    output_input_cost_ratio: float | None,
    target_ttft_milliseconds: int | None,
    target_ntpot_milliseconds: int | None,
    use_case: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    pricing_model: str,
    model_server: str | None,
    model_server_version: str | None,
    family: str,
    sort_by: str,
    format_type: str,
    apply: bool,
    list_models: bool,
) -> None:
    """Recommends hardware configurations and engine parameters using GKE Recommender."""
    # Validate custom input/output token options first
    if (input_tokens is not None and output_tokens is None) or (
        input_tokens is None and output_tokens is not None
    ):
        console.print(
            "[bold red]❌ Both --input-tokens (-I) and --output-tokens (-O) must be provided together for custom distribution.[/bold red]"
        )
        raise click.exceptions.Exit(1)
    if input_tokens is not None and input_tokens <= 0:
        console.print(
            "[bold red]❌ --input-tokens must be a positive integer.[/bold red]"
        )
        raise click.exceptions.Exit(1)
    if output_tokens is not None and output_tokens <= 0:
        console.print(
            "[bold red]❌ --output-tokens must be a positive integer.[/bold red]"
        )
        raise click.exceptions.Exit(1)

    if not ensure_authenticated(interactive=True):
        raise click.exceptions.Exit(1)

    if list_models:
        try:
            supported = fetch_supported_models()
        except Exception as err:
            if is_auth_error(err):
                console.print(
                    "\n[bold red]❌ Google Cloud authentication required.[/bold red]\n"
                    "[yellow]Please run the following command to authenticate:[/yellow]\n"
                    "  • [cyan]gcloud auth application-default login[/cyan]\n"
                )
                raise click.exceptions.Exit(1)
            console.print(
                f"[bold red]❌ Failed to fetch supported models: {err}[/bold red]"
            )
            raise click.exceptions.Exit(1)

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
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            family=fam_enum,
            sort_by=sort_by,
        )
    except google.api_core.exceptions.PermissionDenied as err:
        console.print(
            f"\n[bold red]❌ Permission denied accessing GKE Recommender service: {err}[/bold red]\n"
        )
        raise click.exceptions.Exit(1)
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
        raise click.exceptions.Exit(1)
    except Exception as err:
        if is_auth_error(err):
            console.print(
                "\n[bold red]❌ Google Cloud authentication required.[/bold red]\n"
                "[yellow]Please run the following command to authenticate:[/yellow]\n"
                "  • [cyan]gcloud auth application-default login[/cyan]\n"
            )
            raise click.exceptions.Exit(1)
        console.print(
            f"[bold red]❌ Error retrieving recommendations: {err}[/bold red]"
        )
        raise click.exceptions.Exit(1)

    if not recs:
        console.print(
            f"[bold red]❌ No hardware configurations found matching the criteria for '{target_model}'.[/bold red]"
        )
        raise click.exceptions.Exit(1)

    is_custom_dist = input_tokens is not None and output_tokens is not None

    if format_type.lower() == "json":
        output_payload = {
            "model": target_model,
            "sort_by": sort_by.lower(),
            "use_case": use_case,
            "output_input_cost_ratio": output_input_cost_ratio,
            "custom_distribution": (
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "ratio": _format_distribution_ratio(input_tokens, output_tokens),  # type: ignore
                    "is_interpolated": True,
                }
                if is_custom_dist
                else None
            ),
            "recommendations": recs,
        }
        print(json.dumps(output_payload, indent=2))
        return

    # Rich Table Display
    console.print(
        f"\n[bold cyan]🚀 Hardware & Engine Recommendations for[/bold cyan] [bold yellow]{target_model}[/bold yellow]"
    )

    if is_custom_dist:
        ratio_val = _format_distribution_ratio(input_tokens, output_tokens)  # type: ignore
        console.print(
            Panel.fit(
                f"[bold yellow]ℹ️  Custom Workload Distribution:[/bold yellow] [bold cyan]{input_tokens:,}[/bold cyan] input tokens / [bold cyan]{output_tokens:,}[/bold cyan] output tokens (Ratio: [bold blue]{ratio_val}[/bold blue])\n"
                f"[dim]⚠️  Notice: Performance metrics below are [bold yellow]interpolated[/bold yellow] from empirical GKE Recommender benchmarks for this model.[/dim]",
                border_style="yellow",
            )
        )

    meta_parts = [f"🎯 Sort By: [bold green]{sort_by.upper()}[/bold green]"]
    active_pricing = (
        pricing_model or (recs[0].get("pricing_model") if recs else None) or "spot"
    )
    meta_parts.append(f"Pricing: [bold cyan]{active_pricing.upper()}[/bold cyan]")
    if use_case:
        meta_parts.append(f"Workload: [bold magenta]{use_case}[/bold magenta]")
    if is_custom_dist:
        meta_parts.append(f"Profile: [bold yellow]INTERPOLATED[/bold yellow]")
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
    table.add_column(
        "Cost/M In" if not is_custom_dist else "Cost/M In (Interp)",
        justify="right",
        style="bold yellow",
    )
    table.add_column(
        "Cost/M Out" if not is_custom_dist else "Cost/M Out (Interp)",
        justify="right",
        style="bold yellow",
    )
    table.add_column("TTFT" if not is_custom_dist else "TTFT (Interp)", justify="right")
    table.add_column(
        "NTPOT" if not is_custom_dist else "NTPOT (Interp)", justify="right"
    )
    table.add_column(
        "Output Tok/s" if not is_custom_dist else "Output Tok/s (Interp)",
        justify="right",
    )
    table.add_column("Server (TP)", justify="center")
    table.add_column(
        "Use Case (In/Out)" if not is_custom_dist else "Profile Status", style="dim"
    )

    for idx, r in enumerate(recs, start=1):
        rank_str = f"#{idx}" if idx > 1 else "⭐ #1"
        is_interp_row = r.get("is_interpolated", False)
        prefix = "~" if is_interp_row else ""
        cost_in_str = (
            f"{prefix}${r['input_cost_per_m']:.3f}"
            if r["input_cost_per_m"] is not None
            else "-"
        )
        cost_out_str = (
            f"{prefix}${r['output_cost_per_m']:.3f}"
            if r["output_cost_per_m"] is not None
            else "-"
        )
        ttft_str = f"{prefix}{r['ttft_ms']} ms" if r["ttft_ms"] is not None else "-"
        ntpot_str = f"{prefix}{r['ntpot_ms']} ms" if r["ntpot_ms"] is not None else "-"
        tok_s_str = (
            f"{prefix}{r['output_tokens_per_sec']:,}"
            if r["output_tokens_per_sec"] is not None
            else "-"
        )
        server_tp = (
            f"{r['model_server']} (TP={r['engine_params']['tensor_parallel_size']})"
        )
        if is_interp_row:
            use_case_str = "[yellow]interpolated[/yellow]"
        else:
            uc_name = r.get("use_case") or "-"
            in_len = r.get("average_input_length")
            out_len = r.get("average_output_length")
            if in_len is not None and out_len is not None:
                use_case_str = f"{uc_name} ({in_len}/{out_len})"
            else:
                use_case_str = uc_name

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
    if is_custom_dist:
        console.print(
            "[dim]~ Indicates interpolated metrics calculated for your custom token distribution.[/dim]"
        )

    top = recs[0]
    top_cost_in = (
        f"${top['input_cost_per_m']:.3f}/M in" if top.get("input_cost_per_m") else ""
    )
    top_cost_out = (
        f"(${top['output_cost_per_m']:.3f}/M out)"
        if top.get("output_cost_per_m")
        else ""
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
        selected_rec = None
        if apply:
            selected_rec = top
        elif format_type == "table":
            if len(recs) == 1:
                if Confirm.ask(
                    f"\nApply '{top['machine_type']}' to config/deployment_spec.yaml?",
                    default=True,
                ):
                    selected_rec = top
            else:
                max_rank = len(recs)
                choice = (
                    Prompt.ask(
                        f"\nSelect a recommendation to apply to config/deployment_spec.yaml (1-{max_rank}, [bold]y[/bold]=#1, [bold]n[/bold]=cancel)",
                        default="1",
                    )
                    .strip()
                    .lower()
                )

                if choice in ("y", "yes", "1"):
                    selected_rec = recs[0]
                elif choice in ("n", "no"):
                    selected_rec = None
                elif choice.isdigit():
                    num = int(choice)
                    if 1 <= num <= max_rank:
                        selected_rec = recs[num - 1]
                    else:
                        console.print(
                            f"[yellow]Invalid selection '{choice}'. Skipping configuration update.[/yellow]\n"
                        )
                else:
                    console.print(
                        f"[yellow]Invalid selection '{choice}'. Skipping configuration update.[/yellow]\n"
                    )

        if selected_rec:
            updated = apply_recommendation(selected_rec, Path.cwd())
            if updated:
                console.print(
                    f"[bold green]✅ Updated config/deployment_spec.yaml with machine type '{selected_rec['machine_type']}'.[/bold green]\n"
                )
    else:
        console.print(
            "[dim]Run inside your project workspace or pass --apply to update config/deployment_spec.yaml[/dim]\n"
        )


recommend_cmd = recommend
