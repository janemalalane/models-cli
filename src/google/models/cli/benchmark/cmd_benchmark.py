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

"""Benchmarking command for models-cli."""

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from google.models.cli._gcp_project import get_gcp_access_token
from google.models.cli.common.auth import ensure_authenticated
from google.models.cli.benchmark.benchmark_utils import (
    find_inference_perf_command,
    generate_benchmark_config,
    generate_markdown_report,
    parse_lifecycle_metrics,
)
from google.models.cli.common.config import settings
from google.models.cli.common.constants import DEFAULT_MODEL_REPO

console = Console()
benchmark_cmd = typer.Typer(
    name="benchmark",
    help="Run performance & load testing with inference-perf and generate analysis reports",
)


@benchmark_cmd.callback(invoke_without_command=True)
def benchmark(
    endpoint: Optional[str] = typer.Option(
        None,
        "--endpoint",
        "-e",
        help="Target model serving endpoint base URL.",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to inference-perf config YAML file.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name identifier.",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save benchmark metrics and reports.",
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Run against inference-perf mock server without requiring a live cloud endpoint.",
    ),
) -> None:
    """Runs performance benchmarks with inference-perf and outputs comparison metrics."""
    target_model = model or settings.model_id or DEFAULT_MODEL_REPO
    target_endpoint = endpoint or settings.endpoint_url or "http://localhost:8080"

    # Automatically fetch fresh GCP access token for remote endpoints
    auth_token = settings.auth_token or ""
    if (
        not auth_token
        and not mock
        and (
            "aiplatform.googleapis.com" in target_endpoint
            or "http" in target_endpoint
            and "localhost" not in target_endpoint
            and "127.0.0.1" not in target_endpoint
        )
    ):
        if not ensure_authenticated(interactive=True):
            raise typer.Exit(1)
        fetched_token = get_gcp_access_token()
        if fetched_token:
            auth_token = fetched_token

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = (
        Path(output_dir) if output_dir else Path(f"reports/benchmark-{timestamp}")
    )
    out_path.mkdir(parents=True, exist_ok=True)

    # Locate or generate config (without saving sensitive auth tokens to disk)
    input_config_path = (
        Path(config)
        if config
        else (
            Path("tests/benchmark/config.yaml")
            if Path("tests/benchmark/config.yaml").is_file()
            else None
        )
    )
    active_config_path = out_path / "inference_perf_config.yaml"

    generate_benchmark_config(
        template_config_path=input_config_path,
        output_config_path=active_config_path,
        model_name=target_model,
        endpoint_url=target_endpoint,
        mock=mock,
    )

    console.print(
        f"\n[bold cyan]⚡ Running Performance Benchmark for[/bold cyan] [bold yellow]{target_model}[/bold yellow]"
    )
    console.print(f"   • Endpoint:   [bold]{target_endpoint}[/bold]")
    console.print(f"   • Config:     [bold]{active_config_path}[/bold]")
    console.print(f"   • Output Dir: [bold]{out_path}[/bold]")
    console.print(f"   • Mock Mode:  [bold]{mock}[/bold]\n")

    # Run inference-perf if available
    inf_perf_cmd = find_inference_perf_command()
    if inf_perf_cmd:
        cmd = list(inf_perf_cmd) + [
            "--config",
            str(active_config_path),
            "--storage.local_storage.path",
            str(out_path),
        ]
        # Pass auth token via CLI flag at runtime to avoid storing secrets in YAML files
        if auth_token:
            bearer_hdr = (
                auth_token
                if auth_token.startswith("Bearer ")
                else f"Bearer {auth_token}"
            )
            cmd.extend(["--server.http_headers.Authorization", bearer_hdr])

        # Redact token for safe console logging
        display_cmd = []
        skip_next = False
        for idx, arg in enumerate(cmd):
            if skip_next:
                display_cmd.append("[REDACTED_AUTH_TOKEN]")
                skip_next = False
            elif arg == "--server.http_headers.Authorization":
                display_cmd.append(arg)
                skip_next = True
            else:
                display_cmd.append(arg)

        console.print(f"Executing: [dim]{' '.join(display_cmd)}[/dim]")
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            console.print(
                f"[bold yellow]⚠️ inference-perf execution warning: {e}[/bold yellow]"
            )
    else:
        if mock:
            console.print(
                "[dim]inference-perf CLI binary not found in PATH; parsing synthetic metrics for mock mode.[/dim]"
            )
        else:
            console.print(
                "[bold red]❌ inference-perf is required to benchmark a live endpoint.[/bold red]"
            )
            console.print(
                "Please install dependencies with: [cyan]pip install google-models-cli[/cyan] or [cyan]pip install inference-perf[/cyan]"
            )
            raise typer.Exit(1)

    metrics = parse_lifecycle_metrics(out_path)
    report_file = generate_markdown_report(out_path, target_model, metrics)

    # Display summary table
    ttft = metrics.get("ttft_ms", {})
    tpot = metrics.get("tpot_ms", {})

    table = Table(
        title="Inference Benchmark Summary",
        border_style="dim",
        header_style="bold magenta",
    )
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="bold green", justify="right")
    table.add_column("Target SLA", justify="right", style="dim")

    table.add_row("Total Requests", str(metrics.get("total_requests", 0)), "-")
    table.add_row(
        "Throughput (QPS)", f"{metrics.get('request_throughput_qps', 0):.2f}", "> 5.0"
    )
    table.add_row(
        "Output Tokens/s",
        f"{metrics.get('token_throughput_tokens_sec', 0):.1f}",
        "> 200",
    )
    table.add_row("TTFT (p50)", f"{ttft.get('p50', 0):.1f} ms", "< 80 ms")
    table.add_row("TTFT (p90)", f"{ttft.get('p90', 0):.1f} ms", "< 150 ms")
    table.add_row("TPOT (p50)", f"{tpot.get('p50', 0):.1f} ms", "< 20 ms")
    table.add_row("TPOT (p90)", f"{tpot.get('p90', 0):.1f} ms", "< 30 ms")

    console.print(table)
    console.print(
        f"\n[bold green]✅ Benchmark Completed![/bold green] Detailed report generated at: [cyan]{report_file}[/cyan]\n"
    )
