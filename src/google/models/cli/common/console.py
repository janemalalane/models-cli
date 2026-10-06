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

"""Rich console helpers and UI styling for models-cli."""

from typing import Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()
error_console = Console(stderr=True)


def print_banner(title: str = "models-cli", subtitle: str | None = None) -> None:
    """Renders formatted application banner."""
    content = Text()
    content.append("⚡ ", style="bold yellow")
    content.append(title, style="bold cyan")
    if subtitle:
        content.append(f"\n{subtitle}", style="dim")
    console.print(Panel(content, border_style="blue", expand=False))


def print_success(message: str) -> None:
    """Prints green success message."""
    console.print(f"[bold green]✅ Success:[/bold green] {message}")


def print_warning(message: str) -> None:
    """Prints yellow warning message."""
    console.print(f"[bold yellow]⚠️  Warning:[/bold yellow] {message}")


def print_error(message: str, hint: str | None = None) -> None:
    """Prints red error panel with optional resolution hint."""
    text = Text()
    text.append(f"❌ {message}\n", style="bold red")
    if hint:
        text.append(f"\n💡 Hint: {hint}", style="yellow")
    error_console.print(Panel(text, border_style="red", title="Error", expand=False))


def print_info(message: str) -> None:
    """Prints informational message."""
    console.print(f"[bold blue]ℹ️  Info:[/bold blue] {message}")


def create_table(title: str, columns: list[tuple[str, str]]) -> Table:
    """Creates formatted Rich table with column styles.

    Args:
        title: Table header title.
        columns: List of (column_name, style) tuples.

    Returns:
        Rich Table instance.
    """
    table = Table(title=title, header_style="bold magenta", border_style="dim")
    for name, style in columns:
        table.add_column(name, style=style)
    return table
