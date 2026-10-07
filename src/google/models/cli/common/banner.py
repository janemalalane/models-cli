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

"""Banner and welcome UI styling for models-cli."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel

from google.models.cli import __version__

BANNER_LINE_1 = " █▀▄▀█ █▀█ █▀▄ █▀▀ █   █▀   █▀▀ █  █"
BANNER_LINE_2 = " █ █ █ █▄█ █▄▀ ██▄ █▄  ▄█   █▄▄ █▄ █"

TAGLINE = "Evaluate. Optimize. Deploy. Benchmark."
SUBTITLE = "Lifecycle toolkit for open-weights models on Google Cloud."


def get_banner_panel(version: str | None = None) -> Panel:
    """Creates the signature models-cli banner Panel (for optional boxed display)."""
    ver = version or __version__
    content = (
        f"[blue bold]{BANNER_LINE_1}[/]\n"
        f"[cyan bold]{BANNER_LINE_2}[/]\n\n"
        f" {TAGLINE}\n"
        f" [dim]{SUBTITLE}[/]"
    )
    return Panel(
        content,
        title=f"[bold]models-cli[/] v{ver}",
        border_style="blue",
        padding=(1, 2),
    )


def display_banner(console: Console | None = None) -> None:
    """Print the models-cli ASCII art logo and tagline unboxed, matching agents-cli."""
    c = console or Console(file=sys.stdout)
    c.print(f"[blue bold]{BANNER_LINE_1}[/]")
    c.print(f"[cyan bold]{BANNER_LINE_2}[/]")
    c.print()
    c.print(f" {TAGLINE}")
    c.print(f" [dim]{SUBTITLE}[/]")
