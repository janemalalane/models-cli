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

"""Click helpers: lazy command loading and source-path display in --help."""

from __future__ import annotations

import importlib
import inspect
from typing import Any

import click


class LazyGroup(click.Group):
    """Click group that defers importing subcommand modules until needed.

    Register subcommands with `add_lazy_command(name, "module.path:obj",
    short_help)`. The module is imported only when the command is actually
    invoked (e.g. `tool name ...`) or when its own help is requested
    (`tool name --help`). The parent group's --help (`tool --help`) renders
    the supplied `short_help` strings directly without triggering any imports.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._lazy_commands: dict[str, tuple[str, str, bool]] = {}
        self._command_order: list[str] = []

    def add_command(self, cmd: click.Command, name: str | None = None) -> None:
        cmd_name = name or cmd.name
        if cmd_name and cmd_name not in self._command_order:
            self._command_order.append(cmd_name)
        super().add_command(cmd, name)

    def add_lazy_command(
        self,
        name: str,
        import_path: str,
        short_help: str,
        hidden: bool = False,
    ) -> None:
        self._lazy_commands[name] = (import_path, short_help, hidden)
        if name not in self._command_order:
            self._command_order.append(name)

    def list_commands(self, ctx: click.Context) -> list[str]:
        all_commands = set(super().list_commands(ctx)) | set(self._lazy_commands)
        ordered = [name for name in self._command_order if name in all_commands]
        remaining = [name for name in all_commands if name not in self._command_order]
        return ordered + remaining

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        if cmd_name in self._lazy_commands and cmd_name not in self.commands:
            import_path, _, hidden = self._lazy_commands[cmd_name]
            module_path, attr = import_path.split(":")
            cmd = getattr(importlib.import_module(module_path), attr)
            if hidden:
                cmd.hidden = True
            patch_source_in_help(cmd)
            self.commands[cmd_name] = cmd
        return super().get_command(ctx, cmd_name)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        rows: list[tuple[str, str]] = []
        for name in self.list_commands(ctx):
            if name in self.commands:
                cmd = self.commands[name]
                if getattr(cmd, "hidden", False):
                    continue
                rows.append((name, cmd.get_short_help_str(limit=1000)))
                continue
            lazy = self._lazy_commands.get(name)
            if lazy is not None and not lazy[2]:  # not hidden
                rows.append((name, lazy[1]))
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)


def _source_path(cmd: Any) -> str | None:
    """Resolve the absolute file path of a command's callback module."""
    cb = getattr(cmd, "callback", None)
    if cb is None:
        return None
    try:
        mod = importlib.import_module(cb.__module__)
        return inspect.getfile(mod)
    except (AttributeError, TypeError, OSError, ImportError):
        return None


def patch_source_in_help(cmd: Any) -> None:
    """Recursively patch all commands to show source location in --help epilog."""
    if getattr(cmd, "_source_patched", False):
        return

    original = cmd.format_epilog

    def _patched(ctx: click.Context, formatter: click.HelpFormatter) -> None:
        original(ctx, formatter)
        path = _source_path(cmd)
        if path:
            formatter.write("\n")
            formatter.write(f"Source: {path}\n")

    cmd.format_epilog = _patched
    cmd._source_patched = True

    if isinstance(cmd, click.Group):
        for sub in cmd.commands.values():
            patch_source_in_help(sub)
