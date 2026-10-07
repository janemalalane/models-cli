import getpass
import sys

import pytest
from click.testing import CliRunner

from google.models.cli import __version__
from google.models.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def mock_getpass_stdin(monkeypatch):
    monkeypatch.setattr(
        getpass,
        "getpass",
        lambda prompt="", stream=None: sys.stdin.readline().rstrip("\n"),
    )


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "models-cli" in result.stdout
    assert "create" in result.stdout
    assert "recommend" in result.stdout
    assert "deploy" in result.stdout
    assert "benchmark" in result.stdout
    assert "eval" in result.stdout
    # Unboxed logo displayed on root --help
    assert "█▀▄▀█" in result.stdout
    assert "Evaluate. Optimize. Deploy. Benchmark." in result.stdout


def test_cli_subcommand_help_no_banner():
    result = runner.invoke(app, ["deploy", "--help"])
    assert result.exit_code == 0
    assert "Deploys an open model container" in result.stdout
    assert "█▀▄▀█" not in result.stdout


def test_cli_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_cli_info():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "models-cli Environment Information" in result.stdout
    # Banner should NOT be in info command (matching agents-cli)
    assert "█▀▄▀█" not in result.stdout


def test_banner_rendering():
    from io import StringIO

    from rich.console import Console

    from google.models.cli.common.banner import display_banner, get_banner_panel

    panel = get_banner_panel()
    assert panel.title is not None
    assert "models-cli" in str(panel.title)

    buf = StringIO()
    c = Console(file=buf, force_terminal=False, color_system=None)
    display_banner(c)
    output = buf.getvalue()
    assert "█▀▄▀█" in output
    assert "Evaluate. Optimize. Deploy. Benchmark." in output
    assert "Lifecycle toolkit for open-weights models on Google Cloud." in output


def test_non_interactive_create_no_banner(tmp_path):
    result = runner.invoke(
        app,
        [
            "create",
            "test-non-interactive",
            "-y",
            "--output-dir",
            str(tmp_path),
            "--skip-checks",
            "--bucket",
            "test-bucket",
        ],
    )
    assert result.exit_code == 0
    assert "█▀▄▀█" not in result.output
    assert "Setting up..." not in result.output
    assert "Workspace ready" in result.output


def test_interactive_create_shows_banner(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "google.models.cli.scaffold.cmd_scaffold.resolve_gcp_project",
        lambda override_project=None: "active-gcp-proj",
    )
    user_inputs = (
        "google/gemma-4-31B-it\n"  # 1. model
        "vllm\n"  # 2. engine
        "us-central1\n"  # 3. region
        "my-proj\n"  # 4. project id
        "my-bucket\n"  # 5. bucket
        "\n"  # 6. base path
        "\n"  # 7. service account
        "\n"  # 8. hf token
        "\n"  # 9. endpoint url
    )
    result = runner.invoke(
        app,
        [
            "create",
            "test-interactive-banner",
            "-i",
            "--output-dir",
            str(tmp_path),
            "--skip-checks",
        ],
        input=user_inputs,
    )
    assert result.exit_code == 0
    assert "█▀▄▀█" in result.output
    assert "Setting up..." in result.output
    assert "Workspace ready" in result.output
