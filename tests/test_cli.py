# Copyright 2026 Google LLC
from typer.testing import CliRunner
from google.models.cli import __version__
from google.models.cli.main import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "models-cli" in result.stdout
    assert "create" in result.stdout
    assert "recommend" in result.stdout
    assert "deploy" in result.stdout
    assert "benchmark" in result.stdout
    assert "eval" in result.stdout


def test_cli_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_cli_info():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "models-cli Environment Information" in result.stdout
