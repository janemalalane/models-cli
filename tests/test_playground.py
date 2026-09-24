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

"""Unit tests for models-cli playground command."""

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from google.models.cli.main import app
from google.models.cli.playground.cmd_playground import (
    resolve_endpoint_and_base_url,
)

runner = CliRunner()


def test_playground_help():
    """Verify that playground --help returns 0 and displays options."""
    result = runner.invoke(app, ["playground", "--help"])
    assert result.exit_code == 0
    assert "Interactive chat playground" in result.output
    assert "--endpoint" in result.output
    assert "--model" in result.output
    assert "--stream" in result.output


def test_playground_missing_endpoint_error(monkeypatch):
    """Verify ClickException when no endpoint can be found or resolved."""
    import google.models.cli.playground.cmd_playground as pg_mod

    monkeypatch.setattr(pg_mod, "ensure_authenticated", lambda interactive: True)
    monkeypatch.setattr(pg_mod, "read_endpoint", lambda: None)
    monkeypatch.setattr(pg_mod.settings, "endpoint_url", None)

    # Mock aiplatform.Endpoint.list to return empty list
    with patch("google.cloud.aiplatform.Endpoint.list", return_value=[]):
        result = runner.invoke(app, ["playground", "hello"])
        assert result.exit_code != 0
        assert "No deployed endpoint found" in result.output


def test_playground_single_message_stream(monkeypatch):
    """Verify single message with streaming output."""
    import google.models.cli.playground.cmd_playground as pg_mod

    monkeypatch.setattr(pg_mod, "ensure_authenticated", lambda interactive: True)
    monkeypatch.setattr(
        pg_mod,
        "resolve_endpoint_and_base_url",
        lambda **kwargs: (
            "https://example-dns/v1/projects/1/locations/us-central1/endpoints/2/invoke/v1",
            "projects/1/locations/us-central1/endpoints/2",
            "gemma",
        ),
    )

    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [MagicMock(delta=MagicMock(content="Hello "))]
    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [MagicMock(delta=MagicMock(content="world!"))]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = [mock_chunk1, mock_chunk2]
    monkeypatch.setattr(pg_mod, "_get_openai_client", lambda base_url: mock_client)

    result = runner.invoke(app, ["playground", "Say hi"])
    assert result.exit_code == 0
    assert "Hello world!" in result.output

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["stream"] is True
    assert call_kwargs["messages"] == [{"role": "user", "content": "Say hi"}]


def test_playground_single_message_non_stream(monkeypatch):
    """Verify single message with --no-stream."""
    import google.models.cli.playground.cmd_playground as pg_mod

    monkeypatch.setattr(pg_mod, "ensure_authenticated", lambda interactive: True)
    monkeypatch.setattr(
        pg_mod,
        "resolve_endpoint_and_base_url",
        lambda **kwargs: (
            "https://example-dns/v1/projects/1/locations/us-central1/endpoints/2/invoke/v1",
            "projects/1/locations/us-central1/endpoints/2",
            "gemma",
        ),
    )

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="Non-stream response"))]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    monkeypatch.setattr(pg_mod, "_get_openai_client", lambda base_url: mock_client)

    result = runner.invoke(
        app, ["playground", "--no-stream", "-m", "Tell me something"]
    )
    assert result.exit_code == 0
    assert "Non-stream response" in result.output

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["stream"] is False
    assert call_kwargs["messages"] == [{"role": "user", "content": "Tell me something"}]


def test_playground_with_system_prompt(monkeypatch):
    """Verify system prompt is prepended to messages."""
    import google.models.cli.playground.cmd_playground as pg_mod

    monkeypatch.setattr(pg_mod, "ensure_authenticated", lambda interactive: True)
    monkeypatch.setattr(
        pg_mod,
        "resolve_endpoint_and_base_url",
        lambda **kwargs: ("https://test-url/invoke/v1", "ep-123", "gemma"),
    )

    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock(delta=MagicMock(content="Poetic answer"))]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = [mock_chunk]
    monkeypatch.setattr(pg_mod, "_get_openai_client", lambda base_url: mock_client)

    result = runner.invoke(
        app,
        ["playground", "-s", "You are a poet.", "Write a poem"],
    )
    assert result.exit_code == 0
    assert "Poetic answer" in result.output

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["messages"] == [
        {"role": "system", "content": "You are a poet."},
        {"role": "user", "content": "Write a poem"},
    ]


def test_playground_raw_output(monkeypatch):
    """Verify --raw outputs the serialized chat completion."""
    import google.models.cli.playground.cmd_playground as pg_mod

    monkeypatch.setattr(pg_mod, "ensure_authenticated", lambda interactive: True)
    monkeypatch.setattr(
        pg_mod,
        "resolve_endpoint_and_base_url",
        lambda **kwargs: ("https://test-url/invoke/v1", "ep-123", "gemma"),
    )

    mock_resp = MagicMock()
    mock_resp.model_dump_json.return_value = json.dumps(
        {"id": "chatcmpl-test", "content": "raw data"}
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    monkeypatch.setattr(pg_mod, "_get_openai_client", lambda base_url: mock_client)

    result = runner.invoke(app, ["playground", "--raw", "test prompt"])
    assert result.exit_code == 0
    assert '"id": "chatcmpl-test"' in result.output


def test_playground_interactive_repl_exit(monkeypatch):
    """Verify interactive REPL displays header and exits cleanly on 'exit'."""
    import google.models.cli.playground.cmd_playground as pg_mod

    monkeypatch.setattr(pg_mod, "ensure_authenticated", lambda interactive: True)
    monkeypatch.setattr(
        pg_mod,
        "resolve_endpoint_and_base_url",
        lambda **kwargs: (
            "https://test-url/invoke/v1",
            "projects/123/locations/us-central1/endpoints/456",
            "google/gemma-4-31B-it",
        ),
    )

    result = runner.invoke(app, ["playground"], input="exit\n")
    assert result.exit_code == 0
    assert "models-cli Playground" in result.output
    assert "projects/123/locations/us-central1/endpoints/456" in result.output
    assert "Goodbye!" in result.output


def test_resolve_endpoint_and_base_url_url_argument():
    """Verify direct HTTP(S) URL resolution and formatting."""
    url, _, _ = resolve_endpoint_and_base_url(
        endpoint_arg="https://my-host.vertexai.goog/invoke"
    )
    assert url == "https://my-host.vertexai.goog/invoke/v1"

    url2, _, _ = resolve_endpoint_and_base_url(
        endpoint_arg="https://my-host.vertexai.goog/invoke/v1/chat/completions"
    )
    assert url2 == "https://my-host.vertexai.goog/invoke/v1"


def test_resolve_endpoint_dedicated_dns_from_attribute():
    """Verify dedicated endpoint resolution when dedicated_endpoint_dns is provided."""
    mock_ep = MagicMock()
    mock_ep.resource_name = "projects/555587849335/locations/us-central1/endpoints/4414007571648610304"
    mock_ep.dedicated_endpoint_dns = "4414007571648610304.us-central1-555587849335.prediction.vertexai.goog"
    mock_ep.gca_resource.deployed_models = [MagicMock(display_name="kimi-k2.5")]

    with patch("google.cloud.aiplatform.Endpoint", return_value=mock_ep):
        url, rn, model = resolve_endpoint_and_base_url(
            endpoint_arg="projects/555587849335/locations/us-central1/endpoints/4414007571648610304"
        )
        assert (
            url
            == "https://4414007571648610304.us-central1-555587849335.prediction.vertexai.goog/v1/projects/555587849335/locations/us-central1/endpoints/4414007571648610304/invoke/v1"
        )
        assert rn == "projects/555587849335/locations/us-central1/endpoints/4414007571648610304"
        assert model == "kimi-k2.5"


def test_resolve_endpoint_dedicated_dns_constructed_when_empty_string():
    """Verify dedicated endpoint DNS is constructed when dedicated_endpoint_dns is empty but dedicated_endpoint_enabled is True."""
    mock_ep = MagicMock()
    mock_ep.resource_name = "projects/555587849335/locations/us-central1/endpoints/4414007571648610304"
    mock_ep.dedicated_endpoint_dns = ""
    mock_ep.gca_resource.dedicated_endpoint_dns = ""
    mock_ep.dedicated_endpoint_enabled = True
    mock_ep.gca_resource.deployed_models = []

    with patch("google.cloud.aiplatform.Endpoint", return_value=mock_ep):
        url, rn, model = resolve_endpoint_and_base_url(
            endpoint_arg="projects/555587849335/locations/us-central1/endpoints/4414007571648610304"
        )
        assert (
            url
            == "https://4414007571648610304.us-central1-555587849335.prediction.vertexai.goog/v1/projects/555587849335/locations/us-central1/endpoints/4414007571648610304/invoke/v1"
        )


def test_resolve_endpoint_rewrites_shared_aiplatform_url_for_dedicated():
    """Verify shared aiplatform.googleapis.com URL is rewritten to dedicated domain when endpoint is dedicated."""
    mock_ep = MagicMock()
    mock_ep.resource_name = "projects/555587849335/locations/us-central1/endpoints/4414007571648610304"
    mock_ep.dedicated_endpoint_dns = "4414007571648610304.us-central1-555587849335.prediction.vertexai.goog"
    mock_ep.gca_resource.deployed_models = []

    shared_url = "https://us-central1-aiplatform.googleapis.com/v1/projects/555587849335/locations/us-central1/endpoints/4414007571648610304/invoke"

    with patch("google.cloud.aiplatform.Endpoint", return_value=mock_ep):
        url, rn, _ = resolve_endpoint_and_base_url(endpoint_arg=shared_url)
        assert (
            url
            == "https://4414007571648610304.us-central1-555587849335.prediction.vertexai.goog/v1/projects/555587849335/locations/us-central1/endpoints/4414007571648610304/invoke/v1"
        )
        assert rn == "projects/555587849335/locations/us-central1/endpoints/4414007571648610304"
