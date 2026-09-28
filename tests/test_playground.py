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
    assert "Interactive playground" in result.output
    assert "--endpoint" in result.output
    assert "--model" in result.output
    assert "--stream" in result.output
    assert "--api" in result.output


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


def test_playground_single_message_stream_chat(monkeypatch):
    """Verify single message with streaming output using default Chat Completions API."""
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


def test_playground_single_message_non_stream_chat(monkeypatch):
    """Verify single message with --no-stream using default Chat Completions API."""
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
    mock_resp.choices = [MagicMock(message=MagicMock(content="Non-stream chat response"))]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_resp
    monkeypatch.setattr(pg_mod, "_get_openai_client", lambda base_url: mock_client)

    result = runner.invoke(
        app, ["playground", "--no-stream", "-m", "Tell me something"]
    )
    assert result.exit_code == 0
    assert "Non-stream chat response" in result.output

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["stream"] is False
    assert call_kwargs["messages"] == [{"role": "user", "content": "Tell me something"}]


def test_playground_with_system_prompt_chat(monkeypatch):
    """Verify system prompt is included in Chat Completions API mode."""
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


def test_playground_completions_api_flag(monkeypatch):
    """Verify --api completions uses raw completions."""
    import google.models.cli.playground.cmd_playground as pg_mod

    monkeypatch.setattr(pg_mod, "ensure_authenticated", lambda interactive: True)
    monkeypatch.setattr(
        pg_mod,
        "resolve_endpoint_and_base_url",
        lambda **kwargs: ("https://test-url/invoke/v1", "ep-123", "gemma"),
    )

    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock(text="Completions response")]

    mock_client = MagicMock()
    mock_client.completions.create.return_value = [mock_chunk]
    monkeypatch.setattr(pg_mod, "_get_openai_client", lambda base_url: mock_client)

    result = runner.invoke(
        app,
        ["playground", "--api", "completions", "-s", "System info", "Hi"],
    )
    assert result.exit_code == 0
    assert "Completions response" in result.output

    call_kwargs = mock_client.completions.create.call_args[1]
    assert call_kwargs["prompt"] == "System info\n\nHi"


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
        {"id": "chatcmpl-test", "text": "raw data"}
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

    url3, _, _ = resolve_endpoint_and_base_url(
        endpoint_arg="https://my-host.vertexai.goog/v1/projects/123/locations/us-central1/endpoints/456/invoke"
    )
    assert (
        url3
        == "https://my-host.vertexai.goog/v1beta1/projects/123/locations/us-central1/endpoints/456/invoke/v1"
    )


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
            == "https://4414007571648610304.us-central1-555587849335.prediction.vertexai.goog/v1beta1/projects/555587849335/locations/us-central1/endpoints/4414007571648610304/invoke/v1"
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
        url, _, _ = resolve_endpoint_and_base_url(
            endpoint_arg="projects/555587849335/locations/us-central1/endpoints/4414007571648610304"
        )
        assert (
            url
            == "https://4414007571648610304.us-central1-555587849335.prediction.vertexai.goog/v1beta1/projects/555587849335/locations/us-central1/endpoints/4414007571648610304/invoke/v1"
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
            == "https://4414007571648610304.us-central1-555587849335.prediction.vertexai.goog/v1beta1/projects/555587849335/locations/us-central1/endpoints/4414007571648610304/invoke/v1"
        )
        assert rn == "projects/555587849335/locations/us-central1/endpoints/4414007571648610304"


def test_extract_reasoning_and_content_attribute():
    """Verify _extract_reasoning_and_content extracts reasoning_content attribute when present."""
    from google.models.cli.playground.cmd_playground import _extract_reasoning_and_content

    mock_obj = MagicMock()
    mock_obj.reasoning_content = "Step-by-step thinking"
    mock_obj.content = "Final response"
    reasoning, content = _extract_reasoning_and_content(mock_obj)
    assert reasoning == "Step-by-step thinking"
    assert content == "Final response"


def test_extract_reasoning_and_content_content_only():
    """Verify _extract_reasoning_and_content returns None for reasoning when only content is present."""
    from google.models.cli.playground.cmd_playground import _extract_reasoning_and_content

    mock_obj = MagicMock(spec=["content"])
    mock_obj.content = "Direct answer"
    reasoning, content = _extract_reasoning_and_content(mock_obj)
    assert reasoning is None
    assert content == "Direct answer"


def test_playground_repl_with_api_reasoning_content(monkeypatch):
    """Verify REPL displays [Thought] when API returns reasoning_content attribute."""
    import google.models.cli.playground.cmd_playground as pg_mod

    monkeypatch.setattr(pg_mod, "ensure_authenticated", lambda interactive: True)
    monkeypatch.setattr(
        pg_mod,
        "resolve_endpoint_and_base_url",
        lambda **kwargs: ("https://test-url/invoke/v1", "ep-123", "deepseek-r1"),
    )

    mock_delta1 = MagicMock(spec=["reasoning_content", "content"])
    mock_delta1.reasoning_content = "Thinking about greeting"
    mock_delta1.content = None

    mock_delta2 = MagicMock(spec=["reasoning_content", "content"])
    mock_delta2.reasoning_content = None
    mock_delta2.content = "Hello there!"

    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [MagicMock(delta=mock_delta1)]
    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [MagicMock(delta=mock_delta2)]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = [mock_chunk1, mock_chunk2]
    monkeypatch.setattr(pg_mod, "_get_openai_client", lambda base_url: mock_client)

    result = runner.invoke(app, ["playground"], input="hi\nexit\n")
    assert result.exit_code == 0
    assert "[Thought]" in result.output
    assert "Thinking about greeting" in result.output
    assert "Model > Hello there!" in result.output


def test_playground_repl_content_only_no_splitting(monkeypatch):
    """Verify REPL displays raw content under Model > when only content is provided by API."""
    import google.models.cli.playground.cmd_playground as pg_mod

    monkeypatch.setattr(pg_mod, "ensure_authenticated", lambda interactive: True)
    monkeypatch.setattr(
        pg_mod,
        "resolve_endpoint_and_base_url",
        lambda **kwargs: ("https://test-url/invoke/v1", "ep-123", "muse-glimmer"),
    )

    mock_delta = MagicMock(spec=["content"])
    mock_delta.content = "to=selfhi\nWe need to respond.assistant to=userHi!"

    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock(delta=mock_delta)]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = [mock_chunk]
    monkeypatch.setattr(pg_mod, "_get_openai_client", lambda base_url: mock_client)

    result = runner.invoke(app, ["playground"], input="hi\nexit\n")
    assert result.exit_code == 0
    assert "[Thought]" not in result.output
    assert "Model > to=selfhi\nWe need to respond.assistant to=userHi!" in result.output
