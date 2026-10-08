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

"""Playground command for sending messages to deployed endpoints via Completions or Chat Completions API."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import click
import google.auth
import google.auth.transport.requests
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from google.models.cli._gcp_project import resolve_gcp_project
from google.models.cli._project import read_deployment_config
from google.models.cli.common.auth import ensure_authenticated
from google.models.cli.common.config import settings
from google.models.cli.common.constants import DEFAULT_MODEL_REPO, DEFAULT_REGION
from google.models.cli.deploy._operation import read_endpoint, read_endpoint_url

logger = logging.getLogger(__name__)
console = Console()


def _extract_reasoning_and_content(obj: Any) -> tuple[str | None, str | None]:
    """Extracts reasoning_content and content attributes from an API message or delta."""
    if not obj:
        return None, None

    # 1. Reasoning content attribute (e.g. vLLM with --reasoning-parser, DeepSeek-R1)
    reasoning = getattr(obj, "reasoning_content", None)
    if not isinstance(reasoning, str):
        extra = getattr(obj, "model_extra", None)
        if isinstance(extra, dict):
            extra_val = extra.get("reasoning_content")
            reasoning = extra_val if isinstance(extra_val, str) else None
        else:
            reasoning = None
    if not reasoning and isinstance(obj, dict):
        dict_val = obj.get("reasoning_content")
        reasoning = dict_val if isinstance(dict_val, str) else None

    # 2. Main content attribute
    content = getattr(obj, "content", None)
    if not isinstance(content, str):
        if isinstance(obj, dict):
            dict_val = obj.get("content")
            content = dict_val if isinstance(dict_val, str) else None
        else:
            content = None

    return (
        reasoning if reasoning else None,
        content if content is not None else None,
    )


def _get_auth_token() -> str:
    """Retrieve and refresh Google Cloud credentials token."""
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    if not creds.valid:
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)
    return str(creds.token)


def _get_openai_client(base_url: str) -> OpenAI:
    """Instantiate OpenAI client configured for the Gemini Enterprise Online Prediction endpoint."""
    cleaned_url = base_url.rstrip("/")
    if not cleaned_url.endswith("/v1"):
        cleaned_url = f"{cleaned_url}/v1"
    if "localhost" in cleaned_url or "127.0.0.1" in cleaned_url:
        try:
            token = _get_auth_token()
        except Exception:
            token = "dummy-token"
    else:
        token = _get_auth_token()
    return OpenAI(base_url=cleaned_url, api_key=token)


def _resolve_endpoint_dns_and_model(
    ep: Any,
    rn: str,
) -> tuple[str | None, str | None, str]:
    """Extracts dedicated DNS, model name, and normalized resource name from an Endpoint object."""
    actual_rn = getattr(ep, "resource_name", None) or rn
    model_name = None
    gca = getattr(ep, "gca_resource", None)
    if gca and getattr(gca, "deployed_models", None):
        model_name = gca.deployed_models[0].display_name

    dns = getattr(ep, "dedicated_endpoint_dns", None)
    if not dns and gca:
        dns = getattr(gca, "dedicated_endpoint_dns", None)

    is_dedicated = (
        getattr(ep, "dedicated_endpoint_enabled", False)
        or (gca and getattr(gca, "dedicated_endpoint_enabled", False))
    )
    if not dns and is_dedicated:
        ep_match = re.search(
            r"projects/([^/]+)/locations/([^/]+)/endpoints/([^/]+)",
            actual_rn,
        )
        if ep_match:
            dns = f"{ep_match.group(3)}.{ep_match.group(2)}-{ep_match.group(1)}.prediction.vertexai.goog"

    return dns, model_name, actual_rn


def resolve_endpoint_and_base_url(
    endpoint_arg: str | None = None,
    project: str | None = None,
    location: str | None = None,
) -> tuple[str, str, str | None]:
    """Resolves target endpoint to a Chat Completions base_url.

    Returns:
        tuple of (base_url, endpoint_display_name, default_model_name)
    """
    from google.cloud import aiplatform

    resolved_project = resolve_gcp_project(
        override_project=project or settings.google_cloud_project
    )
    resolved_location = location or settings.google_cloud_location or DEFAULT_REGION

    endpoint_target: str | None = (
        endpoint_arg
        or settings.endpoint_url
        or os.environ.get("ENDPOINT_URL")
        or read_endpoint_url()
        or read_endpoint()
        or os.environ.get("ENDPOINT")
    )

    # If no explicit endpoint provided or recorded, discover via Gemini Enterprise Online Prediction
    if not endpoint_target:
        try:
            aiplatform.init(project=resolved_project, location=resolved_location)
            deploy_cfg = read_deployment_config()
            target_display = deploy_cfg.display_name

            endpoints = aiplatform.Endpoint.list(
                project=resolved_project,
                location=resolved_location,
                order_by="create_time desc",
            )
            # Find endpoint with matching display_name or any active endpoint with deployed models
            chosen_ep = None
            if target_display:
                for ep in endpoints:
                    if (
                        ep.display_name == target_display
                        and ep.gca_resource.deployed_models
                    ):
                        chosen_ep = ep
                        break

            if not chosen_ep:
                for ep in endpoints:
                    if ep.gca_resource.deployed_models:
                        chosen_ep = ep
                        break

            if not chosen_ep and target_display:
                for ep in endpoints:
                    if ep.display_name == target_display:
                        chosen_ep = ep
                        break

            if chosen_ep:
                endpoint_target = chosen_ep.resource_name
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not auto-discover endpoints from Gemini Enterprise Online Prediction: %s", e)

    if not endpoint_target:
        raise click.ClickException(
            "No deployed endpoint found.\n"
            "Please specify --endpoint <RESOURCE_NAME_OR_URL> or deploy a model first using 'models-cli deploy'."
        )

    # 1. Check if endpoint_target is a full HTTP(S) URL
    if endpoint_target.startswith(("http://", "https://")):
        match = re.search(
            r"(projects/[^/]+/locations/[^/]+/endpoints/\d+)", endpoint_target
        )
        rn = match.group(1) if match else endpoint_target
        model_name = None

        if match:
            rn = match.group(1)
            try:
                aiplatform.init(project=resolved_project, location=resolved_location)
                ep = aiplatform.Endpoint(rn)
                dns, discovered_model, actual_rn = _resolve_endpoint_dns_and_model(ep, rn)
                if discovered_model:
                    model_name = discovered_model

                if "aiplatform.googleapis.com" in endpoint_target and dns:
                    return f"https://{dns}/v1beta1/{actual_rn}/invoke/v1", actual_rn, model_name
            except Exception as e:  # noqa: BLE001
                logger.debug("Failed inspecting endpoint from URL %s: %s", rn, e)

        # Direct URL formatting
        url = endpoint_target.rstrip("/")
        url = re.sub(
            r"/(?:v1|v1beta1)/(projects/[^/]+/locations/[^/]+/endpoints/)",
            r"/v1beta1/\1",
            url,
        )
        url = url.removesuffix("/chat/completions").removesuffix("/completions")
        if url.endswith("/invoke"):
            url = f"{url}/v1"
        elif not url.endswith("/v1"):
            if (
                "endpoints/" in url
                or "aiplatform.googleapis.com" in url
                or "prediction.vertexai.goog" in url
            ):
                url = f"{url}/invoke/v1"
            else:
                url = f"{url}/v1"
        return url, rn, model_name

    # 2. Check if endpoint_target is an endpoint resource name
    if endpoint_target.startswith("projects/"):
        rn = endpoint_target
    else:
        # Assume endpoint_target is an endpoint ID
        rn = f"projects/{resolved_project}/locations/{resolved_location}/endpoints/{endpoint_target}"

    aiplatform.init(project=resolved_project, location=resolved_location)
    ep = aiplatform.Endpoint(rn)
    dns, model_name, actual_rn = _resolve_endpoint_dns_and_model(ep, rn)

    if dns:
        base_url = f"https://{dns}/v1beta1/{actual_rn}/invoke/v1"
    else:
        # Extract location from resource name
        loc_match = re.search(r"/locations/([^/]+)/", actual_rn)
        loc = loc_match.group(1) if loc_match else resolved_location
        base_url = f"https://{loc}-aiplatform.googleapis.com/v1beta1/{actual_rn}/invoke/v1"

    return base_url, actual_rn, model_name


@click.command("playground")
@click.argument("prompt", required=False, default=None)
@click.option(
    "--message",
    "-m",
    default=None,
    help="Single prompt message to send to the endpoint.",
)
@click.option(
    "--system",
    "-s",
    "--system-prompt",
    default=None,
    help="System prompt to guide the model behavior.",
)
@click.option(
    "--endpoint",
    "-e",
    default=None,
    help="Gemini Enterprise Online Prediction endpoint resource name, ID, or dedicated URL. Defaults to deployment_metadata.json or active endpoint.",
)
@click.option(
    "--model",
    default=None,
    help="Model identifier to specify in the Chat Completions request. Defaults to MODEL_ID or default model repo.",
)
@click.option(
    "--stream/--no-stream",
    default=True,
    help="Stream response tokens in real-time (default: True).",
)
@click.option(
    "--project",
    "-p",
    default=None,
    help="GCP Project ID. Defaults to environment or gcloud config.",
)
@click.option(
    "--region",
    "-r",
    "--location",
    default=None,
    help="GCP Region / location (e.g. us-central1).",
)
@click.option(
    "--raw",
    is_flag=True,
    default=False,
    help="Print raw JSON response from endpoint instead of formatted text.",
)
@click.option(
    "--api",
    type=click.Choice(["chat", "completions"], case_sensitive=False),
    default="chat",
    show_default=True,
    help="Target OpenAI API endpoint ('chat' or 'completions').",
)
def playground(
    prompt: str | None = None,
    message: str | None = None,
    system: str | None = None,
    endpoint: str | None = None,
    model: str | None = None,
    stream: bool = True,
    project: str | None = None,
    region: str | None = None,
    raw: bool = False,
    api: str = "chat",
) -> None:
    """Interactive playground and test interface for deployed endpoints.

    Sends requests to the deployed model serving container via the OpenAI-compatible
    Chat Completions or Completions API.

    \b
    Examples:
      # Send a single prompt via Chat Completions API (default)
      models-cli playground "Tell me a joke about computers"

      # Send a prompt with a system prompt
      models-cli playground -s "You are a poet." "Explain neural networks"

      # Use raw Completions API
      models-cli playground --api completions "Tell me a joke about computers"

      # Launch interactive multi-turn session
      models-cli playground

      # Send message to a specific endpoint
      models-cli playground -e projects/123/locations/us-central1/endpoints/456 "Hello"
    """
    if not ensure_authenticated(interactive=True):
        raise click.exceptions.Exit(1)

    base_url, ep_identifier, deployed_model_name = resolve_endpoint_and_base_url(
        endpoint_arg=endpoint,
        project=project,
        location=region,
    )

    initial_model = (
        model or settings.model_id or deployed_model_name or DEFAULT_MODEL_REPO
    )
    resolved_model = initial_model

    client = _get_openai_client(base_url)

    # Auto-detect actual model served on the endpoint to avoid 404 model not found errors
    try:
        remote_models = [
            m.id for m in client.models.list().data if m and getattr(m, "id", None)
        ]
    except Exception:
        if "/v1beta1/" in base_url:
            try:
                alt_client = _get_openai_client(base_url.replace("/v1beta1/", "/v1/"))
                remote_models = [
                    m.id
                    for m in alt_client.models.list().data
                    if m and getattr(m, "id", None)
                ]
            except Exception:
                remote_models = []
        else:
            remote_models = []

    if remote_models and resolved_model not in remote_models:
        if len(remote_models) == 1:
            logger.debug(
                "Requested model '%s' not recognized by endpoint. Auto-selecting served model '%s'.",
                resolved_model,
                remote_models[0],
            )
            resolved_model = remote_models[0]
        elif not model:
            resolved_model = remote_models[0]

    effective_prompt = message if message is not None else prompt
    use_chat = api.lower() == "chat"

    # 1. Single-shot prompt mode
    if effective_prompt is not None:
        if use_chat:
            messages: list[dict[str, Any]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": effective_prompt})

            if raw:
                resp = client.chat.completions.create(
                    model=resolved_model,
                    messages=messages,
                    stream=False,
                )
                click.echo(resp.model_dump_json(indent=2))
                return

            if stream:
                response_stream = client.chat.completions.create(
                    model=resolved_model,
                    messages=messages,
                    stream=True,
                )
                thought_header_printed = False
                content_started = False
                for chunk in response_stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    reasoning, content = _extract_reasoning_and_content(delta)
                    if reasoning:
                        if not thought_header_printed:
                            console.print("[dim][Thought][/dim]")
                            thought_header_printed = True
                        console.print(reasoning, style="dim", end="", highlight=False, markup=False)
                    if content:
                        if thought_header_printed and not content_started:
                            console.print()
                            console.print()
                            content_started = True
                        console.print(content, end="", highlight=False, markup=False)
                console.print()
            else:
                response = client.chat.completions.create(
                    model=resolved_model,
                    messages=messages,
                    stream=False,
                )
                msg = response.choices[0].message if response.choices else None
                reasoning, content = _extract_reasoning_and_content(msg)
                if reasoning:
                    console.print("[dim][Thought][/dim]")
                    console.print(reasoning, style="dim", highlight=False, markup=False)
                    console.print()
                console.print(Markdown(content or ""))
            return

        # Completions API (default)
        prompt_text = (
            f"{system}\n\n{effective_prompt}" if system else effective_prompt
        )

        if raw:
            resp = client.completions.create(
                model=resolved_model,
                prompt=prompt_text,
                stream=False,
            )
            click.echo(resp.model_dump_json(indent=2))
            return

        if stream:
            comp_stream = client.completions.create(
                model=resolved_model,
                prompt=prompt_text,
                stream=True,
            )
            for chunk in comp_stream:
                if chunk.choices and chunk.choices[0].text:
                    console.print(chunk.choices[0].text, end="")
            console.print()
        else:
            comp_resp = client.completions.create(
                model=resolved_model,
                prompt=prompt_text,
                stream=False,
            )
            content = (
                comp_resp.choices[0].text if comp_resp.choices else ""
            ) or ""
            console.print(Markdown(content))
            return

    # 2. Interactive session (REPL)
    display_model_str = (
        f"{initial_model} [dim](serving ID: {resolved_model})[/dim]"
        if initial_model and initial_model != resolved_model
        else resolved_model
    )
    header_info = (
        f"[bold cyan]Endpoint:[/bold cyan]  {ep_identifier}\n"
        f"[bold cyan]Model:[/bold cyan]     {display_model_str}\n"
        f"[bold cyan]Base URL:[/bold cyan]  {base_url}\n"
        f"[bold cyan]API:[/bold cyan]       {api.lower()}\n\n"
        "[dim]Commands: [bold]'exit'[/bold], [bold]'quit'[/bold], or [bold]'q'[/bold] to leave • [bold]'/clear'[/bold] to reset history[/dim]"
    )
    console.print(
        Panel(header_info, title="🤖 models-cli Playground", border_style="cyan")
    )

    history: list[dict[str, Any]] = []
    if system:
        history.append({"role": "system", "content": system})

    while True:
        try:
            user_input = console.input("\n[bold cyan]You > [/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session ended.[/yellow]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[yellow]Goodbye![/yellow]")
            break
        if user_input.lower() in ("/clear", "clear"):
            history = [{"role": "system", "content": system}] if system else []
            console.print("[dim]Conversation history cleared.[/dim]")
            continue

        # Refresh client token for long-running sessions
        client = _get_openai_client(base_url)

        if use_chat:
            history.append({"role": "user", "content": user_input})
            if stream:
                thought_header_printed = False
                model_label_printed = False
                collected_content: list[str] = []
                try:
                    chat_stream = client.chat.completions.create(
                        model=resolved_model,
                        messages=history,
                        stream=True,
                    )
                    for chunk in chat_stream:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        reasoning, content = _extract_reasoning_and_content(delta)

                        if reasoning:
                            if not thought_header_printed:
                                console.print("[dim][Thought][/dim]")
                                thought_header_printed = True
                            console.print(
                                reasoning,
                                style="dim",
                                end="",
                                highlight=False,
                                markup=False,
                            )

                        if content:
                            if thought_header_printed and not model_label_printed:
                                console.print()
                                console.print()
                            if not model_label_printed:
                                console.print(
                                    "[bold green]Model > [/bold green]", end=""
                                )
                                model_label_printed = True
                            console.print(
                                content, end="", highlight=False, markup=False
                            )
                            collected_content.append(content)

                    if not model_label_printed and not thought_header_printed:
                        console.print("[bold green]Model > [/bold green]", end="")

                    console.print()
                    assistant_reply = "".join(collected_content)
                    history.append({"role": "assistant", "content": assistant_reply})
                except Exception as e:  # noqa: BLE001
                    console.print(f"\n[red]Error from model endpoint:[/red] {e}")
            else:
                try:
                    resp = client.chat.completions.create(
                        model=resolved_model,
                        messages=history,
                        stream=False,
                    )
                    msg = resp.choices[0].message if resp.choices else None
                    reasoning, content = _extract_reasoning_and_content(msg)
                    if reasoning:
                        console.print("[dim][Thought][/dim]")
                        console.print(
                            reasoning,
                            style="dim",
                            highlight=False,
                            markup=False,
                        )
                        console.print()
                    reply_text = content or ""
                    console.print(f"[bold green]Model > [/bold green]{reply_text}")
                    history.append({"role": "assistant", "content": reply_text})
                except Exception as e:  # noqa: BLE001
                    console.print(f"\n[red]Error from model endpoint:[/red] {e}")
        else:
            # Completions API (build sequential transcript or prompt)
            history.append({"role": "user", "content": user_input})
            prompt_parts: list[str] = []
            for item in history:
                role = item["role"]
                cnt = item["content"]
                if role == "system":
                    prompt_parts.append(f"{cnt}\n")
                elif role == "user":
                    prompt_parts.append(f"User: {cnt}\nAssistant: ")
                elif role == "assistant":
                    prompt_parts.append(f"{cnt}\n")
            session_prompt = "".join(prompt_parts)

            console.print("[bold green]Model > [/bold green]", end="")
            if stream:
                collected_chunks: list[str] = []
                try:
                    comp_stream = client.completions.create(
                        model=resolved_model,
                        prompt=session_prompt,
                        stop=["\nUser:", "\nAssistant:", "\n\nUser:"],
                        stream=True,
                    )
                    for chunk in comp_stream:
                        if chunk.choices and chunk.choices[0].text:
                            text = chunk.choices[0].text
                            console.print(text, end="")
                            collected_chunks.append(text)
                    console.print()
                    assistant_reply = "".join(collected_chunks).strip()
                    history.append({"role": "assistant", "content": assistant_reply})
                except Exception as e:  # noqa: BLE001
                    console.print(f"\n[red]Error from model endpoint:[/red] {e}")
            else:
                try:
                    resp = client.completions.create(
                        model=resolved_model,
                        prompt=session_prompt,
                        stop=["\nUser:", "\nAssistant:", "\n\nUser:"],
                        stream=False,
                    )
                    content = (
                        resp.choices[0].text if resp.choices else ""
                    ) or ""
                    console.print(content)
                    history.append({"role": "assistant", "content": content.strip()})
                except Exception as e:  # noqa: BLE001
                    console.print(f"\n[red]Error from model endpoint:[/red] {e}")
