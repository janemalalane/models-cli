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

"""Lightweight Google Cloud authentication management for models-cli."""

import shutil
import subprocess
import sys
from typing import Optional

from rich.console import Console
from rich.prompt import Confirm

from google.models.cli._gcp_project import get_active_gcp_account, resolve_gcp_project

console = Console()


def check_adc_validity() -> bool:
    """Fast check to determine whether Google Application Default Credentials are valid.

    Uses `gcloud auth application-default print-access-token --quiet` as a fast
    pre-flight check (mirroring agents-cli), falling back to google.auth.
    """
    if shutil.which("gcloud"):
        try:
            res = subprocess.run(
                ["gcloud", "auth", "application-default", "print-access-token", "--quiet"],
                capture_output=True,
                check=True,
                timeout=5,
                text=True,
            )
            return bool(res.stdout.strip())
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
        except Exception:
            pass

    # Fallback to google.auth in python
    try:
        import google.auth
        from google.auth.transport.requests import Request

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(Request())
        return bool(credentials.valid and credentials.token)
    except Exception:
        return False


def is_authenticated() -> tuple[bool, str]:
    """Checks authentication status and returns a descriptive status string."""
    valid = check_adc_validity()
    account = get_active_gcp_account() or "Google Cloud user"
    project = resolve_gcp_project()

    if valid:
        proj_str = f" for project '{project}'" if project else ""
        return True, f"{account}{proj_str} (Application Default Credentials)"
    return False, "Not authenticated or credentials expired"


def is_auth_error(exc: Exception) -> bool:
    """Detects whether an exception is due to missing or expired GCP credentials.

    Covers:
    - google.auth.exceptions.GoogleAuthError (e.g. RefreshError, DefaultCredentialsError)
    - google.api_core.exceptions.Unauthenticated (401)
    - gRPC ServiceUnavailable (503) when falling back to GCE metadata on Cloudtop/workstations
    """
    import google.auth.exceptions
    import google.api_core.exceptions

    if isinstance(
        exc,
        (
            google.auth.exceptions.GoogleAuthError,
            google.api_core.exceptions.Unauthenticated,
        ),
    ):
        return True

    # Check error message and chained exceptions for GCE metadata fallback or auth expiry
    current: Optional[BaseException] = exc
    while current:
        msg = str(current).lower()
        if any(
            pattern in msg
            for pattern in (
                "getting metadata from plugin failed",
                "computemetadata",
                "metadata.google.internal",
                "no service account scopes",
                "could not automatically determine credentials",
                "reauthentication required",
                "invalid_grant",
                "token has expired",
            )
        ):
            return True
        current = current.__cause__ or current.__context__

    return False


def ensure_authenticated(interactive: bool = True) -> bool:
    """Verifies authentication.

    If credentials are missing or expired and running in an interactive terminal,
    prompts the user to authenticate via `gcloud auth application-default login`.
    """
    valid, _ = is_authenticated()
    if valid:
        return True

    if interactive and sys.stdin.isatty() and shutil.which("gcloud"):
        console.print("\n[bold yellow]⚠️  Google Cloud authentication is required or has expired.[/bold yellow]")
        if Confirm.ask("Would you like to run 'gcloud auth application-default login' now?", default=True):
            console.print("[cyan]Opening Google Cloud authentication...[/cyan]\n")
            try:
                res = subprocess.run(["gcloud", "auth", "application-default", "login"], check=False)
                if res.returncode == 0 and check_adc_validity():
                    console.print("[bold green]✓ Authentication successful![/bold green]\n")
                    return True
            except Exception as e:
                console.print(f"[bold red]❌ Failed to run gcloud: {e}[/bold red]\n")
                return False

    console.print(
        "\n[bold red]❌ Google Cloud authentication required.[/bold red]\n"
        "[yellow]Please run the following command to authenticate:[/yellow]\n"
        "  • [cyan]gcloud auth application-default login[/cyan]\n"
    )
    return False


