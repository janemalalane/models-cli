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

import subprocess
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from google.models.cli.common.auth import (
    check_adc_validity,
    ensure_authenticated,
    is_authenticated,
)
from google.models.cli.main import app

runner = CliRunner()


def test_check_adc_validity_success():
    with patch("shutil.which", return_value="/usr/bin/gcloud"), patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="mock-valid-access-token\n")
        assert check_adc_validity() is True


def test_check_adc_validity_failure():
    with patch("shutil.which", return_value="/usr/bin/gcloud"), patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")
        with patch("google.auth.default") as mock_auth_default:
            mock_auth_default.side_effect = Exception("No ADC")
            assert check_adc_validity() is False


def test_is_authenticated_true():
    with (
        patch("google.models.cli.common.auth.check_adc_validity", return_value=True),
        patch("google.models.cli.common.auth.get_active_gcp_account", return_value="user@google.com"),
        patch("google.models.cli.common.auth.resolve_gcp_project", return_value="my-project"),
    ):
        authed, msg = is_authenticated()
        assert authed is True
        assert "user@google.com" in msg
        assert "my-project" in msg


def test_is_authenticated_false():
    with patch("google.models.cli.common.auth.check_adc_validity", return_value=False):
        authed, msg = is_authenticated()
        assert authed is False
        assert "Not authenticated" in msg


def test_ensure_authenticated_already_valid():
    with patch("google.models.cli.common.auth.is_authenticated", return_value=(True, "Authenticated")):
        assert ensure_authenticated() is True


def test_ensure_authenticated_non_interactive_fails():
    with (
        patch("google.models.cli.common.auth.is_authenticated", return_value=(False, "Not authed")),
        patch("sys.stdin.isatty", return_value=False),
    ):
        assert ensure_authenticated(interactive=True) is False


def test_ensure_authenticated_interactive_user_accepts():
    with (
        patch("google.models.cli.common.auth.is_authenticated", side_effect=[(False, "Not authed"), (True, "Authed")]),
        patch("sys.stdin.isatty", return_value=True),
        patch("shutil.which", return_value="/usr/bin/gcloud"),
        patch("rich.prompt.Confirm.ask", return_value=True),
        patch("subprocess.run") as mock_run,
        patch("google.models.cli.common.auth.check_adc_validity", return_value=True),
    ):
        mock_run.return_value = MagicMock(returncode=0)
        assert ensure_authenticated(interactive=True) is True
        mock_run.assert_called_once()


def test_login_command_not_registered():
    result = runner.invoke(app, ["login"])
    assert result.exit_code != 0
    assert "No such command" in result.output or "Error" in result.output


def test_info_command_includes_auth_status():
    with patch("google.models.cli.main.is_authenticated", return_value=(True, "test-user@google.com")):
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "Authentication Status" in result.output
        assert "test-user@google.com" in result.output


def test_info_command_unauthenticated_guidance():
    with patch("google.models.cli.main.is_authenticated", return_value=(False, "Not authenticated")):
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "Authentication Status" in result.output
        assert "application-default login" in result.output


def test_is_auth_error_detection():
    from google.models.cli.common.auth import is_auth_error
    import google.auth.exceptions
    import google.api_core.exceptions

    # GoogleAuthError
    assert is_auth_error(google.auth.exceptions.RefreshError("Token expired")) is True
    assert is_auth_error(google.auth.exceptions.DefaultCredentialsError("No creds")) is True

    # Unauthenticated
    assert is_auth_error(google.api_core.exceptions.Unauthenticated("401 Unauthorized")) is True

    # 503 GCE metadata plugin failure (Cloudtop without ADC file)
    gce_metadata_err = google.api_core.exceptions.ServiceUnavailable(
        "503 Getting metadata from plugin failed with error: ('Failed to retrieve "
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token "
        "from the Google Compute Engine metadata service. Status: 404 Response:\\nb\\'\"No service account scopes specified.\"\\'')"
    )
    assert is_auth_error(gce_metadata_err) is True

    # Unrelated error
    assert is_auth_error(ValueError("Invalid syntax")) is False
    assert is_auth_error(RuntimeError("Connection refused to database")) is False

