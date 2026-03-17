"""Tests for tract.llm.claude_code — ClaudeCodeClient and credential helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tract.llm.claude_code import (
    ClaudeCodeClient,
    load_claude_code_credentials,
    create_claude_code_client,
)
from tract.llm.errors import LLMAuthError, LLMConfigError, LLMResponseError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_creds(tmp_path: Path, oauth: dict | None = None, *, extra: dict | None = None) -> Path:
    """Write a fake credentials file and return its path."""
    data = extra or {}
    if oauth is not None:
        data["claudeAiOauth"] = oauth
    cred_file = tmp_path / ".credentials.json"
    cred_file.write_text(json.dumps(data), encoding="utf-8")
    return cred_file


def _valid_oauth(*, expires_in_ms: float = 3_600_000) -> dict:
    """Return a valid-looking OAuth dict expiring in the future."""
    return {
        "accessToken": "sk-ant-oat01-test-token-abc123",
        "refreshToken": "sk-ant-ort01-refresh-token-xyz",
        "expiresAt": time.time() * 1000 + expires_in_ms,
        "scopes": ["user:inference"],
        "subscriptionType": "team",
    }


def _mock_claude_response(
    result: str = "Hello!",
    model: str = "claude-sonnet-4-6",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> str:
    """Build a JSON string mimicking ``claude -p --output-format json``."""
    return json.dumps({
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": result,
        "stop_reason": "end_turn",
        "session_id": "test-session-123",
        "total_cost_usd": 0.001,
        "duration_ms": 500,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "modelUsage": {
            model: {"inputTokens": input_tokens, "outputTokens": output_tokens},
        },
    })


# ---------------------------------------------------------------------------
# load_claude_code_credentials
# ---------------------------------------------------------------------------

class TestLoadCredentials:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(LLMConfigError, match="not found"):
            load_claude_code_credentials(str(tmp_path / "nope.json"))

    def test_malformed_json(self, tmp_path: Path) -> None:
        bad = tmp_path / ".credentials.json"
        bad.write_text("{{{bad json", encoding="utf-8")
        with pytest.raises(LLMConfigError, match="Failed to read"):
            load_claude_code_credentials(str(bad))

    def test_no_oauth_key(self, tmp_path: Path) -> None:
        path = _write_creds(tmp_path, oauth=None, extra={"other": "stuff"})
        with pytest.raises(LLMConfigError, match="No 'claudeAiOauth'"):
            load_claude_code_credentials(str(path))

    def test_empty_access_token(self, tmp_path: Path) -> None:
        path = _write_creds(tmp_path, {"accessToken": "", "expiresAt": 0})
        with pytest.raises(LLMConfigError, match="no accessToken"):
            load_claude_code_credentials(str(path))

    def test_expired_token(self, tmp_path: Path) -> None:
        oauth = _valid_oauth()
        oauth["expiresAt"] = time.time() * 1000 - 60_000
        path = _write_creds(tmp_path, oauth)
        with pytest.raises(LLMAuthError, match="expired"):
            load_claude_code_credentials(str(path))

    def test_valid_token(self, tmp_path: Path) -> None:
        oauth = _valid_oauth()
        path = _write_creds(tmp_path, oauth)
        result = load_claude_code_credentials(str(path))
        assert result["accessToken"] == "sk-ant-oat01-test-token-abc123"
        assert result["scopes"] == ["user:inference"]

    def test_no_expiry_still_works(self, tmp_path: Path) -> None:
        """Token with expiresAt=0 should not raise."""
        oauth = _valid_oauth()
        oauth["expiresAt"] = 0
        path = _write_creds(tmp_path, oauth)
        result = load_claude_code_credentials(str(path))
        assert result["accessToken"] == "sk-ant-oat01-test-token-abc123"


# ---------------------------------------------------------------------------
# ClaudeCodeClient
# ---------------------------------------------------------------------------

class TestClaudeCodeClient:
    def test_missing_claude_binary(self) -> None:
        with patch("shutil.which", return_value=None):
            with pytest.raises(LLMConfigError, match="Could not find 'claude'"):
                ClaudeCodeClient()

    def test_custom_binary_path(self) -> None:
        client = ClaudeCodeClient(claude_bin="/usr/bin/claude")
        assert client._claude_bin == "/usr/bin/claude"

    def test_chat_success(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = _mock_claude_response("Test response")
        mock_result.stderr = ""

        with patch("shutil.which", return_value="/usr/bin/claude"):
            client = ClaudeCodeClient(model="sonnet")

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            response = client.chat([
                {"role": "user", "content": "Hello"},
            ])

        # Verify response format
        assert response["choices"][0]["message"]["content"] == "Test response"
        assert response["choices"][0]["message"]["role"] == "assistant"
        assert response["choices"][0]["finish_reason"] == "stop"
        assert response["model"] == "claude-sonnet-4-6"
        assert response["usage"]["prompt_tokens"] == 10
        assert response["usage"]["completion_tokens"] == 5

        # Verify subprocess was called
        cmd = mock_run.call_args[0][0]
        assert "-p" in cmd
        assert "--output-format" in cmd
        assert "json" in cmd

    def test_chat_cli_error(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Something went wrong"

        with patch("shutil.which", return_value="/usr/bin/claude"):
            client = ClaudeCodeClient()

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(LLMResponseError, match="exited with code 1"):
                client.chat([{"role": "user", "content": "Hello"}])

    def test_chat_api_error_in_response(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "type": "result", "is_error": True,
            "result": "API rate limited",
        })

        with patch("shutil.which", return_value="/usr/bin/claude"):
            client = ClaudeCodeClient()

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(LLMResponseError, match="API rate limited"):
                client.chat([{"role": "user", "content": "Hello"}])

    def test_chat_timeout(self) -> None:
        import subprocess

        with patch("shutil.which", return_value="/usr/bin/claude"):
            client = ClaudeCodeClient(timeout=5.0)

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("claude", 5)):
            with pytest.raises(LLMResponseError, match="timed out"):
                client.chat([{"role": "user", "content": "Hello"}])

    def test_extract_content(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/claude"):
            client = ClaudeCodeClient()

        response = {
            "choices": [{"message": {"content": "Hello world"}}],
        }
        assert client.extract_content(response) == "Hello world"

    def test_extract_usage(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/claude"):
            client = ClaudeCodeClient()

        response = {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        assert client.extract_usage(response) == {"prompt_tokens": 10, "completion_tokens": 5}

    def test_system_prompt_handling(self) -> None:
        """System messages are separated and passed via --append-system-prompt."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = _mock_claude_response("OK")

        with patch("shutil.which", return_value="/usr/bin/claude"):
            client = ClaudeCodeClient()

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            client.chat([
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ])

        cmd = mock_run.call_args[0][0]
        # System prompt should be passed via --append-system-prompt
        idx = cmd.index("--append-system-prompt")
        assert cmd[idx + 1] == "You are helpful."

    def test_stats_tracking(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = _mock_claude_response("R1", input_tokens=20, output_tokens=10)

        with patch("shutil.which", return_value="/usr/bin/claude"):
            client = ClaudeCodeClient()

        assert client.stats["calls"] == 0

        with patch("subprocess.run", return_value=mock_result):
            client.chat([{"role": "user", "content": "Hello"}])

        assert client.stats["calls"] == 1
        assert client.stats["total_input_tokens"] == 20
        assert client.stats["total_output_tokens"] == 10

    def test_context_manager(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/claude"):
            with ClaudeCodeClient() as client:
                assert isinstance(client, ClaudeCodeClient)

    def test_no_session_persistence_flag(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = _mock_claude_response("OK")

        with patch("shutil.which", return_value="/usr/bin/claude"):
            client = ClaudeCodeClient(no_session_persistence=True)

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            client.chat([{"role": "user", "content": "Hi"}])

        cmd = mock_run.call_args[0][0]
        assert "--no-session-persistence" in cmd

    def test_message_formatting(self) -> None:
        """Multi-turn conversations are formatted correctly."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = _mock_claude_response("Response")

        with patch("shutil.which", return_value="/usr/bin/claude"):
            client = ClaudeCodeClient()

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            client.chat([
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ])

        cmd = mock_run.call_args[0][0]
        prompt = cmd[-1]  # Last arg is the prompt
        assert "Hello" in prompt
        assert "[Assistant]: Hi there!" in prompt
        assert "How are you?" in prompt


# ---------------------------------------------------------------------------
# create_claude_code_client factory
# ---------------------------------------------------------------------------

class TestCreateClient:
    def test_creates_claude_code_client(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/claude"):
            client = create_claude_code_client(model="opus")
        assert isinstance(client, ClaudeCodeClient)
        assert client._model == "opus"

    def test_passes_kwargs(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/claude"):
            client = create_claude_code_client(
                model="haiku",
                timeout=30.0,
                system_prompt="Test prompt",
            )
        assert client._model == "haiku"
        assert client._timeout == 30.0
        assert client._system_prompt == "Test prompt"
