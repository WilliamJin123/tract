"""Tests for tract.llm.claude_code credential helper."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from tract.llm.claude_code import (
    load_claude_code_credentials,
    create_claude_code_client,
)
from tract.llm.errors import LLMAuthError, LLMConfigError


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


# ---------------------------------------------------------------------------
# load_claude_code_credentials
# ---------------------------------------------------------------------------

class TestLoadCredentials:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(LLMConfigError, match="not found"):
            load_claude_code_credentials(tmp_path / "nope.json")

    def test_malformed_json(self, tmp_path: Path) -> None:
        bad = tmp_path / ".credentials.json"
        bad.write_text("{{{bad json", encoding="utf-8")
        with pytest.raises(LLMConfigError, match="Failed to read"):
            load_claude_code_credentials(bad)

    def test_no_oauth_key(self, tmp_path: Path) -> None:
        path = _write_creds(tmp_path, oauth=None, extra={"other": "stuff"})
        with pytest.raises(LLMConfigError, match="No 'claudeAiOauth'"):
            load_claude_code_credentials(path)

    def test_empty_access_token(self, tmp_path: Path) -> None:
        path = _write_creds(tmp_path, {"accessToken": "", "expiresAt": 0})
        with pytest.raises(LLMConfigError, match="no accessToken"):
            load_claude_code_credentials(path)

    def test_expired_token(self, tmp_path: Path) -> None:
        oauth = _valid_oauth()
        oauth["expiresAt"] = time.time() * 1000 - 60_000  # 1 min ago
        path = _write_creds(tmp_path, oauth)
        with pytest.raises(LLMAuthError, match="expired"):
            load_claude_code_credentials(path)

    def test_valid_token(self, tmp_path: Path) -> None:
        oauth = _valid_oauth()
        path = _write_creds(tmp_path, oauth)
        result = load_claude_code_credentials(path)
        assert result["accessToken"] == "sk-ant-oat01-test-token-abc123"
        assert result["scopes"] == ["user:inference"]

    def test_no_expiry_still_works(self, tmp_path: Path) -> None:
        """Token with expiresAt=0 should not raise."""
        oauth = _valid_oauth()
        oauth["expiresAt"] = 0
        path = _write_creds(tmp_path, oauth)
        result = load_claude_code_credentials(path)
        assert result["accessToken"] == "sk-ant-oat01-test-token-abc123"


# ---------------------------------------------------------------------------
# create_claude_code_client
# ---------------------------------------------------------------------------

class TestCreateClient:
    def test_creates_anthropic_client(self, tmp_path: Path) -> None:
        """Factory should return an AnthropicClient with the OAuth token."""
        oauth = _valid_oauth()
        path = _write_creds(tmp_path, oauth)
        client = create_claude_code_client(
            credentials_path=path,
            model="claude-haiku-4-5-20251001",
        )
        from tract.llm.anthropic_client import AnthropicClient
        assert isinstance(client, AnthropicClient)
        assert client._api_key == "sk-ant-oat01-test-token-abc123"
        assert client._default_model == "claude-haiku-4-5-20251001"
        client.close()

    def test_expired_token_raises(self, tmp_path: Path) -> None:
        oauth = _valid_oauth()
        oauth["expiresAt"] = time.time() * 1000 - 1000
        path = _write_creds(tmp_path, oauth)
        with pytest.raises(LLMAuthError, match="expired"):
            create_claude_code_client(credentials_path=path)

    def test_missing_creds_raises(self, tmp_path: Path) -> None:
        with pytest.raises(LLMConfigError):
            create_claude_code_client(credentials_path=tmp_path / "nope.json")
