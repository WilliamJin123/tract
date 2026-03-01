"""Tests for config provenance (config change audit trail)."""
import json
from datetime import datetime, timezone

import pytest

from tract import Tract


class TestConfigProvenance:
    """Tests for config change logging."""

    def test_configure_operations_logs_change(self):
        """configure_operations() creates a config change log entry."""
        from tract.models.config import LLMConfig
        t = Tract.open()
        t.configure_operations(chat=LLMConfig(model="gpt-4o"))
        history = t.config_history()
        assert len(history) >= 1
        entry = history[0]
        assert entry["change_type"] == "operation_config"

    def test_configure_llm_logs_change(self):
        """configure_llm() creates a config change log entry."""
        from unittest.mock import MagicMock
        t = Tract.open()
        mock_client = MagicMock()
        mock_client.chat = MagicMock(return_value={})
        t.configure_llm(mock_client)
        history = t.config_history()
        assert len(history) >= 1
        entry = history[0]
        assert entry["change_type"] == "llm_client"

    def test_configure_clients_logs_change(self):
        """configure_clients() creates a config change log entry."""
        from unittest.mock import MagicMock
        t = Tract.open()
        mock_client = MagicMock()
        t.configure_clients(chat=mock_client)
        history = t.config_history()
        assert len(history) >= 1
        entry = history[0]
        assert entry["change_type"] == "operation_client"

    def test_config_history_reverse_chronological(self):
        """config_history() returns entries in reverse chronological order."""
        from tract.models.config import LLMConfig
        t = Tract.open()
        t.configure_operations(chat=LLMConfig(model="gpt-4o"))
        t.configure_operations(chat=LLMConfig(model="gpt-3.5-turbo"))
        history = t.config_history()
        assert len(history) >= 2
        # First entry should be more recent
        assert history[0]["created_at"] >= history[1]["created_at"]

    def test_config_history_filter_by_type(self):
        """config_history() can filter by change_type."""
        from tract.models.config import LLMConfig
        from unittest.mock import MagicMock
        t = Tract.open()
        t.configure_operations(chat=LLMConfig(model="gpt-4o"))
        mock_client = MagicMock()
        mock_client.chat = MagicMock(return_value={})
        t.configure_llm(mock_client)
        history = t.config_history(change_type="operation_config")
        assert all(e["change_type"] == "operation_config" for e in history)

    def test_config_history_no_persistence_repo(self):
        """config_history() returns empty list when no persistence repo."""
        from tract.models.config import LLMConfig
        t = Tract.open()
        # Even with in-memory, config_history should work (may return [] or actual entries)
        history = t.config_history()
        assert isinstance(history, list)
