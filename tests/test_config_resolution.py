"""Tests for config resolution source tracking (Fix 9)."""
import pytest

from tract import Tract, LLMConfig, OperationConfigs


class TestConfigResolutionSource:
    """Tests for _resolve_llm_config include_sources parameter."""

    def test_sources_not_included_by_default(self):
        """Default behavior: no _resolution_sources in result."""
        t = Tract.open()
        t._default_config = LLMConfig(model="gpt-4o")
        result = t._resolve_llm_config("chat")
        assert "_resolution_sources" not in result

    def test_sugar_source(self):
        """Sugar params are tracked as source='sugar'."""
        t = Tract.open()
        result = t._resolve_llm_config("chat", model="gpt-4o", include_sources=True)
        sources = result["_resolution_sources"]
        assert sources["model"] == "sugar"

    def test_tract_default_source(self):
        """Tract default config is tracked as source='tract_default'."""
        t = Tract.open()
        t._default_config = LLMConfig(model="gpt-4o", temperature=0.7)
        result = t._resolve_llm_config("chat", include_sources=True)
        sources = result["_resolution_sources"]
        assert sources["model"] == "tract_default"
        assert sources["temperature"] == "tract_default"

    def test_operation_source(self):
        """Operation config is tracked as source='operation:{name}'."""
        t = Tract.open()
        t.configure_operations(chat=LLMConfig(model="gpt-4o"))
        result = t._resolve_llm_config("chat", include_sources=True)
        sources = result["_resolution_sources"]
        assert sources["model"] == "operation:chat"

    def test_llm_config_source(self):
        """LLMConfig param is tracked as source='llm_config'."""
        t = Tract.open()
        cfg = LLMConfig(model="gpt-4o")
        result = t._resolve_llm_config("chat", llm_config=cfg, include_sources=True)
        sources = result["_resolution_sources"]
        assert sources["model"] == "llm_config"

    def test_mixed_sources(self):
        """Multiple sources in a single resolution."""
        t = Tract.open()
        t._default_config = LLMConfig(temperature=0.5)
        t.configure_operations(chat=LLMConfig(top_p=0.9))
        result = t._resolve_llm_config(
            "chat", model="gpt-4o", include_sources=True,
        )
        sources = result["_resolution_sources"]
        assert sources["model"] == "sugar"
        assert sources["top_p"] == "operation:chat"
        assert sources["temperature"] == "tract_default"
