"""Tests for workflow profiles system."""

from __future__ import annotations

import pytest

from tract import Tract
from tract.profiles import (
    BUILT_IN_PROFILES,
    CODING,
    ECOMMERCE,
    RESEARCH,
    WorkflowProfile,
    get_profile,
    list_profiles,
    register_profile,
)


# ---------------------------------------------------------------------------
# Built-in profiles exist
# ---------------------------------------------------------------------------


class TestBuiltInProfiles:
    def test_coding_profile_exists(self):
        assert "coding" in BUILT_IN_PROFILES
        assert BUILT_IN_PROFILES["coding"] is CODING

    def test_research_profile_exists(self):
        assert "research" in BUILT_IN_PROFILES
        assert BUILT_IN_PROFILES["research"] is RESEARCH

    def test_ecommerce_profile_exists(self):
        assert "ecommerce" in BUILT_IN_PROFILES
        assert BUILT_IN_PROFILES["ecommerce"] is ECOMMERCE

    def test_coding_profile_has_stages(self):
        assert set(CODING.stages.keys()) == {"design", "implement", "test", "review"}

    def test_research_profile_has_stages(self):
        assert set(RESEARCH.stages.keys()) == {"ingest", "organize", "synthesize", "validate"}

    def test_ecommerce_profile_has_stages(self):
        assert set(ECOMMERCE.stages.keys()) == {"research", "creative", "campaign", "analysis", "optimize"}

    def test_coding_profile_has_directives(self):
        assert "methodology" in CODING.directives
        assert "code_quality" in CODING.directives

    def test_research_profile_has_directives(self):
        assert "methodology" in RESEARCH.directives
        assert "synthesis" in RESEARCH.directives

    def test_ecommerce_profile_has_directives(self):
        assert "brand_consistency" in ECOMMERCE.directives
        assert "data_driven" in ECOMMERCE.directives

    def test_profiles_are_frozen(self):
        with pytest.raises(AttributeError):
            CODING.name = "oops"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# get_profile()
# ---------------------------------------------------------------------------


class TestGetProfile:
    def test_returns_correct_profile(self):
        p = get_profile("coding")
        assert p is CODING

    def test_returns_research(self):
        p = get_profile("research")
        assert p is RESEARCH

    def test_returns_ecommerce(self):
        p = get_profile("ecommerce")
        assert p is ECOMMERCE

    def test_unknown_name_raises_key_error(self):
        with pytest.raises(KeyError, match="not found"):
            get_profile("nonexistent")

    def test_error_message_lists_available(self):
        with pytest.raises(KeyError, match="coding"):
            get_profile("nope")


# ---------------------------------------------------------------------------
# list_profiles()
# ---------------------------------------------------------------------------


class TestListProfiles:
    def test_returns_all(self):
        profiles = list_profiles()
        names = {p.name for p in profiles}
        assert {"coding", "research", "ecommerce"}.issubset(names)

    def test_returns_list_of_workflow_profiles(self):
        for p in list_profiles():
            assert isinstance(p, WorkflowProfile)


# ---------------------------------------------------------------------------
# register_profile()
# ---------------------------------------------------------------------------


class TestRegisterProfile:
    def test_register_custom_profile(self):
        custom = WorkflowProfile(
            name="custom_test",
            description="A custom test profile",
            config={"temperature": 0.42},
            stages={"alpha": {"temperature": 0.1}},
        )
        register_profile(custom)
        try:
            assert get_profile("custom_test") is custom
            assert custom in list_profiles()
        finally:
            # Clean up to avoid polluting other tests
            BUILT_IN_PROFILES.pop("custom_test", None)

    def test_register_overwrites_existing(self):
        original = get_profile("coding")
        replacement = WorkflowProfile(
            name="coding",
            description="Replaced coding profile",
        )
        register_profile(replacement)
        try:
            assert get_profile("coding") is replacement
        finally:
            # Restore original
            BUILT_IN_PROFILES["coding"] = original


# ---------------------------------------------------------------------------
# WorkflowProfile dataclass
# ---------------------------------------------------------------------------


class TestWorkflowProfile:
    def test_default_fields(self):
        p = WorkflowProfile(name="bare", description="Bare profile")
        assert p.config == {}
        assert p.directives == {}
        assert p.directive_templates == {}
        assert p.tool_profile == "self"
        assert p.stages == {}

    def test_with_all_fields(self):
        p = WorkflowProfile(
            name="full",
            description="Full profile",
            config={"temperature": 0.5},
            directive_templates={"review_protocol": {"criteria": "accuracy"}},
            directives={"rule1": "Do X"},
            tool_profile="supervisor",
            stages={"phase1": {"temperature": 0.1}},
        )
        assert p.name == "full"
        assert p.tool_profile == "supervisor"
        assert p.stages["phase1"] == {"temperature": 0.1}


# ---------------------------------------------------------------------------
# Tract.load_profile()
# ---------------------------------------------------------------------------


class TestTractLoadProfile:
    def test_applies_config(self):
        with Tract.open() as t:
            t.templates.load_profile("coding")
            assert t.config.get("temperature") == 0.3
            assert t.config.get("compile_strategy") == "messages"

    def test_applies_directives(self):
        with Tract.open() as t:
            t.templates.load_profile("coding")
            # Directives are committed as InstructionContent
            log = t.log()
            messages = [ci.message for ci in log]
            assert any("directive: methodology" in m for m in messages)
            assert any("directive: code_quality" in m for m in messages)

    def test_skip_directives(self):
        with Tract.open() as t:
            t.templates.load_profile("coding", apply_directives=False)
            # Config should still be applied
            assert t.config.get("temperature") == 0.3
            # But no directives
            log = t.log()
            messages = [ci.message for ci in log]
            assert not any("directive:" in m for m in messages)

    def test_stores_active_profile(self):
        with Tract.open() as t:
            t.templates.load_profile("research")
            assert t.templates.active_profile is not None
            assert t.templates.active_profile.name == "research"

    def test_unknown_profile_raises(self):
        with Tract.open() as t:
            with pytest.raises(KeyError, match="not found"):
                t.templates.load_profile("nonexistent")

    def test_load_research_profile(self):
        with Tract.open() as t:
            t.templates.load_profile("research")
            assert t.config.get("temperature") == 0.5
            assert t.config.get("compile_strategy") == "full"

    def test_load_ecommerce_profile(self):
        with Tract.open() as t:
            t.templates.load_profile("ecommerce")
            assert t.config.get("temperature") == 0.6
            assert t.config.get("compile_strategy") == "messages"

    def test_load_profile_with_templates(self):
        """Profile with directive_templates should apply them."""
        custom = WorkflowProfile(
            name="_tmpl_test",
            description="Template test",
            directive_templates={"output_format": {"format": "json", "max_words": "200"}},
        )
        register_profile(custom)
        try:
            with Tract.open() as t:
                t.templates.load_profile("_tmpl_test")
                log = t.log()
                messages = [ci.message for ci in log]
                assert any("directive: output_format" in m for m in messages)
        finally:
            BUILT_IN_PROFILES.pop("_tmpl_test", None)


# ---------------------------------------------------------------------------
# Tract.apply_stage()
# ---------------------------------------------------------------------------


class TestTractApplyStage:
    def test_applies_stage_config(self):
        with Tract.open() as t:
            t.templates.load_profile("coding")
            t.templates.apply_stage("design")
            assert t.config.get("temperature") == 0.5
            assert t.config.get("compile_strategy") == "full"

    def test_apply_different_stages(self):
        with Tract.open() as t:
            t.templates.load_profile("coding")
            t.templates.apply_stage("test")
            assert t.config.get("temperature") == 0.1

            t.templates.apply_stage("review")
            assert t.config.get("temperature") == 0.4
            assert t.config.get("compile_strategy") == "adaptive"

    def test_error_without_profile(self):
        with Tract.open() as t:
            with pytest.raises(ValueError, match="No workflow profile loaded"):
                t.templates.apply_stage("design")

    def test_error_for_unknown_stage(self):
        with Tract.open() as t:
            t.templates.load_profile("coding")
            with pytest.raises(ValueError, match="not in profile"):
                t.templates.apply_stage("nonexistent_stage")

    def test_error_lists_available_stages(self):
        with Tract.open() as t:
            t.templates.load_profile("coding")
            with pytest.raises(ValueError, match="design"):
                t.templates.apply_stage("bogus")

    def test_apply_research_stages(self):
        with Tract.open() as t:
            t.templates.load_profile("research")
            t.templates.apply_stage("ingest")
            assert t.config.get("temperature") == 0.3

            t.templates.apply_stage("synthesize")
            assert t.config.get("temperature") == 0.6
            assert t.config.get("compile_strategy") == "adaptive"

    def test_apply_ecommerce_stages(self):
        with Tract.open() as t:
            t.templates.load_profile("ecommerce")
            t.templates.apply_stage("creative")
            assert t.config.get("temperature") == 0.8

            t.templates.apply_stage("analysis")
            assert t.config.get("temperature") == 0.2


# ---------------------------------------------------------------------------
# Tract.active_profile property
# ---------------------------------------------------------------------------


class TestActiveProfileProperty:
    def test_none_by_default(self):
        with Tract.open() as t:
            assert t.templates.active_profile is None

    def test_set_after_load(self):
        with Tract.open() as t:
            t.templates.load_profile("coding")
            assert t.templates.active_profile is CODING

    def test_replaced_on_second_load(self):
        with Tract.open() as t:
            t.templates.load_profile("coding")
            assert t.templates.active_profile is CODING
            t.templates.load_profile("research")
            assert t.templates.active_profile is RESEARCH


# ---------------------------------------------------------------------------
# Custom profile with stages end-to-end
# ---------------------------------------------------------------------------


class TestCustomProfileEndToEnd:
    def test_custom_profile_with_stages(self):
        custom = WorkflowProfile(
            name="_e2e_test",
            description="End-to-end custom profile test",
            config={"temperature": 0.5, "compile_strategy": "full"},
            directives={"rule": "Always double-check results."},
            stages={
                "explore": {"temperature": 0.9},
                "execute": {"temperature": 0.1, "compile_strategy": "messages"},
            },
        )
        register_profile(custom)
        try:
            with Tract.open() as t:
                t.templates.load_profile("_e2e_test")

                # Base config applied
                assert t.config.get("temperature") == 0.5
                assert t.config.get("compile_strategy") == "full"

                # Directive applied
                log_msgs = [ci.message for ci in t.log()]
                assert any("directive: rule" in m for m in log_msgs)

                # Switch to explore stage
                t.templates.apply_stage("explore")
                assert t.config.get("temperature") == 0.9
                # compile_strategy not overridden in this stage, still "full"
                assert t.config.get("compile_strategy") == "full"

                # Switch to execute stage
                t.templates.apply_stage("execute")
                assert t.config.get("temperature") == 0.1
                assert t.config.get("compile_strategy") == "messages"
        finally:
            BUILT_IN_PROFILES.pop("_e2e_test", None)


# ---------------------------------------------------------------------------
# Module-level imports from tract package
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_workflow_profile_importable(self):
        from tract import WorkflowProfile as WP
        assert WP is WorkflowProfile

    def test_get_workflow_profile_importable(self):
        from tract import get_workflow_profile
        p = get_workflow_profile("coding")
        assert p.name == "coding"

    def test_list_workflow_profiles_importable(self):
        from tract import list_workflow_profiles
        profiles = list_workflow_profiles()
        assert len(profiles) >= 3

    def test_register_workflow_profile_importable(self):
        from tract import register_workflow_profile
        assert callable(register_workflow_profile)
