"""Tests for behavioral spec persistence.

Covers:
- Gate spec round-trip (create gate, persist, restore)
- Maintainer spec round-trip
- Profile spec round-trip
- Template spec round-trip
- Auto-persist on gate/maintain calls
- Load on open (profiles and templates restored to registries)
- List and remove behavioral specs
- behavioral_specs table exists in fresh DB
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tract import Tract
from tract.gate import SemanticGate
from tract.maintain import SemanticMaintainer
from tract.profiles import WorkflowProfile
from tract.templates import DirectiveTemplate


# ============================================================
# Helpers
# ============================================================


_TRACT_ID = "test-behavioral-tract-001"


def _make_file_tract(tmp_path: Path, **kwargs) -> Tract:
    """Create a Tract backed by a file DB in tmp_path."""
    db_path = str(tmp_path / "test.db")
    kwargs.setdefault("tract_id", _TRACT_ID)
    return Tract.open(path=db_path, **kwargs)


def _reopen_tract(tmp_path: Path, **kwargs) -> Tract:
    """Re-open the same DB file to test persistence across sessions."""
    db_path = str(tmp_path / "test.db")
    kwargs.setdefault("tract_id", _TRACT_ID)
    return Tract.open(path=db_path, **kwargs)


# ============================================================
# Table existence
# ============================================================


class TestBehavioralSpecsTable:
    """behavioral_specs table is created for both file and memory DBs."""

    def test_table_exists_in_fresh_db(self, tmp_path: Path) -> None:
        from sqlalchemy import text

        t = _make_file_tract(tmp_path)
        with t._engine.connect() as conn:
            tables = [
                r[0]
                for r in conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                ).fetchall()
            ]
        assert "behavioral_specs" in tables
        t.close()

    def test_table_exists_in_memory_db(self) -> None:
        from sqlalchemy import text

        t = Tract.open()
        with t._engine.connect() as conn:
            tables = [
                r[0]
                for r in conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                ).fetchall()
            ]
        assert "behavioral_specs" in tables
        t.close()


# ============================================================
# Gate spec round-trip
# ============================================================


class TestGateSpecRoundTrip:
    """SemanticGate to_spec/from_spec and persistence through tract."""

    def test_gate_to_spec_basic(self) -> None:
        gate = SemanticGate(
            name="quality-check",
            check="At least 3 commits tagged 'key-finding' exist",
            model="gpt-4o-mini",
            temperature=0.2,
            max_log_entries=20,
        )
        spec = gate.to_spec()
        assert spec["name"] == "quality-check"
        assert spec["check"] == "At least 3 commits tagged 'key-finding' exist"
        assert spec["model"] == "gpt-4o-mini"
        assert spec["temperature"] == 0.2
        assert spec["max_log_entries"] == 20
        assert spec["has_condition"] is False

    def test_gate_to_spec_with_condition(self) -> None:
        gate = SemanticGate(
            name="cond-gate",
            check="something",
            condition=lambda ctx: True,
        )
        spec = gate.to_spec()
        assert spec["has_condition"] is True

    def test_gate_from_spec(self) -> None:
        spec = {
            "name": "restored-gate",
            "check": "Check something",
            "model": "gpt-4o",
            "temperature": 0.3,
            "max_log_entries": 15,
            "has_condition": True,
        }
        gate = SemanticGate.from_spec(spec)
        assert gate.name == "restored-gate"
        assert gate.check == "Check something"
        assert gate.model == "gpt-4o"
        assert gate.temperature == 0.3
        assert gate.max_log_entries == 15
        assert gate.condition is None  # not restorable

    def test_gate_spec_round_trip(self) -> None:
        original = SemanticGate(
            name="round-trip",
            check="Test criterion",
            model="test-model",
            temperature=0.5,
            max_log_entries=10,
        )
        spec = original.to_spec()
        restored = SemanticGate.from_spec(spec)
        assert restored.name == original.name
        assert restored.check == original.check
        assert restored.model == original.model
        assert restored.temperature == original.temperature
        assert restored.max_log_entries == original.max_log_entries

    def test_gate_spec_persisted_and_loaded(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        gate = SemanticGate(name="persist-test", check="criterion")
        spec_data = gate.to_spec()
        spec_data["event"] = "pre_commit"
        t.persist_behavioral_spec("gate", "persist-test", spec_data)
        t.close()

        t2 = _reopen_tract(tmp_path)
        specs = t2.load_behavioral_specs(spec_type="gate")
        assert len(specs) == 1
        assert specs[0]["spec_name"] == "persist-test"
        assert specs[0]["spec_data"]["check"] == "criterion"
        assert specs[0]["spec_data"]["event"] == "pre_commit"
        t2.close()


# ============================================================
# Maintainer spec round-trip
# ============================================================


class TestMaintainerSpecRoundTrip:
    """SemanticMaintainer to_spec/from_spec and persistence."""

    def test_maintainer_to_spec_basic(self) -> None:
        m = SemanticMaintainer(
            name="cleanup",
            instructions="Mark stale commits as SKIP",
            actions=["annotate", "compress"],
            model="gpt-4o-mini",
            temperature=0.2,
            max_log_entries=20,
            max_peeks=3,
        )
        spec = m.to_spec()
        assert spec["name"] == "cleanup"
        assert spec["instructions"] == "Mark stale commits as SKIP"
        assert sorted(spec["actions"]) == ["annotate", "compress"]
        assert spec["model"] == "gpt-4o-mini"
        assert spec["temperature"] == 0.2
        assert spec["max_log_entries"] == 20
        assert spec["max_peeks"] == 3
        assert spec["has_condition"] is False

    def test_maintainer_to_spec_with_condition(self) -> None:
        m = SemanticMaintainer(
            name="cond-m",
            instructions="test",
            actions=["gc"],
            condition=lambda ctx: False,
        )
        spec = m.to_spec()
        assert spec["has_condition"] is True

    def test_maintainer_from_spec(self) -> None:
        spec = {
            "name": "restored-m",
            "instructions": "Do maintenance",
            "actions": ["annotate", "tag"],
            "model": "gpt-4o",
            "temperature": 0.3,
            "max_log_entries": 15,
            "max_peeks": 5,
            "has_condition": True,
        }
        m = SemanticMaintainer.from_spec(spec)
        assert m.name == "restored-m"
        assert m.instructions == "Do maintenance"
        assert sorted(m.actions) == ["annotate", "tag"]
        assert m.model == "gpt-4o"
        assert m.temperature == 0.3
        assert m.max_log_entries == 15
        assert m.max_peeks == 5
        assert m.condition is None

    def test_maintainer_spec_round_trip(self) -> None:
        original = SemanticMaintainer(
            name="round-trip-m",
            instructions="Maintain context",
            actions=["annotate", "compress", "gc"],
            model="test-model",
            temperature=0.4,
            max_log_entries=25,
            max_peeks=2,
        )
        spec = original.to_spec()
        restored = SemanticMaintainer.from_spec(spec)
        assert restored.name == original.name
        assert restored.instructions == original.instructions
        assert sorted(restored.actions) == sorted(original.actions)
        assert restored.model == original.model
        assert restored.temperature == original.temperature
        assert restored.max_peeks == original.max_peeks

    def test_maintainer_spec_persisted_and_loaded(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        m = SemanticMaintainer(name="persist-m", instructions="test", actions=["gc"])
        spec_data = m.to_spec()
        spec_data["event"] = "post_commit"
        t.persist_behavioral_spec("maintainer", "persist-m", spec_data)
        t.close()

        t2 = _reopen_tract(tmp_path)
        specs = t2.load_behavioral_specs(spec_type="maintainer")
        assert len(specs) == 1
        assert specs[0]["spec_name"] == "persist-m"
        assert specs[0]["spec_data"]["instructions"] == "test"
        t2.close()


# ============================================================
# Profile spec round-trip
# ============================================================


class TestProfileSpecRoundTrip:
    """WorkflowProfile to_spec/from_spec and persistence."""

    def test_profile_to_spec(self) -> None:
        profile = WorkflowProfile(
            name="custom",
            description="Custom workflow",
            config={"temperature": 0.5},
            directives={"method": "Be systematic"},
            stages={"design": {"temperature": 0.7}},
        )
        spec = profile.to_spec()
        assert spec["name"] == "custom"
        assert spec["description"] == "Custom workflow"
        assert spec["config"]["temperature"] == 0.5
        assert spec["directives"]["method"] == "Be systematic"
        assert spec["stages"]["design"]["temperature"] == 0.7

    def test_profile_from_spec(self) -> None:
        spec = {
            "name": "restored",
            "description": "Restored profile",
            "config": {"temperature": 0.3},
            "directives": {"focus": "Stay focused"},
            "stages": {"impl": {"temperature": 0.2}},
            "tool_profile": "supervisor",
        }
        profile = WorkflowProfile.from_spec(spec)
        assert profile.name == "restored"
        assert profile.description == "Restored profile"
        assert profile.config["temperature"] == 0.3
        assert profile.tool_profile == "supervisor"

    def test_profile_spec_round_trip(self) -> None:
        original = WorkflowProfile(
            name="rt",
            description="Round trip",
            config={"a": 1},
            directive_templates={"tmpl": {"param": "val"}},
            directives={"d1": "text"},
            tool_profile="full",
            stages={"s1": {"x": 10}},
        )
        spec = original.to_spec()
        restored = WorkflowProfile.from_spec(spec)
        assert restored.name == original.name
        assert restored.config == original.config
        assert restored.directive_templates == original.directive_templates
        assert restored.directives == original.directives
        assert restored.tool_profile == original.tool_profile
        assert restored.stages == original.stages

    def test_profile_persisted_and_restored_on_open(self, tmp_path: Path) -> None:
        """Profile persisted via persist_behavioral_spec is restored into registry on reopen."""
        t = _make_file_tract(tmp_path)
        profile = WorkflowProfile(
            name="my-custom",
            description="Persisted custom profile",
            config={"temperature": 0.4},
            stages={"phase1": {"temperature": 0.1}},
        )
        t.persist_behavioral_spec("profile", profile.name, profile.to_spec())
        t.close()

        t2 = _reopen_tract(tmp_path)
        # Should be in the registry
        restored = t2.get_profile("my-custom")
        assert restored.name == "my-custom"
        assert restored.description == "Persisted custom profile"
        assert restored.config["temperature"] == 0.4
        t2.close()


# ============================================================
# Template spec round-trip
# ============================================================


class TestTemplateSpecRoundTrip:
    """DirectiveTemplate to_spec/from_spec and persistence."""

    def test_template_to_spec(self) -> None:
        tmpl = DirectiveTemplate(
            name="custom-tmpl",
            description="A custom template",
            content="Focus on {topic} with {depth} analysis",
            parameters={"topic": "The research topic", "depth": "Analysis depth"},
        )
        spec = tmpl.to_spec()
        assert spec["name"] == "custom-tmpl"
        assert spec["description"] == "A custom template"
        assert "{topic}" in spec["content"]
        assert spec["parameters"]["topic"] == "The research topic"

    def test_template_from_spec(self) -> None:
        spec = {
            "name": "restored-tmpl",
            "description": "Restored template",
            "content": "Do {action}",
            "parameters": {"action": "What to do"},
        }
        tmpl = DirectiveTemplate.from_spec(spec)
        assert tmpl.name == "restored-tmpl"
        assert tmpl.content == "Do {action}"
        assert tmpl.render(action="testing") == "Do testing"

    def test_template_spec_round_trip(self) -> None:
        original = DirectiveTemplate(
            name="rt-tmpl",
            description="Round trip template",
            content="Template {a} and {b}",
            parameters={"a": "Param A", "b": "Param B"},
        )
        spec = original.to_spec()
        restored = DirectiveTemplate.from_spec(spec)
        assert restored.name == original.name
        assert restored.content == original.content
        assert restored.parameters == original.parameters
        assert restored.render(a="X", b="Y") == "Template X and Y"

    def test_template_persisted_and_restored_on_open(self, tmp_path: Path) -> None:
        """Template persisted via persist_behavioral_spec is restored into registry on reopen."""
        t = _make_file_tract(tmp_path)
        tmpl = DirectiveTemplate(
            name="my-tmpl",
            description="Custom",
            content="Do {thing}",
            parameters={"thing": "What to do"},
        )
        t.persist_behavioral_spec("template", tmpl.name, tmpl.to_spec())
        t.close()

        t2 = _reopen_tract(tmp_path)
        # Should be in the registry
        restored = t2.get_template("my-tmpl")
        assert restored.name == "my-tmpl"
        assert restored.content == "Do {thing}"
        assert restored.render(thing="work") == "Do work"
        t2.close()


# ============================================================
# Auto-persist on gate/maintain calls
# ============================================================


class TestAutoPersist:
    """Calling t.gate() or t.maintain() auto-persists the spec."""

    def test_gate_auto_persists(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        t.gate(
            "auto-gate",
            event="pre_commit",
            check="Something must be true",
            model="gpt-4o-mini",
            temperature=0.3,
        )
        specs = t.load_behavioral_specs(spec_type="gate")
        assert len(specs) == 1
        assert specs[0]["spec_name"] == "auto-gate"
        data = specs[0]["spec_data"]
        assert data["check"] == "Something must be true"
        assert data["event"] == "pre_commit"
        assert data["model"] == "gpt-4o-mini"
        assert data["temperature"] == 0.3
        t.close()

    def test_maintain_auto_persists(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        t.maintain(
            "auto-maintain",
            event="post_commit",
            instructions="Clean up old stuff",
            actions=["annotate", "gc"],
            max_peeks=2,
        )
        specs = t.load_behavioral_specs(spec_type="maintainer")
        assert len(specs) == 1
        assert specs[0]["spec_name"] == "auto-maintain"
        data = specs[0]["spec_data"]
        assert data["instructions"] == "Clean up old stuff"
        assert data["event"] == "post_commit"
        assert sorted(data["actions"]) == ["annotate", "gc"]
        assert data["max_peeks"] == 2
        t.close()

    def test_remove_gate_removes_spec(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        t.gate("rm-gate", event="pre_commit", check="test")
        assert len(t.load_behavioral_specs(spec_type="gate")) == 1
        t.remove_gate("rm-gate")
        assert len(t.load_behavioral_specs(spec_type="gate")) == 0
        t.close()

    def test_remove_maintainer_removes_spec(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        t.maintain("rm-m", event="post_commit", instructions="test", actions=["gc"])
        assert len(t.load_behavioral_specs(spec_type="maintainer")) == 1
        t.remove_maintainer("rm-m")
        assert len(t.load_behavioral_specs(spec_type="maintainer")) == 0
        t.close()


# ============================================================
# List and remove behavioral specs
# ============================================================


class TestListAndRemove:
    """list_behavioral_specs and remove_behavioral_spec."""

    def test_list_all_types(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        t.persist_behavioral_spec("gate", "g1", {"name": "g1", "check": "c"})
        t.persist_behavioral_spec("maintainer", "m1", {"name": "m1", "instructions": "i"})
        t.persist_behavioral_spec("profile", "p1", {"name": "p1"})
        t.persist_behavioral_spec("template", "t1", {"name": "t1", "content": "c"})

        all_specs = t.list_behavioral_specs()
        assert len(all_specs) == 4

        gate_specs = t.list_behavioral_specs(spec_type="gate")
        assert len(gate_specs) == 1
        assert gate_specs[0]["spec_name"] == "g1"

        t.close()

    def test_remove_spec(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        t.persist_behavioral_spec("gate", "g1", {"name": "g1"})
        assert len(t.list_behavioral_specs()) == 1

        removed = t.remove_behavioral_spec("gate", "g1")
        assert removed is True
        assert len(t.list_behavioral_specs()) == 0

        # Remove non-existent
        removed = t.remove_behavioral_spec("gate", "nonexistent")
        assert removed is False

        t.close()

    def test_upsert_overwrites(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        t.persist_behavioral_spec("gate", "g1", {"check": "v1"})
        t.persist_behavioral_spec("gate", "g1", {"check": "v2"})

        specs = t.load_behavioral_specs(spec_type="gate")
        assert len(specs) == 1
        assert specs[0]["spec_data"]["check"] == "v2"
        t.close()

    def test_invalid_spec_type_raises(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        with pytest.raises(ValueError, match="Invalid spec_type"):
            t.persist_behavioral_spec("invalid_type", "test", {})
        t.close()


# ============================================================
# Load on open
# ============================================================


class TestLoadOnOpen:
    """Behavioral specs are loaded from DB when a tract is reopened."""

    def test_profile_restored_to_registry(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        profile = WorkflowProfile(
            name="restored-profile",
            description="Test",
            config={"temperature": 0.5},
        )
        t.persist_behavioral_spec("profile", profile.name, profile.to_spec())
        t.close()

        t2 = _reopen_tract(tmp_path)
        restored = t2.get_profile("restored-profile")
        assert restored.name == "restored-profile"
        assert restored.config["temperature"] == 0.5
        t2.close()

    def test_template_restored_to_registry(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        tmpl = DirectiveTemplate(
            name="restored-tmpl",
            description="Test",
            content="Template {x}",
            parameters={"x": "param"},
        )
        t.persist_behavioral_spec("template", tmpl.name, tmpl.to_spec())
        t.close()

        t2 = _reopen_tract(tmp_path)
        restored = t2.get_template("restored-tmpl")
        assert restored.name == "restored-tmpl"
        assert restored.render(x="hello") == "Template hello"
        t2.close()

    def test_gate_spec_not_auto_wired(self, tmp_path: Path) -> None:
        """Gate specs are loaded but NOT auto-wired as middleware (callables not restorable)."""
        t = _make_file_tract(tmp_path)
        t.gate("test-gate", event="pre_commit", check="test criterion")
        t.close()

        t2 = _reopen_tract(tmp_path)
        # Spec is in the DB
        specs = t2.load_behavioral_specs(spec_type="gate")
        assert len(specs) == 1
        assert specs[0]["spec_name"] == "test-gate"
        # But NOT in active gates (not auto-wired)
        assert "test-gate" not in t2.list_gates()
        t2.close()

    def test_maintainer_spec_not_auto_wired(self, tmp_path: Path) -> None:
        """Maintainer specs are loaded but NOT auto-wired as middleware."""
        t = _make_file_tract(tmp_path)
        t.maintain("test-m", event="post_commit", instructions="test", actions=["gc"])
        t.close()

        t2 = _reopen_tract(tmp_path)
        specs = t2.load_behavioral_specs(spec_type="maintainer")
        assert len(specs) == 1
        assert specs[0]["spec_name"] == "test-m"
        assert "test-m" not in t2.list_maintainers()
        t2.close()

    def test_corrupted_spec_quarantined(self, tmp_path: Path) -> None:
        """If a spec has invalid JSON, it is quarantined and skipped."""
        t = _make_file_tract(tmp_path)
        # Manually insert a corrupted spec
        from tract.storage.schema import BehavioralSpecRow
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        row = BehavioralSpecRow(
            tract_id=t._tract_id,
            spec_type="profile",
            spec_name="broken",
            spec_json="not valid json {{{",
            created_at=now,
            updated_at=now,
        )
        t._behavioral_spec_repo.save(row)
        t._session.commit()
        t.close()

        # Reopen -- the corrupted spec should be quarantined
        t2 = _reopen_tract(tmp_path)
        quarantined = t2.quarantined
        assert any("spec:profile:broken" in q for q in quarantined)
        # The profile should NOT be in the registry
        with pytest.raises(KeyError):
            t2.get_profile("broken")
        t2.close()


# ============================================================
# Memory-only tract (no persistence repo impact)
# ============================================================


class TestMemoryTract:
    """Behavioral spec operations work on in-memory tracts."""

    def test_persist_and_load_in_memory(self) -> None:
        t = Tract.open()
        t.persist_behavioral_spec("gate", "mem-gate", {"check": "test"})
        specs = t.load_behavioral_specs(spec_type="gate")
        assert len(specs) == 1
        assert specs[0]["spec_name"] == "mem-gate"
        t.close()

    def test_remove_in_memory(self) -> None:
        t = Tract.open()
        t.persist_behavioral_spec("gate", "mem-gate", {"check": "test"})
        assert t.remove_behavioral_spec("gate", "mem-gate") is True
        assert len(t.list_behavioral_specs()) == 0
        t.close()
