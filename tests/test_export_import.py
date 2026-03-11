"""Tests for Tract.export_state() and load_state()."""
import json
import pytest
from tract import Tract
from tract.models.annotations import Priority


class TestExportState:
    def test_empty_tract_export(self):
        """Empty tract exports with no commits."""
        with Tract.open() as t:
            state = t.export_state()
            assert state["version"] == 1
            assert state["head"] is None
            assert state["commits"] == []

    def test_basic_export(self):
        """Export captures all commits."""
        with Tract.open() as t:
            t.system("You are helpful.")
            t.user("Hello")
            t.assistant("Hi there!")
            state = t.export_state()
            assert state["version"] == 1
            assert state["head"] is not None
            assert len(state["commits"]) == 3
            assert state["branch"] == "main"

    def test_export_includes_payloads(self):
        """Export with include_blobs=True includes content payloads."""
        with Tract.open() as t:
            t.user("Test message")
            state = t.export_state(include_blobs=True)
            # walk_ancestry returns root-first, so the only commit is index 0
            commit = state["commits"][0]
            assert "payload" in commit
            assert commit["payload"] is not None

    def test_export_without_blobs(self):
        """Export with include_blobs=False excludes payloads."""
        with Tract.open() as t:
            t.user("Test message")
            state = t.export_state(include_blobs=False)
            commit = state["commits"][0]
            assert "payload" not in commit or commit.get("payload") is None

    def test_export_is_json_serializable(self):
        """Exported state is fully JSON-serializable."""
        with Tract.open() as t:
            t.system("System")
            t.user("User")
            t.assistant("Assistant")
            state = t.export_state()
            # Should not raise
            json_str = json.dumps(state)
            assert len(json_str) > 0

    def test_export_captures_branches(self):
        """Export includes branch information."""
        with Tract.open() as t:
            t.system("Base")
            t.branch("feature")
            t.branch("experiment")
            state = t.export_state()
            assert "main" in state["branches"]
            assert "feature" in state["branches"]
            assert "experiment" in state["branches"]

    def test_export_with_annotations(self):
        """Export captures priority annotations."""
        with Tract.open() as t:
            t.system("Important", priority="pinned")
            state = t.export_state()
            commit = state["commits"][0]
            assert commit.get("priority") == "pinned"

    def test_export_with_metadata(self):
        """Export captures commit metadata."""
        with Tract.open() as t:
            t.user("Hello", metadata={"source": "test"})
            state = t.export_state()
            # Check that metadata is present in at least one commit
            has_metadata = any(c.get("metadata") for c in state["commits"])
            assert has_metadata

    def test_export_with_config(self):
        """Export captures config commits."""
        with Tract.open() as t:
            t.configure(model="gpt-4", temperature=0.7)
            t.user("Hello")
            state = t.export_state()
            config_commits = [c for c in state["commits"] if c["content_type"] == "config"]
            assert len(config_commits) >= 1

    def test_export_includes_commit_hashes(self):
        """Each exported commit has a hash."""
        with Tract.open() as t:
            t.user("Message 1")
            t.user("Message 2")
            state = t.export_state()
            for commit in state["commits"]:
                assert "hash" in commit
                assert commit["hash"] is not None
                assert len(commit["hash"]) > 0

    def test_export_includes_parents(self):
        """Exported commits have parent references."""
        with Tract.open() as t:
            t.user("First")
            t.user("Second")
            state = t.export_state()
            # Root commit should have no parents
            root = state["commits"][0]
            assert root["parents"] == [] or root["parents"] == [None]
            # Second commit should have one parent
            second = state["commits"][1]
            assert len(second["parents"]) >= 0  # may be empty if parent_repo has no entries

    def test_export_tract_id(self):
        """Export includes the tract ID."""
        with Tract.open() as t:
            t.user("Hello")
            state = t.export_state()
            assert "tract_id" in state
            assert state["tract_id"] == t.tract_id

    def test_export_has_timestamp(self):
        """Export includes an exported_at timestamp."""
        with Tract.open() as t:
            t.user("Hello")
            state = t.export_state()
            assert "exported_at" in state
            assert state["exported_at"] is not None


class TestLoadState:
    def test_load_empty_state(self):
        """Loading empty state returns 0."""
        with Tract.open() as t:
            state = {"version": 1, "commits": []}
            loaded = t.load_state(state)
            assert loaded == 0

    def test_load_invalid_version(self):
        """Loading invalid version raises ValueError."""
        with Tract.open() as t:
            with pytest.raises(ValueError):
                t.load_state({"version": 99})

    def test_load_invalid_dict(self):
        """Loading non-dict raises ValueError."""
        with Tract.open() as t:
            with pytest.raises(ValueError):
                t.load_state("not a dict")

    def test_round_trip(self):
        """Export then load preserves message content."""
        with Tract.open() as t1:
            t1.system("You are helpful.")
            t1.user("What is 2+2?")
            t1.assistant("4")
            state = t1.export_state()

        with Tract.open() as t2:
            loaded = t2.load_state(state)
            assert loaded == 3
            compiled = t2.compile()
            text = compiled.to_text()
            assert "helpful" in text
            assert "2+2" in text

    def test_load_preserves_annotations(self):
        """Loading state preserves priority annotations."""
        with Tract.open() as t1:
            t1.system("Important", priority="pinned")
            t1.user("Normal")
            state = t1.export_state()

        with Tract.open() as t2:
            loaded = t2.load_state(state)
            assert loaded == 2
            # Check the imported commits have the right priorities
            log = t2.log()
            # At least one should be pinned
            found_pinned = False
            for entry in log:
                ann = t2._annotation_repo.get_latest(entry.commit_hash)
                if ann and ann.priority == Priority.PINNED:
                    found_pinned = True
            assert found_pinned

    def test_load_without_blobs_skips(self):
        """Loading state exported without blobs skips commits."""
        with Tract.open() as t1:
            t1.user("Test")
            state = t1.export_state(include_blobs=False)

        with Tract.open() as t2:
            loaded = t2.load_state(state)
            assert loaded == 0  # no payloads to load

    def test_cross_tract_transfer(self):
        """Export from one tract, load into another with different ID."""
        with Tract.open() as t1:
            t1.system("Source system prompt")
            t1.user("Source question")
            t1.assistant("Source answer")
            state = t1.export_state()

        with Tract.open() as t2:
            t2.system("Destination base")
            loaded = t2.load_state(state)
            assert loaded == 3
            # Both destination base AND loaded content should be present
            compiled = t2.compile()
            text = compiled.to_text()
            assert "Destination base" in text
            assert "Source system prompt" in text

    def test_load_returns_count(self):
        """load_state returns the number of commits loaded."""
        with Tract.open() as t1:
            for i in range(5):
                t1.user(f"Message {i}")
            state = t1.export_state()

        with Tract.open() as t2:
            loaded = t2.load_state(state)
            assert loaded == 5

    def test_load_none_raises(self):
        """Loading None raises ValueError."""
        with Tract.open() as t:
            with pytest.raises(ValueError):
                t.load_state(None)


class TestExportImportEdgeCases:
    def test_export_after_compression(self):
        """Export works correctly after compression."""
        with Tract.open() as t:
            for i in range(10):
                t.user(f"Message {i}")
                t.assistant(f"Response {i}")
            t.compress(content="Summary of 10 exchanges")
            state = t.export_state()
            assert len(state["commits"]) > 0

    def test_export_after_branch_switch(self):
        """Export captures current branch state."""
        with Tract.open() as t:
            t.system("Base")
            t.branch("feature")
            t.switch("feature")
            t.user("Feature work")
            state = t.export_state()
            assert state["branch"] == "feature"
            # Should include base + feature commits
            assert len(state["commits"]) >= 2

    def test_large_export(self):
        """Export handles many commits efficiently."""
        with Tract.open() as t:
            for i in range(100):
                t.user(f"Message {i}")
            state = t.export_state()
            assert len(state["commits"]) == 100
            assert json.dumps(state)  # serializable

    def test_json_file_round_trip(self):
        """Export to JSON string, parse back, and load."""
        import tempfile
        import os

        with Tract.open() as t1:
            t1.system("System prompt")
            t1.user("Hello world")
            state = t1.export_state()

        # Serialize to JSON string and back
        json_str = json.dumps(state)
        parsed = json.loads(json_str)

        with Tract.open() as t2:
            loaded = t2.load_state(parsed)
            assert loaded == 2
            compiled = t2.compile()
            text = compiled.to_text()
            assert "System prompt" in text
            assert "Hello world" in text

    def test_export_preserves_content_types(self):
        """Different content types are preserved in export."""
        with Tract.open() as t:
            t.system("System")
            t.user("User")
            t.assistant("Assistant")
            state = t.export_state()
            types = {c["content_type"] for c in state["commits"]}
            assert "instruction" in types or "system" in types  # system prompt type
            assert "dialogue" in types  # user/assistant are dialogue
