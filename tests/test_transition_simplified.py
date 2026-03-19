"""Tests for simplified transition: middleware gates and handoff modes.

Covers:
- transition(target) creates branch and switches
- transition with handoff="none" produces no handoff commit
- transition with handoff="full" produces handoff commit with compiled context
- transition with handoff="summary" uses adaptive strategy
- transition with handoff=custom_text produces handoff with that text
- pre_transition middleware can block
- post_transition middleware fires after switch
"""

from __future__ import annotations

import pytest

from tract import (
    BlockedError,
    Tract,
)


# ---------------------------------------------------------------------------
# Basic transition behavior
# ---------------------------------------------------------------------------


class TestTransitionBasic:
    """transition(target) creates branch and switches."""

    def test_transition_creates_branch(self):
        """transition() creates the target branch if it does not exist."""
        with Tract.open() as t:
            t.user("Setup")
            t.transition("feature")
            branches = {b.name for b in t.list_branches()}
            assert "feature" in branches

    def test_transition_switches_to_target(self):
        """transition() switches to the target branch."""
        with Tract.open() as t:
            t.user("Setup")
            t.transition("feature")
            assert t.current_branch == "feature"

    def test_transition_to_existing_branch(self):
        """transition() switches to an existing branch without error."""
        with Tract.open() as t:
            t.user("Setup")
            t.branch("feature", switch=False)
            t.transition("feature")
            assert t.current_branch == "feature"

    def test_transition_returns_none_for_no_handoff(self):
        """transition() with default handoff='none' returns None."""
        with Tract.open() as t:
            t.user("Setup")
            result = t.transition("feature")
            assert result is None


# ---------------------------------------------------------------------------
# Handoff modes
# ---------------------------------------------------------------------------


class TestTransitionHandoff:
    """Handoff parameter controls what context is transferred."""

    def test_handoff_none_no_commit(self):
        """handoff='none' produces no handoff commit on target."""
        with Tract.open() as t:
            t.user("Setup on main")
            main_head = t.head
            t.transition("feature", handoff="none")
            # On feature branch, head should be same as main (shared ancestry)
            # No extra handoff commit
            feature_head = t.head
            assert feature_head == main_head

    def test_handoff_full_creates_commit(self):
        """handoff='full' creates a handoff commit with compiled context."""
        with Tract.open() as t:
            t.system("You are helpful.")
            t.user("Hello world")
            t.assistant("Hi there!")
            result = t.transition("feature", handoff="full")
            assert result is not None
            assert result.content_type == "instruction"
            assert "handoff" in result.message.lower()

    def test_handoff_full_contains_context(self):
        """handoff='full' commit includes compiled context text."""
        with Tract.open() as t:
            t.system("You are helpful.")
            t.user("Hello world")
            result = t.transition("feature", handoff="full")
            # The handoff commit should contain the compiled context
            compiled = t.compile()
            texts = [m.content for m in compiled.messages]
            # The handoff commit text should reference the original content
            assert any("handoff" in txt.lower() or "Hello world" in txt for txt in texts)

    def test_handoff_summary_creates_commit(self):
        """handoff='summary' creates a handoff commit using adaptive strategy."""
        with Tract.open() as t:
            t.system("You are helpful.")
            t.user("Hello world")
            t.assistant("Hi there!")
            result = t.transition("feature", handoff="summary")
            assert result is not None
            assert "handoff" in result.message.lower()

    def test_handoff_custom_text(self):
        """Custom text string as handoff creates a commit with that text."""
        with Tract.open() as t:
            t.user("Setup")
            custom_text = "This is a custom handoff message for the next branch."
            result = t.transition("feature", handoff=custom_text)
            assert result is not None
            # The handoff commit should contain our custom text
            compiled = t.compile()
            texts = [m.content for m in compiled.messages]
            assert any(custom_text in txt for txt in texts)

    def test_handoff_summary_k_config(self):
        """handoff='summary' respects handoff_summary_k config."""
        with Tract.open() as t:
            # Add enough commits to make adaptive strategy meaningful
            t.system("You are helpful.")
            for i in range(5):
                t.user(f"User message {i}")
                t.assistant(f"Reply {i}")
            t.config.set(handoff_summary_k=2)
            result = t.transition("feature", handoff="summary")
            assert result is not None


# ---------------------------------------------------------------------------
# Middleware gates
# ---------------------------------------------------------------------------


class TestTransitionMiddleware:
    """pre_transition and post_transition middleware integration."""

    def test_pre_transition_can_block(self):
        """pre_transition middleware raising BlockedError prevents transition."""
        with Tract.open() as t:
            t.user("Setup")
            def blocker(ctx):
                if ctx.target == "restricted":
                    raise BlockedError("pre_transition", "Branch restricted")
            t.middleware.add("pre_transition", blocker)
            with pytest.raises(BlockedError, match="Branch restricted"):
                t.transition("restricted")
            assert t.current_branch == "main"

    def test_pre_transition_allows_other_branches(self):
        """pre_transition middleware that blocks specific branch allows others."""
        with Tract.open() as t:
            t.user("Setup")
            def blocker(ctx):
                if ctx.target == "restricted":
                    raise BlockedError("pre_transition", "Branch restricted")
            t.middleware.add("pre_transition", blocker)
            # This should succeed
            t.transition("allowed")
            assert t.current_branch == "allowed"

    def test_post_transition_fires_after_switch(self):
        """post_transition fires after the branch switch completes."""
        events = []
        with Tract.open() as t:
            t.user("Setup")
            def post_handler(ctx):
                events.append({
                    "event": ctx.event,
                    "target": ctx.target,
                    "current_branch": t.current_branch,
                })
            t.middleware.add("post_transition", post_handler)
            t.transition("feature")
            assert len(events) == 1
            assert events[0]["event"] == "post_transition"
            assert events[0]["target"] == "feature"
            # After transition, we should be on the target branch
            assert events[0]["current_branch"] == "feature"

    def test_both_pre_and_post_fire(self):
        """Both pre and post transition middleware fire in order."""
        order = []
        with Tract.open() as t:
            t.user("Setup")
            t.middleware.add("pre_transition", lambda ctx: order.append("pre"))
            t.middleware.add("post_transition", lambda ctx: order.append("post"))
            t.transition("feature")
            assert order == ["pre", "post"]
