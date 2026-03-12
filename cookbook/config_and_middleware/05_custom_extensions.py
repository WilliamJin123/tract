"""Custom Extensions with Middleware

Middleware is the escape hatch for custom logic. Any Python code can run
on any event — just plain functions, no framework to learn.

Patterns shown:
  1. Custom validation  -- pre_commit checks with arbitrary logic
  2. Custom counters    -- post_commit tracking with closures
  3. Custom gates       -- pre_transition with domain logic
  4. Composable guards  -- multiple middleware on the same event

Demonstrates: t.use() for custom logic, closures as state,
              composing multiple handlers, BlockedError

No LLM required.
"""

from tract import Tract, BlockedError


def main():
    with Tract.open() as t:

        # --- Custom validation: language detection ---

        print("=== Custom Validation: Language Detection ===\n")

        PYTHON_MARKERS = ["def ", "import ", "class ", "print("]
        JS_MARKERS = ["function ", "const ", "let ", "=>"]

        detected_languages = []

        def detect_language(ctx):
            """Post-commit: detect programming language in message."""
            if ctx.commit is None:
                return
            text = (ctx.commit.message or "").lower()
            for marker in PYTHON_MARKERS:
                if marker in text:
                    detected_languages.append("python")
                    return
            for marker in JS_MARKERS:
                if marker in text:
                    detected_languages.append("javascript")
                    return

        lang_id = t.use("post_commit", detect_language)
        print(f"  Registered language detector: {lang_id}")

        # --- Custom counter: track user messages ---

        print("\n=== Custom Counter: User Message Tracking ===\n")

        user_count = {"n": 0, "total_chars": 0}

        def count_user_messages(ctx):
            """Pre-commit: count user dialogue messages.

            The pre_commit ctx.pending holds the content model (e.g.
            DialogueContent) before it is persisted, so we can inspect
            .role and .text without touching private storage APIs.
            """
            pending = ctx.pending
            if pending is None:
                return
            if getattr(pending, "content_type", None) == "dialogue":
                if getattr(pending, "role", None) == "user":
                    user_count["n"] += 1
                    user_count["total_chars"] += len(getattr(pending, "text", ""))

        counter_id = t.use("pre_commit", count_user_messages)
        print(f"  Registered user counter: {counter_id}")

        # --- Build conversation ---

        print("\n=== Building Conversation ===\n")

        t.system("You are a Python tutor.")
        t.user("What are generators?")
        t.assistant("Generators yield values lazily using def and yield.")
        t.user("Show me an example with def and yield.")
        t.assistant("def countdown(n): yield from range(n, 0, -1)")
        t.user("Can you show a JavaScript const example?")
        t.assistant("const add = (a, b) => a + b")

        print(f"  Total commits: {len(t.log())}")
        print(f"  User messages: {user_count['n']}")
        print(f"  Total user chars: {user_count['total_chars']}")
        print(f"  Languages detected: {detected_languages}")

        # --- Custom gate: minimum user engagement ---

        print("\n=== Custom Gate: User Engagement ===\n")

        def engagement_gate(ctx):
            """Require 3+ user messages before any transition."""
            if user_count["n"] < 3:
                raise BlockedError(
                    "pre_transition",
                    [f"Need >= 3 user messages (have {user_count['n']})"],
                )

        gate_id = t.use("pre_transition", engagement_gate)
        print(f"  Registered engagement gate: {gate_id}")

        # Test the gate
        result = t.transition("review")
        print(f"  Transition to 'review': succeeded (3 user messages)")
        print(f"  Current branch: {t.current_branch}")

        # --- Composable guards: multiple handlers on same event ---

        print("\n=== Composable Guards ===\n")

        t.switch("main")

        blocked_reasons = []

        def content_length_guard(ctx):
            """Block commits with very long messages."""
            if ctx.commit and len(ctx.commit.message or "") > 1000:
                raise BlockedError("pre_commit", ["Message exceeds 1000 chars"])

        def profanity_guard(ctx):
            """Block commits containing banned words."""
            if ctx.commit is None:
                return
            text = (ctx.commit.message or "").lower()
            banned = ["badword1", "badword2"]
            for word in banned:
                if word in text:
                    raise BlockedError("pre_commit", [f"Banned word: {word}"])

        len_id = t.use("pre_commit", content_length_guard)
        prof_id = t.use("pre_commit", profanity_guard)
        print(f"  Guard 1 (length): {len_id}")
        print(f"  Guard 2 (profanity): {prof_id}")
        print("  Both run on pre_commit -- first BlockedError wins")

        # Normal commit goes through both guards
        t.user("This is a normal, clean commit.")
        print("  Normal commit: OK (passed both guards)")

        # --- Cleanup ---

        print("\n=== Cleanup ===\n")

        for hid in [lang_id, counter_id, gate_id, len_id, prof_id]:
            t.remove_middleware(hid)
        print("  All middleware removed")

        # --- Summary ---

        print("\n=== Summary ===\n")
        print("  Middleware replaces custom conditions/actions/metrics.")
        print("  Use closures for state, BlockedError for blocking,")
        print("  and multiple handlers on the same event for composition.")


if __name__ == "__main__":
    main()
