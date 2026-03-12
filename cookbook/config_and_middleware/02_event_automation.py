"""Event Automation with Middleware

Middleware handlers fire when the agent does something: commit, compile,
compress, merge, gc, or transition. They let you configure automatic
reactions -- logging, validation, rate limiting -- without manual
intervention.

Valid middleware events (12 total):
  pre_commit, post_commit, pre_compile, pre_compress,
  pre_merge, pre_gc, pre_transition, post_transition,
  pre_generate, post_generate, pre_tool_execute, post_tool_execute

Pre-event handlers can raise BlockedError to prevent the operation.
Post-event handlers run after the operation completes.

Demonstrates: t.use(), pre_commit validation, post_commit logging,
              pre_compile hooks, BlockedError, t.remove_middleware()

No LLM required.
"""

from tract import Tract, BlockedError


def main():
    with Tract.open() as t:

        # --- Post-commit logging middleware ---

        print("=== Post-Commit Logging ===\n")

        commit_log = []

        def log_commits(ctx):
            """Log every commit with its hash and branch."""
            entry = {
                "hash": ctx.commit.commit_hash[:8] if ctx.commit else "?",
                "branch": ctx.branch,
            }
            commit_log.append(entry)

        log_id = t.use("post_commit", log_commits)
        print(f"  Registered post_commit logger: {log_id}")

        t.system("You are a helpful assistant.")
        t.user("What is Python?")
        t.assistant("Python is a high-level programming language.")

        print(f"  Commits logged: {len(commit_log)}")
        for entry in commit_log:
            print(f"    {entry['hash']} on {entry['branch']}")

        # --- Pre-commit validation middleware ---

        print("\n=== Pre-Commit Validation ===\n")

        def block_secrets(ctx):
            """Block commits that look like they contain secrets.

            For pre_commit events, ctx.pending holds the content model
            being committed (e.g. DialogueContent) while ctx.commit is None.
            """
            if ctx.pending is None:
                return
            text = getattr(ctx.pending, "text", "") or ""
            text = text.lower()
            for pattern in ["api_key=", "secret=", "password="]:
                if pattern in text:
                    raise BlockedError(
                        "pre_commit",
                        [f"Potential secret detected: '{pattern}' in content"],
                    )

        secrets_id = t.use("pre_commit", block_secrets)
        print(f"  Registered pre_commit validator: {secrets_id}")

        # This commit goes through fine
        t.user("Tell me about security best practices.")
        print("  Normal commit: OK")

        # This commit should be BLOCKED -- contains a secret pattern
        blocked = False
        try:
            t.user("Set api_key=sk-12345 in the config file.")
        except BlockedError as e:
            blocked = True
            print(f"  Blocked: {e.reasons[0]}")
        assert blocked, "Secret-containing commit should have been blocked"

        # --- Pre-commit token limit middleware ---

        print("\n=== Pre-Commit Token Limit ===\n")

        def limit_message_length(ctx):
            """Block commits with content longer than 500 characters.

            For pre_commit events, ctx.pending holds the content model
            being committed while ctx.commit is None.
            """
            if ctx.pending is None:
                return
            text = getattr(ctx.pending, "text", "") or ""
            if len(text) > 500:
                raise BlockedError(
                    "pre_commit",
                    [f"Content too long: {len(text)} chars (limit 500)"],
                )

        limit_id = t.use("pre_commit", limit_message_length)
        print(f"  Registered pre_commit length limiter: {limit_id}")

        # Short commit goes through
        t.assistant("Here are some security tips.")
        print("  Short commit: OK")

        # Long commit should be BLOCKED -- exceeds 500 chars
        long_text = "A" * 501
        blocked = False
        try:
            t.user(long_text)
        except BlockedError as e:
            blocked = True
            print(f"  Blocked: {e.reasons[0]}")
        assert blocked, "Over-length commit should have been blocked"

        # --- Pre-compile middleware ---

        print("\n=== Pre-Compile Middleware ===\n")

        compile_count = {"n": 0}

        def track_compiles(ctx):
            """Track how many times compile is called."""
            compile_count["n"] += 1

        compile_id = t.use("pre_compile", track_compiles)
        print(f"  Registered pre_compile tracker: {compile_id}")

        t.compile()
        t.compile()
        print(f"  Compiles tracked: {compile_count['n']}")

        # --- Removing middleware ---

        print("\n=== Removing Middleware ===\n")

        print(f"  Removing logger: {log_id}")
        t.remove_middleware(log_id)

        print(f"  Removing secrets filter: {secrets_id}")
        t.remove_middleware(secrets_id)

        print(f"  Removing length limiter: {limit_id}")
        t.remove_middleware(limit_id)

        print(f"  Removing compile tracker: {compile_id}")
        t.remove_middleware(compile_id)

        # Now commits go through without any middleware
        count_before = len(commit_log)
        t.user("This commit is not logged by middleware.")
        print(f"  Commits logged (unchanged): {len(commit_log)} (was {count_before})")

        # --- Summary ---

        print("\n=== Summary ===\n")
        print("  Middleware events:  pre_commit, post_commit, pre_compile,")
        print("                     pre_compress, pre_merge, pre_gc,")
        print("                     pre_transition, post_transition,")
        print("                     pre_generate, post_generate,")
        print("                     pre_tool_execute, post_tool_execute")
        print("  Block pattern:     raise BlockedError(event, [reasons])")
        print("  Remove pattern:    t.remove_middleware(handler_id)")


if __name__ == "__main__":
    main()
