"""Middleware Reference: Events, Gates, Preservation, Observability, Autonomy

Quick reference for the middleware surface area:
- Event automation: pre/post_commit handlers, BlockedError, handler removal
- Transition gates: pre_transition guards, handoff modes, config-based gates
- Data preservation: Priority.PINNED, directives, pre_compress guards
- Observability: operation audit trail, context growth alerting
- Autonomous behaviors: auto-skip, compression triggers, keyword routing,
  adaptive config (circuit breaker), tool budget guard

All 12 middleware events:
  pre_commit, post_commit, pre_compile, pre_compress,
  pre_merge, pre_gc, pre_transition, post_transition,
  pre_generate, post_generate, pre_tool_execute, post_tool_execute

No LLM required.
"""

from datetime import datetime, timezone

from tract import (
    Tract,
    BlockedError,
    DialogueContent,
    MiddlewareContext,
    Priority,
)


# =====================================================================
# 1. EVENT AUTOMATION -- pre/post commit handlers, BlockedError
# =====================================================================

def event_automation() -> None:
    """Post-commit logging, pre-commit validation, and handler removal."""

    print("=" * 60)
    print("1. Event Automation")
    print("=" * 60, "\n")

    with Tract.open() as t:

        # --- Post-commit logging ---
        commit_log: list[dict] = []

        def log_commits(ctx: MiddlewareContext):
            """Log every commit with its hash and branch."""
            commit_log.append({
                "hash": ctx.commit.commit_hash[:8] if ctx.commit else "?",
                "branch": ctx.branch,
            })

        log_id = t.middleware.add("post_commit", log_commits)

        t.system("You are a helpful assistant.")
        t.user("What is Python?")
        t.assistant("Python is a high-level programming language.")

        print(f"  Post-commit logger captured {len(commit_log)} commits:")
        for entry in commit_log:
            print(f"    {entry['hash']} on {entry['branch']}")

        # --- Pre-commit validation (secret detection + length limit) ---
        print("\n  Pre-commit validators:")

        def block_secrets(ctx: MiddlewareContext):
            """Block commits containing secret-like patterns.
            For pre_commit, ctx.pending holds the content; ctx.commit is None.
            """
            if ctx.pending is None:
                return
            text = (getattr(ctx.pending, "text", "") or "").lower()
            for pattern in ["api_key=", "secret=", "password="]:
                if pattern in text:
                    raise BlockedError(
                        "pre_commit",
                        [f"Potential secret: '{pattern}' in content"],
                    )

        def limit_length(ctx: MiddlewareContext):
            """Block commits with content longer than 500 characters."""
            if ctx.pending is None:
                return
            text = getattr(ctx.pending, "text", "") or ""
            if len(text) > 500:
                raise BlockedError(
                    "pre_commit",
                    [f"Content too long: {len(text)} chars (limit 500)"],
                )

        secrets_id = t.middleware.add("pre_commit", block_secrets)
        limit_id = t.middleware.add("pre_commit", limit_length)

        # Normal commit passes both validators
        t.user("Tell me about security best practices.")
        print("    Normal commit: OK")

        # Secret blocked
        blocked = False
        try:
            t.user("Set api_key=sk-12345 in the config file.")
        except BlockedError as e:
            blocked = True
            print(f"    Blocked (secret): {e.reasons[0]}")
        assert blocked

        # Length blocked
        blocked = False
        try:
            t.user("A" * 501)
        except BlockedError as e:
            blocked = True
            print(f"    Blocked (length): {e.reasons[0]}")
        assert blocked

        # --- Pre-compile tracking ---
        compile_count = {"n": 0}

        def track_compiles(ctx: MiddlewareContext):
            compile_count["n"] += 1

        compile_id = t.middleware.add("pre_compile", track_compiles)
        t.compile()
        t.compile()
        print(f"\n  Pre-compile tracker: {compile_count['n']} compiles")

        # --- Removing middleware ---
        print("\n  Removing middleware:")
        for mid, name in [
            (log_id, "logger"), (secrets_id, "secrets"),
            (limit_id, "length"), (compile_id, "compile tracker"),
        ]:
            t.middleware.remove(mid)
            print(f"    Removed {name}")

        count_before = len(commit_log)
        t.user("This commit is not logged.")
        assert len(commit_log) == count_before
        print(f"  Post-removal: log unchanged ({count_before} entries)")

    print("\nPASSED")


# =====================================================================
# 2. TRANSITION GATES -- pre_transition guards, handoff modes
# =====================================================================

def transition_gates() -> None:
    """Block transitions with middleware, handoff modes, config-based gates."""

    print("\n" + "=" * 60)
    print("2. Transition Gates")
    print("=" * 60, "\n")

    with Tract.open() as t:
        t.system("You are a coding assistant.")

        # --- Commit-count gate ---
        def review_gate(ctx: MiddlewareContext):
            if ctx.target != "review":
                return
            count = len(ctx.tract.log())
            if count < 5:
                raise BlockedError(
                    "pre_transition",
                    [f"Need >= 5 commits for review (have {count})"],
                )

        t.middleware.add("pre_transition", review_gate)

        transition_log: list[str] = []
        t.middleware.add("post_transition", lambda ctx: transition_log.append(ctx.target))

        # Blocked: too few commits
        try:
            t.transition("review")
        except BlockedError as e:
            print(f"  Blocked: {e.reasons[0]}")

        t.user("Implement fibonacci")
        t.assistant("Implementation...")
        t.user("Add error handling")
        t.assistant("Validation added...")

        t.transition("review")
        print(f"  After 5+ commits: transitioned to {t.current_branch}")
        assert t.current_branch == "review"

        # --- Config-based gate ---
        t.switch("main")

        def production_gate(ctx: MiddlewareContext):
            if ctx.target != "production":
                return
            if not ctx.tract.config.get("approved"):
                raise BlockedError(
                    "pre_transition",
                    ["Production requires approved=True"],
                )

        t.middleware.add("pre_transition", production_gate)

        try:
            t.transition("production")
        except BlockedError as e:
            print(f"  Blocked: {e.reasons[0]}")

        t.config.set(approved=True)
        t.transition("production")
        print(f"  After approval: transitioned to {t.current_branch}")

        # --- Handoff modes ---
        t.switch("main")
        t.user("Context for handoff demo")
        t.assistant("Acknowledged")

        result = t.transition("full-handoff", handoff="full")
        if result:
            print(f"  full    -> {result.commit_hash[:8]} on {t.current_branch}")

        t.switch("main")
        t.config.set(handoff_summary_k=3)
        result = t.transition("summary-handoff", handoff="summary")
        if result:
            print(f"  summary -> {result.commit_hash[:8]} on {t.current_branch}")

        t.switch("main")
        result = t.transition("custom-handoff", handoff="Key context: fibonacci")
        if result:
            print(f"  custom  -> {result.commit_hash[:8]} on {t.current_branch}")

        t.switch("main")
        result = t.transition("bare-branch", handoff="none")
        print(f"  none    -> result={result}, branch={t.current_branch}")

        print(f"\n  Transition log: {transition_log}")

    print("\nPASSED")


# =====================================================================
# 3. DATA PRESERVATION -- PINNED, directives, pre_compress guards
# =====================================================================

def data_preservation() -> None:
    """PINNED priority, directives, pre_compress middleware guards."""

    print("\n" + "=" * 60)
    print("3. Data Preservation")
    print("=" * 60, "\n")

    with Tract.open() as t:
        t.register_tag("credentials")

        t.system("You are a helpful assistant.")
        t.user("What is Python?")
        t.assistant("Python is a high-level programming language.")
        important = t.commit(
            DialogueContent(role="user", text="CRITICAL: API key format: sk-xxx"),
            tags=["credentials"],
        )
        t.assistant("I will remember that format.")

        # Layer 1: PINNED annotation
        t.annotate(important.commit_hash, Priority.PINNED, reason="credential data")
        print(f"  PINNED commit: {important.commit_hash[:8]}")
        print("    Hard engine-level protection -- survives all compression")

        # Layer 2: Directives (PINNED by default)
        t.directive("api-format", "API key format is sk-xxx. Always validate.")
        print("  Directive 'api-format': PINNED by default, survives compression")

        # Layer 3: Pre-compress middleware guards
        def protect_credentials(ctx: MiddlewareContext):
            for ci in ctx.tract.log():
                if "credentials" in (ci.tags or []):
                    raise BlockedError(
                        "pre_compress",
                        ["Cannot compress: credential-tagged commits present"],
                    )

        def require_min_history(ctx: MiddlewareContext):
            count = len(ctx.tract.log())
            if count < 20:
                raise BlockedError(
                    "pre_compress",
                    [f"Too few commits: {count} (need >= 20)"],
                )

        t.middleware.add("pre_compress", protect_credentials)
        t.middleware.add("pre_compress", require_min_history)
        print("  pre_compress guards: tag-based + minimum-history threshold")

        # Layer 4: Config threshold
        t.config.set(auto_compress_threshold=100)
        print(f"  auto_compress_threshold: {t.config.get('auto_compress_threshold')}")

        print("\n  Layered defense:")
        print("    1. PINNED priority   -> engine-level hard protection")
        print("    2. Directives        -> PINNED by default, compiled into context")
        print("    3. Middleware         -> programmatic guards (tags, thresholds)")
        print("    4. Config            -> auto_compress_threshold controls triggers")

    print("\nPASSED")


# =====================================================================
# 4. OBSERVABILITY -- audit trail, context growth alerting
# =====================================================================

def observability() -> None:
    """Operation audit trail and context growth alerting."""

    print("\n" + "=" * 60)
    print("4. Observability")
    print("=" * 60, "\n")

    # --- Audit trail: single handler on 6 events ---
    print("--- Operation Audit Trail ---\n")

    with Tract.open() as t:
        log: list[dict] = []

        def record(ctx: MiddlewareContext) -> None:
            entry: dict = {
                "time": datetime.now(timezone.utc).isoformat(),
                "event": ctx.event, "branch": ctx.branch,
            }
            if ctx.commit:
                entry.update(op="commit", hash=ctx.commit.commit_hash[:8],
                             type=ctx.commit.content_type, tokens=ctx.commit.token_count)
            elif "transition" in ctx.event:
                entry.update(op="transition", target=ctx.target or "?")
            elif "compile" in ctx.event:
                entry.update(op="compile")
            elif "compress" in ctx.event:
                entry.update(op="compress")
            elif "merge" in ctx.event:
                entry.update(op="merge")
            log.append(entry)

        ids = [t.middleware.add(evt, record) for evt in [
            "post_commit", "pre_compile", "pre_compress",
            "pre_merge", "pre_transition", "post_transition",
        ]]

        t.system("You are a project assistant.")
        t.user("Start planning.")
        t.assistant("Planning initiated.")
        t.transition("implementation")
        t.user("Implement login.")
        t.assistant("Login module done.")
        t.compile()
        t.compress(content="[Summary] Planning and implementation done.")
        t.branch("hotfix")
        t.user("Fix auth bypass.")
        t.assistant("Patched session validation.")
        t.switch("implementation")
        t.merge("hotfix")

        print(f"  {len(log)} audit entries:")
        for i, e in enumerate(log, 1):
            op = e.get("op", "?")
            detail = ""
            if op == "commit":
                detail = f"[{e['hash']}] {e['type']} ({e['tokens']}tok)"
            elif op == "transition":
                detail = f"-> {e.get('target', '?')}"
            print(f"    {i:>2}. {e['event']:<18} {op:<11} {e['branch']:<16} {detail}")

        ops_found = {e.get("op") for e in log}
        assert {"commit", "transition", "compile", "compress", "merge"} <= ops_found
        print(f"  All 5 operation types captured")

        for mid in ids:
            t.middleware.remove(mid)

    # --- Context growth alerting ---
    print("\n--- Context Growth Alerting ---\n")

    MAX_CTX, ALERT_PCT = 2000, 30

    with Tract.open() as t:
        t.system("You are a technical architect.")
        prev = t.compile().token_count

        exchanges = [
            ("Review auth.", "JWT with 24h expiry and refresh rotation."),
            ("Database layer?", "PostgreSQL, pooling, read replicas."),
            ("Full caching arch: Redis cluster, eviction, TTLs, warming.",
             "3-node Redis Cluster. allkeys-lru. TTLs: 30m/5m/1h/4h. Circuit breaker."),
            ("API rate limiting?", "Token bucket 100 req/s, sliding window."),
        ]

        alerts = []
        for i, (q, a) in enumerate(exchanges):
            t.user(q)
            t.assistant(a)
            cur = t.compile().token_count
            delta = cur - prev
            gpct = delta / MAX_CTX * 100
            flag = " << ALERT" if gpct > ALERT_PCT else ""
            print(f"    #{i+1}: {cur:>4} tok (+{delta:>3}, {gpct:.1f}%){flag}")
            if gpct > ALERT_PCT:
                alerts.append(i + 1)
            prev = cur

        if alerts:
            print(f"  {len(alerts)} alert(s) at exchanges {alerts}")

        assert cur > exchanges[0][0].__len__()  # context grew

    print("\nPASSED")


# =====================================================================
# 5. AUTONOMOUS BEHAVIORS -- self-managing middleware patterns
# =====================================================================

def autonomous_behaviors() -> None:
    """Auto-skip, compression trigger, keyword routing, adaptive config."""

    print("\n" + "=" * 60)
    print("5. Autonomous Behaviors")
    print("=" * 60, "\n")

    # --- Auto-skip low-value content types ---
    print("--- Auto-Skip by Content Type ---\n")

    with Tract.open() as t:
        skip_types = {"config", "reasoning"}

        def auto_skip(ctx: MiddlewareContext):
            if ctx.commit and ctx.commit.content_type in skip_types:
                ctx.tract.annotate(ctx.commit.commit_hash, Priority.SKIP)

        t.middleware.add("post_commit", auto_skip)

        t.system("You are a helpful assistant.")
        t.config.set(temperature=0.5)    # config -> auto-skipped
        t.reasoning("Let me think...")   # reasoning -> auto-skipped
        t.user("What is Python?")
        t.assistant("A programming language.")

        skipped = len(t.search.skipped())
        print(f"  {len(t.log())} commits, {t.compile().commit_count} compiled, {skipped} skipped")
        assert skipped == 2

    # --- Commit-count compression trigger ---
    print("\n--- Commit-Count Compression Trigger ---\n")

    with Tract.open() as t:
        state = {"triggered": False, "count": 0}

        def auto_compress(ctx: MiddlewareContext):
            state["count"] += 1
            if state["count"] >= 6:
                # Production: ctx.tract.compress(strategy="sliding_window")
                state["triggered"] = True
                state["count"] = 0

        t.middleware.add("post_commit", auto_compress)

        t.system("You are a research analyst.")
        for i in range(7):
            t.user(f"Finding #{i}")
            t.assistant(f"Noted #{i}.")

        print(f"  {len(t.log())} commits, trigger fired: {state['triggered']}")
        assert state["triggered"]

    # --- Keyword-based stage routing ---
    print("\n--- Keyword Routing ---\n")

    with Tract.open() as t:
        transitions: list[str] = []
        routes = {
            "implementation": ["implement", "code", "function", "class"],
            "validation": ["test", "verify", "assert", "check"],
        }

        def keyword_router(ctx: MiddlewareContext):
            if not ctx.commit:
                return
            text = (ctx.commit.message or "").lower()
            content = str(ctx.tract.get_content(ctx.commit) or "").lower()
            combined = text + " " + content
            current = ctx.tract.config.get("stage") or "research"
            for target, keywords in routes.items():
                if target != current and any(kw in combined for kw in keywords):
                    ctx.tract.config.set(stage=target)
                    transitions.append(f"{current} -> {target}")
                    break

        t.middleware.add("post_commit", keyword_router)
        t.config.set(stage="research")
        t.system("You are a software engineer.")

        t.user("Research stack data structures.")
        t.assistant("A stack is a LIFO structure...")

        t.user("Now implement a Stack class with push and pop.")
        print(f"  Routed: {transitions[-1]}")

        t.assistant("Here's the implementation.")
        t.user("Now write tests to verify the Stack works.")
        print(f"  Routed: {transitions[-1]}")
        print(f"  Final stage: {t.config.get('stage')}")
        assert t.config.get("stage") == "validation"

    # --- Adaptive config / circuit breaker ---
    print("\n--- Adaptive Config (Circuit Breaker) ---\n")

    with Tract.open() as t:
        errors = {"consecutive": 0}

        def adaptive_config(ctx: MiddlewareContext):
            if not ctx.commit or "tool_result" not in (ctx.commit.tags or []):
                return
            is_error = (ctx.commit.metadata or {}).get("is_error", False)
            if is_error:
                errors["consecutive"] += 1
                if errors["consecutive"] >= 2:
                    ctx.tract.config.set(temperature=0.1)
            else:
                if errors["consecutive"] >= 2:
                    ctx.tract.config.set(temperature=0.7)
                errors["consecutive"] = 0

        t.middleware.add("post_commit", adaptive_config)
        t.config.set(temperature=0.7)
        t.system("You are a deployment agent.")

        # Two errors -> lower temperature
        for call_id, err_msg in [("c1", "Connection refused"), ("c2", "Timeout")]:
            t.assistant("Deploying...", metadata={
                "tool_calls": [{"id": call_id, "name": "deploy", "arguments": {}}]
            })
            t.tool_result(call_id, "deploy", err_msg, is_error=True)

        temp = t.config.get("temperature")
        print(f"  After 2 errors: temp={temp}")
        assert temp == 0.1

        # Success restores temperature
        t.assistant("Retry...", metadata={
            "tool_calls": [{"id": "c3", "name": "deploy", "arguments": {}}]
        })
        t.tool_result("c3", "deploy", "Deployed successfully.")
        temp = t.config.get("temperature")
        print(f"  After success:  temp={temp}")
        assert temp == 0.7

    # --- Tool result budget guard ---
    print("\n--- Tool Result Budget Guard ---\n")

    with Tract.open() as t:
        tool_state: dict = {"tokens": 0, "compaction_needed": False}

        def tool_budget(ctx: MiddlewareContext):
            if not ctx.commit or "tool_result" not in (ctx.commit.tags or []):
                return
            tool_state["tokens"] += ctx.commit.token_count
            if tool_state["tokens"] > 20:
                # Production: ctx.tract.compression.compress_tool_calls()
                tool_state["compaction_needed"] = True
                tool_state["tokens"] = 0

        t.middleware.add("post_commit", tool_budget)
        t.system("You are a search agent.")

        for i in range(5):
            t.assistant(f"Search {i}...", metadata={
                "tool_calls": [{"id": f"c{i}", "name": "search", "arguments": {}}]
            })
            t.tool_result(f"c{i}", "search", f"Result {i}: " + "x" * 100)

        print(f"  5 tool calls, compaction flagged: {tool_state['compaction_needed']}")
        assert tool_state["compaction_needed"]

    print("\nPASSED")


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    event_automation()
    transition_gates()
    data_preservation()
    observability()
    autonomous_behaviors()

    print("\n" + "=" * 60)
    print("Middleware Reference Summary")
    print("=" * 60)
    print("""
  Section                  Key Primitives
  -----------------------  -----------------------------------------------
  1. Event automation      t.middleware.add/remove, BlockedError,
                           pre_commit, post_commit, pre_compile
  2. Transition gates      pre_transition, post_transition, t.transition(),
                           handoff modes (none/full/summary/custom)
  3. Data preservation     Priority.PINNED, t.directive(), pre_compress,
                           auto_compress_threshold config
  4. Observability         6-event audit trail, context growth alerting
  5. Autonomous behaviors  auto-skip, compression trigger, keyword routing,
                           circuit breaker, tool budget guard

  All patterns: stateful closures, composable, zero new APIs.
""")
    print("Done.")


if __name__ == "__main__":
    main()
