"""Core API Reference: Content Types, Config, Directives, and Formatting

Quick reference for the most-used tract surface area in one file:

  1. Content types     — all 10 types and when to use each
  2. Shorthands        — t.system(), t.user(), t.assistant()
  3. Config basics     — t.config.set(), t.config.get(), t.config.get_all()
  4. DAG precedence    — closer to HEAD wins
  5. Branch isolation  — each branch resolves config independently
  6. Unset semantics   — setting a key to None removes it
  7. Directives        — named standing instructions (deduplicated by name)
  8. Format methods    — to_dicts(), to_openai(), to_anthropic(), pprint()
  9. Batch commits     — atomic multi-commit (all-or-nothing)
 10. Status            — token counting and budget tracking
 11. Compile strategies — full, messages, adaptive

No LLM required — every scenario runs without API keys.
"""

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import (
    Tract,
    Priority,
    MiddlewareContext,
    InstructionContent,
    DialogueContent,
    ToolIOContent,
    ReasoningContent,
    ArtifactContent,
    OutputContent,
    FreeformContent,
    ConfigContent,
    MetadataContent,
    TractConfig,
    TokenBudgetConfig,
)
from tract.formatting import pprint_log


# === Section: Content Types ================================================


def content_types():
    """All 10 content types committed manually via t.commit(ContentModel(...))."""
    with Tract.open() as t:
        print("=== 1. Content Types ===\n")

        ci = t.commit(InstructionContent(text="You are a helpful assistant."))
        print(f"  InstructionContent:  {ci.commit_hash[:8]}  {ci.message}")

        ci = t.commit(DialogueContent(role="user", text="Hello"))
        print(f"  DialogueContent:     {ci.commit_hash[:8]}  {ci.message}")

        ci = t.commit(DialogueContent(role="assistant", text="Hi there!"))
        print(f"  DialogueContent:     {ci.commit_hash[:8]}  {ci.message}")

        ci = t.commit(ToolIOContent(
            tool_name="search", direction="call",
            payload={"query": "weather"}, status="success",
        ))
        print(f"  ToolIOContent:       {ci.commit_hash[:8]}  {ci.message}")

        ci = t.commit(ReasoningContent(text="Let me think step by step..."))
        print(f"  ReasoningContent:    {ci.commit_hash[:8]}  {ci.message}")

        ci = t.commit(ArtifactContent(
            artifact_type="code", content="print('hello')", language="python",
        ))
        print(f"  ArtifactContent:     {ci.commit_hash[:8]}  {ci.message}")

        ci = t.commit(OutputContent(text="Result: 42", format="text"))
        print(f"  OutputContent:       {ci.commit_hash[:8]}  {ci.message}")

        ci = t.commit(FreeformContent(payload={"custom_key": "custom_value"}))
        print(f"  FreeformContent:     {ci.commit_hash[:8]}  {ci.message}")

        ci = t.commit(ConfigContent(
            settings={"temperature": 0.7, "model": "gpt-4o"},
        ))
        print(f"  ConfigContent:       {ci.commit_hash[:8]}  {ci.message}")

        ci = t.commit(MetadataContent(kind="tag", data={"version": "1.0"}))
        print(f"  MetadataContent:     {ci.commit_hash[:8]}  {ci.message}")

    print("  Content types: PASSED")


# === Section: Shorthands ===================================================


def shorthands():
    """t.system(), t.user(), t.assistant() — no imports needed."""
    with Tract.open() as t:
        print("\n=== 2. Shorthands ===\n")

        ci = t.system("You are a helpful assistant.")
        print(f"  t.system():    {ci.commit_hash[:8]}  {ci.content_type}  {ci.message}")

        ci = t.user("What is Python?")
        print(f"  t.user():      {ci.commit_hash[:8]}  {ci.content_type}  {ci.message}")

        ci = t.assistant("A programming language.")
        print(f"  t.assistant(): {ci.commit_hash[:8]}  {ci.content_type}  {ci.message}")

    print("  Shorthands: PASSED")


# === Section: Config Basics ================================================


def config_basics():
    """Set and retrieve config values, including defaults for missing keys."""
    with Tract.open() as t:
        print("\n=== 3. Config Basics ===\n")

        t.config.set(model="gpt-4", temperature=0.7, max_tokens=4096)

        assert t.config.get("model") == "gpt-4"
        assert t.config.get("temperature") == 0.7
        assert t.config.get("max_tokens") == 4096
        print(f"  model:       {t.config.get('model')}")
        print(f"  temperature: {t.config.get('temperature')}")
        print(f"  max_tokens:  {t.config.get('max_tokens')}")

        # Missing keys return None, or an explicit default
        assert t.config.get("nonexistent") is None
        assert t.config.get("nonexistent", "fallback") == "fallback"
        print(f"  missing key: {t.config.get('nonexistent', 'fallback')}")

        # Complex values work too
        t.config.set(stop=["END", "DONE", "---"])
        assert t.config.get("stop") == ["END", "DONE", "---"]
        print(f"  complex val: {t.config.get('stop')}")

        # Resolve all active configs at once
        all_cfg = t.config.get_all()
        print(f"  get_all():   {sorted(all_cfg.keys())}")

    print("  Config basics: PASSED")


# === Section: DAG Precedence ===============================================


def dag_precedence():
    """When the same key is set multiple times, the commit nearest HEAD wins."""
    with Tract.open() as t:
        print("\n=== 4. DAG Precedence ===\n")

        t.config.set(model="gpt-4", temperature=0.7)
        t.user("Hello, world!")
        t.assistant("Hi there!")

        # Override just one key — closer to HEAD wins
        t.config.set(temperature=0.2)

        assert t.config.get("model") == "gpt-4"       # inherited (unchanged)
        assert t.config.get("temperature") == 0.2      # overridden
        print(f"  model (unchanged):   {t.config.get('model')}")
        print(f"  temperature (0.2):   {t.config.get('temperature')}")

        all_cfg = t.config.get_all()
        assert all_cfg["model"] == "gpt-4"
        assert all_cfg["temperature"] == 0.2
        assert "missing_key" not in all_cfg

    print("  DAG precedence: PASSED")


# === Section: Branch Isolation =============================================


def branch_isolation():
    """Configs on different branches resolve independently."""
    with Tract.open() as t:
        print("\n=== 5. Branch Isolation ===\n")

        t.config.set(model="gpt-4", temperature=0.7)

        # Experiment branch gets its own overrides
        t.branch("experiment")
        t.switch("experiment")
        t.config.set(model="claude-3", temperature=0.0)

        assert t.config.get("model") == "claude-3"
        assert t.config.get("temperature") == 0.0
        print(f"  experiment:  model={t.config.get('model')}, temp={t.config.get('temperature')}")

        # Main still has original values
        t.switch("main")
        assert t.config.get("model") == "gpt-4"
        assert t.config.get("temperature") == 0.7
        print(f"  main:        model={t.config.get('model')}, temp={t.config.get('temperature')}")

        # Feature branch inherits from main at fork point
        t.branch("feature")
        t.switch("feature")
        assert t.config.get("model") == "gpt-4"        # inherited
        t.config.set(max_tokens=2048)                   # branch-specific addition
        assert t.config.get("max_tokens") == 2048
        print(f"  feature:     model={t.config.get('model')}, max_tokens={t.config.get('max_tokens')}")

    print("  Branch isolation: PASSED")


# === Section: Unset Semantics ==============================================


def unset_semantics():
    """Setting a key to None hides it from get() and get_all()."""
    with Tract.open() as t:
        print("\n=== 6. Unset Semantics ===\n")

        t.config.set(model="gpt-4", temperature=0.5)

        # "Unset" temperature
        t.config.set(temperature=None)
        assert t.config.get("temperature") is None
        assert t.config.get("temperature", 0.7) == 0.7  # default kicks in
        assert t.config.get("model") == "gpt-4"         # unaffected
        print(f"  After unset: temperature={t.config.get('temperature')}, model={t.config.get('model')}")

        all_cfg = t.config.get_all()
        assert "temperature" not in all_cfg
        assert all_cfg["model"] == "gpt-4"

        # Custom (non-well-known) keys follow the same pattern
        t.config.set(custom_key="hello")
        assert t.config.get("custom_key") == "hello"
        t.config.set(custom_key=None)
        assert t.config.get("custom_key") is None
        print(f"  Custom keys: same unset behavior (custom_key={t.config.get('custom_key')})")

    print("  Unset semantics: PASSED")


# === Section: Directives ===================================================


def directives():
    """Named standing instructions — deduplicated by name, PINNED priority."""
    with Tract.open() as t:
        print("\n=== 7. Directives ===\n")

        t.directive(
            "tone",
            "Always respond in a professional, concise tone. "
            "Avoid filler words and unnecessary qualifiers.",
        )
        t.directive(
            "format",
            "Use markdown formatting for all responses. "
            "Include code blocks with language tags.",
        )
        print("  Committed 'tone' and 'format' directives (PINNED by default)")

        # Override by name: same name -> closest to HEAD wins
        t.directive(
            "tone",
            "Respond in a friendly, casual tone. Use analogies and humor.",
        )
        print("  Overrode 'tone' directive (casual instead of formal)")

        # Directives appear in the compiled context as system messages
        ctx = t.compile()
        dicts = ctx.to_dicts()
        system_texts = [d["content"] for d in dicts if d["role"] == "system"]
        all_system = " ".join(system_texts)
        assert "casual" in all_system or "humor" in all_system  # latest tone wins
        assert "markdown" in all_system                         # format still present
        print(f"  Compiled: {len(system_texts)} system messages with active directives")

        # Directives are visible in the log
        log = t.log()
        directive_commits = [c for c in log if c.content_type == "instruction"]
        print(f"  Directive commits in log: {len(directive_commits)}")

    print("  Directives: PASSED")


# === Section: Format Methods ===============================================


def format_methods():
    """compile() -> to_dicts(), to_openai(), to_anthropic(), pprint()."""
    with Tract.open() as t:
        print("\n=== 8. Format Methods ===\n")

        t.system("You are a helpful coding assistant.")
        t.user("What is a decorator in Python?")
        t.assistant("A decorator wraps a function to modify its behavior.")
        t.user("Show me an example.")

        ctx = t.compile()

        # Generic list of {"role": ..., "content": ...}
        dicts = ctx.to_dicts()
        print(f"  to_dicts():     {len(dicts)} messages")
        for d in dicts:
            print(f"    [{d['role']}] {d['content'][:60]}")

        # OpenAI format — ready for openai.chat.completions.create(messages=...)
        openai_msgs = ctx.to_openai()
        print(f"\n  to_openai():    {len(openai_msgs)} messages")
        assert all("role" in m and "content" in m for m in openai_msgs)

        # Anthropic format — system extracted separately
        anthropic_fmt = ctx.to_anthropic()
        sys_text = anthropic_fmt["system"]
        msgs = anthropic_fmt["messages"]
        print(f"  to_anthropic(): system='{sys_text[:40]}...', {len(msgs)} messages")
        assert "system" in anthropic_fmt
        assert "messages" in anthropic_fmt

        # Pretty-print: three styles available
        print(f"\n  pprint styles: 'table' (default), 'chat', 'compact'")
        print(f"  str(ctx):      {ctx}")  # compact one-liner

    print("  Format methods: PASSED")


# === Section: Batch Commits ================================================


def batch_commits():
    """t.batch() for atomic multi-commit — all-or-nothing semantics."""
    with Tract.open() as t:
        print("\n=== 9. Batch Commits ===\n")

        t.system("You are an assistant.")
        count_before = len(t.compile().messages)

        # Failure inside batch -> rollback, no commits land
        try:
            with t.batch():
                t.user("Question 1")
                t.user("Question 2")
                raise RuntimeError("Simulated failure")
        except RuntimeError:
            pass

        count_after = len(t.compile().messages)
        assert count_before == count_after
        print(f"  Rollback: {count_before} -> {count_after} messages (unchanged)")

        # Success -> all commits land atomically
        with t.batch():
            t.user("Question 1")
            t.user("Question 2")

        count_success = len(t.compile().messages)
        assert count_success == 3  # system + 2 questions
        print(f"  Success:  {count_success} messages (system + 2 questions)")

    print("  Batch commits: PASSED")


# === Section: Status =======================================================


def status_tracking():
    """t.status() for token counting and budget tracking."""
    with Tract.open() as t:
        print("\n=== 10. Status ===\n")

        t.system("You are helpful.")
        t.user("Hello")
        t.assistant("Hi there!")

        status = t.status()
        print(f"  str(status):      {status}")
        print(f"  commit_count:     {status.commit_count}")
        print(f"  token_count:      {status.token_count}")
        print(f"  token_budget_max: {status.token_budget_max}")
        assert status.commit_count == 3
        assert status.token_count > 0

    # With a token budget configured
    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=4096))
    with Tract.open(config=config) as t2:
        t2.system("You are helpful.")
        t2.user("Hello")
        s = t2.status()
        if s.token_budget_max:
            pct = s.token_count / s.token_budget_max * 100
            print(f"  Budget usage:     {pct:.1f}% of {s.token_budget_max} tokens")

    print("  Status tracking: PASSED")


# === Section: Compile Strategies ===========================================


def compile_strategies():
    """Config drives which compile strategy builds the context window.

    Strategies differ in content detail, not message count:
      full     -- every commit's full content
      messages -- lightweight commit-message summaries only
      adaptive -- last k full, rest lightweight
    """
    with Tract.open() as t:
        print("\n=== 11. Compile Strategies ===\n")

        t.system("You are a helpful assistant.")
        for i in range(6):
            t.user(f"Question {i + 1}: Tell me about topic {i + 1}.")
            t.assistant(f"Answer {i + 1}: Information about topic {i + 1}.")

        # Full strategy — every commit in full
        t.config.set(compile_strategy="full")
        ctx_full = t.compile(strategy=t.config.get("compile_strategy"))

        # Messages strategy — lightweight summaries
        t.config.set(compile_strategy="messages", compile_strategy_k=4)
        ctx_msg = t.compile(strategy=t.config.get("compile_strategy"))

        # Adaptive strategy — last k full, rest lightweight
        t.config.set(compile_strategy="adaptive")
        k = t.config.get("compile_strategy_k")
        ctx_adapt = t.compile(
            strategy=t.config.get("compile_strategy"), strategy_k=k,
        )

        print(f"  full:     {len(ctx_full.messages)} msgs (full content)")
        print(f"  messages: {len(ctx_msg.messages)} msgs (commit summaries)")
        print(f"  adaptive: {len(ctx_adapt.messages)} msgs (last {k} full, rest light)")

    print("  Compile strategies: PASSED")


# === Section: Config in Middleware =========================================


def middleware_reads_config():
    """Middleware handlers can read live config during event hooks."""
    with Tract.open() as t:
        print("\n=== 12. Middleware Reads Config ===\n")

        t.config.set(model="gpt-4", max_tokens=4096)
        snapshots = []

        def on_commit(ctx: MiddlewareContext):
            snapshots.append({
                "model": ctx.tract.config.get("model"),
                "max_tokens": ctx.tract.config.get("max_tokens"),
            })

        mid_id = t.middleware.add("post_commit", on_commit)

        t.user("Hello")
        t.assistant("Hi")

        # Override mid-conversation
        t.config.set(model="gpt-4o")
        t.user("Updated model")

        assert snapshots[0]["model"] == "gpt-4"
        assert snapshots[-1]["model"] == "gpt-4o"
        print(f"  Snapshot 0: model={snapshots[0]['model']}")
        print(f"  Snapshot 2: model={snapshots[-1]['model']} (after override)")

        t.middleware.remove(mid_id)

    print("  Middleware reads config: PASSED")


# === Section: Log Visibility ===============================================


def log_visibility():
    """Configs and directives are visible commits in the log."""
    with Tract.open() as t:
        print("\n=== 13. Log Visibility ===\n")

        t.config.set(model="gpt-4o", temperature=0.7)
        t.directive("tone", "Be concise and direct.")
        t.user("Hello")
        t.assistant("Hi!")

        log = t.log()
        for entry in log:
            print(f"  {entry.commit_hash[:8]}  [{entry.content_type:12}]  {entry.message}")

        config_commits = [c for c in log if c.content_type == "config"]
        directive_commits = [c for c in log if c.content_type == "instruction"]
        dialogue_commits = [c for c in log if c.content_type == "dialogue"]
        print(f"\n  config={len(config_commits)}, directive={len(directive_commits)}, dialogue={len(dialogue_commits)}")

    print("  Log visibility: PASSED")


# === Runner ================================================================


def main() -> None:
    content_types()
    shorthands()
    config_basics()
    dag_precedence()
    branch_isolation()
    unset_semantics()
    directives()
    format_methods()
    batch_commits()
    status_tracking()
    compile_strategies()
    middleware_reads_config()
    log_visibility()
    print("\n" + "=" * 60)
    print("All core API reference scenarios: PASSED")


# Alias for pytest discovery
test_core_api = main


if __name__ == "__main__":
    main()
