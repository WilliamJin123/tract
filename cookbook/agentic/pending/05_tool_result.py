"""Agentic Tool Result Control

An LLM agent autonomously controls a PendingToolResult using ALL of its
actions: approve, reject, edit_result, summarize.  Three scenarios show
the agent detecting sensitive data and redacting it, summarizing verbose
output to fit a token budget, and rejecting useless error results.

Demonstrates: PendingToolResult lifecycle, to_dict()/to_tools() for LLM
              consumption, edit_result with original_content preservation,
              summarize with target_tokens, reject for error results,
              httpx direct LLM calls for agentic decision-making
"""

import json
import sys
from pathlib import Path
from typing import Any

import httpx

from tract import Tract
from tract.hooks.tool_result import PendingToolResult

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import groq as llm

MODEL_ID = llm.large


# -- helpers ---------------------------------------------------------

def ask_llm(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """Single LLM call via httpx. Returns the raw response dict."""
    body: dict[str, Any] = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": 1024,
    }
    if tools:
        body["tools"] = tools
    resp = httpx.post(
        f"{llm.base_url}/chat/completions",
        headers={"Authorization": f"Bearer {llm.api_key}"},
        json=body,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def extract_tool_call(raw: dict) -> dict[str, Any] | None:
    """Pull the first tool call from an LLM response, or None."""
    tc_list = raw["choices"][0]["message"].get("tool_calls", [])
    if not tc_list:
        return None
    tc = tc_list[0]
    return {
        "action": tc["function"]["name"],
        "args": json.loads(tc["function"].get("arguments", "{}")),
    }


# =====================================================================
# SCENARIO A -- Agent edits result (redaction)
# =====================================================================

def scenario_a_edit():
    """Tool result contains an API key. The agent detects sensitive data,
    uses edit_result to redact it, then approves."""
    print("=" * 60)
    print("SCENARIO A -- Agent Edits Result (Redaction)")
    print("=" * 60)
    print()
    print("  The tool result contains 'API_KEY=sk-12345'.")
    print("  The agent inspects via to_dict(), detects the secret,")
    print("  and uses edit_result to redact before approving.")
    print()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system(
            "You are a security-conscious code assistant. You NEVER allow "
            "secrets, API keys, passwords, or tokens into the conversation "
            "context. If a tool result contains sensitive data, redact it."
        )
        t.user("Show me the contents of the .env file.")

        # Assistant calls a tool
        t.assistant("Let me read that file for you.", metadata={
            "tool_calls": [{
                "id": "tc1",
                "name": "read_file",
                "arguments": {"path": ".env"},
            }],
        })

        # The tool result contains sensitive data
        raw_content = (
            "APP_NAME=my-service\n"
            "DEBUG=false\n"
            "API_KEY=sk-12345-secret-key-do-not-share\n"
            "DATABASE_URL=postgres://admin:hunter2@db.internal:5432/prod\n"
            "LOG_LEVEL=info"
        )

        pending: PendingToolResult = t.tool_result(
            "tc1", "read_file", raw_content, review=True,
        )

        print("  PendingToolResult before agent acts:")
        pending.pprint()

        # Give the agent the pending state and tool schemas
        state = pending.to_dict()
        tools = pending.to_tools()

        decision_messages = [
            {"role": "system", "content": (
                "You are a security agent reviewing tool results before they "
                "enter the LLM context window. You have tools to control the "
                "pending result. If the content contains secrets (API keys, "
                "passwords, tokens), use edit_result to replace the content "
                "with a redacted version, then approve. Redact by replacing "
                "secret values with '***REDACTED***'."
            )},
            {"role": "user", "content": (
                f"A tool result is pending. Review it and decide.\n\n"
                f"State:\n{json.dumps(state, indent=2)}"
            )},
        ]

        # Agent loop: may need edit_result then approve (two tool calls)
        for turn in range(5):
            raw_resp = ask_llm(decision_messages, tools)
            decision = extract_tool_call(raw_resp)

            if not decision:
                # Agent responded with text -- done
                text = raw_resp["choices"][0]["message"]["content"]
                print(f"\n  Agent (text): {text[:120]}")
                break

            action = decision["action"]
            args = decision["args"]
            print(f"  Agent calls: {action}({json.dumps(args)[:100]})")

            # Execute the action on the pending object
            result = pending.apply_decision(decision)

            if action == "edit_result":
                print(f"  -> Content replaced ({len(pending.content)} chars)")
                print(f"  -> Original preserved: {pending.original_content is not None}")
                # Feed result back so the agent can call approve next
                decision_messages.append(
                    raw_resp["choices"][0]["message"]
                )
                decision_messages.append({
                    "role": "tool",
                    "tool_call_id": raw_resp["choices"][0]["message"]["tool_calls"][0]["id"],
                    "content": "edit_result applied successfully. Now approve if ready.",
                })
                continue

            if action == "approve":
                print(f"  -> Approved! Committed.")
                break

            if action == "reject":
                print(f"  -> Rejected: {args.get('reason', '')}")
                break

        print(f"\n  Final pending status: {pending.status}")
        print(f"\n  Committed content (redacted):")
        ctx = t.compile()
        for msg in ctx.messages:
            if msg.role == "tool":
                print(f"    tool ({msg.metadata.get('tool_name', '?')}): "
                      f"{msg.content[:120]}")

        print(f"\n  original_content preserved for provenance:")
        print(f"    {(pending.original_content or '')[:80]}...")
        print()


# =====================================================================
# SCENARIO B -- Agent summarizes verbose output
# =====================================================================

def scenario_b_summarize():
    """Tool result is a huge directory listing. The agent uses
    summarize(target_tokens=100) to condense it, then approves."""
    print("=" * 60)
    print("SCENARIO B -- Agent Summarizes Verbose Output")
    print("=" * 60)
    print()
    print("  The tool result is a large directory listing (500+ lines).")
    print("  The agent decides it is too verbose and calls summarize()")
    print("  with target_tokens=100 to condense it, then approves.")
    print()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system(
            "You are a code navigation assistant. Keep tool results concise. "
            "If a tool returns more than ~200 tokens of output, summarize it "
            "to preserve context budget."
        )
        t.user("List all files in the project.")

        t.assistant("Let me list the project directory.", metadata={
            "tool_calls": [{
                "id": "tc2",
                "name": "list_directory",
                "arguments": {"path": "."},
            }],
        })

        # Generate a large directory listing
        lines = []
        for d in ["src", "tests", "docs", "scripts", "config", "data"]:
            for i in range(50):
                ext = ["py", "md", "json", "yaml", "txt"][i % 5]
                size = 1000 + i * 137
                lines.append(f"{d}/module_{i:03d}.{ext}    {size:>8} bytes")
        big_listing = "\n".join(lines)

        pending: PendingToolResult = t.tool_result(
            "tc2", "list_directory", big_listing, review=True,
        )

        print(f"  PendingToolResult: {pending.token_count} tokens, "
              f"{len(big_listing)} chars")
        pending.pprint()

        # Agent inspects
        state = pending.to_dict()
        tools = pending.to_tools()

        decision_messages = [
            {"role": "system", "content": (
                "You are a context budget manager reviewing tool results. "
                "You have tools to control a pending tool result. If the "
                "token_count exceeds 200 tokens, call summarize with "
                "target_tokens=100. After summarizing, call approve."
            )},
            {"role": "user", "content": (
                f"A tool result is pending review.\n\n"
                f"State:\n{json.dumps(state, indent=2)}"
            )},
        ]

        for turn in range(5):
            raw_resp = ask_llm(decision_messages, tools)
            decision = extract_tool_call(raw_resp)

            if not decision:
                text = raw_resp["choices"][0]["message"]["content"]
                print(f"\n  Agent (text): {text[:120]}")
                break

            action = decision["action"]
            args = decision["args"]
            print(f"  Agent calls: {action}({json.dumps(args)[:80]})")

            if action == "summarize":
                # summarize() calls the LLM internally via tract's LLM client
                pending.summarize(
                    target_tokens=args.get("target_tokens", 100),
                )
                print(f"  -> Summarized: {len(pending.content)} chars")
                print(f"  -> Original preserved: {pending.original_content is not None}")
                decision_messages.append(
                    raw_resp["choices"][0]["message"]
                )
                decision_messages.append({
                    "role": "tool",
                    "tool_call_id": raw_resp["choices"][0]["message"]["tool_calls"][0]["id"],
                    "content": (
                        f"Summarized. New content: {len(pending.content)} chars. "
                        f"Original preserved. Call approve to commit."
                    ),
                })
                continue

            result = pending.apply_decision(decision)

            if action == "approve":
                print(f"  -> Approved! Committed.")
                break

        print(f"\n  Final pending status: {pending.status}")
        print(f"\n  Summarized content committed:")
        ctx = t.compile()
        for msg in ctx.messages:
            if msg.role == "tool":
                print(f"    tool: {msg.content[:200]}")

        print(f"\n  Original was {len(pending.original_content or '')} chars, "
              f"summary is {len(pending.content)} chars")
        print()


# =====================================================================
# SCENARIO C -- Agent rejects useless result
# =====================================================================

def scenario_c_reject():
    """Tool result is an error message. The agent rejects it."""
    print("=" * 60)
    print("SCENARIO C -- Agent Rejects Useless Result")
    print("=" * 60)
    print()
    print("  The tool result is a stack trace / error message.")
    print("  The agent inspects it, decides it is useless, and rejects.")
    print()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system(
            "You are a coding assistant. When a tool fails, do not waste "
            "context tokens on error stack traces. Reject the result and "
            "explain the failure concisely."
        )
        t.user("Run the test suite for the auth module.")

        t.assistant("Running the tests now.", metadata={
            "tool_calls": [{
                "id": "tc3",
                "name": "run_tests",
                "arguments": {"module": "auth"},
            }],
        })

        # The tool returned an error
        error_output = (
            "Traceback (most recent call last):\n"
            "  File \"/app/tests/test_auth.py\", line 42, in test_login\n"
            "    response = client.post('/login', json=credentials)\n"
            "  File \"/app/lib/http.py\", line 118, in post\n"
            "    raise ConnectionError(f'Failed to connect to {url}')\n"
            "ConnectionError: Failed to connect to http://localhost:8080/login\n"
            "\n"
            "During handling of the above exception, another exception occurred:\n"
            "\n"
            "  File \"/app/tests/test_auth.py\", line 45, in test_login\n"
            "    self.fail(f'Login endpoint unreachable: {e}')\n"
            "AssertionError: Login endpoint unreachable: "
            "Failed to connect to http://localhost:8080/login\n"
            "\n"
            "----------------------------------------------------------------------\n"
            "Ran 1 test in 0.003s\n"
            "\n"
            "FAILED (failures=1)"
        )

        pending: PendingToolResult = t.tool_result(
            "tc3", "run_tests", error_output, is_error=True, review=True,
        )

        print("  PendingToolResult (error):")
        pending.pprint()

        # Agent inspects
        state = pending.to_dict()
        tools = pending.to_tools()

        decision_messages = [
            {"role": "system", "content": (
                "You are a context management agent. A tool result is pending "
                "review. If the result is an error (is_error=true) and the "
                "content is just a stack trace with no useful data, reject it "
                "with a concise reason. Only approve error results if they "
                "contain actionable information worth keeping in context."
            )},
            {"role": "user", "content": (
                f"Review this tool result and decide.\n\n"
                f"State:\n{json.dumps(state, indent=2)}"
            )},
        ]

        raw_resp = ask_llm(decision_messages, tools)
        decision = extract_tool_call(raw_resp)

        if decision:
            action = decision["action"]
            args = decision["args"]
            print(f"\n  Agent calls: {action}({json.dumps(args)[:100]})")
            pending.apply_decision(decision)
        else:
            # Fallback: if LLM did not use tools, reject manually
            print("\n  Agent did not use tools, applying fallback reject.")
            pending.reject("Error stack trace -- no useful data for context")

        print(f"\n  Final pending status: {pending.status}")

        print(f"\n  Context (rejected result NOT included):")
        ctx = t.compile()
        ctx.pprint(style="chat")

        print(f"\n  The error stack trace ({len(error_output)} chars) was kept")
        print(f"  out of the context window, saving tokens.")
        print()


# =====================================================================
# Main
# =====================================================================

def main():
    if not llm.api_key:
        print("ERROR: No API key configured.")
        print("Set GROQ_API_KEY in your environment or .env file.")
        return

    scenario_a_edit()
    scenario_b_summarize()
    scenario_c_reject()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    print("  Scenario A (edit_result): Agent detected 'API_KEY=sk-12345',")
    print("    redacted it via edit_result(), original preserved, then approved.")
    print()
    print("  Scenario B (summarize): Agent detected 500+ line listing,")
    print("    called summarize(target_tokens=100), original preserved,")
    print("    then approved the condensed version.")
    print()
    print("  Scenario C (reject): Agent detected an error stack trace,")
    print("    rejected it to keep the context window clean.")
    print()
    print("  Key patterns:")
    print("    - to_dict() gives the agent full visibility into pending state")
    print("    - to_tools() generates tool schemas for the LLM to call")
    print("    - apply_decision() safely dispatches the LLM's choice")
    print("    - edit_result() preserves original_content for provenance")
    print("    - summarize() calls LLM internally, also preserves original")
    print("    - reject() prevents useless content from entering context")


if __name__ == "__main__":
    main()


# --- See also ---
# cookbook/hooks/02_pending/08_tool_result_basics.py  -- Hook basics, review=True
# cookbook/hooks/02_pending/09_tool_result_edit.py    -- edit_result + summarize
# cookbook/hooks/03_agent_interface/04_dispatch.py    -- apply_decision full pipeline
# cookbook/hooks/02_pending/10_tool_result_config.py  -- Tool result configuration
