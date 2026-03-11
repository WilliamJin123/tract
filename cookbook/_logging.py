"""Shared step logger for cookbook examples.

Prints comprehensive real-time information about what happens during
a t.run() agent loop: assistant messages, tool calls, and tool results.

Usage:
    from _logging import StepLogger

    log = StepLogger()
    result = t.run("...", on_step=log.on_step, on_tool_result=log.on_tool_result)
"""

from __future__ import annotations

import json
from typing import Any


class StepLogger:
    """Real-time logger for agent loop steps.

    Prints assistant text and tool calls (on_step), plus tool results
    (on_tool_result) as they happen. Designed for cookbook readability.

    Args:
        indent: Prefix string for all output lines.
        max_text: Maximum characters of assistant text to display.
        max_result: Maximum characters of tool result to display.
        show_args: Whether to show tool call arguments.
    """

    def __init__(
        self,
        *,
        indent: str = "  ",
        max_text: int = 200,
        max_result: int = 150,
        show_args: bool = True,
    ):
        self.indent = indent
        self.max_text = max_text
        self.max_result = max_result
        self.show_args = show_args

    def on_step(self, step_num: int, response: Any) -> None:
        """on_step callback for t.run().

        Extracts and prints assistant text and tool calls from the
        raw LLM response object.
        """
        content = _extract_content(response)
        tool_calls = _extract_tool_calls(response)

        print(f"\n{self.indent}{'-' * 40}")
        print(f"{self.indent}Step {step_num}")
        print(f"{self.indent}{'-' * 40}")

        if content:
            text = content.replace("\n", "\n" + self.indent + "  ")
            if len(content) > self.max_text:
                text = content[:self.max_text].replace("\n", "\n" + self.indent + "  ") + "..."
            print(f"{self.indent}  Assistant: {text}")

        if tool_calls:
            for tc in tool_calls:
                if self.show_args:
                    args = _format_args(tc.get("arguments", {}))
                    print(f"{self.indent}  >> tool call: {tc['name']}({args})")
                else:
                    print(f"{self.indent}  >> tool call: {tc['name']}")
        elif not content:
            print(f"{self.indent}  (empty response)")

    def on_tool_result(self, tool_name: str, output: str, status: str) -> None:
        """on_tool_result callback for t.run().

        Prints tool results as they are returned.
        """
        icon = "  <<" if status == "success" else "  !! ERROR"
        text = output
        if len(text) > self.max_result:
            text = text[:self.max_result] + "..."
        text = text.replace("\n", "\n" + self.indent + "     ")
        print(f"{self.indent}{icon} {tool_name}: {text}")


def _extract_content(response: Any) -> str | None:
    """Extract text content from an LLM response (OpenAI format)."""
    # OpenAI object
    if hasattr(response, "choices"):
        try:
            return response.choices[0].message.content
        except (IndexError, AttributeError):
            return None
    # Dict
    if isinstance(response, dict):
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return response.get("content")
    if isinstance(response, str):
        return response
    return None


def _extract_tool_calls(response: Any) -> list[dict]:
    """Extract tool calls from an LLM response (OpenAI format)."""
    # OpenAI object
    if hasattr(response, "choices"):
        try:
            msg = response.choices[0].message
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                return [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": (
                            json.loads(tc.function.arguments)
                            if isinstance(tc.function.arguments, str)
                            else tc.function.arguments
                        ),
                    }
                    for tc in msg.tool_calls
                ]
        except (IndexError, AttributeError):
            pass
    # Dict
    if isinstance(response, dict):
        try:
            msg = response["choices"][0]["message"]
            tcs = msg.get("tool_calls", [])
            if tcs:
                result = []
                for tc in tcs:
                    args = tc.get("function", {}).get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    result.append({
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": args,
                    })
                return result
        except (KeyError, IndexError, TypeError):
            pass
    return []


def _format_args(args: dict, max_len: int = 80) -> str:
    """Format tool call arguments concisely."""
    if not args:
        return ""
    parts = []
    for k, v in list(args.items())[:4]:
        v_str = json.dumps(v) if not isinstance(v, str) else repr(v)
        if len(v_str) > 40:
            v_str = v_str[:37] + "..."
        parts.append(f"{k}={v_str}")
    result = ", ".join(parts)
    if len(args) > 4:
        result += ", ..."
    return result
