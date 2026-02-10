"""Token counting implementations for Trace.

Provides TiktokenCounter (production use) and NullTokenCounter (testing).
Both implement the TokenCounter protocol from protocols.py.
"""

from __future__ import annotations


class TiktokenCounter:
    """Token counter using tiktoken (OpenAI's tokenizer).

    Lazily imports tiktoken and caches the Encoding instance.
    Falls back to o200k_base encoding if model is unknown.

    Implements the TokenCounter protocol.
    """

    def __init__(self, model: str = "gpt-4o", encoding_name: str | None = None) -> None:
        import tiktoken

        if encoding_name is not None:
            self._enc = tiktoken.get_encoding(encoding_name)
        else:
            try:
                self._enc = tiktoken.encoding_for_model(model)
            except KeyError:
                self._enc = tiktoken.get_encoding("o200k_base")

        self._encoding_name = self._enc.name

    @property
    def encoding_name(self) -> str:
        """Name of the tiktoken encoding being used."""
        return self._encoding_name

    def count_text(self, text: str) -> int:
        """Count tokens in a plain text string.

        Args:
            text: The text to tokenize.

        Returns:
            Number of tokens. Returns 0 for empty string.
        """
        if not text:
            return 0
        return len(self._enc.encode(text))

    def count_messages(self, messages: list[dict]) -> int:
        """Count tokens in a structured message list including overhead.

        Uses the OpenAI cookbook formula:
        - 3 tokens per message (role/content/separator overhead)
        - 1 token per name field (if present)
        - 3 tokens for the response primer (after all messages)

        Args:
            messages: List of message dicts with "role", "content", and
                optional "name" keys.

        Returns:
            Total token count including overhead.
        """
        if not messages:
            return 0

        total = 0
        for message in messages:
            total += 3  # per-message overhead
            for key, value in message.items():
                if isinstance(value, str):
                    total += len(self._enc.encode(value))
                if key == "name":
                    total += 1  # name field costs an extra token
        total += 3  # response primer
        return total


class NullTokenCounter:
    """Token counter that always returns 0.

    Useful for testing when token counts are irrelevant.

    Implements the TokenCounter protocol.
    """

    def count_text(self, text: str) -> int:
        """Always returns 0."""
        return 0

    def count_messages(self, messages: list[dict]) -> int:
        """Always returns 0."""
        return 0
