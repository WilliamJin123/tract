"""Compile cache manager for Tract.

Owns the LRU snapshot cache and all incremental patching logic.
Extracted from tract.py to keep the facade class focused on orchestration.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

from tract.engine.compiler import DefaultContextCompiler
from tract.models.annotations import Priority
from tract.models.config import LLMConfig
from tract.protocols import CompiledContext, CompileSnapshot, Message, TokenCounter

if TYPE_CHECKING:
    from tract.models.commit import CommitInfo
    from tract.protocols import ContextCompiler
    from tract.storage.repositories import CommitRepository
    from tract.storage.schema import CommitRow

logger = logging.getLogger(__name__)

# OpenAI cookbook: 3 tokens appended after all messages as the response primer.
RESPONSE_PRIMER_TOKENS = 3


class CacheManager:
    """LRU compile-snapshot cache with incremental patching.

    Manages an OrderedDict-based LRU cache of CompileSnapshot objects.
    Supports O(1) incremental extension for APPEND commits, in-memory
    patching for EDIT commits, and annotation-aware invalidation.

    Token counts use per-message tracking: each message's token count
    (including per-message overhead) is stored in the snapshot.  The
    conversation total equals ``sum(message_token_counts) + RESPONSE_PRIMER_TOKENS``.
    After ``record_usage()`` calibrates the total with API-reported counts,
    subsequent incremental operations preserve that API base and only add
    tiktoken-computed deltas for new/changed messages.
    """

    def __init__(
        self,
        *,
        maxsize: int,
        compiler: ContextCompiler,
        token_counter: TokenCounter,
        commit_repo: CommitRepository,
    ) -> None:
        self._cache: OrderedDict[str, CompileSnapshot] = OrderedDict()
        self._maxsize = maxsize
        self._compiler = compiler
        self._token_counter = token_counter
        self._commit_repo = commit_repo

    # ------------------------------------------------------------------
    # LRU primitives
    # ------------------------------------------------------------------

    def get(self, head_hash: str) -> CompileSnapshot | None:
        """Get snapshot from LRU cache.  Returns None on miss."""
        if head_hash not in self._cache:
            logger.debug("Cache miss: %s", head_hash[:12])
            return None
        self._cache.move_to_end(head_hash)
        logger.debug("Cache hit: %s", head_hash[:12])
        return self._cache[head_hash]

    def put(self, head_hash: str, snapshot: CompileSnapshot) -> None:
        """Store snapshot in LRU cache, evicting LRU entry if at capacity."""
        if head_hash in self._cache:
            self._cache.move_to_end(head_hash)
        self._cache[head_hash] = snapshot
        while len(self._cache) > self._maxsize:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug("Cache evict: %s", evicted_key[:12])
        logger.debug("Cache put: %s (size=%d)", head_hash[:12], len(self._cache))

    def clear(self) -> None:
        """Clear all cached snapshots."""
        size = len(self._cache)
        self._cache.clear()
        if size > 0:
            logger.debug("Cache cleared (%d entries)", size)

    # ------------------------------------------------------------------
    # Snapshot <-> CompiledContext conversion
    # ------------------------------------------------------------------

    @staticmethod
    def to_compiled(snapshot: CompileSnapshot) -> CompiledContext:
        """Convert a CompileSnapshot to a CompiledContext for return.

        Uses copy-on-output for generation_configs to prevent user mutations
        of the returned CompiledContext from corrupting the cached snapshot.
        """
        return CompiledContext(
            messages=list(snapshot.messages),
            token_count=snapshot.token_count,
            commit_count=snapshot.commit_count,
            token_source=snapshot.token_source,
            generation_configs=[LLMConfig.from_dict(c) if c else None for c in snapshot.generation_configs],
            commit_hashes=list(snapshot.commit_hashes),
        )

    def build_snapshot(
        self, head_hash: str, result: CompiledContext
    ) -> CompileSnapshot | None:
        """Build a CompileSnapshot from a full compile result.

        Computes per-message token counts for O(1) incremental updates.
        Returns None if the compiler is not a DefaultContextCompiler
        (custom compilers bypass incremental cache).
        """
        if not isinstance(self._compiler, DefaultContextCompiler):
            return None
        messages = tuple(result.messages)
        per_msg = self._compute_per_message_counts(messages)
        token_count = sum(per_msg) + RESPONSE_PRIMER_TOKENS if per_msg else 0
        return CompileSnapshot(
            head_hash=head_hash,
            messages=messages,
            commit_count=result.commit_count,
            token_count=token_count,
            token_source=result.token_source,
            generation_configs=tuple(c.to_dict() if c is not None else {} for c in result.generation_configs),
            commit_hashes=tuple(result.commit_hashes),
            message_token_counts=per_msg,
        )

    # ------------------------------------------------------------------
    # Token counting helpers
    # ------------------------------------------------------------------

    def _token_source(self) -> str:
        """Return the token_source string for tiktoken-based counts."""
        from tract.engine.tokens import TiktokenCounter

        if isinstance(self._token_counter, TiktokenCounter):
            return f"tiktoken:{self._token_counter._encoding_name}"
        return ""

    @staticmethod
    def _message_to_dict(m: Message) -> dict:
        """Convert a Message to the dict format expected by TokenCounter."""
        if m.name is None:
            return {"role": m.role, "content": m.content}
        return {"role": m.role, "content": m.content, "name": m.name}

    def _count_single_message_tokens(self, message: Message) -> int:
        """Count tokens for a single message including per-message overhead.

        Returns the per-message token count *without* the response primer.
        The conversation total is ``sum(per_message) + RESPONSE_PRIMER_TOKENS``.
        """
        msg_dict = self._message_to_dict(message)
        return self._token_counter.count_messages([msg_dict]) - RESPONSE_PRIMER_TOKENS

    def _compute_per_message_counts(
        self, messages: tuple[Message, ...] | list[Message]
    ) -> tuple[int, ...]:
        """Compute per-message token counts for all messages."""
        return tuple(self._count_single_message_tokens(m) for m in messages)

    @property
    def uses_default_compiler(self) -> bool:
        """Whether the compiler supports incremental caching."""
        return isinstance(self._compiler, DefaultContextCompiler)

    # ------------------------------------------------------------------
    # Incremental patching
    # ------------------------------------------------------------------

    def extend_for_append(
        self, commit_info: CommitInfo, parent_snapshot: CompileSnapshot
    ) -> None:
        """Incrementally extend a cached snapshot for an APPEND commit.

        Builds the message for the new commit, appends it (no aggregation),
        and computes the token delta in O(1) using per-message counts.
        The parent snapshot stays in the LRU cache under its own HEAD
        (useful for future checkout back).
        """
        commit_row = self._commit_repo.get(commit_info.commit_hash)
        if commit_row is None:
            return

        assert isinstance(self._compiler, DefaultContextCompiler)
        new_message = self._compiler.build_message_for_commit(commit_row)
        new_config = dict(commit_row.generation_config_json or {})

        new_messages = parent_snapshot.messages + (new_message,)
        new_commit_hashes = parent_snapshot.commit_hashes + (commit_info.commit_hash,)

        # O(1) token delta: count only the new message
        new_msg_tokens = self._count_single_message_tokens(new_message)
        if parent_snapshot.message_token_counts:
            # Delta-based: preserves API-calibrated base from record_usage()
            new_token_count = parent_snapshot.token_count + new_msg_tokens
            new_msg_counts = parent_snapshot.message_token_counts + (new_msg_tokens,)
        else:
            # Fallback: no per-message counts (legacy snapshot), full recount
            new_msg_counts = self._compute_per_message_counts(new_messages)
            new_token_count = sum(new_msg_counts) + RESPONSE_PRIMER_TOKENS if new_msg_counts else 0

        self.put(
            commit_info.commit_hash,
            CompileSnapshot(
                head_hash=commit_info.commit_hash,
                messages=new_messages,
                commit_count=parent_snapshot.commit_count + 1,
                token_count=new_token_count,
                token_source=parent_snapshot.token_source,
                generation_configs=parent_snapshot.generation_configs + (new_config,),
                commit_hashes=new_commit_hashes,
                message_token_counts=new_msg_counts,
            ),
        )

    def patch_for_edit(
        self,
        parent_snapshot: CompileSnapshot,
        new_head_hash: str,
        edit_row: CommitRow,
    ) -> CompileSnapshot | None:
        """Patch a cached snapshot for an EDIT commit in-memory.

        Finds the message corresponding to the edited target (via response_to),
        replaces it with the new message, and computes the token delta in O(1)
        using per-message counts.

        Returns None if patching is not possible (missing commit_hashes, target
        not found), signaling caller to fall back to full recompile on next
        compile().
        """
        if not parent_snapshot.commit_hashes:
            return None

        target_hash = edit_row.response_to
        if target_hash is None:
            return None

        # Find position of the target commit in the snapshot
        try:
            target_idx = list(parent_snapshot.commit_hashes).index(target_hash)
        except ValueError:
            return None  # Target not in snapshot

        assert isinstance(self._compiler, DefaultContextCompiler)
        new_message = self._compiler.build_message_for_commit(edit_row)

        # Replace message at target position
        new_messages = list(parent_snapshot.messages)
        new_messages[target_idx] = new_message

        # Handle generation_config: edit-inherits-original rule
        new_configs = list(parent_snapshot.generation_configs)
        if edit_row.generation_config_json is not None:
            new_configs[target_idx] = dict(edit_row.generation_config_json)  # copy-on-input
        # else: keep original config at target_idx (edit-inherits-original)

        # O(1) token delta
        new_msg_tokens = self._count_single_message_tokens(new_message)
        if parent_snapshot.message_token_counts and len(parent_snapshot.message_token_counts) > target_idx:
            old_msg_tokens = parent_snapshot.message_token_counts[target_idx]
            new_token_count = parent_snapshot.token_count - old_msg_tokens + new_msg_tokens
            new_msg_counts = list(parent_snapshot.message_token_counts)
            new_msg_counts[target_idx] = new_msg_tokens
            new_msg_counts_tuple = tuple(new_msg_counts)
        else:
            # Fallback: full recount
            new_msg_counts_tuple = self._compute_per_message_counts(new_messages)
            new_token_count = sum(new_msg_counts_tuple) + RESPONSE_PRIMER_TOKENS if new_msg_counts_tuple else 0

        return CompileSnapshot(
            head_hash=new_head_hash,
            messages=tuple(new_messages),
            commit_count=parent_snapshot.commit_count,  # Same count (EDIT replaces, doesn't add)
            token_count=new_token_count,
            token_source=parent_snapshot.token_source,
            generation_configs=tuple(new_configs),
            commit_hashes=parent_snapshot.commit_hashes,  # Same positions
            message_token_counts=new_msg_counts_tuple,
        )

    def patch_for_annotate(
        self,
        snapshot: CompileSnapshot,
        target_hash: str,
        new_priority: Priority,
    ) -> CompileSnapshot | None:
        """Patch a cached snapshot for an annotation change.

        SKIP: remove the target's message from the snapshot.
        NORMAL/PINNED on already-included commit: no change needed.
        NORMAL/PINNED on previously-SKIP commit: return None (full recompile).
        """
        if not snapshot.commit_hashes:
            return None

        # Find target position
        target_idx = None
        for i, ch in enumerate(snapshot.commit_hashes):
            if ch == target_hash:
                target_idx = i
                break

        if new_priority == Priority.SKIP:
            if target_idx is None:
                return snapshot  # Already not in snapshot

            # Remove message, config, hash, and token count at target position
            new_messages = list(snapshot.messages)
            new_configs = list(snapshot.generation_configs)
            new_hashes = list(snapshot.commit_hashes)
            new_msg_counts = list(snapshot.message_token_counts) if snapshot.message_token_counts else []
            del new_messages[target_idx]
            del new_configs[target_idx]
            del new_hashes[target_idx]

            # O(1) token delta
            if new_msg_counts and len(new_msg_counts) > target_idx:
                removed_tokens = new_msg_counts[target_idx]
                del new_msg_counts[target_idx]
                new_token_count = snapshot.token_count - removed_tokens
            else:
                new_msg_counts = list(self._compute_per_message_counts(new_messages))
                new_token_count = sum(new_msg_counts) + RESPONSE_PRIMER_TOKENS if new_msg_counts else 0

            return CompileSnapshot(
                head_hash=snapshot.head_hash,
                messages=tuple(new_messages),
                commit_count=snapshot.commit_count - 1,
                token_count=new_token_count,
                token_source=snapshot.token_source,
                generation_configs=tuple(new_configs),
                commit_hashes=tuple(new_hashes),
                message_token_counts=tuple(new_msg_counts),
            )
        else:
            # NORMAL or PINNED
            if target_idx is not None:
                return snapshot  # Already included, no change
            else:
                return None  # Was skipped, need full recompile (don't have message content)
