# Copyright 2025-2026 Strands RL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Token management for token-in/token-out training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Token:
    """A single token with its ID, logprob, and loss mask."""

    token_id: int
    logprob: float | None = None
    loss_mask: bool = True


class TokenManager:
    """Manages token accumulation with segment-based prompt/response tracking.

    Notes:
        - Tokens are organized into `segments`, where each segment is either:
            - `PROMPT`: System messages, user input, tool results (loss_mask=False)
            - `RESPONSE`: Model outputs (loss_mask=True)
        - During an agent loop with the `SGLangModel` backend, segments are added in this order:
            - `segments[0]`: `PROMPT`   — initial prompt (system + tools + user message / conversation history)
            - `segments[1]`: `RESPONSE` — first model output (may include tool calls)
            - `segments[2]`: `PROMPT`   — tool results (if tool use occurred)
            - `segments[3]`: `RESPONSE` — next model output
            - ...                   — alternating `PROMPT`/`RESPONSE` until the loop ends
        - `segments[0]` always contains the full initial prompt from the first generation call. Everything after it is the rollout.

    Example:
        >>> manager = TokenManager()
        >>> manager.add_prompt([1, 2, 3])
        >>> manager.add_response([4, 5], [0.1, 0.2])
        >>> manager.token_ids      # [1, 2, 3, 4, 5]
        >>> manager.loss_mask      # [0, 0, 0, 1, 1]
        >>> manager.logprobs       # [None, None, None, 0.1, 0.2]
    """

    def __init__(self) -> None:
        """Create a TokenManager."""
        # Flat, index-aligned storage (as upstream strands-sglang's `Rollout`, #65): ints/floats are not
        # GC-tracked, so a whole trajectory is a handful of tracked lists instead of one tracked `Token`
        # per token — the RolloutManager's full collections took 85-250 s with ~1e8-1e9 Token objects live.
        # `Token` objects are only built on demand by the `tokens` / `segments` / `initial_prompt` views.
        self._ids: list[int] = []
        self._lps: list[float | None] = []
        self._mask = bytearray()
        self._seg_bounds: list[tuple[bool, int, int]] = []  # turn-segments: (is_output, start, end)
        # Trajectory-segment boundaries (token offsets), for in-loop compaction reseeds
        # (design/segment_tree_trajectories.md). A *trajectory-segment* is one context window
        # between reseeds — coarser than the PROMPT/RESPONSE turn-segments in `_seg_bounds`. Always
        # starts with [0]; `reseed_inplace` appends the token offset where each new segment begins.
        # With no reseed there is exactly one trajectory-segment ([0]) and behavior is unchanged.
        self._traj_seg_starts: list[int] = [0]

    def reset(self) -> None:
        """Reset token accumulation for a new episode."""
        self._ids, self._lps, self._mask, self._seg_bounds = [], [], bytearray(), []
        self._traj_seg_starts = [0]

    def _push(self, token_ids: list[int], logprobs: list[float | None] | None, is_output: bool) -> None:
        """Append one turn-segment (PROMPT if not `is_output`, else RESPONSE)."""
        start = len(self._ids)
        self._ids.extend(token_ids)
        self._lps.extend(logprobs if logprobs is not None else [None] * len(token_ids))
        self._mask.extend(b"\x01" * len(token_ids) if is_output else bytes(len(token_ids)))
        self._seg_bounds.append((is_output, start, len(self._ids)))

    def _append_runs(
        self,
        token_ids: list[int],
        loss_mask: list[int],
        logprobs: list[float | None] | None,
    ) -> None:
        """Append `token_ids` as alternating PROMPT/RESPONSE turn-segments split at `loss_mask`
        run boundaries (0 -> PROMPT/masked, 1 -> RESPONSE/trained). Shared by `seed_prefix` and
        `reseed_inplace`."""
        n = len(token_ids)
        i = 0
        while i < n:
            is_output = bool(loss_mask[i])
            j = i
            while j < n and bool(loss_mask[j]) == is_output:
                j += 1
            self._push(token_ids[i:j], logprobs[i:j] if logprobs is not None else None, is_output)
            i = j

    def add_prompt(self, token_ids: list[int], logprobs: list[float] | None = None) -> None:
        """Add a prompt segment (system messages, user input, tool results)."""
        if not token_ids:
            return
        if logprobs is not None and len(logprobs) != len(token_ids):
            raise ValueError(f"logprobs length ({len(logprobs)}) must match token_ids length ({len(token_ids)})")

        self._push(list(token_ids), list(logprobs) if logprobs is not None else None, False)

    def add_response(self, token_ids: list[int], logprobs: list[float] | None = None) -> None:
        """Add a response segment (model output)."""
        if not token_ids:
            return
        if not self._seg_bounds:
            raise RuntimeError("First segment must be a prompt. Call add_prompt() before add_response().")
        if logprobs is not None and len(logprobs) != len(token_ids):
            raise ValueError(f"logprobs length ({len(logprobs)}) must match token_ids length ({len(token_ids)})")

        self._push(list(token_ids), list(logprobs) if logprobs is not None else None, True)

    def seed_prefix(
        self,
        token_ids: list[int],
        loss_mask: list[int],
        logprobs: list[float | None] | None = None,
    ) -> None:
        """Seed with a pre-existing token stream as alternating `PROMPT`/`RESPONSE` segments.

        Splits `token_ids` into segments at `loss_mask` run boundaries (0 -> `PROMPT`,
        1 -> `RESPONSE`), reproducing the segment structure an in-order `add_prompt`/`add_response`
        sequence would build. Use to resume a trajectory from its exact token-in/token-out stream
        without re-rendering the conversation (e.g. an aborted-trajectory replay). Must be called
        on a fresh manager (before any `add_*`, or right after `reset()`).

        Args:
            token_ids: the accumulated token ids of the prefix.
            loss_mask: per-token mask (1 = model output / `RESPONSE`, 0 = prompt/tool / `PROMPT`).
            logprobs: per-token log-probs (the behavior policy's); a `None` entry is allowed.

        Raises:
            RuntimeError: if the manager is not fresh.
            ValueError: on a length mismatch.
        """
        if self._seg_bounds:
            raise RuntimeError("seed_prefix() requires a fresh TokenManager (call before any add_*/after reset()).")
        if len(loss_mask) != len(token_ids):
            raise ValueError(f"loss_mask length ({len(loss_mask)}) must match token_ids length ({len(token_ids)})")
        if logprobs is not None and len(logprobs) != len(token_ids):
            raise ValueError(f"logprobs length ({len(logprobs)}) must match token_ids length ({len(token_ids)})")

        self._append_runs(token_ids, loss_mask, logprobs)

    def reseed_inplace(
        self,
        token_ids: list[int],
        loss_mask: list[int],
        logprobs: list[float | None] | None = None,
    ) -> None:
        """Close the ACTIVE trajectory-segment and start a fresh one seeded with `token_ids`.

        The in-loop compaction reseed (design/segment_tree_trajectories.md): after a
        length-triggered summary turn, drop the history the model conditions on and continue in
        the SAME rollout session on a compact `[problem + summary]` context. Unlike `seed_prefix`
        (which needs a fresh manager), this runs MID-episode:

          * the already-accumulated turn-segments STAY in `_seg_bounds` — they are the archived
            trajectory-segment, emitted later as its own training `Sample` (the bridge slices
            per-segment via `trajectory_segment_bounds`);
          * a new boundary is recorded at the current token count, so `active_base_offset` moves
            to the start of the seed — the model then sends only `token_ids[active_base_offset:]`
            to `/generate`, dropping the old KV;
          * the seed is appended as PROMPT/RESPONSE runs. A compaction summary is passed with
            `loss_mask=0` (masked context — regenerated as prompt for the child, never trained
            in the child), matching the "trained in the emitting segment, masked in the seed"
            rule.

        Args:
            token_ids: the compacted seed prefix (e.g. problem + summary [+ code]).
            loss_mask: per-token mask (0 = masked/prompt, 1 = trained/response).
            logprobs: per-token log-probs; `None` (or `None` entries) allowed for masked tokens.

        Raises:
            RuntimeError: on an empty manager (nothing to compact).
            ValueError: on a length mismatch.
        """
        if not self._seg_bounds:
            raise RuntimeError("reseed_inplace() requires an existing trajectory (nothing to compact).")
        if len(loss_mask) != len(token_ids):
            raise ValueError(f"loss_mask length ({len(loss_mask)}) must match token_ids length ({len(token_ids)})")
        if logprobs is not None and len(logprobs) != len(token_ids):
            raise ValueError(f"logprobs length ({len(logprobs)}) must match token_ids length ({len(token_ids)})")

        self._traj_seg_starts.append(len(self))  # boundary at the current end (token units)
        self._append_runs(token_ids, loss_mask, logprobs)

    @property
    def active_base_offset(self) -> int:
        """Token index where the ACTIVE trajectory-segment begins (0 before any reseed).

        The `SGLangModel` sends only `token_ids[active_base_offset:]` to `/generate` after a
        compaction reseed, so the dropped history no longer costs KV / prefill.
        """
        return self._traj_seg_starts[-1]

    @property
    def n_trajectory_segments(self) -> int:
        """Number of trajectory-segments (1 + number of compaction reseeds)."""
        return len(self._traj_seg_starts)

    @property
    def trajectory_segment_bounds(self) -> list[tuple[int, int]]:
        """`[start, end)` token ranges of each trajectory-segment; tiles `token_ids` exactly.

        The bridge slices `token_ids` / `loss_mask` / `logprobs` by these ranges to emit one
        training `Sample` per trajectory-segment (each trained with its OWN compacted context).
        """
        starts = self._traj_seg_starts
        ends = starts[1:] + [len(self)]
        return list(zip(starts, ends))

    def _tokens(self, start: int, end: int) -> list[Token]:
        """`Token` view of `[start, end)` (built on demand; nothing per-token is stored)."""
        ids, lps, mask = self._ids, self._lps, self._mask
        return [Token(token_id=ids[k], logprob=lps[k], loss_mask=bool(mask[k])) for k in range(start, end)]

    @property
    def tokens(self) -> list[Token]:
        """Get all tokens as a flat list (a fresh view)."""
        return self._tokens(0, len(self._ids))

    @property
    def token_ids(self) -> list[int]:
        """Get all token IDs as a flat list (a copy)."""
        return list(self._ids)

    def token_ids_from(self, start: int) -> list[int]:
        """Token IDs from `start` to the end (e.g. the active window `active_base_offset:`)."""
        return self._ids[start:]

    @property
    def loss_mask(self) -> list[int]:
        """Get loss mask for all tokens (1 = model output, 0 = prompt/tool).

        Notes:
            Only compute loss on tokens where mask is 1 (model outputs).
        """
        return list(self._mask)

    @property
    def logprobs(self) -> list[float | None]:
        """Get log probabilities for all tokens."""
        return list(self._lps)

    @property
    def initial_prompt(self) -> list[Token]:
        """Get the initial prompt tokens (`segments[0]`).

        Notes:
            Contains the full input context from the first generation call:
            system prompt + tool definitions + user message (or conversation history).
        """
        if not self._seg_bounds:
            return []
        _, start, end = self._seg_bounds[0]
        return self._tokens(start, end)

    @property
    def initial_prompt_length(self) -> int:
        """Length of `initial_prompt`, without building it."""
        return self._seg_bounds[0][2] - self._seg_bounds[0][1] if self._seg_bounds else 0

    @property
    def segments(self) -> list[list[Token]]:
        """Get tokens organized by segment (a fresh view; appending to it does not change the manager)."""
        return [self._tokens(start, end) for _, start, end in self._seg_bounds]

    @property
    def segment_info(self) -> list[tuple[bool, int]]:
        """Get segment metadata as `(is_output, length)` tuples."""
        return [(is_output, end - start) for is_output, start, end in self._seg_bounds]

    def __len__(self) -> int:
        """Return total number of tokens."""
        return len(self._ids)

    def __repr__(self) -> str:
        """Return string representation."""
        n_segments = len(self._seg_bounds)
        n_tokens = len(self)
        n_output = self._mask.count(1)
        return f"TokenManager(segments={n_segments}, tokens={n_tokens}, output_tokens={n_output})"
