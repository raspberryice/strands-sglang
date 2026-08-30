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

"""Unit tests for token module."""

import pytest

from strands_sglang import Token, TokenManager


class TestToken:
    """Tests for Token dataclass."""

    def test_token_defaults(self):
        """Token has correct default values."""
        token = Token(token_id=42)
        assert token.token_id == 42
        assert token.logprob is None
        assert token.loss_mask is True


class TestTokenManagerBasic:
    """Basic TokenManager tests."""

    def test_init(self):
        """TokenManager starts empty."""
        manager = TokenManager()
        assert len(manager) == 0

    def test_empty_manager(self):
        """Empty manager returns empty lists."""
        manager = TokenManager()
        assert manager.tokens == []
        assert manager.token_ids == []
        assert manager.loss_mask == []
        assert manager.logprobs == []
        assert manager.segments == []
        assert manager.segment_info == []


class TestTokenManagerAddPrompt:
    """Tests for add_prompt method."""

    def test_add_prompt_basic(self):
        """add_prompt adds tokens with loss_mask=False."""
        manager = TokenManager()
        manager.add_prompt([1, 2, 3])

        assert manager.token_ids == [1, 2, 3]
        assert manager.loss_mask == [False, False, False]
        assert manager.logprobs == [None, None, None]

    def test_add_prompt_with_logprobs(self):
        """add_prompt accepts logprobs."""
        manager = TokenManager()
        manager.add_prompt([10, 20], logprobs=[-0.1, -0.2])

        assert manager.token_ids == [10, 20]
        assert manager.logprobs == [-0.1, -0.2]

    def test_add_prompt_empty(self):
        """add_prompt with empty list does nothing."""
        manager = TokenManager()
        manager.add_prompt([])
        assert len(manager) == 0
        assert manager.segments == []

    def test_add_prompt_mismatched_logprobs_raises(self):
        """add_prompt raises ValueError if logprobs length doesn't match token_ids."""
        manager = TokenManager()
        with pytest.raises(ValueError, match="logprobs length"):
            manager.add_prompt([1, 2, 3], logprobs=[-0.1])


class TestTokenManagerAddResponse:
    """Tests for add_response method."""

    def test_add_response_basic(self):
        """add_response adds tokens with loss_mask=True."""
        manager = TokenManager()
        manager.add_prompt([1])
        manager.add_response([4, 5, 6])

        assert manager.token_ids == [1, 4, 5, 6]
        assert manager.loss_mask == [False, True, True, True]

    def test_add_response_with_logprobs(self):
        """add_response accepts logprobs."""
        manager = TokenManager()
        manager.add_prompt([1])
        manager.add_response([100, 200], logprobs=[-0.5, -0.6])

        assert manager.logprobs[1:] == [-0.5, -0.6]

    def test_add_response_empty(self):
        """add_response with empty list does nothing."""
        manager = TokenManager()
        manager.add_response([])
        assert len(manager) == 0

    def test_add_response_without_prompt_raises(self):
        """add_response raises RuntimeError if no prompt segment exists."""
        manager = TokenManager()
        with pytest.raises(RuntimeError, match="First segment must be a prompt"):
            manager.add_response([4, 5, 6])

    def test_add_response_mismatched_logprobs_raises(self):
        """add_response raises ValueError if logprobs length doesn't match token_ids."""
        manager = TokenManager()
        manager.add_prompt([0])
        with pytest.raises(ValueError, match="logprobs length"):
            manager.add_response([1, 2, 3], logprobs=[-0.1, -0.2])


class TestTokenManagerMultipleSegments:
    """Tests for multiple segment operations."""

    def test_prompt_response_sequence(self):
        """Typical prompt-response sequence works correctly."""
        manager = TokenManager()
        manager.add_prompt([1, 2, 3])
        manager.add_response([4, 5], logprobs=[-0.1, -0.2])

        assert manager.token_ids == [1, 2, 3, 4, 5]
        assert manager.loss_mask == [False, False, False, True, True]
        assert manager.logprobs == [None, None, None, -0.1, -0.2]

    def test_multi_turn_conversation(self):
        """Multi-turn conversation with tool calls."""
        manager = TokenManager()

        # Initial prompt
        manager.add_prompt([1, 2])
        # Model response (tool call)
        manager.add_response([3, 4], logprobs=[-0.1, -0.2])
        # Tool result (treated as prompt)
        manager.add_prompt([5, 6])
        # Final model response
        manager.add_response([7, 8], logprobs=[-0.3, -0.4])

        assert manager.token_ids == [1, 2, 3, 4, 5, 6, 7, 8]
        assert manager.loss_mask == [False, False, True, True, False, False, True, True]
        assert manager.logprobs == [None, None, -0.1, -0.2, None, None, -0.3, -0.4]
        assert len(manager) == 8

    def test_segments_property(self):
        """segments property returns segment data."""
        manager = TokenManager()
        manager.add_prompt([1, 2])
        manager.add_response([3, 4])

        segments = manager.segments
        assert len(segments) == 2
        assert all(not t.loss_mask for t in segments[0])
        assert all(t.loss_mask for t in segments[1])

    def test_initial_prompt(self):
        """initial_prompt returns the first segment."""
        manager = TokenManager()
        manager.add_prompt([1, 2, 3])
        manager.add_response([4, 5])

        prompt = manager.initial_prompt
        assert [t.token_id for t in prompt] == [1, 2, 3]
        assert all(not t.loss_mask for t in prompt)

    def test_initial_prompt_empty(self):
        """initial_prompt returns empty list when no segments exist."""
        manager = TokenManager()
        assert manager.initial_prompt == []

    def test_segment_info(self):
        """segment_info returns correct metadata."""
        manager = TokenManager()
        manager.add_prompt([1, 2, 3])
        manager.add_response([4, 5])
        manager.add_prompt([6])

        assert manager.segment_info == [(False, 3), (True, 2), (False, 1)]


class TestTokenManagerReset:
    """Tests for reset functionality."""

    def test_reset_clears_all(self):
        """reset clears all accumulated tokens."""
        manager = TokenManager()
        manager.add_prompt([1, 2, 3])
        manager.add_response([4, 5])

        manager.reset()

        assert len(manager) == 0
        assert manager.tokens == []
        assert manager.segments == []

    def test_reset_allows_reuse(self):
        """Manager can be reused after reset."""
        manager = TokenManager()
        manager.add_prompt([1, 2])
        manager.add_response([3, 4])

        manager.reset()

        manager.add_prompt([10, 20])
        manager.add_response([30])

        assert manager.token_ids == [10, 20, 30]
        assert len(manager) == 3


class TestTokenManagerSeedPrefix:
    """seed_prefix() reconstructs alternating PROMPT/RESPONSE segments from a stored stream."""

    def test_seed_prefix_basic(self):
        """A 0/1 loss-mask splits into one PROMPT then one RESPONSE segment."""
        manager = TokenManager()
        manager.seed_prefix([1, 2, 3, 4, 5], [0, 0, 0, 1, 1])
        assert manager.token_ids == [1, 2, 3, 4, 5]
        assert manager.loss_mask == [0, 0, 0, 1, 1]
        assert manager.segment_info == [(False, 3), (True, 2)]

    def test_seed_prefix_logprobs(self):
        """Logprobs ride along, attached per token (None allowed)."""
        manager = TokenManager()
        manager.seed_prefix([1, 2, 3], [0, 1, 1], [None, -0.5, -0.2])
        assert manager.logprobs == [None, -0.5, -0.2]
        assert manager.loss_mask == [0, 1, 1]

    def test_seed_prefix_multi_turn_alternation(self):
        """Multiple turns -> a fresh segment at every loss-mask run boundary."""
        manager = TokenManager()
        manager.seed_prefix([1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 0, 1])
        assert manager.segment_info == [(False, 2), (True, 2), (False, 1), (True, 1)]
        assert manager.token_ids == [1, 2, 3, 4, 5, 6]

    def test_seed_prefix_then_continue(self):
        """After seeding, add_* appends the continuation as the next segment(s)."""
        manager = TokenManager()
        manager.seed_prefix([1, 2, 3], [0, 0, 1])
        manager.add_prompt([4, 5])  # e.g. the continue-prompt turn
        manager.add_response([6, 7], [-0.1, -0.2])
        assert manager.token_ids == [1, 2, 3, 4, 5, 6, 7]
        assert manager.loss_mask == [0, 0, 1, 0, 0, 1, 1]
        assert manager.logprobs == [None, None, None, None, None, -0.1, -0.2]

    def test_seed_prefix_requires_fresh_manager(self):
        """seed_prefix() on a non-fresh manager raises (seed-once)."""
        manager = TokenManager()
        manager.add_prompt([1, 2])
        with pytest.raises(RuntimeError, match="fresh TokenManager"):
            manager.seed_prefix([3, 4], [0, 1])

    def test_seed_prefix_length_mismatch(self):
        """Mismatched loss_mask / logprobs lengths raise ValueError."""
        with pytest.raises(ValueError, match="loss_mask length"):
            TokenManager().seed_prefix([1, 2, 3], [0, 1])
        with pytest.raises(ValueError, match="logprobs length"):
            TokenManager().seed_prefix([1, 2], [0, 1], [-0.1])


class TestTokenManagerReseedInplace:
    """reseed_inplace() closes the active trajectory-segment and starts a fresh one (Phase-3
    in-loop compaction, design/segment_tree_trajectories.md)."""

    def test_default_single_trajectory_segment(self):
        """No reseed -> one trajectory-segment spanning the whole stream (unchanged behavior)."""
        m = TokenManager()
        m.add_prompt([1, 2, 3])
        m.add_response([4, 5], [-0.1, -0.2])
        assert m.n_trajectory_segments == 1
        assert m.active_base_offset == 0
        assert m.trajectory_segment_bounds == [(0, 5)]

    def test_reseed_records_boundary_and_active_base(self):
        """Reseed archives turns in place, moves active_base to the seed, adds a boundary."""
        m = TokenManager()
        m.add_prompt([1, 2, 3])            # seg0 turns
        m.add_response([4, 5], [-0.1, -0.2])
        m.reseed_inplace([9, 8, 7], [0, 0, 1], [None, None, -0.3])  # [problem+summary] seed; last=trained
        # Archived turns remain; the flat stream keeps growing.
        assert m.token_ids == [1, 2, 3, 4, 5, 9, 8, 7]
        assert m.loss_mask == [0, 0, 0, 1, 1, 0, 0, 1]
        assert m.n_trajectory_segments == 2
        assert m.active_base_offset == 5                      # seed starts at token 5
        assert m.trajectory_segment_bounds == [(0, 5), (5, 8)]

    def test_reseed_then_continue_extends_active_segment(self):
        """Post-reseed add_* extends the ACTIVE trajectory-segment (base unchanged)."""
        m = TokenManager()
        m.add_prompt([1, 2])
        m.add_response([3], [-0.1])
        m.reseed_inplace([9, 8], [0, 1], [None, -0.2])   # summary masked (0), then a trained token
        m.add_prompt([7])                                 # e.g. a tool result in the new segment
        m.add_response([6, 5], [-0.3, -0.4])
        assert m.token_ids == [1, 2, 3, 9, 8, 7, 6, 5]
        assert m.active_base_offset == 3
        assert m.trajectory_segment_bounds == [(0, 3), (3, 8)]
        # The active window the model would send to /generate is exactly the 2nd segment.
        base = m.active_base_offset
        assert m.token_ids[base:] == [9, 8, 7, 6, 5]

    def test_multiple_reseeds_tile_the_stream(self):
        """Each reseed adds a boundary; bounds tile token_ids exactly."""
        m = TokenManager()
        m.add_prompt([1]); m.add_response([2], [-0.1])
        m.reseed_inplace([3, 4], [0, 1])
        m.add_response([5], [-0.2])
        m.reseed_inplace([6], [0])
        m.add_response([7], [-0.3])
        assert m.n_trajectory_segments == 3
        assert m.trajectory_segment_bounds == [(0, 2), (2, 5), (5, 7)]
        # Concatenated per-segment slices == the full stream.
        flat = [t for a, b in m.trajectory_segment_bounds for t in m.token_ids[a:b]]
        assert flat == m.token_ids == [1, 2, 3, 4, 5, 6, 7]

    def test_reseed_requires_existing_trajectory(self):
        with pytest.raises(RuntimeError, match="requires an existing trajectory"):
            TokenManager().reseed_inplace([1, 2], [0, 1])

    def test_reseed_length_mismatch(self):
        m = TokenManager(); m.add_prompt([1, 2])
        with pytest.raises(ValueError, match="loss_mask length"):
            m.reseed_inplace([1, 2, 3], [0, 1])
        with pytest.raises(ValueError, match="logprobs length"):
            m.reseed_inplace([1, 2], [0, 1], [-0.1])

    def test_reset_clears_trajectory_segments(self):
        m = TokenManager()
        m.add_prompt([1]); m.add_response([2], [-0.1]); m.reseed_inplace([3], [0])
        assert m.n_trajectory_segments == 2
        m.reset()
        assert m.n_trajectory_segments == 1
        assert m.active_base_offset == 0
        assert m.trajectory_segment_bounds == [(0, 0)]
