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

"""Custom exceptions for SGLangClient."""


class SGLangClientError(Exception):
    """Base exception for all SGLangClient errors."""


class SGLangHTTPError(SGLangClientError):
    """HTTP error from SGLang server."""

    def __init__(self, message: str, *, status: int, body: str = ""):
        """Initialize an `SGLangHTTPError` instance."""
        super().__init__(message)
        self.status = status
        self.body = body


class SGLangContextLengthError(SGLangHTTPError):
    """Prompt/context exceeds the model's maximum length (400 + length keywords)."""


class SGLangThrottledError(SGLangHTTPError):
    """Rate-limited or temporarily unavailable (429, 503)."""


class SGLangConnectionError(SGLangClientError):
    """Connection-level failure (connect, timeout, DNS)."""


class SGLangDecodingError(SGLangClientError):
    """Server returned non-JSON response body."""


class GenerationAbortedException(SGLangClientError):
    """SGLang aborted an in-flight generation mid-stream (`finish_reason="abort"`).

    Returned by `/generate` as an HTTP-200 *partial* (not an HTTP error) when the
    server aborts all running requests — most commonly an `abort_request(abort_all=True)`
    issued by a weight-sync `pause_generation(mode="abort")` during fully-async RL
    rollout. The partial is incomplete and off-policy, so `SGLangModel.stream` raises
    this instead of yielding it as a clean `end_turn`. `strands_env`'s
    `TerminationReason.from_error` maps it to `GENERATION_ABORTED`, which the slime
    bridge marks `ABORTED` (neutralized from training + re-queued for retry) rather
    than mislabeling it a reward-0 `task_complete` / "no submission". See
    `notes/training_logs/ioi/2026-06-23_qwen3_6_curriculum_stage1_format.md`.
    """
