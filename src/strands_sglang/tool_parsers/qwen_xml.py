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

"""Qwen XML tool call parser."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from typing_extensions import override

from .base import ToolParser, ToolParseResult, register_tool_parser

logger = logging.getLogger(__name__)


@register_tool_parser("qwen_xml")
class QwenXMLToolParser(ToolParser):
    r"""Parser for Qwen3.5/Qwen3-Coder attribute-style XML tool call format.

    Format:

        <tool_call>
        <function=function_name>
        <parameter=param1>value1</parameter>
        <parameter=param2>value2</parameter>
        </function>
        </tool_call>

    Notes:
        - Function and parameter names are embedded in tag attributes.
        - Only content after a complete ``</think>`` is parsed, so draft tool
          calls drawn up during reasoning are never surfaced (see ``parse``).
    """

    _FUNCTION_PATTERN = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
    _PARAMETER_PATTERN = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)

    @override
    def parse(self, text: str) -> list[ToolParseResult]:
        """Parse tool calls from model output.

        Only content emitted *after* a complete ``</think>`` is parsed. Qwen
        thinking models seed the opening ``<think>`` in the prompt, so model
        output is ``reasoning… </think> answer`` — the opening tag is usually
        absent from ``text`` and a complete block can't be matched by removal.
        Keying off the closing tag: take only what follows the first
        ``</think>``; if there is no close, the whole output is still
        reasoning, so parse nothing (avoids surfacing draft tool calls the
        model drew up mid-thought but never committed to).
        """
        end = self.think_end_token
        if end in text:
            text = text.split(end, 1)[1]  # content = post-</think> only
        else:
            text = ""  # no close => all reasoning => no tool calls

        tool_calls: list[ToolParseResult] = []

        for i, match in enumerate(self.tool_pattern.finditer(text)):
            raw_content = match.group(1).strip()
            tool_call_id = f"call_{i:04d}"  # Sequential IDs for sortability

            # Parse the function tag
            func_match = self._FUNCTION_PATTERN.search(raw_content)
            if not func_match:
                logger.warning("Tool parse error: missing <function=...> tag")
                tool_calls.append(ToolParseResult.from_parse_error(id=tool_call_id, raw=raw_content))
                continue

            func_name = func_match.group(1).strip()
            func_body = func_match.group(2)

            if not func_name:
                logger.warning("Tool parse error: empty function name")
                tool_calls.append(ToolParseResult.from_parse_error(id=tool_call_id, raw=raw_content))
                continue

            # Parse all parameters from the function body
            arguments: dict[str, Any] = {}
            for param_match in self._PARAMETER_PATTERN.finditer(func_body):
                param_name = param_match.group(1).strip()
                param_value = param_match.group(2).strip()
                if param_name:
                    try:
                        arguments[param_name] = json.loads(param_value)
                    except (json.JSONDecodeError, ValueError):
                        arguments[param_name] = param_value

            tool_calls.append(
                ToolParseResult(
                    id=tool_call_id,
                    name=func_name,
                    input=arguments,
                )
            )

        return tool_calls
