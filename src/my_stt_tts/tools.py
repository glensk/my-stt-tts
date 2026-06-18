"""In-conversation tool / function calling (R2-7): a registry + example tools.

The model can call a *tool* mid-conversation — the brain executes it and feeds the
result back so the model can finish its spoken answer. This is the inline upgrade
of the legacy "agent, <task>" trigger (which is still supported in
:mod:`my_stt_tts.brain`).

A :class:`Tool` bundles a name, a human description, a JSON-Schema parameter spec,
and a Python ``run`` callable. A :class:`ToolRegistry` dispatches by name and
serializes the whole set to **both** provider wire formats (Anthropic's ``tools``
shape and OpenAI's ``function`` shape) so the same registry works with either
backend. Tool execution is pure Python and tested directly; the provider
round-trip is tested with a mocked client in ``tests/test_tools.py``.

Shipped example tools (so the feature is demonstrably real):

* ``get_time``     — current local date/time (ISO-8601), optional timezone.
* ``calculator``   — evaluate a safe arithmetic expression.
* ``home_control`` — route a natural-language home command to the existing agent /
  Home Assistant dispatch (reuses :func:`my_stt_tts.agent.dispatch_to_agent`).
"""

from __future__ import annotations

import ast
import datetime as _dt
import logging
import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("my_stt_tts.tools")

# JSON-Schema "object" with no properties — a tool that takes no arguments.
_NO_ARGS: dict[str, Any] = {"type": "object", "properties": {}}


@dataclass
class Tool:
    """One callable tool: name, description, JSON-Schema params, and a runner.

    ``run`` receives the parsed argument dict and returns a string (what is fed
    back to the model). It must not raise — wrap fallible work; the registry also
    guards execution so a tool bug degrades to an error string, never a crash.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[dict[str, Any]], str]

    def anthropic_schema(self) -> dict[str, Any]:
        """This tool as an Anthropic ``tools`` entry (``input_schema``)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def openai_schema(self) -> dict[str, Any]:
        """This tool as an OpenAI ``tools`` entry (``function`` wrapper)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """A name→:class:`Tool` map with provider serialization + guarded dispatch."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Add a tool (overwrites a previous tool of the same name)."""
        self._tools[tool.name] = tool

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def anthropic_tools(self) -> list[dict[str, Any]]:
        """All tools in Anthropic wire format (pass as ``tools=...``)."""
        return [t.anthropic_schema() for t in self._tools.values()]

    def openai_tools(self) -> list[dict[str, Any]]:
        """All tools in OpenAI wire format (pass as ``tools=...``)."""
        return [t.openai_schema() for t in self._tools.values()]

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        """Run the named tool with ``arguments``; return its string result.

        Unknown tools and tool exceptions both yield a short error string (which is
        fed back to the model) rather than raising, so one misbehaving tool can't
        abort the conversation.
        """
        tool = self._tools.get(name)
        if tool is None:
            log.warning("model called unknown tool %r", name)
            return f"error: unknown tool {name!r}"
        try:
            return str(tool.run(arguments or {}))
        except Exception as exc:  # noqa: BLE001 - surface to the model, never crash
            log.warning("tool %r failed: %s", name, exc)
            return f"error: tool {name!r} failed: {exc}"


# --- example tools -------------------------------------------------------------


def _get_time(args: dict[str, Any]) -> str:
    """Current local date+time as ISO-8601 (Swiss/EU default), optional ``tz`` label."""
    now = _dt.datetime.now().astimezone()
    stamp = now.replace(microsecond=0).isoformat()
    tz = args.get("tz")
    return f"{stamp} ({tz})" if tz else stamp


# A tiny safe arithmetic evaluator: only numeric literals + the four operators,
# power, unary minus, and parentheses. NO names, calls, or attribute access — so
# it cannot be used to reach arbitrary Python.
_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def _calculate(args: dict[str, Any]) -> str:
    """Evaluate a safe arithmetic ``expression`` (numbers + - * / // % ** and parens)."""
    expr = str(args.get("expression", "")).strip()
    if not expr:
        return "error: no expression"
    try:
        result = _safe_eval(ast.parse(expr, mode="eval"))
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as exc:
        return f"error: cannot evaluate {expr!r}: {exc}"
    # Present whole numbers without a trailing .0 for natural speech.
    return str(int(result)) if result == int(result) else str(result)


def make_home_control_tool(dispatch: Callable[[str], str], *, name: str = "home_control") -> Tool:
    """Build the ``home_control`` tool wired to a ``dispatch(command) -> str`` callable.

    The brain passes a dispatcher that routes to the existing agent / Home Assistant
    path (the same machinery as the "agent, ..." trigger), so a model tool-call like
    ``home_control(command="turn off the kitchen light")`` actually acts on the home.
    """
    return Tool(
        name=name,
        description=(
            "Control smart-home devices (lights, music, scenes, thermostats) by "
            "passing a short natural-language command. Use for any 'turn on/off', "
            "'play', 'set', or 'dim' style home request."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The home command in plain language, e.g. 'turn off the lights'.",
                }
            },
            "required": ["command"],
        },
        run=lambda args: dispatch(str(args.get("command", "")).strip()),
    )


def default_tools(home_dispatch: Callable[[str], str] | None = None) -> list[Tool]:
    """The shipped example tools. ``home_dispatch`` enables ``home_control`` if given."""
    tools = [
        Tool(
            name="get_time",
            description="Get the current local date and time as an ISO-8601 string.",
            parameters={
                "type": "object",
                "properties": {
                    "tz": {"type": "string", "description": "Optional timezone label to echo back."}
                },
            },
            run=_get_time,
        ),
        Tool(
            name="calculator",
            description="Evaluate a simple arithmetic expression and return the result.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic like '12 * (3 + 4)' or '2 ** 10'.",
                    }
                },
                "required": ["expression"],
            },
            run=_calculate,
        ),
    ]
    if home_dispatch is not None:
        tools.append(make_home_control_tool(home_dispatch))
    return tools


@dataclass
class ToolCall:
    """A normalized tool-call request emitted by either provider's API."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
