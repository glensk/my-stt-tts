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
* ``get_weather``  — real current weather for a place via Open-Meteo (NO API KEY):
  geocode the location, fetch the forecast, summarize in metric or imperial units.
* ``home_control`` — route a natural-language home command to the existing agent /
  Home Assistant dispatch (reuses :func:`my_stt_tts.agent.dispatch_to_agent`).
"""

from __future__ import annotations

import ast
import datetime as _dt
import json
import logging
import operator
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("my_stt_tts.tools")

# Open-Meteo public endpoints — free, NO API KEY required. Geocoding resolves a
# place name to lat/lon; the forecast endpoint returns current conditions.
_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_WEATHER_TIMEOUT = 10  # seconds; a slow/unreachable network must not hang the turn

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


# --- weather (Open-Meteo, no API key) ------------------------------------------

# WMO weather interpretation codes -> short spoken phrases. Coarse on purpose:
# the model turns these into a natural sentence, so a handful of buckets suffice.
_WMO_CODES: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "heavy freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "violent rain showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}


def _wmo_description(code: int) -> str:
    """Human phrase for a WMO weather code (falls back to a neutral label)."""
    return _WMO_CODES.get(code, "unsettled weather")


def _http_get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """GET ``url?params`` and parse the JSON body (dependency-light urllib)."""
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(  # noqa: S310 — pinned HTTPS Open-Meteo endpoints
        f"{url}?{query}", method="GET", headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=_WEATHER_TIMEOUT) as resp:  # noqa: S310 — pinned HTTPS
        return dict(json.loads(resp.read().decode("utf-8")))


def _geocode(location: str) -> dict[str, Any] | None:
    """Resolve a place name to a result dict (name/lat/lon) via Open-Meteo, or None."""
    data = _http_get_json(_GEOCODE_URL, {"name": location, "count": 1, "format": "json"})
    results = data.get("results") or []
    return dict(results[0]) if results else None


def _format_weather(place: str, current: dict[str, Any], *, units: str) -> str:
    """Render an Open-Meteo ``current`` block as a concise, units-aware summary."""
    imperial = units == "imperial"
    temp_unit = "°F" if imperial else "°C"
    speed_unit = "mph" if imperial else "km/h"
    desc = _wmo_description(int(current.get("weather_code", -1)))
    temp = current.get("temperature_2m")
    feels = current.get("apparent_temperature")
    wind = current.get("wind_speed_10m")
    parts = [f"Weather in {place}: {desc}"]
    if temp is not None:
        parts.append(f"{round(float(temp))}{temp_unit}")
    if feels is not None and feels != temp:
        parts.append(f"feels like {round(float(feels))}{temp_unit}")
    if wind is not None:
        parts.append(f"wind {round(float(wind))} {speed_unit}")
    return ", ".join(parts) + "."


def get_weather(location: str, *, units: str = "metric") -> str:
    """Fetch current weather for ``location`` via Open-Meteo; concise units-aware text.

    Geocodes the place name, fetches current conditions, and returns a one-line
    summary in metric (°C / km·h) or imperial (°F / mph). NO API KEY is required.
    Never raises: any network/parse failure returns a clear "weather unavailable"
    message so a tool call can't crash the conversation turn.
    """
    place = location.strip()
    if not place:
        return "error: no location given"
    imperial = units == "imperial"
    try:
        match = _geocode(place)
        if match is None:
            return f"Weather unavailable: could not find a place called {place!r}."
        label = str(match.get("name") or place)
        country = match.get("country")
        if country:
            label = f"{label}, {country}"
        forecast = _http_get_json(
            _FORECAST_URL,
            {
                "latitude": match["latitude"],
                "longitude": match["longitude"],
                "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                "temperature_unit": "fahrenheit" if imperial else "celsius",
                "wind_speed_unit": "mph" if imperial else "kmh",
            },
        )
    except (urllib.error.URLError, TimeoutError, OSError):
        log.warning("weather request failed for %r", place, exc_info=True)
        return (
            f"Weather for {place} is unavailable right now (could not reach the weather service)."
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        log.warning("weather response could not be parsed for %r", place, exc_info=True)
        return f"Weather for {place} is unavailable right now (unexpected response)."
    current = forecast.get("current") or {}
    if not current:
        return f"Weather for {place} is unavailable right now (no current conditions)."
    return _format_weather(label, dict(current), units=units)


def make_weather_tool(*, default_location: str, default_units: str = "metric") -> Tool:
    """Build the ``get_weather`` tool bound to the configured location + units.

    ``location`` is optional in the call: when omitted the assistant's configured
    ``default_location`` is used, so "what's the weather?" answers for home while
    "weather in Tokyo?" overrides it. Units always follow the configured system.
    """

    def _run(args: dict[str, Any]) -> str:
        location = str(args.get("location") or "").strip() or default_location
        return get_weather(location, units=default_units)

    return Tool(
        name="get_weather",
        description=(
            "Get the current weather for a place (temperature, conditions, wind). "
            "Omit 'location' to use the user's configured home location. Use this for "
            "any 'what's the weather', 'is it raining', 'how hot/cold is it' question."
        ),
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": (
                        "Optional place name, e.g. 'Tokyo' or 'Paris, France'. "
                        "Defaults to the user's configured location when omitted."
                    ),
                }
            },
        },
        run=_run,
    )


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


def make_music_tools(*, player: str = "auto", volume: int | None = None) -> list[Tool]:
    """Build the ``play_music`` / ``stop_music`` tools for API (anthropic/openai) brains.

    These let a tool-calling provider trigger the SAME stoppable YouTube player the
    local intent router uses (the router stays the primary path because the default
    ``claude-cli`` brain does no tool-calling). ``player`` / ``volume`` come from
    config and select the playback backend (mpv -> ffplay -> yt-dlp download). The
    music module is imported lazily inside ``run`` so importing :mod:`tools` never
    needs yt-dlp (CORE stays importable without the ``music`` extra)."""

    def _play(args: dict[str, Any]) -> str:
        from .music import get_player

        query = str(args.get("query", "")).strip()
        if not query:
            return "error: no song given"
        result = get_player(player=player, volume=volume).play(query)
        return f"Playing {result.title}." if result.ok else result.reason

    def _stop(_args: dict[str, Any]) -> str:
        from .music import get_player

        stopped = get_player(player=player, volume=volume).stop()
        return "Stopped the music." if stopped else "Nothing was playing."

    return [
        Tool(
            name="play_music",
            description=(
                "Play a song or piece of music from YouTube through the speakers. "
                "Pass the song and artist as 'query' (e.g. 'We Will Rock You by Queen'). "
                "Use for any 'play <song>', 'put on <music>' request."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The song to play, e.g. 'Bohemian Rhapsody by Queen'.",
                    }
                },
                "required": ["query"],
            },
            run=_play,
        ),
        Tool(
            name="stop_music",
            description="Stop the music currently playing. Use for 'stop the music' / 'stop'.",
            parameters=_NO_ARGS,
            run=_stop,
        ),
    ]


def default_tools(
    home_dispatch: Callable[[str], str] | None = None,
    *,
    location: str = "Lausanne, Switzerland",
    units: str = "metric",
    music_enabled: bool = True,
    music_player: str = "auto",
    music_volume: int | None = None,
) -> list[Tool]:
    """The shipped example tools (get_time, calculator, get_weather).

    ``get_weather`` is bound to the configured ``location`` + ``units`` so it
    defaults to the user's home but accepts an explicit place per call.
    ``home_dispatch`` additionally enables ``home_control`` when supplied.
    ``music_enabled`` adds ``play_music`` / ``stop_music`` (YouTube) so a
    tool-calling API brain can play music too (the local intent router is primary).
    """
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
        make_weather_tool(default_location=location, default_units=units),
    ]
    if music_enabled:
        tools.extend(make_music_tools(player=music_player, volume=music_volume))
    if home_dispatch is not None:
        tools.append(make_home_control_tool(home_dispatch))
    return tools


@dataclass
class ToolCall:
    """A normalized tool-call request emitted by either provider's API."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
