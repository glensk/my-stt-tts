"""Tests for in-conversation tool calling + optional cloud backends (R2-7):

* the tool registry: dispatch, unknown-tool + error handling, provider schemas
* the shipped example tools (get_time, calculator, home_control routing)
* the Anthropic + OpenAI tool-call round-trip with a MOCKED provider client
  (tool requested -> executed -> result fed back -> final spoken answer)
* backend selection: cloud STT/TTS are local-first and gated on an API key

No real API calls, keys, mic, or network — every provider boundary is faked.
"""
# pylint: disable=missing-function-docstring,protected-access,missing-class-docstring
# pylint: disable=too-few-public-methods,redefined-builtin,unused-argument
# (test doubles mirror SDK shapes: tiny fakes + the OpenAI tool-call `id` field name)

from typing import Any

import numpy as np

from my_stt_tts.brain import Brain
from my_stt_tts.config import Config
from my_stt_tts.tools import (
    Tool,
    ToolCall,
    ToolRegistry,
    default_tools,
    make_home_control_tool,
)

# --- registry + example tools --------------------------------------------------


def test_registry_dispatch_runs_tool():
    reg = ToolRegistry([Tool("echo", "echo it", {"type": "object"}, lambda a: f"got {a.get('x')}")])
    assert reg.dispatch("echo", {"x": 7}) == "got 7"
    assert "echo" in reg
    assert len(reg) == 1


def test_registry_unknown_tool_returns_error_not_raises():
    reg = ToolRegistry()
    out = reg.dispatch("nope", {})
    assert "unknown tool" in out


def test_registry_tool_exception_is_caught():
    def _boom(_args):
        raise RuntimeError("kaboom")

    reg = ToolRegistry([Tool("boom", "boom", {"type": "object"}, _boom)])
    out = reg.dispatch("boom", {})
    assert "failed" in out and "kaboom" in out


def test_get_time_returns_iso_string():
    reg = ToolRegistry(default_tools())
    out = reg.dispatch("get_time", {})
    # ISO-8601-ish: starts with YYYY-MM-DD and contains a 'T' separator.
    assert out[:4].isdigit() and out[4] == "-" and "T" in out


def test_calculator_evaluates_safely():
    reg = ToolRegistry(default_tools())
    assert reg.dispatch("calculator", {"expression": "12 * (3 + 4)"}) == "84"
    assert reg.dispatch("calculator", {"expression": "2 ** 10"}) == "1024"
    assert reg.dispatch("calculator", {"expression": "10 / 4"}) == "2.5"


def test_calculator_rejects_non_arithmetic():
    reg = ToolRegistry(default_tools())
    out = reg.dispatch("calculator", {"expression": "__import__('os').system('ls')"})
    assert out.startswith("error")
    out2 = reg.dispatch("calculator", {"expression": "1/0"})
    assert out2.startswith("error")


def test_home_control_routes_to_dispatch():
    seen: list[str] = []

    def _dispatch(cmd: str) -> str:
        seen.append(cmd)
        return "ok"

    tool = make_home_control_tool(_dispatch)
    assert tool.run({"command": "turn off the lights"}) == "ok"
    assert seen == ["turn off the lights"]


def test_default_tools_includes_home_only_with_dispatch():
    assert "home_control" not in ToolRegistry(default_tools())
    assert "home_control" in ToolRegistry(default_tools(home_dispatch=lambda c: "ok"))


def test_provider_schemas_have_right_shape():
    reg = ToolRegistry(default_tools(home_dispatch=lambda c: "ok"))
    a = reg.anthropic_tools()
    o = reg.openai_tools()
    assert all("input_schema" in t and "name" in t for t in a)
    assert all(t["type"] == "function" and "parameters" in t["function"] for t in o)
    assert {t["name"] for t in a} == {t["function"]["name"] for t in o}


# --- Anthropic tool-call round-trip (mocked client) ----------------------------


class _Block:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class _Msg:
    def __init__(self, content: list[_Block]) -> None:
        self.content = content


class _StreamCtx:
    """A fake Anthropic streaming context manager yielding fixed text deltas."""

    def __init__(self, parts: list[str]) -> None:
        self.text_stream = iter(parts)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None


class _FakeAnthropicClient:
    """Returns a tool_use message once, then streams the final answer.

    Records the messages it was sent so the test can assert the tool *result* was
    fed back before the final streamed answer.
    """

    def __init__(self, final_parts: list[str]) -> None:
        self._final_parts = final_parts
        self._create_calls = 0
        self.last_messages: list[Any] = []
        self.messages = self  # client.messages.create / .stream both live here

    def create(self, **kwargs: Any):
        self._create_calls += 1
        self.last_messages = kwargs["messages"]
        if self._create_calls == 1:
            return _Msg(
                [_Block(type="tool_use", id="t1", name="calculator", input={"expression": "2+2"})]
            )
        return _Msg([_Block(type="text", text="")])  # no more tools

    def stream(self, **kwargs: Any):
        self.last_messages = kwargs["messages"]
        return _StreamCtx(self._final_parts)


def test_anthropic_tool_round_trip_executes_and_feeds_result_back():
    cfg = Config(llm_provider="anthropic", anthropic_api_key="x")
    brain = Brain(cfg)  # default tools include the calculator
    client = _FakeAnthropicClient(final_parts=["The ", "answer ", "is ", "four."])
    brain._client = client  # inject the fake provider client

    out = "".join(brain.stream("what is two plus two?"))
    assert out == "The answer is four."
    # The tool result (calculator -> "4") was fed back as a user tool_result block.
    fed = client.last_messages
    tool_results = [
        block
        for m in fed
        if isinstance(m.get("content"), list)
        for block in m["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert tool_results and tool_results[0]["content"] == "4"


# --- OpenAI tool-call round-trip (mocked client) -------------------------------


class _Fn:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _TC:
    def __init__(self, id: str, name: str, arguments: str) -> None:  # noqa: A002
        self.id = id
        self.function = _Fn(name, arguments)


class _OAIMessage:
    def __init__(self, content: str | None, tool_calls: list[_TC] | None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message=None, delta=None) -> None:  # noqa: ANN001
        self.message = message
        self.delta = delta


class _Completion:
    def __init__(self, message: _OAIMessage) -> None:
        self.choices = [_Choice(message=message)]


class _Delta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _StreamChunk:
    def __init__(self, content: str | None) -> None:
        self.choices = [_Choice(delta=_Delta(content))]


class _FakeOpenAICompletions:
    def __init__(self, final_parts: list[str]) -> None:
        self._final_parts = final_parts
        self._calls = 0
        self.tool_messages: list[Any] = []

    def create(self, **kwargs: Any):
        if kwargs.get("stream"):
            return iter(_StreamChunk(p) for p in self._final_parts)
        self._calls += 1
        self.tool_messages = kwargs["messages"]
        if self._calls == 1:
            return _Completion(
                _OAIMessage(
                    content=None,
                    tool_calls=[_TC("c1", "calculator", '{"expression": "6*7"}')],
                )
            )
        return _Completion(_OAIMessage(content="", tool_calls=None))


class _FakeOpenAIClient:
    def __init__(self, final_parts: list[str]) -> None:
        self.chat = self
        self.completions = _FakeOpenAICompletions(final_parts)


def test_openai_tool_round_trip_executes_and_feeds_result_back():
    cfg = Config(llm_provider="openai", openai_api_key="x")
    brain = Brain(cfg)
    client = _FakeOpenAIClient(final_parts=["Forty-", "two."])
    brain._client = client

    out = "".join(brain.stream("six times seven?"))
    assert out == "Forty-two."
    # A tool-role message carrying the calculator result ("42") was fed back.
    tool_msgs = [m for m in client.completions.tool_messages if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["content"] == "42"


def test_tools_disabled_uses_plain_stream():
    cfg = Config(llm_provider="anthropic", anthropic_api_key="x", tools_enabled=False)
    brain = Brain(cfg)
    assert brain.tools is None
    brain._stream_anthropic = lambda model: iter(["plain ", "reply"])  # type: ignore[assignment]
    assert "".join(brain.stream("hi")) == "plain reply"


# --- backend selection (R2-7) --------------------------------------------------


def test_make_transcriber_defaults_to_local():
    from my_stt_tts.stt import ParakeetSTT, make_transcriber

    cfg = Config()  # stt_backend defaults to local
    assert isinstance(make_transcriber(cfg), ParakeetSTT)


def test_make_transcriber_cloud_needs_key_else_falls_back():
    from my_stt_tts.stt import CloudTranscriber, ParakeetSTT, make_transcriber

    # Cloud requested but no key -> local fallback (graceful, no crash).
    no_key = Config(stt_backend="cloud", stt_cloud_api_key=None)
    assert isinstance(make_transcriber(no_key), ParakeetSTT)
    # Cloud requested WITH a key -> the cloud adapter is selected.
    with_key = Config(stt_backend="cloud", stt_cloud_api_key="sk-test")
    assert isinstance(make_transcriber(with_key), CloudTranscriber)


def test_cloud_tts_gated_on_api_key():
    from my_stt_tts.tts import TTSRouter

    # Cloud requested, no key -> router uses local (no cloud adapter wired).
    router_local = TTSRouter(Config(tts_backend="cloud", tts_cloud_api_key=None))
    assert router_local._cloud is None
    # Cloud requested, key present -> the cloud adapter is wired in.
    router_cloud = TTSRouter(Config(tts_backend="cloud", tts_cloud_api_key="sk-test"))
    assert router_cloud._cloud is not None


def test_synth_pcm_prefers_cloud_when_active():
    from my_stt_tts.tts import TTSRouter

    router = TTSRouter(Config(tts_backend="cloud", tts_cloud_api_key="sk-test"))

    class _CloudStub:
        def render(self, text: str):  # noqa: ARG002
            return np.full(16, 0.2, dtype=np.float32), 24000

    router._cloud = _CloudStub()  # type: ignore[assignment]
    pcm, sr = router.synth_pcm("hallo welt")
    assert sr == 24000
    assert pcm.size == 16


def test_tool_call_dataclass_defaults():
    tc = ToolCall(id="x", name="t")
    assert not tc.arguments  # defaults to an empty dict
