#!/usr/bin/env -S uv run --no-sync --project /Users/albert/obsidian/42-Git/infra/my-stt-tts python
"""Spikes S-ELEVEN + synthetic S-AUDIO (PLAN_claude-bridge.md, Phase 0, tp#855).

Creates a DISPOSABLE ElevenLabs agent (a clone of the real agent's conversation_config,
named ``spike-claude-bridge-<ts>``, plus one client tool ``spike_echo``), talks to it
through a fake ``AudioInterface`` that feeds macOS ``say`` speech (16 kHz PCM16) at
real-time pace, captures the agent's audio and transcribes it with ElevenLabs STT
(scribe) to verify what was actually SPOKEN. The agent is always deleted at the end
(``-k`` keeps it). The real agent (``ELEVENLABS_AGENT_ID``) is only read, never changed.

S-ELEVEN:
  (a) first-message override — rejected while the platform setting is off (negative
      test), accepted + spoken once ``platform_settings.overrides.
      conversation_config_override.agent.first_message`` is true;
  (b) client tool round-trip — "run the echo test" → ``spike_echo`` → spoken result;
  (c) ``send_user_message("[system notice] …")`` spoken? vs ``send_contextual_update``.

S-AUDIO (synthetic): N user turns (en/de/fr) while Silero VAD + SilenceEndpointer run on
exactly the fed PCM; per turn: local utterance end vs user_transcript / agent_response /
client_tool_call times, and whether the 7.5 binding rule (latest finished utterance
ending <= 3 s before the transcript) holds. A proxy only: the real-mic run is attended.

Secrets: the API key is loaded from the repo ``.env`` in-process and never printed or
written; the artifacts hold no keys.

Examples:
    scripts/spikes/s_eleven.py                  # both spikes, 20 audio turns, delete agent
    scripts/spikes/s_eleven.py -a 5 -k          # 5 audio turns, keep the agent
    scripts/spikes/s_eleven.py -E -A agent_xxx  # S-AUDIO only, reuse a kept spike agent
    scripts/spikes/s_eleven.py -d agent_xxx     # delete a leftover spike agent
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import logging
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

REPO = Path(__file__).resolve().parents[2]

API = "https://api.elevenlabs.io"
SR = 16000
CHUNK = 4000  # 250 ms, same as MacAudioInterface's input chunk
VAD_FRAME = 512  # Silero's fixed frame at 16 kHz (32 ms)
PREFIX = "spike-claude-bridge-"
ECHO_WORD = "pineapple"
FIRST_MESSAGE = "Test briefing: scratch session needs a decision."
NOTICE_SPOKEN = "[system notice] The delivery to scratch-dead failed."
NOTICE_CONTEXT = "[system notice] The delivery to scratch-two failed."
PROMPT_ADDENDUM = (
    "\n\n# Spike test instructions\n"
    "- When the user says 'run the echo test', call the client tool spike_echo with "
    "text='echo test' and then tell the user exactly what the tool returned.\n"
    "- A user message that starts with [system notice] is not from the user but from the "
    "system: read the notice to the user in one short sentence.\n"
    "- Keep every other reply under ten words.\n"
)
ECHO_TOOL = {
    "type": "client",
    "name": "spike_echo",
    "description": "Echo test tool. Call it when the user says 'run the echo test'.",
    "expects_response": True,
    "response_timeout_secs": 10,
    "parameters": {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "The text to echo back."}},
        "required": ["text"],
    },
}
EN, DE, FR = "Samantha", "Anna", "Thomas"
# (text, voice, pause_inside_s): a "|" splits the text into two parts spoken with a
# pause in between — the merge/split probe (two VAD utterances, one intended turn).
AUDIO_TURNS: list[tuple[str, str]] = [
    ("Open YouTube.", EN),
    ("Lauter.", DE),
    ("Mets le volume à trente.", FR),
    ("Which sessions are running?", EN),
    ("Run the echo test.", EN),
    ("Mach die Musik leiser.", DE),
    ("Ouvre YouTube.", FR),
    ("Open YouTube. | And play the first video.", EN),
    ("Skip ten seconds.", EN),
    ("Welche Sitzungen laufen gerade?", DE),
    ("Plus fort.", FR),
    ("Make the screen brighter.", EN),
    ("Öffne den Taschenrechner. | Und rechne zwei plus zwei.", DE),
    ("Quelles sessions sont actives ?", FR),
    ("Run the echo test.", EN),
    ("Pause the video.", EN),
    ("Nächstes Video.", DE),
    ("Mets en pause.", FR),
    ("Open Calculator.", EN),
    ("Volume dreißig.", DE),
]
INNER_PAUSE_S = 0.9  # > the endpointer's silence, < a typical turn timeout

log = logging.getLogger("s_eleven")


# ---------------------------------------------------------------- ElevenLabs REST


class Api:
    """Minimal REST client; the key lives only in this object."""

    def __init__(self, key: str) -> None:
        self._h = {"xi-api-key": key}
        self.key = key

    def req(self, method: str, path: str, **kw: Any) -> httpx.Response:
        r = httpx.request(method, API + path, headers=self._h, timeout=60, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:500]}")
        return r

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        return self.req("GET", f"/v1/convai/agents/{agent_id}").json()

    def stt(self, pcm: bytes, language: str | None = None) -> str:
        """Transcribe int16 mono 16 kHz PCM with scribe_v2; '' for no audio."""
        if len(pcm) < SR // 5:  # < 0.1 s
            return ""
        buf = io.BytesIO()
        with wave.Wave_write(buf) as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm)
        data = {"model_id": "scribe_v2"}
        if language:
            data["language_code"] = language
        r = self.req(
            "POST", "/v1/speech-to-text", data=data, files={"file": ("a.wav", buf.getvalue())}
        )
        return str(r.json().get("text", "")).strip()


def build_agent_payload(real: dict[str, Any], *, first_message_override: bool) -> dict[str, Any]:
    """Clone the real agent's conversation_config into a disposable spike agent."""
    cc = copy.deepcopy(real["conversation_config"])
    prompt = cc["agent"]["prompt"]
    prompt["prompt"] = (prompt.get("prompt") or "") + PROMPT_ADDENDUM
    prompt["tools"] = [*(prompt.get("tools") or []), ECHO_TOOL]
    cc["conversation"]["max_duration_seconds"] = 900
    overrides = copy.deepcopy(real["platform_settings"].get("overrides") or {})
    agent_ovr = overrides.setdefault("conversation_config_override", {}).setdefault("agent", {})
    agent_ovr["first_message"] = first_message_override
    return {
        "name": f"{PREFIX}{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "tags": ["spike"],
        "conversation_config": cc,
        "platform_settings": {"overrides": overrides},
    }


def safe_delete(api: Api, agent_id: str, real_id: str) -> bool:
    """Delete ``agent_id`` only if it is a spike agent (name prefix) and not the real one."""
    if agent_id == real_id:
        raise RuntimeError("refusing to delete the real agent")
    name = api.get_agent(agent_id).get("name", "")
    if not name.startswith(PREFIX):
        raise RuntimeError(f"refusing to delete {agent_id}: name {name!r} is not a spike agent")
    tool_ids = spike_tool_ids(api, api.get_agent(agent_id))
    api.req("DELETE", f"/v1/convai/agents/{agent_id}")
    try:
        api.get_agent(agent_id)
        return False
    except RuntimeError as exc:
        if "404" not in str(exc) and "not_found" not in str(exc).lower():
            return False
    # The API turns an inline client tool into a workspace tool (tool_ids) that outlives
    # the agent; its only dependent is the deleted agent's branch, so force-delete it.
    for tid in tool_ids:
        deps = api.req("GET", f"/v1/convai/tools/{tid}/dependent-agents").json()
        users = {d.get("agent_id") for d in [*deps.get("agents", []), *deps.get("branches", [])]}
        if users - {agent_id}:
            print(f"⚠️  tool {tid} still used by {sorted(users - {agent_id})}; not deleted")
            continue
        api.req("DELETE", f"/v1/convai/tools/{tid}", params={"force": "true"})
        print(f"✅ deleted spike tool {tid}")
    return True


def spike_tool_ids(api: Api, agent: dict[str, Any]) -> list[str]:
    """The agent's workspace tools named like the spike tool."""
    out = []
    for tid in agent["conversation_config"]["agent"]["prompt"].get("tool_ids") or []:
        cfg = api.req("GET", f"/v1/convai/tools/{tid}").json().get("tool_config", {})
        if cfg.get("name") == ECHO_TOOL["name"]:
            out.append(tid)
    return out


# ---------------------------------------------------------------- audio helpers


_TTS_CACHE: dict[tuple[str, str], bytes] = {}


def say_pcm(text: str, voice: str) -> bytes:
    """macOS ``say`` → int16 mono 16 kHz PCM (cached)."""
    key = (text, voice)
    if key not in _TTS_CACHE:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "s.wav"
            subprocess.run(
                ["say", "-v", voice, "-o", str(out), "--data-format=LEI16@16000", text],
                check=True,
                timeout=30,
            )
            with wave.open(str(out), "rb") as w:
                assert w.getframerate() == SR and w.getnchannels() == 1
                _TTS_CACHE[key] = w.readframes(w.getnframes())
    return _TTS_CACHE[key]


def turn_pcm(text: str, voice: str) -> bytes:
    """PCM for a turn; ``a | b`` = two parts separated by INNER_PAUSE_S of silence."""
    parts = [p.strip() for p in text.split("|")]
    gap = b"\x00\x00" * int(INNER_PAUSE_S * SR)
    return gap.join(say_pcm(p, voice) for p in parts)


class VadTracker:
    """Silero VAD + SilenceEndpointer on the fed stream → utterances (start, end, closed)."""

    def __init__(self, silence_s: float) -> None:
        from my_stt_tts import vad as vad_mod  # pylint: disable=import-outside-toplevel
        from my_stt_tts.audio import reframe  # pylint: disable=import-outside-toplevel

        self._reframe = reframe
        self.vad = vad_mod.SileroVad(SR)
        self.ep = vad_mod.SilenceEndpointer(silence_s, VAD_FRAME / SR)
        self.silence_s = silence_s
        self.cur_start: float | None = None
        self.last_end = 0.0
        self.utterances: list[dict[str, float]] = []
        self.vad.is_speech(np.zeros(VAD_FRAME, dtype=np.float32))  # load the model now

    def process(self, pcm: bytes, t_start: float) -> None:
        arr = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        for i, frame in enumerate(self._reframe(arr, VAD_FRAME)):
            ft = t_start + i * VAD_FRAME / SR
            speech = self.vad.is_speech(frame)
            if speech:
                if self.cur_start is None:
                    self.cur_start = ft
                self.last_end = ft + VAD_FRAME / SR
            if self.ep.update(speech):
                assert self.cur_start is not None
                self.utterances.append(
                    {"start": self.cur_start, "end": self.last_end, "closed": ft + VAD_FRAME / SR}
                )
                self.cur_start = None
                self.ep.reset()


class EventLog:
    """Thread-safe event list with monotonic timestamps relative to T0."""

    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def now(self) -> float:
        return time.monotonic() - self.t0

    def add(self, ev: str, **kw: Any) -> dict[str, Any]:
        rec = {"t": round(self.now(), 3), "ev": ev, **kw}
        with self._lock:
            self.events.append(rec)
        log.info("%7.3f %-22s %s", rec["t"], ev, json.dumps(kw, ensure_ascii=False)[:160])
        return rec

    def note(self, ev: str, **kw: Any) -> None:
        """Like :meth:`add`, for SDK callbacks that must return None."""
        self.add(ev, **kw)

    def since(self, t: float, ev: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            return [e for e in self.events if e["t"] >= t and (ev is None or e["ev"] == ev)]


class _FakeAudioCore:  # pylint: disable=too-many-instance-attributes
    """Feeds queued PCM (silence otherwise) at real-time pace; captures agent audio."""

    def __init__(self, elog: EventLog, vad: VadTracker | None) -> None:
        self._elog = elog
        self._vad = vad
        self.queue: deque[tuple[str, bytes, threading.Event]] = deque()
        self.out: list[tuple[float, bytes]] = []  # (t rel T0, pcm)
        self.playback_end = 0.0  # rel T0: when simulated playback would finish
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.started = threading.Event()

    def start(self, input_callback: Callable[[bytes], None]) -> None:
        self._thread = threading.Thread(
            target=self._feed, args=(input_callback,), name="fake-mic", daemon=True
        )
        self._thread.start()
        self.started.set()
        self._elog.add("audio_interface_start")

    def _feed(self, cb: Callable[[bytes], None]) -> None:
        n = 0
        t0 = time.monotonic()
        cur: tuple[str, bytes, threading.Event] | None = None
        pos = 0
        silence = b"\x00\x00" * CHUNK
        while not self._stop.is_set():
            if cur is None and self.queue:
                cur, pos = self.queue.popleft(), 0
                self._elog.add("feed_start", label=cur[0])
            if cur is not None:
                chunk = cur[1][pos : pos + CHUNK * 2]
                pos += CHUNK * 2
                chunk += b"\x00" * (CHUNK * 2 - len(chunk))
                if pos >= len(cur[1]):
                    self._elog.add("feed_end", label=cur[0])
                    cur[2].set()
                    cur = None
            else:
                chunk = silence
            # a real mic delivers a chunk once it has been recorded
            target = t0 + (n + 1) * CHUNK / SR
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            chunk_start = target - CHUNK / SR - self._elog.t0
            try:
                cb(chunk)
            except Exception:  # pylint: disable=broad-exception-caught
                log.debug("input callback failed", exc_info=True)
            if self._vad is not None:
                self._vad.process(chunk, chunk_start)
            n += 1

    def enqueue(self, label: str, pcm: bytes) -> threading.Event:
        done = threading.Event()
        self.queue.append((label, pcm, done))
        return done

    def stop(self) -> None:
        self._stop.set()

    def output(self, audio: bytes) -> None:
        now = self._elog.now()
        with self._lock:
            if now > self.playback_end + 0.3:
                self._elog.add("agent_audio_start")
            self.out.append((now, audio))
            self.playback_end = max(now, self.playback_end) + len(audio) / (2 * SR)

    def interrupt(self) -> None:
        with self._lock:
            self.playback_end = self._elog.now()
        self._elog.add("interruption")

    def speaking(self) -> bool:
        return self._elog.now() < self.playback_end

    def audio_since(self, t: float) -> bytes:
        with self._lock:
            return b"".join(a for ts, a in self.out if ts >= t)

    def audio_seconds_since(self, t: float) -> float:
        return len(self.audio_since(t)) / (2 * SR)


def make_fake_audio(elog: EventLog, vad: VadTracker | None) -> Any:
    """Build the fake SDK AudioInterface (the SDK base is mixed in so its import stays lazy)."""
    # pylint: disable-next=import-outside-toplevel
    from elevenlabs.conversational_ai.conversation import AudioInterface

    return type("FakeAudio", (_FakeAudioCore, AudioInterface), {})(elog, vad)


# ---------------------------------------------------------------- one conversation


class Call:
    """One SDK Conversation against the spike agent, fully instrumented."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        api: Api,
        agent_id: str,
        elog: EventLog,
        *,
        vad: VadTracker | None = None,
        override: dict[str, Any] | None = None,
        tool_reply: str = f"Echo received. The echo word is {ECHO_WORD}.",
    ) -> None:
        from elevenlabs.client import ElevenLabs  # pylint: disable=import-outside-toplevel
        from elevenlabs.conversational_ai.conversation import (  # pylint: disable=import-outside-toplevel
            ClientTools,
            Conversation,
            ConversationInitiationData,
        )

        self.elog = elog
        self.audio = make_fake_audio(elog, vad)
        self.ended = threading.Event()
        self.tool_calls: list[dict[str, Any]] = []
        tools = ClientTools()

        def echo(params: dict[str, Any]) -> str:
            rec = elog.add(
                "client_tool_call",
                tool="spike_echo",
                params={k: v for k, v in params.items() if k != "tool_call_id"},
            )
            self.tool_calls.append(rec)
            elog.add("client_tool_return", tool="spike_echo")
            return tool_reply

        tools.register("spike_echo", echo)
        config = ConversationInitiationData(conversation_config_override=override or {})
        self.conv = Conversation(
            ElevenLabs(api_key=api.key),
            agent_id,
            requires_auth=True,
            audio_interface=self.audio,
            config=config,
            client_tools=tools,
            callback_user_transcript=lambda t: elog.note("user_transcript", text=t),
            callback_agent_response=lambda t: elog.note("agent_response", text=t),
            callback_agent_response_correction=lambda o, c: elog.note(
                "agent_response_correction", original=o, corrected=c
            ),
            callback_latency_measurement=lambda ms: elog.note("ping_latency", ms=ms),
            callback_end_session=self._on_end,
        )

    def _on_end(self) -> None:
        if not self.ended.is_set():
            self.ended.set()
            self.elog.add("conversation_end")

    def start(self, timeout: float = 15.0) -> str | None:
        self.elog.add("start_session")
        self.conv.start_session()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.ended.is_set():
            cid = self.conv._conversation_id  # pylint: disable=protected-access
            if cid:
                self.elog.add("conversation_started", conversation_id=cid)
                return str(cid)
            time.sleep(0.05)
        return None

    def end(self) -> None:
        with contextlib.suppress(Exception):
            self.conv.end_session()
        if self.conv._thread is not None:  # pylint: disable=protected-access
            self.conv._thread.join(timeout=5)  # pylint: disable=protected-access

    def wait_idle(self, quiet: float = 1.2, first_audio_timeout: float = 12.0) -> bool:
        """Wait for agent audio (up to ``first_audio_timeout``), then ``quiet`` s of silence.

        Returns True when the agent produced audio during the wait.
        """
        t_begin = self.elog.now()
        deadline = time.monotonic() + first_audio_timeout
        while time.monotonic() < deadline and not self.ended.is_set():
            if self.audio.audio_seconds_since(t_begin) > 0 or self.audio.speaking():
                break
            time.sleep(0.05)
        else:
            return False
        while not self.ended.is_set():
            if self.elog.now() - self.audio.playback_end >= quiet:
                return True
            time.sleep(0.05)
        return True

    def say(self, label: str, pcm: bytes) -> None:
        self.audio.enqueue(label, pcm).wait(timeout=30)


# ---------------------------------------------------------------- S-ELEVEN


def contains_any(text: str, words: list[str]) -> bool:
    low = text.lower()
    return any(w in low for w in words)


def sanitized_config_facts(agent: dict[str, Any]) -> dict[str, Any]:
    cc = agent["conversation_config"]
    ovr = agent["platform_settings"].get("overrides") or {}
    return {
        "version_id": agent.get("version_id"),
        "branch_id": agent.get("branch_id"),
        "client_events": cc["conversation"].get("client_events"),
        "prompt_tools": [t.get("name") for t in cc["agent"]["prompt"].get("tools") or []],
        "tool_ids": cc["agent"]["prompt"].get("tool_ids"),
        "first_message_override_allowed": (
            ovr.get("conversation_config_override", {}).get("agent", {}).get("first_message")
        ),
        "updated_at_unix_secs": agent.get("metadata", {}).get("updated_at_unix_secs"),
    }


def run_negative(api: Api, agent_id: str) -> dict[str, Any]:
    """(a-) first-message override while the platform setting is OFF."""
    elog = EventLog()
    errors: list[str] = []
    handler = _ListHandler(errors)
    sdk_log = logging.getLogger("elevenlabs")
    sdk_log.addHandler(handler)
    call = Call(api, agent_id, elog, override={"agent": {"first_message": FIRST_MESSAGE}})
    try:
        cid = call.start(timeout=10)
        time.sleep(6)
        spoken = api.stt(call.audio.audio_since(0)) if call.audio.out else ""
    finally:
        call.end()
        sdk_log.removeHandler(handler)
    return {
        "conversation_id": cid,
        "ended_by_server": call.ended.is_set(),
        "sdk_errors": errors,
        "agent_audio_s": round(call.audio.audio_seconds_since(0), 2),
        "spoken_stt": spoken,
        "events": elog.events,
    }


def run_tool_probe(api: Api, agent_id: str) -> dict[str, Any]:
    """(b-) is a client tool call delivered while client_events lacks client_tool_call?"""
    elog = EventLog()
    call = Call(api, agent_id, elog)
    try:
        cid = call.start()
        time.sleep(1.0)
        t = elog.now()
        call.say("run the echo test", say_pcm("Run the echo test.", EN))
        call.wait_idle(quiet=1.5, first_audio_timeout=12)
        time.sleep(3)
        spoken = api.stt(call.audio.audio_since(t), "en")
    finally:
        call.end()
    return {
        "conversation_id": cid,
        "tool_called": bool(call.tool_calls),
        "user_transcript": [e["text"] for e in elog.since(0, "user_transcript")],
        "agent_response_text": [e["text"] for e in elog.since(0, "agent_response")],
        "spoken_stt": spoken,
        "events": elog.events,
    }


class _ListHandler(logging.Handler):
    def __init__(self, sink: list[str]) -> None:
        super().__init__(logging.DEBUG)
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self.sink.append(record.getMessage()[:400])


def run_eleven(  # pylint: disable=too-many-locals
    api: Api, agent_id: str, res: dict[str, Any]
) -> dict[str, Any]:
    """S-ELEVEN (a)+, (b), (c) in one conversation; fills ``res`` in place (partial on error)."""
    elog = EventLog()
    vad = VadTracker(0.5)
    errors: list[str] = []
    handler = _ListHandler(errors)
    logging.getLogger("elevenlabs").addHandler(handler)
    call = Call(api, agent_id, elog, vad=vad, override={"agent": {"first_message": FIRST_MESSAGE}})
    try:
        cid = call.start()
        res["conversation_id"] = cid
        # (a) first message
        got = call.wait_idle(quiet=1.0, first_audio_timeout=10)
        first_resp = [e["text"] for e in elog.since(0, "agent_response")]
        spoken = api.stt(call.audio.audio_since(0), "en")
        res["a_first_message"] = {
            "accepted": cid is not None and not call.ended.is_set(),
            "agent_audio": got,
            "agent_response_text": first_resp,
            "spoken_stt": spoken,
            "spoken_matches": contains_any(spoken, ["briefing"])
            and contains_any(spoken, ["decision"]),
        }
        res["b_client_tool"] = _probe_client_tool(api, call, elog, vad)
        time.sleep(1.0)
        # (c1) send_user_message
        t_c = elog.now()
        elog.add("send_user_message", text=NOTICE_SPOKEN)
        call.conv.send_user_message(NOTICE_SPOKEN)
        got_c = call.wait_idle(quiet=1.5, first_audio_timeout=12)
        spoken_c = api.stt(call.audio.audio_since(t_c), "en")
        first_audio_c = elog.since(t_c, "agent_audio_start")
        res["c_send_user_message"] = {
            "agent_audio": got_c,
            "agent_response_text": [e["text"] for e in elog.since(t_c, "agent_response")],
            "user_transcript_events": [e["text"] for e in elog.since(t_c, "user_transcript")],
            "spoken_stt": spoken_c,
            "spoken_mentions_notice": contains_any(spoken_c, ["scratch", "deliver"]),
            "latency_to_first_audio_s": _d(first_audio_c[0]["t"] if first_audio_c else None, t_c),
        }
        time.sleep(1.0)
        # (c2) send_contextual_update
        t_u = elog.now()
        elog.add("send_contextual_update", text=NOTICE_CONTEXT)
        call.conv.send_contextual_update(NOTICE_CONTEXT)
        time.sleep(8)
        spoken_u = api.stt(call.audio.audio_since(t_u), "en")
        res["c_contextual_update"] = {
            "agent_audio_s_within_8s": round(call.audio.audio_seconds_since(t_u), 2),
            "agent_response_text": [e["text"] for e in elog.since(t_u, "agent_response")],
            "spoken_stt": spoken_u,
        }
        # does the LLM know about it when asked?
        t_q = elog.now()
        call.say("anything new", say_pcm("Is there anything new?", EN))
        call.wait_idle(quiet=1.5, first_audio_timeout=12)
        spoken_q = api.stt(call.audio.audio_since(t_q), "en")
        res["c_contextual_update"]["followup_question"] = "Is there anything new?"
        res["c_contextual_update"]["followup_spoken_stt"] = spoken_q
        res["c_contextual_update"]["followup_mentions_notice"] = contains_any(
            spoken_q, ["scratch", "deliver"]
        )
    finally:
        call.end()
        logging.getLogger("elevenlabs").removeHandler(handler)
        res["sdk_errors"] = errors
        res["vad_utterances"] = vad.utterances
        res["events"] = elog.events
    return res


def _probe_client_tool(api: Api, call: Call, elog: EventLog, vad: VadTracker) -> dict[str, Any]:
    """(b) client tool round-trip: "run the echo test" -> spike_echo -> spoken result."""
    t_b = elog.now()
    call.say("run the echo test", say_pcm("Run the echo test.", EN))
    call.wait_idle(quiet=1.5, first_audio_timeout=15)
    # wait for the post-tool reply if the agent spoke a preamble first
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and call.tool_calls:
        ret = elog.since(t_b, "client_tool_return")
        after = [e for e in elog.since(t_b, "agent_audio_start") if ret and e["t"] > ret[0]["t"]]
        if after and not call.audio.speaking() and elog.now() - call.audio.playback_end > 1.0:
            break
        time.sleep(0.1)
    spoken_b = api.stt(call.audio.audio_since(t_b), "en")
    utt = [u for u in vad.utterances if u["start"] >= t_b]
    tr = elog.since(t_b, "user_transcript")
    ret = elog.since(t_b, "client_tool_return")
    audio_after_tool = [
        e for e in elog.since(t_b, "agent_audio_start") if ret and e["t"] >= ret[0]["t"]
    ]
    u_end = utt[-1]["end"] if utt else None
    return {
        "tool_called": bool(call.tool_calls),
        "tool_args": call.tool_calls[0]["params"] if call.tool_calls else None,
        "user_transcript": [e["text"] for e in tr],
        "agent_response_text": [e["text"] for e in elog.since(t_b, "agent_response")],
        "spoken_stt": spoken_b,
        "spoken_reflects_tool_result": ECHO_WORD in spoken_b.lower(),
        "latency_s": {
            "utterance_end_to_transcript": _d(tr[0]["t"] if tr else None, u_end),
            "utterance_end_to_tool_call": _d(
                call.tool_calls[0]["t"] if call.tool_calls else None, u_end
            ),
            "tool_return_to_agent_audio": _d(
                audio_after_tool[0]["t"] if audio_after_tool else None,
                ret[0]["t"] if ret else None,
            ),
            "utterance_end_to_first_agent_audio": _d(
                (elog.since(t_b, "agent_audio_start") or [{"t": None}])[0]["t"], u_end
            ),
        },
    }


def _d(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else round(a - b, 3)


# ---------------------------------------------------------------- S-AUDIO


def run_audio(api: Api, agent_id: str, n_turns: int, silence_s: float) -> dict[str, Any]:
    """S-AUDIO (synthetic): N turns, VAD vs SDK event timing."""
    elog = EventLog()
    vad = VadTracker(silence_s)
    turns = [AUDIO_TURNS[i % len(AUDIO_TURNS)] for i in range(n_turns)]
    pcms = [turn_pcm(t, v) for t, v in turns]  # synthesize before the call starts
    call = Call(api, agent_id, elog, vad=vad)
    windows: list[dict[str, Any]] = []
    try:
        cid = call.start()
        time.sleep(1.5)
        for i, ((text, voice), pcm) in enumerate(zip(turns, pcms, strict=True)):
            if call.ended.is_set():
                break
            # the mic chunk being recorded when we enqueue started up to 250 ms earlier
            w: dict[str, Any] = {"i": i, "text": text, "voice": voice, "t0": elog.now() - 0.3}
            call.say(f"turn {i}", pcm)
            w["fed_end"] = elog.now()
            w["agent_spoke"] = call.wait_idle(quiet=1.2, first_audio_timeout=10)
            time.sleep(0.8)
            windows.append(w)
        time.sleep(1.0)
    finally:
        call.end()
    return analyse_audio(cid, windows, vad, elog, silence_s)


def analyse_audio(  # pylint: disable=too-many-locals
    cid: str | None,
    windows: list[dict[str, Any]],
    vad: VadTracker,
    elog: EventLog,
    silence_s: float,
) -> dict[str, Any]:
    """Per-turn table, binding-rule check and summary stats."""
    utts = vad.utterances
    transcripts = [e for e in elog.events if e["ev"] == "user_transcript" and e["text"].strip()]
    rows = []
    for k, w in enumerate(windows):
        t_hi = windows[k + 1]["t0"] if k + 1 < len(windows) else float("inf")
        in_win = [u for u in utts if w["t0"] <= u["start"] < t_hi]
        trs = [e for e in transcripts if w["t0"] <= e["t"] < t_hi]
        resp = [e for e in elog.events if e["ev"] == "agent_response" and w["t0"] <= e["t"] < t_hi]
        tools = [
            e for e in elog.events if e["ev"] == "client_tool_call" and w["t0"] <= e["t"] < t_hi
        ]
        u_end = in_win[-1]["end"] if in_win else None
        tr_t = trs[0]["t"] if trs else None
        rows.append(
            {
                "i": w["i"],
                "said": w["text"],
                "voice": w["voice"],
                "vad_utterances": len(in_win),
                "utterance_start": round(in_win[0]["start"], 3) if in_win else None,
                "utterance_end": round(u_end, 3) if u_end is not None else None,
                "utterance_closed": round(in_win[-1]["closed"], 3) if in_win else None,
                "utterance_len_s": (
                    round(in_win[-1]["end"] - in_win[0]["start"], 3) if in_win else None
                ),
                "transcripts": [e["text"] for e in trs],
                "transcript_t": [e["t"] for e in trs],
                "agent_response_t": resp[0]["t"] if resp else None,
                "client_tool_call_t": tools[0]["t"] if tools else None,
                "delta_transcript_minus_utt_end": _d(tr_t, u_end),
                "delta_transcript_minus_utt_closed": _d(
                    tr_t, in_win[-1]["closed"] if in_win else None
                ),
                "delta_tool_minus_utt_end": _d(tools[0]["t"] if tools else None, u_end),
            }
        )
    binding = [bind_check(e, utts, windows) for e in transcripts]
    deltas = [r["delta_transcript_minus_utt_end"] for r in rows]
    valid = [d for d in deltas if d is not None]
    within = [d for d in valid if 0 <= d <= 3.0]
    n = len(rows)
    summary = {
        "turns": n,
        "turns_with_transcript": sum(1 for r in rows if r["transcripts"]),
        "within_3s": len(within),
        "pct_within_3s_of_all_turns": round(100 * len(within) / n, 1) if n else None,
        "median_delta_s": round(statistics.median(valid), 3) if valid else None,
        "p95_delta_s": round(_pct(valid, 95), 3) if valid else None,
        "max_delta_s": round(max(valid), 3) if valid else None,
        "min_delta_s": round(min(valid), 3) if valid else None,
        "turns_split_by_vad": sum(1 for r in rows if r["vad_utterances"] > 1),
        "turns_with_multiple_transcripts": sum(1 for r in rows if len(r["transcripts"]) > 1),
        "turns_without_transcript": [r["i"] for r in rows if not r["transcripts"]],
        "transcripts_before_local_endpoint": sum(
            1
            for r in rows
            if r["delta_transcript_minus_utt_closed"] is not None
            and r["delta_transcript_minus_utt_closed"] < 0
        ),
        "binding_rule_ok": sum(1 for b in binding if b["ok"]),
        "binding_rule_total": len(binding),
        "binding_multi_candidate": sum(1 for b in binding if b["candidates"] > 1),
        "binding_rule_ok_if_open_utterance_counts": sum(1 for b in binding if b["ok_incl_open"]),
        "threshold_pass": bool(n) and len(within) / n >= 0.95,
    }
    return {
        "conversation_id": cid,
        "endpointer_silence_s": silence_s,
        "inner_pause_s": INNER_PAUSE_S,
        "input_chunk_s": CHUNK / SR,
        "rows": rows,
        "binding": binding,
        "summary": summary,
        "events": elog.events,
        "vad_utterances": utts,
    }


def bind_check(tr: dict[str, Any], utts: list[dict[str, float]], windows: list[dict[str, Any]]):
    """Plan 7.5 rule at transcript time: latest FINISHED utterance ending <= 3 s before it."""
    t = tr["t"]
    finished = [u for u in utts if u["closed"] <= t and t - u["end"] <= 3.0]
    open_or_finished = [u for u in utts if u["start"] <= t and t - u["end"] <= 3.0]

    def window_of(x: float) -> int | None:
        idx = None
        for w in windows:
            if w["t0"] <= x:
                idx = w["i"]
        return idx

    tw = window_of(t)
    ok = bool(finished) and window_of(finished[-1]["start"]) == tw
    ok_open = bool(open_or_finished) and window_of(open_or_finished[-1]["start"]) == tw
    return {
        "t": t,
        "text": tr["text"],
        "turn": tw,
        "candidates": len(finished),
        "latest_end": finished[-1]["end"] if finished else None,
        "ok": ok,
        "ok_incl_open": ok_open,
    }


def _pct(vals: list[float], p: float) -> float:
    s = sorted(vals)
    k = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ---------------------------------------------------------------- main


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"✅ wrote {path}")


def ensure_client_tool_event(api: Api, agent_id: str, agent: dict[str, Any]) -> dict[str, Any]:
    """PATCH: allow the first-message override and send client_tool_call events."""
    events = list(agent["conversation_config"]["conversation"].get("client_events") or [])
    patch_events = "client_tool_call" not in events
    if patch_events:
        events.append("client_tool_call")
    overrides = copy.deepcopy(agent["platform_settings"].get("overrides") or {})
    overrides.setdefault("conversation_config_override", {}).setdefault("agent", {})[
        "first_message"
    ] = True
    body: dict[str, Any] = {"platform_settings": {"overrides": overrides}}
    if patch_events:
        body["conversation_config"] = {"conversation": {"client_events": events}}
    api.req("PATCH", f"/v1/convai/agents/{agent_id}", json=body)
    return {"patched_client_events": patch_events, "patched_first_message_override": True}


def _spike_eleven(api: Api, args: argparse.Namespace, aid: str, agent: dict, res: dict) -> None:
    """S-ELEVEN; the negative (a-)/(b-) probes only run against a freshly created agent."""
    created = args.agent_id is None
    if not args.no_negative and created:
        print("… S-ELEVEN (a-) override with the setting OFF")
        res["a_negative_setting_off"] = run_negative(api, aid)
    if created:
        print("… S-ELEVEN (b-) client tool WITHOUT client_tool_call in client_events")
        res["b_negative_no_client_event"] = run_tool_probe(api, aid)
    res["meta"]["patch"] = ensure_client_tool_event(api, aid, agent)
    res["meta"]["after_patch"] = sanitized_config_facts(api.get_agent(aid))
    print("… S-ELEVEN (a) (b) (c)")
    try:
        run_eleven(api, aid, res)
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        res["error"] = str(exc)[:400]  # keep the partial result, then still write it
        print(f"❌ S-ELEVEN aborted: {res['error']}")
    write_json(args.out / "s_eleven.json", res)


def main(argv: list[str] | None = None) -> int:  # pylint: disable=too-many-statements
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0],
        epilog=__doc__.split("Examples:")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-c", "--create", action="store_true", help="only create the agent (kept)")
    p.add_argument("-k", "--keep", action="store_true", help="keep the disposable agent")
    p.add_argument("-A", "--agent-id", help="reuse an existing spike agent (never the real one)")
    p.add_argument("-d", "--delete", metavar="ID", help="delete a leftover spike agent and exit")
    p.add_argument("-a", "--audio-turns", type=int, default=20, help="S-AUDIO turns (0 = skip)")
    p.add_argument("-E", "--skip-eleven", action="store_true", help="skip S-ELEVEN")
    p.add_argument("-N", "--no-negative", action="store_true", help="skip the negative (a) test")
    p.add_argument("-S", "--silence", type=float, default=0.5, help="endpointer silence (s)")
    p.add_argument(
        "-o", "--out", type=Path, default=REPO / "tests/fixtures/spikes", help="artifact dir"
    )
    p.add_argument("-v", "--verbose", action="store_true", help="log every event")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stderr, force=True)
    log.setLevel(logging.INFO if args.verbose else logging.WARNING)

    from my_stt_tts.eleven_voice import _load_env  # pylint: disable=import-outside-toplevel

    _load_env()
    key, real_id = os.environ.get("ELEVENLABS_API_KEY"), os.environ.get("ELEVENLABS_AGENT_ID")
    if not key or not real_id:
        print("❌ ELEVENLABS_API_KEY / ELEVENLABS_AGENT_ID missing in .env")
        return 2
    api = Api(key)
    if args.delete:
        ok = safe_delete(api, args.delete, real_id)
        print(("✅ deleted " if ok else "❌ delete not confirmed: ") + args.delete)
        return 0 if ok else 1
    if args.agent_id == real_id:
        print("❌ -A must be a spike agent, not the real agent")
        return 2
    sub = api.req("GET", "/v1/user/subscription").json()
    used, limit = sub.get("character_count", 0), sub.get("character_limit", 0)
    if limit and used >= limit:
        reset = datetime.fromtimestamp(sub.get("next_character_count_reset_unix", 0), UTC)
        print(f"❌ ElevenLabs credits exhausted ({used}/{limit}, tier {sub.get('tier')})")
        print(f"   conversations fail with [quota_exceeded] until {reset:%Y-%m-%d} or an upgrade.")
        return 4
    print(f"✅ credits: {used}/{limit} used (tier {sub.get('tier')})")

    meta: dict[str, Any] = {"date": datetime.now(UTC).isoformat(timespec="seconds")}
    agent_id = args.agent_id
    created = False
    try:
        if agent_id is None:
            real = api.get_agent(real_id)  # GET only; never written to disk
            payload = build_agent_payload(real, first_message_override=False)
            del real
            r = api.req("POST", "/v1/convai/agents/create", json=payload).json()
            agent_id, created = r["agent_id"], True
            print(f"✅ created disposable agent {agent_id} ({payload['name']})")
            meta["agent_name"] = payload["name"]
        meta["agent_id"] = agent_id
        after_create = api.get_agent(agent_id)
        meta["after_create"] = sanitized_config_facts(after_create)
        if args.create:
            args.keep = True
            return 0
        eleven: dict[str, Any] = {"meta": meta}
        if not args.skip_eleven:
            _spike_eleven(api, args, agent_id, after_create, eleven)
        elif created:
            meta["patch"] = ensure_client_tool_event(api, agent_id, after_create)
        if args.audio_turns > 0:
            print(f"… S-AUDIO synthetic, {args.audio_turns} turns")
            audio = run_audio(api, agent_id, args.audio_turns, args.silence)
            audio["meta"] = {k: v for k, v in meta.items() if k in ("date", "agent_id")}
            audio["note"] = (
                "Synthetic proxy: macOS say voices fed through a fake AudioInterface at "
                "real-time pace; the real-mic run with Albert stays an attended follow-up."
            )
            write_json(args.out / "s_audio_synthetic.json", audio)
            print(json.dumps(audio["summary"], indent=1))
    finally:
        if agent_id and created and not args.keep:
            ok = safe_delete(api, agent_id, real_id)
            print(
                ("✅ deleted disposable agent " if ok else "❌ delete NOT confirmed: ") + agent_id
            )
        elif agent_id and created:
            print(f"⚠️  kept disposable agent {agent_id} — delete with: -d {agent_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
