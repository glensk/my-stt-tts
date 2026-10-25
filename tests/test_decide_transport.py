"""The decide hub transport (tp#1234): one secret-run child per call, keep-alive HTTPS.

Fakes only: an in-memory serving child on real pipes, a fake ``HTTPSConnection``. No test
starts ``secret-run`` or reaches the real hub.
"""
# pylint: disable=missing-function-docstring

from __future__ import annotations

import http.client
import io
import json
import os
import subprocess
import threading
import time
from collections.abc import Mapping
from typing import Any, ClassVar

import pytest

from my_stt_tts import decide
from my_stt_tts.bridge import Authoriser, BridgeController
from my_stt_tts.decide import (
    DecideDesk,
    HttpReply,
    HubClient,
    HubConfig,
    HubError,
    SecretRunTransport,
)
from my_stt_tts.decide_tools import register

CLIENT_ID = "test-client.access"


def _plain(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float
) -> HttpReply:
    del method, url, headers, body, timeout
    return HttpReply(200, "application/json", b"{}")


class FakeVad:
    def is_speech(self, frame: Any) -> bool:
        del frame
        return False


class FakeScorer:
    def score_against(self, audio: Any, name: str, *, timeout: float = 5.0) -> float | None:
        del audio, name, timeout
        return 0.1


class _FakeChild:  # pylint: disable=too-many-instance-attributes  # pipe + process state
    """A serving child in memory: answers each request line on a real pipe (or stays mute)."""

    def __init__(self, argv: list[str], reply: Any) -> None:
        self.argv = argv
        self.reply = reply  # dict → JSON line; None → never answers; "exit" → closes stdout
        self.first_delay = 0.0  # a cold secret-run: the first reply comes this much later
        self.lines: list[dict[str, Any]] = []
        self.killed = False
        read_fd, self._write_fd = os.pipe()
        self.stdout = os.fdopen(read_fd, "r")
        self.stdin = self
        self._code: int | None = None

    # stdin side
    def write(self, text: str) -> int:
        self.lines.append(json.loads(text))
        if self.reply == "exit":
            self._close_out(1)
        elif self.reply is not None:
            if len(self.lines) == 1 and self.first_delay:
                time.sleep(self.first_delay)
            reply = {"ok": True} if self.lines[-1].get("op") == "connect" else self.reply
            os.write(self._write_fd, (json.dumps(reply) + "\n").encode())
        return len(text)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    # process side
    def _close_out(self, code: int) -> None:
        if self._code is None:
            self._code = code
            os.close(self._write_fd)

    def poll(self) -> int | None:
        return self._code

    def kill(self) -> None:
        self.killed = True
        self._close_out(-9)

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self._code if self._code is not None else 0


class _Spawner:
    """``popen`` for SecretRunTransport: one :class:`_FakeChild` per start, replies in order."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.children: list[_FakeChild] = []
        self.kwargs: list[dict[str, Any]] = []

    def __call__(self, argv: list[str], **kw: Any) -> _FakeChild:
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        child = _FakeChild(argv, reply)
        self.children.append(child)
        self.kwargs.append(kw)
        return child


_OK = {"status": 200, "content_type": "application/json", "body": '{"ok": 1}'}


def _transport(spawn: _Spawner) -> SecretRunTransport:
    return SecretRunTransport(
        "/bin/secret-run", popen=spawn, python="/py", start_s=0.5, margin_s=0.1, warm_s=1.0
    )


def test_secret_run_transport_keeps_the_secret_in_one_child_per_call() -> None:
    spawn = _Spawner(_OK)
    transport = _transport(spawn)
    for _ in range(3):
        reply = transport("GET", "https://x/api/plate", {"CF-Access-Client-Id": "id"}, None, 1.0)
        assert reply == HttpReply(200, "application/json", b'{"ok": 1}')
    assert len(spawn.children) == 1  # secret-run + interpreter + connection paid once
    child = spawn.children[0]
    assert child.argv == [
        "/bin/secret-run",
        "-e",
        "DECIDE_CLI_CF_CLIENT_SECRET=decide-cli-cf-service-token",
        "--",
        "/py",
        "-I",
        "-m",
        "my_stt_tts.decide",
        "-S",
    ]
    assert spawn.kwargs[0]["stderr"] == subprocess.DEVNULL
    assert len(child.lines) == 3 and not any("Secret" in json.dumps(x) for x in child.lines)
    transport.close()
    assert child.killed and not transport.running


def test_a_mute_child_is_killed_and_the_read_retried_on_a_fresh_one() -> None:
    spawn = _Spawner(None, _OK)  # the first child hangs (e.g. DNS on a hotspot)
    client = HubClient(HubConfig(client_id=CLIENT_ID), _transport(spawn), timeout=0.05)
    started = time.monotonic()
    assert client.call("GET", "/api/plate") == {"ok": 1}
    assert time.monotonic() - started < 2.0  # 0.05 s request + 0.5 s start margin, twice at most
    assert len(spawn.children) == 2 and spawn.children[0].killed


def test_a_mutation_on_a_mute_child_is_not_repeated() -> None:
    spawn = _Spawner(None, _OK)
    client = HubClient(HubConfig(client_id=CLIENT_ID), _transport(spawn), timeout=0.05)
    with pytest.raises(HubError) as err:
        client.answer("session:aaaa1111", {"choices": [0], "comment": ""})
    assert err.value.status == 0 and len(spawn.children) == 1


def test_a_child_that_exits_is_replaced_on_the_next_request() -> None:
    spawn = _Spawner("exit", _OK)
    transport = _transport(spawn)
    with pytest.raises(HubError) as err:
        transport("GET", "https://x/", {}, None, 1.0)
    assert "exit 1" in str(err.value)
    assert transport("GET", "https://x/", {}, None, 1.0).status == 200
    assert len(spawn.children) == 2


def test_warm_starts_the_child_and_connects_and_close_stops_it() -> None:
    spawn = _Spawner({"ok": True})
    client = HubClient(HubConfig(client_id=CLIENT_ID), _transport(spawn))
    assert client.warm() is True
    assert spawn.children[0].lines == [
        {"op": "connect", "url": "https://decide.dom42.space", "timeout": decide.TIMEOUT_S}
    ]
    client.close()
    assert spawn.children[0].killed
    assert HubClient(HubConfig(client_id=CLIENT_ID), _plain).warm() is False  # no child


def test_timeout_comes_from_the_environment() -> None:
    assert decide.timeout_from_env({}) == decide.TIMEOUT_S == 4.0
    assert decide.timeout_from_env({decide.TIMEOUT_ENV: "2.5"}) == 2.5
    assert decide.timeout_from_env({decide.TIMEOUT_ENV: "junk"}) == decide.TIMEOUT_S
    env = {"DECIDE_VOICE_CF_CLIENT_ID": "id", decide.SECRET_ENV: "x", decide.TIMEOUT_ENV: "3"}
    client = decide.build_client(env)
    assert client is not None and client.timeout == 3.0


class _FakeResponse:
    def __init__(self, status: int = 200, will_close: bool = False) -> None:
        self.status = status
        self.will_close = will_close

    def read(self) -> bytes:
        return b"{}"

    def getheader(self, name: str, default: str = "") -> str:
        return "application/json" if name == "content-type" else default


class _FakeConn:
    made: ClassVar[list[_FakeConn]] = []
    errors: ClassVar[list[BaseException]] = []  # raised by the next requests, in order

    def __init__(self, host: str, **_kw: Any) -> None:
        self.host = host
        self.sock: Any = None
        self.requests: list[tuple[str, str, dict[str, str]]] = []
        self.closed = False
        self._next = _FakeResponse()
        _FakeConn.made.append(self)

    def connect(self) -> None:
        self.sock = _FakeSock()

    def request(self, method: str, path: str, body: Any = None, headers: Any = None) -> None:
        del body
        self.sock = self.sock or _FakeSock()
        self.requests.append((method, path, dict(headers or {})))
        if _FakeConn.errors:
            raise _FakeConn.errors.pop(0)
        self._next = _FakeResponse()

    def getresponse(self) -> _FakeResponse:
        return self._next

    def close(self) -> None:
        self.closed = True


class _FakeSock:
    def settimeout(self, _t: float) -> None:
        pass


@pytest.fixture(name="fake_https")
def _fake_https(monkeypatch: pytest.MonkeyPatch) -> type[_FakeConn]:
    _FakeConn.made.clear()
    _FakeConn.errors.clear()
    monkeypatch.setattr(http.client, "HTTPSConnection", _FakeConn)
    return _FakeConn


def test_keep_alive_reuses_one_connection_and_adds_the_secret(
    fake_https: type[_FakeConn], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(decide.SECRET_ENV, "s3cr3t-value")
    conn = decide.KeepAlive()
    for _ in range(2):
        headers = {"CF-Access-Client-Id": "i"}
        reply = conn.exchange("GET", "https://h/api/plate?top=3", headers, None, 1.0)
        assert reply == HttpReply(200, "application/json", b"{}")
    assert len(fake_https.made) == 1
    method, path, headers = fake_https.made[0].requests[0]
    assert (method, path) == ("GET", "/api/plate?top=3")
    assert headers["CF-Access-Client-Secret"] == "s3cr3t-value"
    conn.exchange("GET", "https://h/", {}, None, 1.0)
    assert "CF-Access-Client-Secret" not in fake_https.made[0].requests[-1][2]  # no client id


def test_keep_alive_retries_a_read_once_when_the_hub_closed_the_idle_connection(
    fake_https: type[_FakeConn],
) -> None:
    conn = decide.KeepAlive()
    conn.exchange("GET", "https://h/a", {}, None, 1.0)
    fake_https.errors.append(http.client.RemoteDisconnected("idle close"))
    assert conn.exchange("GET", "https://h/a", {}, None, 1.0).status == 200
    assert len(fake_https.made) == 2 and fake_https.made[0].closed
    fake_https.errors.append(http.client.RemoteDisconnected("idle close"))
    with pytest.raises(http.client.RemoteDisconnected):
        conn.exchange("POST", "https://h/a", {}, b"{}", 1.0)  # a mutation is never repeated


def test_keep_alive_replaces_a_stale_connection(fake_https: type[_FakeConn]) -> None:
    conn = decide.KeepAlive(stale_s=0.0)
    conn.exchange("GET", "https://h/a", {}, None, 1.0)
    conn.exchange("GET", "https://h/a", {}, None, 1.0)
    assert len(fake_https.made) == 2
    with pytest.raises(ValueError):
        conn.exchange("GET", "http://h/a", {}, None, 1.0)


def test_child_serve_answers_line_by_line_and_exits_when_idle(
    fake_https: type[_FakeConn],
) -> None:
    read_fd, write_fd = os.pipe()
    requests = [
        {"op": "connect", "url": "https://h", "timeout": 1.0},
        {"method": "GET", "url": "https://h/api/plate", "headers": {}, "timeout": 1.0},
        {"method": "GET", "url": "nonsense", "headers": {}},
    ]
    os.write(write_fd, "".join(json.dumps(r) + "\n" for r in requests).encode())
    out = io.StringIO()
    with os.fdopen(read_fd, "r") as stdin:
        assert decide.child_serve(stdin, out, idle_s=0.05) == 0  # nothing more: idle exit
    os.close(write_fd)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert replies[0] == {"ok": True}
    assert replies[1] == {"status": 200, "content_type": "application/json", "body": "{}"}
    assert replies[2]["status"] == 0 and replies[2]["ok"] is False
    assert len(fake_https.made) == 1  # connect + plate on one connection


def test_decide_tools_warm_the_client_at_call_start_and_close_it_at_the_end() -> None:
    events: list[str] = []

    class Client:
        def warm(self) -> bool:
            events.append("warm")
            return True

        def close(self) -> None:
            events.append("close")

    ctl = BridgeController(Authoriser(FakeScorer(), "albert"), vad_factory=FakeVad)
    desk = DecideDesk(Client())  # type: ignore[arg-type]  # only warm/close are used
    register(ctl, env={"MAC_VOICE_DECIDE": "1"}, desk=desk)
    ctl.begin_call()
    deadline = time.monotonic() + 2.0
    while "warm" not in events and time.monotonic() < deadline:
        time.sleep(0.01)
    ctl.end_call()
    assert events == ["warm", "close"]


def test_warm_gives_a_cold_child_more_time_and_a_request_waits_for_it() -> None:
    spawn = _Spawner(_OK)
    transport = _transport(spawn)
    cold = spawn.__call__

    def slow_start(argv: list[str], **kw: Any) -> _FakeChild:
        child = cold(argv, **kw)
        child.first_delay = 0.7  # longer than start_s (0.5), shorter than warm_s (1.0)
        return child

    transport.popen = slow_start
    warmer = threading.Thread(target=transport.warm, args=("https://h", 0.05))
    warmer.start()
    time.sleep(0.1)  # the plate request arrives while the warm-up runs
    reply = transport("GET", "https://h/api/plate", {}, None, 0.05)
    warmer.join()
    assert reply.status == 200 and len(spawn.children) == 1


def test_warm_on_a_mute_child_gives_up_and_never_raises() -> None:
    spawn = _Spawner(None)
    transport = _transport(spawn)
    assert transport.warm("https://h", 0.01) is False
    assert spawn.children[0].killed and not transport.running
