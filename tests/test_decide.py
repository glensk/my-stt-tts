"""The decide hub by voice: client, spoken summaries, choices, the desk, both front ends.

Everything runs against :class:`FakeHub`, an in-memory stand-in for the hub's JSON API
(``/api/plate``, ``/api/decisions/{id}``, ``…/answer``, ``…/undo``). No test reaches the
real hub, and nothing here is real decision content.
"""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from my_stt_tts import decide, decide_intents
from my_stt_tts.__main__ import maybe_handle_decide
from my_stt_tts.brain import Brain
from my_stt_tts.bridge import Authoriser, BridgeController
from my_stt_tts.config import Config
from my_stt_tts.decide import (
    Choice,
    DecideDesk,
    HttpReply,
    HubClient,
    HubConfig,
    HubError,
    PlateContext,
    Proposal,
    Snoozes,
    plate_summary,
    resolve_choice,
)
from my_stt_tts.decide_intents import LocalDecide
from my_stt_tts.decide_tools import EXECUTOR, register

CLIENT_ID = "test-client.access"
NOW = datetime(2026, 10, 23, 15, 0, tzinfo=UTC)
FRAME = 1600
ALBERT = 0.5

D1: dict[str, Any] = {
    "id": "session:aaaa1111",
    "source": "session",
    "kind": "question",
    "title": "Which storage engine should the example service use? It matters for demos.",
    "background": "The example service stores a few hundred records. Demos must start fast.",
    "goal": "Make the example service start in seconds.",
    "fit": "This picks the storage layer; it unblocks the first demo.",
    "options": [
        {"label": "Embedded file", "description": "No extra process; fine for one user."},
        {"label": "Local server", "description": "Needs a running daemon."},
        {"label": "In-memory only", "description": "All data is lost on restart."},
    ],
    "recommendation": [0],
    "recommendation_why": "It needs no extra process.",
    "multi_select": False,
    "allow_other": True,
    "asker": {"name": "example session", "account": "cpriv", "machine": "mac"},
    "project": {"area": "private", "sub": "infra/example", "guessed": False, "via": "cwd"},
    "view": {"project_key": "infra/example"},
    "state": "open",
}
D2: dict[str, Any] = {
    "id": "tp:bbbb2222",
    "source": "tp",
    "kind": "question",
    "title": "Should the nightly report also go to the team channel?",
    "background": "The nightly report goes to one inbox today.",
    "goal": "",
    "fit": "",
    "options": [],
    "recommendation": [],
    "recommendation_why": "",
    "multi_select": False,
    "allow_other": True,
    "asker": {"name": "tp#47", "account": "cwork", "machine": "mac"},
    "project": {"area": "work", "sub": "sdsc/example", "guessed": False, "via": "tp"},
    "view": {"project_key": "sdsc/example"},
    "state": "open",
}
D3: dict[str, Any] = {
    "id": "action:cccc3333",
    "source": "action",
    "kind": "action",
    "title": "Plug in the backup drive.",
    "background": "",
    "goal": "",
    "fit": "",
    "options": [],
    "recommendation": [],
    "recommendation_why": "",
    "multi_select": False,
    "allow_other": False,
    "asker": {"name": "\\~", "account": "cpriv", "machine": "mac"},
    "project": {"area": "private", "sub": None, "guessed": True, "via": "account"},
    "view": {"project_key": "unfiled:private"},
    "state": "open",
}


def _plate_item(rank: int, d: Mapping[str, Any]) -> dict[str, Any]:
    brief = " ".join(
        f"{k}: {d[f]}" for k, f in (("Background", "background"), ("Building", "goal"),
                                    ("Fit", "fit")) if d[f]
    )  # fmt: skip
    return {
        "id": d["id"],
        "rank": rank,
        "title": d["title"],
        "source": d["source"],
        "kind": d["kind"],
        "project": d["view"]["project_key"],
        "asker": d["asker"]["name"],
        "options": [
            {"n": i + 1, "label": o["label"], "recommended": i in d["recommendation"]}
            for i, o in enumerate(d["options"])
        ],
        "recommendation_why": d["recommendation_why"],
        "brief": brief,
        "answerable": True,
        "why_now": ["blocking a live session 12 min"],
    }


class FakeHub:
    """The hub's JSON API in memory; records every request."""

    def __init__(self, decisions: list[dict[str, Any]] | None = None) -> None:
        self.decisions = {d["id"]: json.loads(json.dumps(d)) for d in decisions or [D1, D2, D3]}
        self.calls: list[tuple[str, str, dict[str, str], Any]] = []
        self.fail: tuple[int, str] | None = None

    # pylint: disable-next=too-many-positional-arguments,too-many-return-statements
    def __call__(
        self, method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float
    ) -> HttpReply:
        del timeout
        parts = urllib.parse.urlsplit(url)
        payload = json.loads(body) if body else None
        self.calls.append((method, parts.path + ("?" + parts.query if parts.query else ""),
                           dict(headers), payload))  # fmt: skip
        if headers.get("CF-Access-Client-Id") != CLIENT_ID:
            return HttpReply(403, "text/html", b"<html>Access</html>")
        if self.fail is not None:
            status, code = self.fail
            return self._json(status, {"detail": {"error": code, "message": code}})
        path = urllib.parse.unquote(parts.path)
        if path == "/api/plate":
            return self._json(200, self._plate(urllib.parse.parse_qs(parts.query)))
        if path.startswith("/api/decisions/"):
            rest = path.removeprefix("/api/decisions/")
            did, _, verb = rest.partition("/")
            d = self.decisions.get(did)
            if d is None:
                return self._json(404, {"detail": {"error": "unknown_decision"}})
            if method == "GET" and not verb:
                return self._json(200, d)
            if method == "POST" and verb == "answer":
                return self._answer(d, payload or {}, headers)
            if method == "POST" and verb == "undo":
                if d["state"] != "queued":
                    return self._json(409, {"detail": {"error": "nothing_to_undo"}})
                d["state"], d["answer"] = "open", None
                return self._json(200, d)
        return self._json(404, {"detail": {"error": "not_found"}})

    def _plate(self, query: Mapping[str, list[str]]) -> dict[str, Any]:
        focus = (query.get("focus") or [""])[0] or None
        open_ = [d for d in self.decisions.values() if d["state"] == "open"]
        if focus:
            open_ = [d for d in open_ if d["project"]["area"] == focus]
        top = int((query.get("top") or ["3"])[0])

        def section(*sources: str) -> int:  # the hub's overlapping section counts
            return sum(1 for d in open_ if d["source"] in sources)

        return {
            "schema_version": 1,
            "focus": focus,
            "top": [_plate_item(i, d) for i, d in enumerate(open_[:top], start=1)],
            "counts": {"inbox": len(open_), "code": section("session", "action"),
                       "agents": section("staff", "tpy"), "tickets": section("tp", "tpy"),
                       "flagged": 0, "overdue": 1, "due_soon": 0, "snoozed": 0},
            "urgent": [],
            "blocked_sessions": [{"session_id": "s1", "name": "example session"}],
        }  # fmt: skip

    def _answer(self, d: dict[str, Any], body: dict[str, Any], headers: Mapping[str, str]) -> Any:
        if headers.get("Origin") != "https://decide.dom42.space":
            return self._json(403, {"error": "bad_origin"})
        if body.get("channel") != "voice":
            return self._json(403, {"detail": {"error": "channel_not_allowed"}})
        if d["state"] != "open":
            return self._json(409, {"detail": {"error": "not_open"}})
        d["state"] = "queued"
        d["answer"] = {**body, "release_at": "2026-10-23T15:00:05Z"}
        return self._json(200, d)

    @staticmethod
    def _json(status: int, data: Any) -> HttpReply:
        return HttpReply(status, "application/json", json.dumps(data).encode())

    def posts(self) -> list[tuple[str, Any]]:
        return [(p, b) for m, p, _, b in self.calls if m == "POST"]


def _client(hub: FakeHub) -> HubClient:
    return HubClient(HubConfig(client_id=CLIENT_ID), hub)


def _desk(hub: FakeHub, tmp_path: Path) -> DecideDesk:
    return DecideDesk(_client(hub), snoozes=Snoozes(tmp_path / "s.json", clock=lambda: NOW),
                      clock=lambda: NOW)  # fmt: skip


# -- client -----------------------------------------------------------------------------------
def test_config_needs_https_and_reads_the_client_id() -> None:
    cfg = HubConfig.from_env({"DECIDE_VOICE_CF_CLIENT_ID": " abc.access "})
    assert (cfg.url, cfg.client_id, cfg.origin) == (
        "https://decide.dom42.space",
        "abc.access",
        "https://decide.dom42.space",
    )
    assert HubConfig.from_env({"DECIDE_CLI_CF_CLIENT_ID": "cli"}).client_id == "cli"
    with pytest.raises(ValueError):
        HubConfig.from_env({"DECIDE_URL": "http://decide.dom42.space"})


def test_no_client_without_a_client_id() -> None:
    assert decide.build_client({}) is None


def test_transport_choice() -> None:
    assert decide.make_transport({decide.SECRET_ENV: "x"}) is decide.urllib_exchange
    chosen = decide.make_transport({decide.SECRET_RUN_ENV: "/opt/secret-run"})
    assert isinstance(chosen, decide.SecretRunTransport) and chosen.secret_run == "/opt/secret-run"


def test_reads_send_the_token_id_and_answers_add_origin_and_voice() -> None:
    hub = FakeHub()
    client = _client(hub)
    client.plate("work", top=3)
    client.answer(D1["id"], {"choices": [1], "comment": ""})
    assert len(hub.calls) == 2
    m1, p1, h1, _ = hub.calls[0]
    m2, p2, h2, b2 = hub.calls[1]
    assert (m1, p1) == ("GET", "/api/plate?top=3&focus=work")
    assert "Origin" not in h1 and h1["CF-Access-Client-Id"] == CLIENT_ID
    assert (m2, p2) == ("POST", "/api/decisions/session:aaaa1111/answer")
    assert h2["Origin"] == "https://decide.dom42.space"
    assert h2["Content-Type"] == "application/json"
    assert b2 == {"choices": [1], "comment": "", "channel": "voice"}
    assert not any("Secret" in k for _, _, h, _ in hub.calls for k in h)


def test_reads_retry_once_mutations_never() -> None:
    hub = FakeHub()
    calls = {"n": 0}

    def flaky(*args: Any) -> HttpReply:
        calls["n"] += 1
        if calls["n"] == 1:
            return HttpReply(0, "", b"URLError")
        return hub(*args)

    client = HubClient(HubConfig(client_id=CLIENT_ID), flaky)
    assert client.plate()["counts"]["inbox"] == 3 and calls["n"] == 2
    calls["n"] = 0
    with pytest.raises(HubError):
        client.answer(D1["id"], {"choices": [0], "comment": ""})
    assert calls["n"] == 1 and not hub.posts()


def test_hub_errors_become_speakable_reasons() -> None:
    hub = FakeHub()
    hub.fail = (409, "undo_too_late")
    with pytest.raises(HubError) as err:
        _client(hub).undo(D1["id"])
    assert err.value.code == "undo_too_late" and "too late" in err.value.reason
    bad = HubClient(HubConfig(client_id="other"), hub)
    with pytest.raises(HubError) as edge:
        bad.plate()
    assert edge.value.code == "edge" and "Cloudflare" in edge.value.reason


def test_child_fetch_adds_the_secret_from_its_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, Any] = {}

    def fake_exchange(*args: Any) -> HttpReply:
        sent.update(args[2])
        return HttpReply(200, "application/json", b"{}")

    monkeypatch.setattr(decide, "urllib_exchange", fake_exchange)
    out = json.loads(
        decide.child_fetch(json.dumps({"method": "GET", "url": "https://x/", "headers": {}}))
    )
    assert out == {"status": 200, "content_type": "application/json", "body": "{}"}


def test_urllib_exchange_adds_the_secret_only_with_a_client_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, str]] = []

    class Resp:
        status = 200
        headers = {"content-type": "application/json"}

        def read(self) -> bytes:
            return b"{}"

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *a: object) -> None:
            return None

    def fake_urlopen(req: Any, **_: Any) -> Any:
        captured.append({k.lower(): v for k, v in req.header_items()})
        return Resp()

    monkeypatch.setenv(decide.SECRET_ENV, "s3cr3t-value")
    monkeypatch.setattr(decide.urllib.request, "urlopen", fake_urlopen)
    decide.urllib_exchange("GET", "https://x/", {"CF-Access-Client-Id": "id"}, None, 1.0)
    decide.urllib_exchange("GET", "https://x/", {}, None, 1.0)
    assert captured[0]["cf-access-client-secret"] == "s3cr3t-value"
    assert "cf-access-client-secret" not in captured[1]


# -- spoken text ------------------------------------------------------------------------------
def test_plate_summary_counts_sections_then_top_three_with_context() -> None:
    plate = FakeHub()._plate({})
    text = plate_summary(plate)
    assert text.startswith("3 decisions wait for you: 2 from Claude sessions and 1 tp item.")
    assert "1 is overdue." in text and "1 session is blocked until you answer." in text
    assert (
        "First, Example session asks: Which storage engine should the example service use?"
        not in text
    )  # the asker keeps its own case
    assert (
        "First, example session asks: Which storage engine should the example service use?" in text
    )
    assert "This picks the storage layer; it unblocks the first demo." in text
    assert "Second, tp 47 asks: Should the nightly report also go to the team channel?" in text
    assert "Third, a Claude session" not in text  # D3 is an action, not a session
    assert "Third, the question: Plug in the backup drive." in text
    assert text.endswith("Say next decision to go through them.")
    assert "#" not in text


def test_plate_summary_empty_and_focus() -> None:
    assert plate_summary({"counts": {"inbox": 0}}, "work") == (
        "No work decisions wait for you on the decide hub."
    )
    assert plate_summary({"counts": {"inbox": 0}}) == "No decisions wait for you on the decide hub."


def test_plate_buckets_are_disjoint_and_add_up() -> None:
    # The live plate that read as 80: t.py approvals sit in both agents and tickets.
    counts = {"inbox": 42, "code": 1, "agents": 41, "tickets": 38}
    assert decide.plate_buckets(counts) == {"session": 1, "tpy": 38, "staff": 3, "tp": 0}
    assert sum(decide.plate_buckets(counts).values()) == 42
    both = {"inbox": 9, "code": 2, "agents": 4, "tickets": 5}  # 2 t.py, 2 staff, 3 tp
    assert decide.plate_buckets(both) == {"session": 2, "tpy": 2, "staff": 2, "tp": 3}
    # Inconsistent or missing numbers never go negative.
    assert decide.plate_buckets({"inbox": 1, "agents": 3, "tickets": 2}) == {
        "session": 0, "tpy": 2, "staff": 1, "tp": 0,
    }  # fmt: skip
    assert decide.plate_buckets({"inbox": "x"}) == {"session": 0, "tpy": 0, "staff": 0, "tp": 0}


def test_plate_summary_parts_add_up_and_read_naturally() -> None:
    counts = {"inbox": 42, "code": 1, "agents": 41, "tickets": 38}
    assert plate_summary({"counts": counts}) == (
        "42 decisions wait for you: 1 from a Claude session, 38 support tickets and 3 from "
        "staff agents."
    )
    work = {"inbox": 38, "code": 0, "agents": 38, "tickets": 38}
    assert plate_summary({"counts": work}, "work") == (
        "38 work decisions wait for you, all support tickets."
    )
    one = {"inbox": 1, "code": 1, "agents": 0, "tickets": 0}
    assert plate_summary({"counts": one}, "private") == (
        "1 private decision waits for you, from a Claude session."
    )
    assert plate_summary({"counts": {"inbox": 1, "tickets": 1}}) == (
        "1 decision waits for you, a tp item."
    )
    mixed = {"inbox": 5, "code": 0, "agents": 2, "tickets": 4}  # 1 t.py, 1 staff, 3 tp
    assert plate_summary({"counts": mixed}) == (
        "5 decisions wait for you: 1 support ticket, 1 from a staff agent and 3 tp items."
    )
    assert plate_summary({"counts": {"inbox": 2}}) == "2 decisions wait for you."


def test_short_and_full_brief() -> None:
    short = decide.short_brief(D1)
    assert short.startswith("Example session asks (example): Which storage engine")
    assert "Options: 1, Embedded file; 2, Local server; 3, In-memory only." in short
    assert "It recommends option 1, Embedded file, because it needs no extra process." in short
    full = decide.full_brief(D1)
    assert "Background: The example service stores a few hundred records." in full
    assert "Building: Make the example service start in seconds." in full
    assert "1, Embedded file — No extra process; fine for one user." in full
    assert "say done" in decide.short_brief(D3)
    assert "say answer, then your words" in decide.short_brief(D2)


def test_speakable_drops_markdown_links_and_secrets() -> None:
    secret = "ghp_" + "A" * 30
    text = decide.speakable(f"Use `tp#12` see https://x.example/y and **{secret}**")
    assert text == "Use tp 12 see a link and [redacted]"


# -- choices ----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("said", "want"),
    [
        ("option two", Choice(choices=(1,))),
        ("the second one", Choice(choices=(1,))),
        ("2", Choice(choices=(1,))),
        ("local server", Choice(choices=(1,))),
        ("the last option", Choice(choices=(2,))),
        ("go with the recommendation", Choice(accept_recommendation=True)),
        ("what you recommended", Choice(accept_recommendation=True)),
    ],
)
def test_resolve_choice(said: str, want: Choice) -> None:
    assert resolve_choice(D1, said) == want


def test_resolve_choice_refusals() -> None:
    assert resolve_choice(D1, "option five") == "There are only 3 options."
    assert "Only one option" in str(resolve_choice(D1, "options one and two"))
    multi = {**D1, "multi_select": True}
    assert resolve_choice(multi, "options one and three") == Choice(choices=(0, 2))
    assert "no recommendation" in str(resolve_choice(D2, "the recommendation"))
    assert resolve_choice(D2, "", free_text="Yes, but only on weekdays") == Choice(
        other_text="Yes, but only on weekdays"
    )
    assert resolve_choice(D3, "done") == Choice(done=True)
    assert "step for you" in str(resolve_choice(D3, "option one"))
    no_free = {**D1, "allow_other": False}
    assert "no free text" in str(resolve_choice(no_free, "", free_text="something else"))


def test_proposal_body_and_readback() -> None:
    p = Proposal("x", "Pick a storage engine.", Choice(choices=(1,)), ("Local server",))
    assert p.body() == {"comment": "", "choices": [1]}
    assert p.readback("yes, do it") == (
        'I will answer "Pick a storage engine." with option 2, Local server.'
        " Say yes, do it to send it, or no to cancel."
    )
    c = p.with_comment("only for the demo")
    assert c.body()["comment"] == "only for the demo"
    assert 'with the comment "only for the demo"' in c.readback("yes")
    assert Proposal.from_args(c.as_args()) == c
    assert Proposal("x", "t", Choice(accept_recommendation=True)).body() == {
        "comment": "",
        "accept_recommendation": True,
    }
    assert Proposal("x", "t", Choice(done=True)).body() == {"comment": "", "done": True}


# -- the desk ---------------------------------------------------------------------------------
def test_desk_next_prepare_send_undo(tmp_path: Path) -> None:
    hub = FakeHub()
    desk = _desk(hub, tmp_path)
    text = desk.next_decision()
    assert text.startswith("2 more after this one. Example session asks")
    proposal = desk.prepare("option two", comment="for now")
    assert isinstance(proposal, Proposal) and not hub.posts()  # prepare never sends
    sent = desk.send(proposal)
    assert sent == "Sent. It waits 5 seconds in the hub — say undo to take it back."
    assert hub.posts() == [
        ("/api/decisions/session:aaaa1111/answer",
         {"comment": "for now", "choices": [1], "channel": "voice"}),
    ]  # fmt: skip
    assert desk.undo() == 'Taken back: "Which storage engine should the example service use?"' \
        " is open again."  # fmt: skip
    assert hub.decisions[D1["id"]]["state"] == "open"
    assert desk.undo() == "There is no voice answer to take back."


def test_desk_without_a_decision_asks_first(tmp_path: Path) -> None:
    desk = _desk(FakeHub(), tmp_path)
    assert desk.prepare("option one") == "Which decision? Say next decision first."
    assert desk.snooze() == "Which decision? Say next decision first."


def test_desk_skip_snooze_and_focus(tmp_path: Path) -> None:
    hub = FakeHub()
    desk = _desk(hub, tmp_path)
    desk.next_decision()
    assert "Tp 47 asks" in desk.skip()
    snoozed = desk.snooze()
    assert snoozed.startswith("Snoozed until tomorrow at 08:00 (for voice only).")
    data = json.loads((tmp_path / "s.json").read_text())
    assert D2["id"] in data["snoozed"]
    assert oct((tmp_path / "s.json").stat().st_mode & 0o777) == "0o600"
    fresh = _desk(hub, tmp_path)
    assert all(t["id"] != D2["id"] for t in fresh.plate()["top"])  # snooze survives
    assert fresh.set_focus("work") == "Focus: work only. "
    assert fresh.summary().startswith("1 work decision waits for you, ")
    assert hub.calls[-1][1].endswith("focus=work")
    assert fresh.set_focus("everything") == "Focus: everything. "
    assert "work, private, or everything" in fresh.set_focus("hobby")


def test_snoozes_expire(tmp_path: Path) -> None:
    now = [NOW]
    snoozes = Snoozes(tmp_path / "s.json", clock=lambda: now[0])
    snoozes.add("a", NOW + timedelta(hours=1))
    assert snoozes.active() == {"a"}
    now[0] = NOW + timedelta(hours=2)
    assert snoozes.active() == set()


def test_desk_speaks_hub_failures(tmp_path: Path) -> None:
    hub = FakeHub()
    desk = _desk(hub, tmp_path)
    desk.next_decision()
    proposal = desk.prepare("option one")
    assert isinstance(proposal, Proposal)
    hub.fail = (409, "not_open")
    assert desk.send(proposal) == "Not sent: it is no longer open."
    assert DecideDesk(None).summary() == (
        "Cannot read your decisions: the decide hub is not set up for voice."
    )


def test_tomorrow_morning_is_zurich_eight() -> None:
    assert decide.tomorrow_morning(NOW).isoformat() == "2026-10-24T08:00:00+02:00"


# -- brain context ----------------------------------------------------------------------------
def test_plate_context_only_for_questions_about_decisions() -> None:
    hub = FakeHub()
    ticks = [0.0]
    ctx = PlateContext(lambda: _client(hub), ttl_s=60, clock=lambda: ticks[0])
    assert ctx.for_text("what is the weather") == ""
    line = ctx.for_text("Which decisions are most urgent?")
    assert line.startswith("Albert's open decisions (decide hub, read-only context): 3")
    ctx.for_text("and my decisions again?")
    assert len(hub.calls) == 1  # cached
    assert CLIENT_ID not in line


def test_brain_system_prompt_gets_the_plate_for_decision_questions() -> None:
    brain = Brain(Config(decide_enabled=True, tools_enabled=False))
    assert brain._decide_ctx is not None  # noqa: SLF001
    brain._decide_ctx = PlateContext(lambda: _client(FakeHub()))  # noqa: SLF001
    brain._decide_line = brain._decide_ctx.for_text("what decisions are open?")  # noqa: SLF001
    assert "Albert's open decisions" in brain._system_prompt()  # noqa: SLF001
    plain = Brain(Config(tools_enabled=False))
    assert plain._decide_ctx is None  # noqa: SLF001
    assert "decide hub" not in plain._system_prompt()  # noqa: SLF001


# -- local intent router ----------------------------------------------------------------------
def _router(hub: FakeHub, tmp_path: Path, authorized: str | None = "albert") -> LocalDecide:
    ticks = [0.0]
    router = LocalDecide(_desk(hub, tmp_path), authorized, clock=lambda: ticks[0])
    router.ticks = ticks  # type: ignore[attr-defined]
    return router


def test_router_plate_next_more_answer_yes(tmp_path: Path) -> None:
    hub = FakeHub()
    r = _router(hub, tmp_path)
    assert r.handle("play some music") is None
    assert r.handle("next") is None  # bare "next" only while a decision is discussed
    assert str(r.handle("What's on my plate?")).startswith("3 decisions wait for you")
    assert "Example session asks" in str(r.handle("Next decision"))
    assert "Background:" in str(r.handle("tell me more"))
    readback = r.handle("Answer option two, please", "albert")
    assert readback == (
        'I will answer "Which storage engine should the example service use?" with option 2,'
        " Local server. Say yes to send it, or no to cancel."
    )
    assert not hub.posts()
    assert "comment" in str(r.handle("Add comment: only until the demo.", "albert"))
    assert r.handle("yes", "albert") == (
        "Sent. It waits 5 seconds in the hub — say undo to take it back."
    )
    assert hub.posts()[0][1] == {
        "comment": "only until the demo.",
        "choices": [1],
        "channel": "voice",
    }
    assert "open again" in str(r.handle("undo", "albert"))


def test_router_no_cancels_and_free_text_keeps_his_words(tmp_path: Path) -> None:
    hub = FakeHub([D2])
    r = _router(hub, tmp_path)
    r.handle("next decision")
    readback = r.handle("Answer: Yes, but only on weekdays.", "albert")
    assert 'your words "Yes, but only on weekdays."' in str(readback)
    assert r.handle("no", "albert") == "Cancelled — nothing was sent."
    assert r.handle("yes", "albert") is None  # nothing waits any more
    assert not hub.posts()


def test_router_takes_answers_only_from_the_authorised_voice(tmp_path: Path) -> None:
    hub = FakeHub()
    r = _router(hub, tmp_path)
    r.handle("next decision")
    assert r.handle("go with the recommendation", "guest") == (
        "I only take decision answers from Albert's voice."
    )
    assert r.handle("go with the recommendation", None) == (
        "I only take decision answers from Albert's voice."
    )
    assert "with the recommendation, Embedded file" in str(
        r.handle("go with the recommendation", "Albert")
    )
    assert r.handle("yes", "guest") == "I only take decision answers from Albert's voice."
    assert not hub.posts()
    off = _router(hub, tmp_path, authorized=None)
    off.handle("next decision")
    assert "DECIDE_VOICE_SPEAKER" in str(off.handle("option one", "albert"))


def test_router_pending_answer_expires(tmp_path: Path) -> None:
    hub = FakeHub()
    r = _router(hub, tmp_path)
    r.handle("next decision")
    r.handle("option one", "albert")
    r.ticks[0] += decide_intents.CONFIRM_TTL_S + 1  # type: ignore[attr-defined]
    assert r.handle("yes", "albert") is None
    assert not hub.posts()


def test_router_focus_and_skip(tmp_path: Path) -> None:
    hub = FakeHub()
    r = _router(hub, tmp_path)
    assert str(r.handle("show work only")).startswith("Focus: work only. 1 work decision waits")
    r.handle("next decision")
    assert "No more decisions" in str(r.handle("skip"))


def test_maybe_handle_decide_is_off_by_default() -> None:
    nothing: Any = None  # never reached: the router is off
    assert maybe_handle_decide(Config(), nothing, nothing, "on my plate", "albert") is False


# -- mac-voice client tools -------------------------------------------------------------------
class FakeVad:
    def is_speech(self, frame: np.ndarray) -> bool:
        return bool(np.max(np.abs(frame)) > 0.05)


class FakeScorer:
    def __init__(self, score: float = 0.52) -> None:
        self.score = score

    def score_against(self, audio: Any, name: str, *, timeout: float = 5.0) -> float | None:
        del audio, timeout
        return self.score if name == "albert" else None


class Rig:
    """A controller in a call + the decide tools on a fake hub."""

    def __init__(self, tmp_path: Path, score: float = 0.52) -> None:
        self.now = 1000.0
        self.ctl = BridgeController(
            Authoriser(FakeScorer(score), "albert"), vad_factory=FakeVad, clock=lambda: self.now
        )
        self.ctl.begin_call()
        self.hub = FakeHub()
        tools = register(self.ctl, env={"MAC_VOICE_DECIDE": "1"}, desk=_desk(self.hub, tmp_path))
        assert tools is not None
        self.seq = 0

    def say(self, text: str) -> None:
        for _ in range(10):
            self.now += 0.1
            self.ctl.feed_audio(np.full(FRAME, ALBERT, dtype=np.float32))
        for _ in range(6):
            self.now += 0.1
            self.ctl.feed_audio(np.zeros(FRAME, dtype=np.float32))
        self.now += 1.5
        self.seq += 1
        self.ctl.on_transcript(self.seq, text, self.now)

    def tool(self, name: str, *args: Any, **kwargs: Any) -> str:
        result = self.ctl.tools[name](*args, **kwargs)
        assert isinstance(result, str)
        self.ctl.note_tool_result(name, result)
        return result


def test_tools_register_only_with_the_flag() -> None:
    ctl = BridgeController(Authoriser(FakeScorer(), "albert"), vad_factory=FakeVad)
    assert register(ctl, env={}) is None
    assert set(ctl.tools) == {"confirm_action"}
    assert register(ctl, env={"MAC_VOICE_DECIDE": "1"}, desk=DecideDesk(None)) is not None
    assert {"decide_plate", "decide_next", "decide_answer", "decide_undo"} <= set(ctl.tools)
    assert EXECUTOR in ctl.executors
    assert "decide_focus" in ctl.own_outcome_tools


def test_tool_answer_reads_back_and_sends_only_after_yes_do_it(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.say("What is on my plate?")
    assert rig.tool("decide_plate").startswith("3 decisions wait for you")
    rig.say("Next decision.")
    assert "Example session asks" in rig.tool("decide_next")
    rig.say("Answer option two.")
    readback = rig.tool("decide_answer", choice="option two")
    assert readback.endswith("Say yes, do it to send it, or no to cancel.")
    assert not rig.hub.posts()
    rig.say("Add the comment only for the demo.")
    assert 'with the comment "only for the demo"' in rig.tool(
        "decide_comment", text="only for the demo"
    )
    rig.say("Yes, do it.")
    assert rig.tool("confirm_action").startswith("Sent.")
    assert rig.hub.posts() == [
        ("/api/decisions/session:aaaa1111/answer",
         {"comment": "only for the demo", "choices": [1], "channel": "voice"}),
    ]  # fmt: skip
    rig.say("Undo that.")
    assert "open again" in rig.tool("decide_undo")


def test_tool_answer_refuses_words_albert_did_not_say(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.say("Next decision.")
    rig.tool("decide_next")
    rig.say("Answer option two.")
    assert rig.tool("decide_answer", choice="option three").startswith("refused:")
    rig.say("Answer option two.")
    assert rig.tool("decide_answer", choice="option two", comment="ship it today").startswith(
        "refused:"
    )
    assert not rig.ctl.pending.live()


def test_tool_answer_refuses_another_voice(tmp_path: Path) -> None:
    rig = Rig(tmp_path, score=0.05)
    rig.say("Next decision.")
    rig.tool("decide_next")
    rig.say("Answer option two.")
    assert rig.tool("decide_answer", choice="option two").startswith("refused:")
    assert not rig.ctl.pending.live() and not rig.hub.posts()


def test_tool_no_drops_the_answer(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.say("Next decision.")
    rig.tool("decide_next")
    rig.say("Go with the recommendation.")
    assert "the recommendation, Embedded file" in rig.tool("decide_answer", choice="recommendation")
    rig.say("No, cancel.")
    assert rig.tool("confirm_action").startswith(("cancelled", "refused"))
    assert not rig.hub.posts()


def test_new_call_forgets_the_decision_in_discussion(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.say("Next decision.")
    rig.tool("decide_next")
    rig.ctl.end_call()
    rig.ctl.begin_call()
    rig.seq = 0
    rig.say("Answer option two.")
    assert rig.tool("decide_answer", choice="option two") == (
        "Which decision? Say next decision first."
    )
