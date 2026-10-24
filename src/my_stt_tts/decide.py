# pylint: disable=too-many-lines  # one cohesive client: transport, speech, choices, desk, CLI
"""Albert's decision hub (``decide.dom42.space``) by voice: what is on his plate, briefs, answers.

The hub (repo ``infra/decide``, API contract ``docs/api.md``) holds every decision that waits
on Albert: questions of Claude Code sessions, staff agents, tp items and t.py tickets. This
module is the voice client of its JSON API:

* ``GET /api/plate?top=&focus=`` — counts per section, the top decisions, urgent items;
* ``GET /api/decisions/{id}`` — one decision with its brief (background, building, fit),
  options and recommendation;
* ``POST /api/decisions/{id}/answer`` with ``channel: voice`` — an option index, the
  recommendation, free text or "done" (an action step), plus a comment;
* ``POST /api/decisions/{id}/undo`` — withdraws the answer during the hub's answer grace
  (``DECIDE_ANSWER_GRACE_S``, currently 3 s; :meth:`DecideDesk.send` reads the window from the
  answer's ``release_at``). An "undo" spoken after "yes, do it" often comes too late: the
  read-back before the yes is the real safety.

Auth: the ``decide-cli`` Cloudflare Access service token. The hub accepts ``channel: voice``
only from a service token on its voice list, so every call goes through Cloudflare — also at
home, never the direct LAN path. The client id comes from ``DECIDE_VOICE_CF_CLIENT_ID`` (or
the decide CLI's ``DECIDE_CLI_CF_CLIENT_ID``). The secret is never read into this process
when it is not already there: without ``DECIDE_CLI_CF_CLIENT_SECRET`` in the environment each
request runs in a short child under ``secret-run -e
DECIDE_CLI_CF_CLIENT_SECRET=decide-cli-cf-service-token`` (``python -m my_stt_tts.decide -X``,
request on stdin, response on stdout), so the value only lives in that child. Nothing here
prints, logs or returns it.

:class:`DecideDesk` keeps the conversation state that both voice front ends share (the mac-voice
client tools in :mod:`my_stt_tts.decide_tools`, the local intent router in
:mod:`my_stt_tts.decide_intents`): the focus (work / private / all), the queue of the last
plate, the decision being discussed, local skips and snoozes, and the last sent answer for
undo. It never sends an answer by itself: :meth:`DecideDesk.prepare` builds a
:class:`Proposal` with the read-back text, and only :meth:`DecideDesk.send` posts it — the
callers send only after Albert's spoken yes.

Voice snoozes stay in this client (``~/.local/state/mac-voice/decide-voice.json``): the hub
lets a service token answer and undo, nothing else.

CLI (read-only)::

    python -m my_stt_tts.decide -p          # the spoken plate summary
    python -m my_stt_tts.decide -p -F work  # ... for work only
    python -m my_stt_tts.decide -n          # the brief of the top decision, as spoken
    python -m my_stt_tts.decide -s ID       # the brief of one decision
    python -m my_stt_tts.decide -p -j       # the raw plate JSON
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import shutil
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .bridge_text import normalise, numbers_in, redact

log = logging.getLogger("my_stt_tts.decide")

DEFAULT_URL = "https://decide.dom42.space"
URL_ENV = "DECIDE_URL"
CLIENT_ID_ENVS = ("DECIDE_VOICE_CF_CLIENT_ID", "DECIDE_CLI_CF_CLIENT_ID")
SECRET_ENV = "DECIDE_CLI_CF_CLIENT_SECRET"
#: Vaultwarden item of the ``decide-cli`` service token (``secret-run`` injects it).
SECRET_ITEM = "decide-cli-cf-service-token"
SECRET_ITEM_ENV = "DECIDE_VOICE_SECRET_ITEM"
USER_AGENT = "my-stt-tts-voice/1"
TIMEOUT_S = 8.0
CHILD_TIMEOUT_S = 5.0  # secret-run + interpreter start on top of the request timeout
SECRET_RUN_ENV = "DECIDE_SECRET_RUN_BIN"
#: Where secret-run lives when launchd's PATH does not have it (the browser-login repo).
SECRET_RUN_FALLBACK = Path.home() / "obsidian" / "42-Git" / "home" / "browser-login" / "bin"
STATE_DIR = Path.home() / ".local" / "state" / "mac-voice"
STATE_FILE = "decide-voice.json"
ZURICH = ZoneInfo("Europe/Zurich")
TOP = 3
FOCUS_SCOPES = ("work", "private")
COMMENT_MAX = 2000
OTHER_MAX = 2000
SAY_TITLE = 160
SAY_CONTEXT = 200
SAY_OPTION = 120
SAY_WHY = 300
BRIEF_PART = 500

#: Speakable reasons for the hub's error codes (docs/api.md).
REASONS = {
    "not_open": "it is no longer open",
    "readonly": "it cannot be answered from here",
    "unknown_decision": "the hub no longer has it",
    "invalid_answer": "the hub did not accept that answer",
    "no_recommendation": "it has no recommendation",
    "nothing_to_undo": "there is no answer waiting to be taken back",
    "undo_too_late": "too late, the relay has already picked it up",
    "channel_not_allowed": "the hub did not see the voice token (the request did not go"
    " through Cloudflare)",
    "service_token_read_only": "the voice token may not do that",
    "bad_origin": "the hub refused the request origin",
    "invalid_focus": "the hub does not know that focus",
    "unreachable": "the hub is not reachable",
    "not_configured": "the decide hub is not set up for voice",
    "edge": "Cloudflare Access refused the voice token",
}


class HubError(Exception):
    """The hub refused (status ≥ 400), Cloudflare blocked, or nothing answered (status 0)."""

    def __init__(self, status: int, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.status = status
        self.code = code

    @property
    def reason(self) -> str:
        """Short and speakable."""
        return REASONS.get(self.code, f"the hub answered {self.status or 'nothing'}")


# -- transport --------------------------------------------------------------------------------
@dataclass(frozen=True)
class HttpReply:
    """What one HTTP exchange returned."""

    status: int
    content_type: str
    body: bytes


Transport = Callable[[str, str, Mapping[str, str], bytes | None, float], HttpReply]


def urllib_exchange(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float
) -> HttpReply:
    """One HTTPS exchange with the stdlib; adds the service-token secret from the environment."""
    sent = dict(headers)
    secret = os.environ.get(SECRET_ENV, "")
    if secret and "CF-Access-Client-Id" in sent:
        sent["CF-Access-Client-Secret"] = secret
    req = urllib.request.Request(url, data=body, headers=sent, method=method)
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:  # noqa: S310
            return HttpReply(resp.status, resp.headers.get("content-type", ""), resp.read())
    except urllib.error.HTTPError as exc:
        return HttpReply(exc.code, exc.headers.get("content-type", ""), exc.read())


Runner = Callable[..., subprocess.CompletedProcess[str]]


class SecretRunTransport:
    """Each exchange in a child under ``secret-run``: the secret stays in the child."""

    def __init__(
        self,
        secret_run: str,
        item: str = SECRET_ITEM,
        runner: Runner = subprocess.run,
        python: str = sys.executable,
    ) -> None:
        self.secret_run = secret_run
        self.item = item
        self.runner = runner
        self.python = python

    def __call__(  # pylint: disable=too-many-positional-arguments  # the Transport signature
        self, method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float
    ) -> HttpReply:
        request = {
            "method": method,
            "url": url,
            "headers": dict(headers),
            "body": body.decode() if body is not None else None,
            "timeout": timeout,
        }
        argv = [
            self.secret_run,
            "-e",
            f"{SECRET_ENV}={self.item}",
            "--",
            self.python,
            "-I",
            "-m",
            "my_stt_tts.decide",
            "-X",
        ]
        try:
            done = self.runner(
                argv,
                input=json.dumps(request),
                capture_output=True,
                text=True,
                timeout=timeout + CHILD_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise HubError(0, "unreachable", type(exc).__name__) from None
        try:
            reply = json.loads(done.stdout)
            return HttpReply(
                int(reply["status"]), str(reply["content_type"]), reply["body"].encode()
            )
        except (ValueError, KeyError, TypeError, AttributeError):
            log.warning("⚠️  decide: secret-run child failed (exit %s)", done.returncode)
            raise HubError(
                0, "unreachable", f"the secret-run child gave no answer (exit {done.returncode})"
            ) from None


def child_fetch(stdin: str) -> str:
    """``-X``: run one exchange (request JSON on stdin) and return the reply JSON."""
    req = json.loads(stdin)
    body = req.get("body")
    try:
        reply = urllib_exchange(
            str(req["method"]),
            str(req["url"]),
            dict(req.get("headers") or {}),
            body.encode() if isinstance(body, str) else None,
            float(req.get("timeout") or TIMEOUT_S),
        )
    except (urllib.error.URLError, OSError) as exc:
        return json.dumps({"status": 0, "content_type": "", "body": type(exc).__name__})
    text = reply.body.decode("utf-8", errors="replace")
    return json.dumps({"status": reply.status, "content_type": reply.content_type, "body": text})


# -- the hub client ---------------------------------------------------------------------------
@dataclass
class HubConfig:
    """Where the hub is and which service token speaks for the voice client."""

    url: str = DEFAULT_URL
    client_id: str = ""
    secret_item: str = SECRET_ITEM

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> HubConfig:
        env = os.environ if env is None else env
        url = (env.get(URL_ENV) or DEFAULT_URL).rstrip("/")
        if urllib.parse.urlsplit(url).scheme != "https":
            raise ValueError(f"refusing {url}: the voice client only talks https")
        client_id = next((env[k].strip() for k in CLIENT_ID_ENVS if env.get(k, "").strip()), "")
        return cls(
            url=url, client_id=client_id, secret_item=env.get(SECRET_ITEM_ENV) or SECRET_ITEM
        )

    @property
    def origin(self) -> str:
        parts = urllib.parse.urlsplit(self.url)
        return f"{parts.scheme}://{parts.netloc}"


def resolve_secret_run(env: Mapping[str, str] | None = None) -> str | None:
    """``$DECIDE_SECRET_RUN_BIN`` → ``$PATH`` → the browser-login repo's ``bin/secret-run``."""
    env = os.environ if env is None else env
    override = env.get(SECRET_RUN_ENV, "").strip()
    if override:
        return override
    found = shutil.which("secret-run")
    if found:
        return found
    fallback = SECRET_RUN_FALLBACK / "secret-run"
    return str(fallback) if fallback.is_file() else None


def make_transport(env: Mapping[str, str] | None = None) -> Transport | None:
    """The stdlib transport when the secret is in the environment, else ``secret-run``."""
    env = os.environ if env is None else env
    if env.get(SECRET_ENV):
        return urllib_exchange
    exe = resolve_secret_run(env)
    if exe is None:
        return None
    return SecretRunTransport(exe, env.get(SECRET_ITEM_ENV) or SECRET_ITEM)


class HubClient:
    """The hub's JSON API as the voice client uses it (through Cloudflare only)."""

    def __init__(self, cfg: HubConfig, transport: Transport, timeout: float = TIMEOUT_S) -> None:
        self.cfg = cfg
        self.transport = transport
        self.timeout = timeout

    def _headers(self, mutation: bool) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
            "CF-Access-Client-Id": self.cfg.client_id,
        }
        if mutation:
            headers["Origin"] = self.cfg.origin
            headers["Content-Type"] = "application/json"
        return headers

    def call(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> Any:
        """One JSON call; raises :class:`HubError`. A read that reached nothing is tried twice."""
        try:
            return self._call(method, path, body)
        except HubError as exc:
            if method != "GET" or exc.status != 0:
                raise  # a mutation is never repeated
            log.info("decide: %s unreachable once (%s), retrying", path.split("?")[0], exc)
            return self._call(method, path, body)

    def _call(self, method: str, path: str, body: Mapping[str, Any] | None) -> Any:
        mutation = method != "GET"
        payload = json.dumps(body or {}).encode() if mutation else None
        try:
            reply = self.transport(
                method, self.cfg.url + path, self._headers(mutation), payload, self.timeout
            )
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise HubError(0, "unreachable", type(exc).__name__) from None
        json_reply = "application/json" in reply.content_type
        if reply.status == 0:
            raise HubError(0, "unreachable", reply.body.decode(errors="replace")[:80])
        if reply.status in {401, 403} and not json_reply:
            raise HubError(reply.status, "edge")  # Cloudflare Access answers with HTML
        try:
            data = json.loads(reply.body) if json_reply else None
        except ValueError:
            data = None
        if reply.status >= 400:
            raise _hub_error(reply.status, data)
        if data is None:
            raise HubError(reply.status, "non_json", "the hub answered without JSON")
        return data

    def plate(self, focus: str | None = None, top: int = TOP) -> dict[str, Any]:
        query = {"top": str(top), **({"focus": focus} if focus else {})}
        return dict(self.call("GET", "/api/plate?" + urllib.parse.urlencode(query)))

    def decision(self, decision_id: str) -> dict[str, Any]:
        return dict(self.call("GET", "/api/decisions/" + urllib.parse.quote(decision_id, safe=":")))

    def answer(self, decision_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        path = f"/api/decisions/{urllib.parse.quote(decision_id, safe=':')}/answer"
        return dict(self.call("POST", path, {**body, "channel": "voice"}))

    def undo(self, decision_id: str) -> dict[str, Any]:
        path = f"/api/decisions/{urllib.parse.quote(decision_id, safe=':')}/undo"
        return dict(self.call("POST", path, {}))


def _hub_error(status: int, data: Any) -> HubError:
    if isinstance(data, Mapping):
        detail = data.get("detail", data)
        if isinstance(detail, Mapping):
            code = str(detail.get("error") or "http_error")
            return HubError(status, code, str(detail.get("message") or code))
    return HubError(status, "http_error")


def build_client(env: Mapping[str, str] | None = None) -> HubClient | None:
    """The client from the environment, or None when the voice token is not configured."""
    try:
        cfg = HubConfig.from_env(env)
    except ValueError:
        log.warning("⚠️  decide: %s is not https — decide by voice is off", URL_ENV)
        return None
    if not cfg.client_id:
        return None
    transport = make_transport(env)
    if transport is None:
        return None
    return HubClient(cfg, transport)


# -- speaking ---------------------------------------------------------------------------------
_MD = re.compile(r"\*\*|__|`|#+\s")
_URL = re.compile(r"https?://\S+")
_TICKET = re.compile(r"\b([A-Za-z]+)#(\d+)")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def speakable(text: object, cap: int = 1500) -> str:
    """Markdown and links out, token shapes redacted, one line, at most ``cap`` characters."""
    plain = _URL.sub("a link", _MD.sub("", str(text or "")))
    plain = _TICKET.sub(r"\1 \2", plain)  # "tp#1142" → "tp 1142" (not "hash")
    return redact(" ".join(plain.split()), cap)


def first_sentence(text: object, cap: int) -> str:
    """The first sentence of ``text`` (speakable), cut at ``cap`` on a word boundary."""
    plain = speakable(text, 4000)
    head = _SENTENCE.split(plain, maxsplit=1)[0] if plain else ""
    if len(head) <= cap:
        return head
    cut = head[:cap].rsplit(" ", 1)[0].rstrip(",;:")
    return cut + " …"


def _end(text: str) -> str:
    """``text`` with a closing full stop (no other change)."""
    text = text.strip()
    return text if not text or text[-1] in ".!?…" else text + "."


def _sentence(text: str) -> str:
    """``text`` as a sentence: capital first letter, closing full stop."""
    text = text.strip()
    return _end(text[:1].upper() + text[1:])


# The plate's section counts overlap: ``code`` = session + action, ``agents`` = staff + t.py,
# ``tickets`` = tp + t.py (a t.py approval sits in both; hub ``docs/api.md`` "Built-in
# perspectives"). Spoken buckets are disjoint, so they add up to the total:
# (key, with count 1, with count > 1, as the only part of a one-decision plate).
_BUCKETS = (
    ("session", "from a Claude session", "from Claude sessions", "from a Claude session"),
    ("tpy", "support ticket", "support tickets", "a support ticket"),
    ("staff", "from a staff agent", "from staff agents", "from a staff agent"),
    ("tp", "tp item", "tp items", "a tp item"),
)
_ORDINAL_WORDS = ("First", "Second", "Third", "Fourth", "Fifth")
_FOCUS_WORDS = {"work": "work", "private": "private"}


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _count(counts: Mapping[str, Any], key: str) -> int:
    try:
        return max(0, int(counts.get(key) or 0))
    except (TypeError, ValueError):
        return 0


def plate_buckets(counts: Mapping[str, Any]) -> dict[str, int]:
    """Disjoint per-source counts from the plate's overlapping sections.

    Every source sits in ``code`` (session, action), ``agents`` (staff, t.py) or ``tickets``
    (tp, t.py); only t.py is in two of them, so ``inbox = code + agents + tickets - tpy``.
    """
    code, agents, tickets = (_count(counts, k) for k in ("code", "agents", "tickets"))
    tpy = min(agents, tickets, max(0, code + agents + tickets - _count(counts, "inbox")))
    return {"session": code, "tpy": tpy, "staff": agents - tpy, "tp": tickets - tpy}


def _plate_head(counts: Mapping[str, Any], total: int, scope: str) -> str:
    """The first sentence, e.g. "38 work decisions wait for you: 1 from a Claude session and
    37 support tickets"; one part alone is "all support tickets"."""
    noun = "decision waits" if total == 1 else "decisions wait"
    head = f"{total}{scope} {noun} for you"
    buckets = plate_buckets(counts)
    parts = [(buckets[key], one, many, solo) for key, one, many, solo in _BUCKETS if buckets[key]]
    if len(parts) == 1 and parts[0][0] == total:
        n, _one, many, solo = parts[0]
        return f"{head}, {solo if n == 1 else 'all ' + many}"
    said = [_plural(n, one, many) for n, one, many, _solo in parts]
    if not said:
        return head
    return f"{head}: " + (", ".join(said[:-1]) + " and " if len(said) > 1 else "") + said[-1]


def _brief_parts(item: Mapping[str, Any]) -> dict[str, str]:
    """Background / Building / Fit from a decision record or a plate item's ``brief``."""
    if any(k in item for k in ("background", "goal", "fit")):
        return {
            "Background": str(item.get("background") or ""),
            "Building": str(item.get("goal") or ""),
            "Fit": str(item.get("fit") or ""),
        }
    brief = str(item.get("brief") or "")
    parts: dict[str, str] = {}
    marks = list(re.finditer(r"\b(Background|Building|Fit):\s*", brief))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(brief)
        parts[m.group(1)] = brief[m.end() : end].strip()
    return parts


def _context(item: Mapping[str, Any]) -> str:
    """One sentence of context: the brief's Fit, else its Background, else the asker."""
    parts = _brief_parts(item)
    for key in ("Fit", "Background", "Building"):
        if parts.get(key):
            return first_sentence(parts[key], SAY_CONTEXT)
    return ""


def _asker(item: Mapping[str, Any]) -> str:
    asker = item.get("asker")
    if isinstance(asker, Mapping):
        asker = asker.get("name") or asker.get("agent")
    name = speakable(asker or "", 60)
    if name.strip("\\~ ") == "":
        return "a Claude session" if item.get("source") == "session" else ""
    return name


def _project(item: Mapping[str, Any]) -> str:
    project = item.get("project")
    if isinstance(project, Mapping):
        view = item.get("view") if isinstance(item.get("view"), Mapping) else {}
        project = (view or {}).get("project_key") or project.get("sub") or project.get("area")
    text = str(project or "")
    return "" if text.startswith("unfiled") else speakable(text.rsplit("/", maxsplit=1)[-1], 60)


def plate_summary(plate: Mapping[str, Any], focus: str | None = None) -> str:
    """The spoken answer to "what's on my plate": counts per source, then the top three."""
    counts = plate.get("counts") if isinstance(plate.get("counts"), Mapping) else {}
    counts = counts or {}
    total = _count(counts, "inbox")
    scope = f" {_FOCUS_WORDS[focus]}" if focus in _FOCUS_WORDS else ""
    if total == 0:
        return f"No{scope} decisions wait for you on the decide hub."
    parts = [_sentence(_plate_head(counts, total, scope))]
    extra = [
        _plural(int(counts.get(k) or 0), one, many)
        for k, one, many in (
            ("overdue", "is overdue", "are overdue"),
            ("due_soon", "is due soon", "are due soon"),
            ("flagged", "is flagged", "are flagged"),
        )
        if int(counts.get(k) or 0)
    ]
    if extra:
        parts.append(_sentence(", ".join(extra)))
    blocked = [b for b in plate.get("blocked_sessions") or [] if isinstance(b, Mapping)]
    if blocked:
        parts.append(
            _sentence(
                f"{_plural(len(blocked), 'session is', 'sessions are')} blocked until you answer"
            )
        )
    top = [t for t in plate.get("top") or [] if isinstance(t, Mapping)]
    if top:
        parts.append("The top ones:" if len(top) > 1 else "The top one:")
    for i, item in enumerate(top[: len(_ORDINAL_WORDS)]):
        who = _asker(item)
        title = first_sentence(item.get("title"), SAY_TITLE)
        line = (
            f"{_ORDINAL_WORDS[i]}, {who + ' asks' if who else 'the question'}: {_sentence(title)}"
        )
        context = _context(item)
        if context:
            line += f" {_sentence(context)}"
        parts.append(line)
    if top:
        parts.append("Say next decision to go through them.")
    return " ".join(parts)


def _options(decision: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [o for o in decision.get("options") or [] if isinstance(o, Mapping)]


def _recommended(decision: Mapping[str, Any]) -> list[int]:
    """0-based recommended option indices (decision record or plate item)."""
    rec = decision.get("recommendation")
    if isinstance(rec, list):
        return [int(i) for i in rec if isinstance(i, int)]
    return [i for i, o in enumerate(_options(decision)) if o.get("recommended")]


def option_list(decision: Mapping[str, Any], with_descriptions: bool) -> str:
    """ "Options: 1, Approve — ships it; 2, …" (numbered from 1, as spoken)."""
    opts = _options(decision)
    if not opts:
        return ""
    said = []
    for i, o in enumerate(opts, start=1):
        text = f"{i}, {first_sentence(o.get('label'), SAY_OPTION)}"
        desc = first_sentence(o.get("description"), SAY_OPTION) if with_descriptions else ""
        said.append(text + (f" — {desc}" if desc else ""))
    multi = " Several can be picked." if decision.get("multi_select") else ""
    return f"Options: {'; '.join(said)}.{multi}"


def recommendation_text(decision: Mapping[str, Any]) -> str:
    rec = _recommended(decision)
    opts = _options(decision)
    why = first_sentence(decision.get("recommendation_why"), SAY_WHY)
    if not rec or not opts:
        return f"Recommendation: {_sentence(why)}" if why and not opts else ""
    names = " and ".join(
        f"option {i + 1}, {first_sentence(opts[i].get('label'), SAY_OPTION)}"
        for i in rec
        if 0 <= i < len(opts)
    )
    return f"It recommends {names}" + (f", because {_end(_lower_first(why))}" if why else ".")


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text[:2] != text[:2].upper() else text


def how_to_answer(decision: Mapping[str, Any]) -> str:
    if decision.get("kind") == "action":
        return "It is a step for you: say done when you have done it."
    if not _options(decision):
        return "It has no options: say answer, then your words."
    return "Say option and a number, go with the recommendation, or answer and your words."


def short_brief(decision: Mapping[str, Any], position: str = "") -> str:
    """ "Next decision": who asks, the question, a sentence of context, options, recommendation."""
    who = _asker(decision)
    project = _project(decision)
    head = f"{who} asks" if who else "The question"
    if project:
        head += f" ({project})"
    parts = [
        position + _sentence(f"{head}: {_sentence(first_sentence(decision.get('title'), 400))}")
    ]
    context = _context(decision)
    if context:
        parts.append(_sentence(context))
    for part in (option_list(decision, False), recommendation_text(decision)):
        if part:
            parts.append(part)
    parts.append(how_to_answer(decision))
    return " ".join(parts)


def full_brief(decision: Mapping[str, Any]) -> str:
    """ "Tell me more": background, building, fit, options with their trade-offs, recommendation."""
    parts = [_sentence(first_sentence(decision.get("title"), 400))]
    for label, text in _brief_parts(decision).items():
        if text:
            parts.append(f"{label}: {_sentence(speakable(text, BRIEF_PART))}")
    for part in (option_list(decision, True), recommendation_text(decision)):
        if part:
            parts.append(part)
    parts.append(how_to_answer(decision))
    return " ".join(parts)


# -- choices ----------------------------------------------------------------------------------
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "erste": 1, "erster": 1, "erstes": 1, "zweite": 2, "zweiter": 2,
    "dritte": 3, "dritter": 3, "vierte": 4, "funfte": 5, "premier": 1, "premiere": 1,
    "deuxieme": 2, "troisieme": 3, "quatrieme": 4, "last": -1, "letzte": -1, "dernier": -1,
}  # fmt: skip
_RECOMMEND = re.compile(
    r"\b(?:recommend\w*|suggest\w*|empfehlung|empfohlen\w*|recommande\w*|your pick|as proposed)\b"
)
_DONE = re.compile(r"\b(?:done|did it|erledigt|gemacht|fait|termine)\b")
_INDEX_FILLER = frozenset(
    {
        "option", "options", "number", "nummer", "numero", "answer", "choice", "choose", "pick",
        "take", "go", "with", "the", "and", "und", "et", "plus", "please", "bitte", "i", "ll",
        "d", "want", "nehme", "die", "den", "das", "la", "le",
    }
)  # fmt: skip
#: "the second ONE", "die zweite EINE": after an ordinal, "one" is filler, not option 1.
_ONE_WORDS = frozenset({"one", "eins", "eine", "un", "une"})


@dataclass(frozen=True)
class Choice:
    """What Albert picked for one decision."""

    choices: tuple[int, ...] = ()
    other_text: str = ""
    accept_recommendation: bool = False
    done: bool = False


def resolve_choice(  # pylint: disable=too-many-return-statements  # one answer per case
    decision: Mapping[str, Any], said: str, free_text: str = ""
) -> Choice | str:
    """The option(s), recommendation, free text or "done" that ``said`` names; else a question."""
    opts = _options(decision)
    n = len(opts)
    norm = normalise(said)
    if free_text.strip():
        if n and not decision.get("allow_other", True):
            return "This question takes no free text — pick one of the options."
        return Choice(other_text=" ".join(free_text.split())[:OTHER_MAX])
    if decision.get("kind") == "action":
        if _DONE.search(norm) or norm in {"yes", "ja", "oui"}:
            return Choice(done=True)
        return "This one is a step for you — say done when you have done it."
    if _RECOMMEND.search(norm):
        if not _recommended(decision):
            return "It has no recommendation — pick an option."
        return Choice(accept_recommendation=True)
    if not n:
        text = " ".join(said.split())
        if not text:
            return "What is your answer?"
        return Choice(other_text=text[:OTHER_MAX])
    picks = _label_picks(norm, opts) or _index_picks(norm, n)
    if isinstance(picks, str):
        return picks
    if not picks:
        return f"Which option? There are {n}: say option and a number."
    if len(picks) > 1 and not decision.get("multi_select"):
        return "Only one option can be picked here — which one?"
    return Choice(choices=tuple(picks))


def _label_picks(norm: str, opts: Sequence[Mapping[str, Any]]) -> list[int]:
    labels = [normalise(str(o.get("label") or "")) for o in opts]
    exact = [i for i, lab in enumerate(labels) if lab and lab == norm]
    if exact:
        return exact
    return [i for i, lab in enumerate(labels) if len(lab) > 3 and f" {lab} " in f" {norm} "]


def _index_picks(norm: str, n: int) -> list[int] | str:
    picks: list[int] = []
    after_ordinal = False
    for word in norm.split():
        if word in _INDEX_FILLER or (after_ordinal and word in _ONE_WORDS):
            after_ordinal = False
            continue
        after_ordinal = word in _ORDINALS
        if after_ordinal:
            value = _ORDINALS[word]
            picks.append(n if value == -1 else value)
        else:
            picks.extend(sorted(numbers_in(word)))
    out: list[int] = []
    for p in picks:
        if not 1 <= p <= n:
            return f"There {'is only one option' if n == 1 else f'are only {n} options'}."
        if p - 1 not in out:
            out.append(p - 1)
    return out


# -- proposals --------------------------------------------------------------------------------
@dataclass(frozen=True)
class Proposal:
    """An answer waiting for Albert's spoken yes."""

    decision_id: str
    title: str
    choice: Choice
    labels: tuple[str, ...] = ()
    comment: str = ""

    def body(self) -> dict[str, Any]:
        """The answer endpoint's body (``channel`` is added by the client)."""
        body: dict[str, Any] = {"comment": self.comment}
        if self.choice.done:
            body["done"] = True
        elif self.choice.accept_recommendation:
            body["accept_recommendation"] = True
        elif self.choice.other_text:
            body["other_text"] = self.choice.other_text
        else:
            body["choices"] = list(self.choice.choices)
        return body

    def as_args(self) -> dict[str, Any]:
        """A plain mapping (a bridge ``PendingAction``'s arguments)."""
        return {
            "decision_id": self.decision_id,
            "title": self.title,
            "choices": list(self.choice.choices),
            "other_text": self.choice.other_text,
            "accept_recommendation": self.choice.accept_recommendation,
            "done": self.choice.done,
            "labels": list(self.labels),
            "comment": self.comment,
        }

    @classmethod
    def from_args(cls, args: Mapping[str, Any]) -> Proposal:
        choice = Choice(
            choices=tuple(int(i) for i in args.get("choices") or []),
            other_text=str(args.get("other_text") or ""),
            accept_recommendation=bool(args.get("accept_recommendation")),
            done=bool(args.get("done")),
        )
        return cls(
            decision_id=str(args["decision_id"]),
            title=str(args.get("title") or ""),
            choice=choice,
            labels=tuple(str(x) for x in args.get("labels") or []),
            comment=str(args.get("comment") or ""),
        )

    def with_comment(self, comment: str) -> Proposal:
        text = " ".join(comment.split())[:COMMENT_MAX]
        joined = f"{self.comment} {text}".strip() if self.comment else text
        return Proposal(
            self.decision_id, self.title, self.choice, self.labels, joined[:COMMENT_MAX]
        )

    def what(self) -> str:
        """What is being answered, in words."""
        c = self.choice
        if c.done:
            return "done"
        if c.other_text:
            return f'your words "{speakable(c.other_text, 300)}"'
        picks = [f"option {i + 1}" for i in c.choices]
        if c.accept_recommendation:
            return "the recommendation" + (f", {' and '.join(self.labels)}" if self.labels else "")
        named = [f"{p}, {lab}" for p, lab in zip(picks, self.labels, strict=False)]
        return " and ".join(named or picks)

    def readback(self, confirm_phrase: str) -> str:
        text = f'I will answer "{first_sentence(self.title, 120)}" with {self.what()}'
        if self.comment:
            text += f', with the comment "{speakable(self.comment, 300)}"'
        return f"{text}. Say {confirm_phrase} to send it, or no to cancel."


# -- local state ------------------------------------------------------------------------------
def tomorrow_morning(now: datetime | None = None) -> datetime:
    """08:00 Zurich time tomorrow (the hub's default defer time)."""
    local = (now or datetime.now(UTC)).astimezone(ZURICH)
    day = local.date() + timedelta(days=1)
    return datetime(day.year, day.month, day.day, 8, 0, tzinfo=ZURICH)


class Snoozes:
    """Decisions Albert snoozed by voice, until when (a 0600 JSON file)."""

    def __init__(self, path: Path, clock: Callable[[], datetime] = lambda: datetime.now(UTC)):
        self.path = path
        self.clock = clock
        self._lock = threading.Lock()

    def _read(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        snoozed = data.get("snoozed") if isinstance(data, dict) else None
        return {str(k): str(v) for k, v in (snoozed or {}).items()}

    def active(self) -> set[str]:
        now = self.clock()
        out = set()
        for key, until in self._read().items():
            with contextlib.suppress(ValueError):
                if datetime.fromisoformat(until) > now:
                    out.add(key)
        return out

    def add(self, decision_id: str, until: datetime) -> None:
        with self._lock:
            now = self.clock()
            data = {
                k: v
                for k, v in self._read().items()
                if _later(v, now)  # forget expired entries
            }
            data[decision_id] = until.astimezone(UTC).isoformat()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"snoozed": data}, indent=1), encoding="utf-8")
            tmp.chmod(0o600)
            tmp.replace(self.path)


def _later(iso: str, now: datetime) -> bool:
    try:
        return datetime.fromisoformat(iso) > now
    except ValueError:
        return False


# -- the conversation -------------------------------------------------------------------------
@dataclass
class Sent:
    """The last answer sent by voice (for undo)."""

    decision_id: str
    title: str
    release_at: str = ""


@dataclass
class DecideDesk:  # pylint: disable=too-many-instance-attributes  # one conversation's state
    """The decision conversation: plate, queue, the decision in discussion, skips, undo."""

    client: HubClient | None
    snoozes: Snoozes = field(default_factory=lambda: Snoozes(STATE_DIR / STATE_FILE))
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    focus: str | None = None
    queue: list[str] = field(default_factory=list)
    skipped: set[str] = field(default_factory=set)
    current: dict[str, Any] | None = None
    last_sent: Sent | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- reads --------------------------------------------------------------------------
    def _client(self) -> HubClient:
        if self.client is None:
            raise HubError(0, "not_configured")
        return self.client

    def plate(self, top: int = 50) -> dict[str, Any]:
        """The plate (focus applied); refreshes the queue (snoozed + skipped left out)."""
        plate = self._client().plate(self.focus, top=top)
        hidden = self.snoozes.active() | self.skipped
        top_items = [t for t in plate.get("top") or [] if isinstance(t, Mapping)]
        plate["top"] = [t for t in top_items if str(t.get("id")) not in hidden]
        with self.lock:
            self.queue = [str(t.get("id")) for t in plate["top"] if t.get("id")]
        return plate

    def summary(self) -> str:
        """ "What's on my plate"."""
        try:
            plate = self.plate()
        except HubError as exc:
            return f"Cannot read your decisions: {exc.reason}."
        top = list(plate["top"])
        plate["top"] = top[:TOP]
        return plate_summary(plate, self.focus)

    def next_decision(self) -> str:
        """The next decision in the queue (refetched when empty): its short brief."""
        try:
            with self.lock:
                current = str(self.current.get("id")) if self.current else None
                queue = [i for i in self.queue if i != current and i not in self.skipped]
            if not queue:
                self.plate()
                with self.lock:
                    queue = [i for i in self.queue if i != current]
            if not queue:
                with self.lock:
                    self.current = None
                return "No more decisions wait for you" + (
                    f" in {self.focus}." if self.focus else "."
                )
            decision = self._client().decision(queue[0])
        except HubError as exc:
            return f"Cannot read the next decision: {exc.reason}."
        with self.lock:
            self.current = decision
            if current and current in self.queue:
                self.queue.remove(current)
            left = len([i for i in self.queue if i != decision.get("id")])
        position = f"{left} more after this one. " if left else ""
        return position + short_brief(decision)

    def tell_more(self) -> str:
        """ "Tell me more": the full brief of the decision in discussion."""
        with self.lock:
            current = self.current
        if current is None:
            return self.next_decision()
        try:
            fresh = self._client().decision(str(current.get("id")))
        except HubError:
            fresh = current
        with self.lock:
            self.current = fresh
        return full_brief(fresh)

    def show(self, decision_id: str) -> str:
        """The brief of one decision (it becomes the one in discussion)."""
        try:
            decision = self._client().decision(decision_id)
        except HubError as exc:
            return f"Cannot read that decision: {exc.reason}."
        with self.lock:
            self.current = decision
        return full_brief(decision)

    # -- local moves --------------------------------------------------------------------
    def skip(self) -> str:
        with self.lock:
            current = self.current
            if current is not None:
                self.skipped.add(str(current.get("id")))
        if current is None:
            return "Nothing to skip yet. " + self.next_decision()
        return "Skipped. " + self.next_decision()

    def snooze(self, until: datetime | None = None) -> str:
        """Hide the decision in discussion from voice until ``until`` (default: tomorrow 08:00)."""
        with self.lock:
            current = self.current
        if current is None:
            return "Which decision? Say next decision first."
        when = until or tomorrow_morning(self.clock())
        self.snoozes.add(str(current.get("id")), when)
        local = when.astimezone(ZURICH)
        day = "tomorrow" if local.date() == tomorrow_morning(self.clock()).date() else local.date()
        return f"Snoozed until {day} at {local:%H:%M} (for voice only). " + self.next_decision()

    def set_focus(self, scope: str | None) -> str:
        scope = (scope or "").strip().lower() or None
        if scope in {"all", "everything", "none", "alles", "tout"}:
            scope = None
        if scope is not None and scope not in FOCUS_SCOPES:
            return "I can focus on work, private, or everything."
        with self.lock:
            self.focus = scope
            self.queue = []
            self.current = None
        return f"Focus: {scope} only. " if scope else "Focus: everything. "

    # -- answers ------------------------------------------------------------------------
    def prepare(self, said: str, comment: str = "", free_text: str = "") -> Proposal | str:
        """A :class:`Proposal` for the decision in discussion, or what to ask instead."""
        with self.lock:
            current = self.current
        if current is None:
            return "Which decision? Say next decision first."
        if current.get("state") not in (None, "open"):
            return f"That decision is {current.get('state')} — say next decision."
        choice = resolve_choice(current, said, free_text)
        if isinstance(choice, str):
            return choice
        opts = _options(current)
        idx = list(choice.choices) or (
            _recommended(current) if choice.accept_recommendation else []
        )
        labels = tuple(
            first_sentence(opts[i].get("label"), SAY_OPTION) for i in idx if 0 <= i < len(opts)
        )
        return Proposal(
            decision_id=str(current.get("id")),
            title=str(current.get("title") or ""),
            choice=choice,
            labels=labels,
            comment=" ".join(comment.split())[:COMMENT_MAX],
        )

    def send(self, proposal: Proposal) -> str:
        """POST the answer (``channel: voice``); only after Albert's spoken yes."""
        try:
            stored = self._client().answer(proposal.decision_id, proposal.body())
        except HubError as exc:
            return f"Not sent: {exc.reason}."
        answer = stored.get("answer") if isinstance(stored.get("answer"), Mapping) else {}
        release = str((answer or {}).get("release_at") or "")
        with self.lock:
            self.last_sent = Sent(proposal.decision_id, proposal.title, release)
            if self.current and self.current.get("id") == proposal.decision_id:
                self.current = stored
            if proposal.decision_id in self.queue:
                self.queue.remove(proposal.decision_id)
        grace = self._grace_s(release)
        tail = f" It waits {grace} seconds in the hub — say undo to take it back." if grace else ""
        return "Sent." + tail

    def _grace_s(self, release_at: str) -> int:
        try:
            left = datetime.fromisoformat(release_at.replace("Z", "+00:00")) - self.clock()
        except ValueError:
            return 0
        return max(0, round(left.total_seconds()))

    def undo(self) -> str:
        with self.lock:
            sent = self.last_sent
        if sent is None:
            return "There is no voice answer to take back."
        try:
            reopened = self._client().undo(sent.decision_id)
        except HubError as exc:
            return f"Cannot take it back: {exc.reason}."
        with self.lock:
            self.last_sent = None
            self.current = reopened
            if sent.decision_id not in self.queue:
                self.queue.insert(0, sent.decision_id)
        return f'Taken back: "{first_sentence(sent.title, 120)}" is open again.'


# -- brain context ----------------------------------------------------------------------------
_ABOUT_DECISIONS = re.compile(
    r"\b(?:decisions?|decide|plate|on my plate|entscheid\w*|decision\w*|questions? for me)\b",
    re.IGNORECASE,
)


def about_decisions(text: str) -> bool:
    """True when a free question is about Albert's open decisions."""
    return bool(_ABOUT_DECISIONS.search(text or ""))


class PlateContext:
    """The plate summary as LLM context for a free question about decisions (cached briefly)."""

    def __init__(
        self,
        client_factory: Callable[[], HubClient | None] = build_client,
        ttl_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client_factory = client_factory
        self.ttl_s = ttl_s
        self.clock = clock
        self._cached: tuple[float, str] | None = None
        self._lock = threading.Lock()

    def for_text(self, text: str) -> str:
        """``"Albert's open decisions (decide hub): …"`` for a question about decisions, else ""."""
        if not about_decisions(text):
            return ""
        with self._lock:
            cached = self._cached
            if cached is not None and self.clock() - cached[0] < self.ttl_s:
                return cached[1]
        client = self.client_factory()
        if client is None:
            return ""
        try:
            summary = plate_summary(client.plate(top=TOP))
        except HubError:
            return ""
        line = f"Albert's open decisions (decide hub, read-only context): {summary}"
        with self._lock:
            self._cached = (self.clock(), line)
        return line


# -- CLI --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """``python -m my_stt_tts.decide`` — read-only checks of the voice client."""
    parser = argparse.ArgumentParser(
        prog="python -m my_stt_tts.decide",
        description="Read Albert's decide hub as the voice client does (read-only).",
        epilog=(
            "examples:\n"
            "  python -m my_stt_tts.decide -p          the spoken plate summary\n"
            "  python -m my_stt_tts.decide -p -F work  ... work only\n"
            "  python -m my_stt_tts.decide -n          the short brief of the top decision\n"
            "  python -m my_stt_tts.decide -s ID       the full brief of one decision\n"
            "  python -m my_stt_tts.decide -p -j       the raw plate JSON\n"
            "\n"
            f"needs {CLIENT_ID_ENVS[0]} (repo .env); the secret comes from {SECRET_ENV} or\n"
            f"secret-run (item {SECRET_ITEM})."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", "--plate", action="store_true", help="the spoken plate summary")
    group.add_argument("-n", "--next", action="store_true", help="the brief of the top decision")
    group.add_argument("-s", "--show", metavar="ID", help="the full brief of one decision")
    group.add_argument("-X", "--fetch", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("-F", "--focus", choices=FOCUS_SCOPES, help="work or private only")
    parser.add_argument("-j", "--json", action="store_true", help="raw JSON instead of speech")
    args = parser.parse_args(argv)
    if args.fetch:
        sys.stdout.write(child_fetch(sys.stdin.read()))
        return 0
    with contextlib.suppress(ImportError):
        from dotenv import load_dotenv  # pylint: disable=import-outside-toplevel

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    client = build_client()
    if client is None:
        print(f"❌ decide by voice is not configured: set {CLIENT_ID_ENVS[0]} in .env")
        return 2
    desk = DecideDesk(client, focus=args.focus)
    try:
        if args.json:
            data = desk.plate(top=TOP) if args.plate or args.next else client.decision(args.show)
            print(json.dumps(data, indent=1, ensure_ascii=False))
        elif args.plate:
            print(desk.summary())
        elif args.next:
            print(desk.next_decision())
        else:
            print(desk.show(args.show))
    except HubError as exc:
        print(f"❌ {exc.reason}")
        return 3 if exc.status == 0 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
