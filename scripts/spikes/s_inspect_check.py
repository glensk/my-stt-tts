#!/usr/bin/env python3
"""Spike S-INSPECT (PLAN_claude-bridge.md §6, Phase 0): reference parser for ``ccc inspect -j``.

Parses Claude Code 2.1.29x transcripts (``~/.claude*/projects/*/<session>.jsonl``) into the
``data`` object of ``ccc inspect -s ID -j``: ``{state, last_reply, last_prompt, decision}``.
It is the reference implementation Phase 1 reuses; this script also checks it against the
fixtures in ccc's ``tests/fixtures/inspect/`` (``<name>.jsonl`` + ``<name>.expected.json``).

Rules fixed by this spike (the contract in §6 leaves them open):

- **Human input** = a ``user`` record typed at the prompt (string or ``text`` blocks; not
  ``isMeta``/sidechain, not a ``<task-notification>``, not an interrupt marker), an
  ``attachment`` of type ``queued_command`` with ``commandMode == "prompt"`` and a human
  (or missing) ``origin`` (a prompt typed while busy and absorbed mid-turn), or the
  ``tool_result`` answering ``AskUserQuestion`` (rendered from top-level
  ``toolUseResult.answers`` as ``"<question>"="<answer>"`` pairs). A pending
  ``queue-operation``/``enqueue`` is NOT a prompt yet.
- ``last_prompt`` = the last human input, head kept, ≤ 500 chars (``…`` on truncation).
- ``last_reply`` = the ``text`` blocks of the assistant records after the last human input,
  joined by a blank line, tail kept, ≤ 1500 chars (``…`` + tail on truncation); ``""``
  when the current turn has produced no text yet.
- ``state``: ``waiting`` when exactly the last assistant message holds an
  ``AskUserQuestion`` ``tool_use`` without a ``tool_result`` (queue-operations and
  attachments after it are ignored — real 2.1.295 transcripts queue task notices there);
  else ``busy`` when the last human input / tool round is not followed by an end-of-turn
  (assistant ``stop_reason`` ``end_turn``/``stop_sequence``/``max_tokens``, ``system``
  ``stop_hook_summary``/``turn_duration``, or an interrupt marker); else ``idle``. The live
  ``claude agents`` status overrides this in ccc; the transcript state is the fallback.
- ``decision``: the pending ``AskUserQuestion`` (source ``ask_user_question``) wins;
  otherwise, only in state ``idle``, the todo-line grammar over the last reply (source
  ``todo_line``). Option labels drop a ``(Recommended)`` marker. With more than one
  question, ``recommendation``/``recommendation_reason`` are ``"<n>) …"`` parts joined by
  ``"; "`` (the §6 contract has one field per decision — a gap Phase 1 may close with
  per-question fields).
- ``decision_id`` = sha256 over compact JSON ``[anchor, [[text, multi_select, [labels…]]…]]``
  with anchor = the ``tool_use`` id (ask) or the uuid of the assistant record holding the
  to-do list (todo).
- ``context.recent_summary`` = the fallback of §6: first 3 sentences of the last reply
  (heading lines dropped, whitespace collapsed), ≤ 800 chars; ``aim``/``ticket`` null here.

Examples:
    scripts/spikes/s_inspect_check.py                 # check every fixture, exit 1 on diff
    scripts/spikes/s_inspect_check.py -f busy -v      # one fixture, print the parsed data
    scripts/spikes/s_inspect_check.py -p ~/.claude/projects/x/y.jsonl   # parse any transcript
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FIXTURES = Path("/Users/albert/obsidian/42-Git/llms/claude-command-center/tests/fixtures/inspect")
FIXTURE_NAMES = ("todo_decision", "pending_ask", "busy", "answered_ask")
DECISION_ID_PLACEHOLDER = "<sha256>"

LAST_REPLY_MAX = 1500
LAST_PROMPT_MAX = 500
SUMMARY_MAX = 800

INTERRUPT_MARKERS = ("[Request interrupted by user]", "[Request interrupted by user for tool use]")
END_STOP_REASONS = frozenset({"end_turn", "stop_sequence", "max_tokens"})
END_SYSTEM_SUBTYPES = frozenset({"stop_hook_summary", "turn_duration"})

TODO_HEADING_RE = re.compile(r"^\s*#{1,6}\s*To-do list\s*$", re.IGNORECASE)
HEADING_RE = re.compile(r"^\s*#{1,6}\s")
TODO_DECISION_RE = re.compile(r"^\s*\d+\.\s+You \[decision\]:\s*(.+)$")
OPTION_MARKER_RE = re.compile(r"(?:(?<=\s)|^)([a-z])\)\s+")
RECOMMENDED_RE = re.compile(r"\s*\(recommended\)", re.IGNORECASE)
CONSEQUENCE_SPLIT_RE = re.compile(r"\s+(?:—|–|--?)\s+")
RECOMMEND_SENTENCE_RE = re.compile(
    r"(?:^|(?<=[.?!])\s+)"
    r"(?:I(?:'d| would)? recommend|My (?:suggestion|recommendation)(?: is)?:?)\s+"
    r"(?P<rec>.+?)(?:\s+because\s+(?P<why>.+?))?[.!]?\s*$",
    re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


# --------------------------------------------------------------------------- records


def load_records(path: Path) -> list[dict[str, Any]]:
    """Every JSON object line of *path*; raises ValueError on a malformed line."""
    records: list[dict[str, Any]] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{n}: not JSON ({exc.msg})") from exc
        if not isinstance(record, dict):
            raise TypeError(f"{path.name}:{n}: not an object")
        records.append(record)
    return records


def _content(record: dict[str, Any]) -> Any:
    message = record.get("message")
    return message.get("content") if isinstance(message, dict) else None


def _blocks(record: dict[str, Any]) -> list[dict[str, Any]]:
    content = _content(record)
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict)]


def _is_main(record: dict[str, Any]) -> bool:
    return not record.get("isSidechain") and not record.get("isMeta")


def _typed_prompt(record: dict[str, Any]) -> str | None:
    """Text of a prompt typed at the idle prompt, or None."""
    if record.get("type") != "user" or not _is_main(record):
        return None
    if record.get("isCompactSummary"):
        return None
    origin = record.get("origin")
    if isinstance(origin, dict) and origin.get("kind") not in (None, "human"):
        return None
    content = _content(record)
    if isinstance(content, str):
        text = content
    else:
        blocks = _blocks(record)
        if not blocks or any(b.get("type") == "tool_result" for b in blocks):
            return None
        text = "\n".join(str(b.get("text") or "") for b in blocks if b.get("type") == "text")
    text = text.strip()
    if not text or text.startswith("<task-notification>") or text in INTERRUPT_MARKERS:
        return None
    return text


def _queued_prompt(record: dict[str, Any]) -> str | None:
    """Text of a prompt queued while busy and absorbed into the turn, or None."""
    if record.get("type") != "attachment" or not _is_main(record):
        return None
    attachment = record.get("attachment")
    if not isinstance(attachment, dict) or attachment.get("type") != "queued_command":
        return None
    if attachment.get("commandMode") != "prompt":
        return None
    origin = attachment.get("origin")
    if isinstance(origin, dict) and origin.get("kind") not in (None, "human"):
        return None
    prompt = attachment.get("prompt")
    if isinstance(prompt, list):
        prompt = "\n".join(
            str(b.get("text") or "")
            for b in prompt
            if isinstance(b, dict) and b.get("type") == "text"
        )
    text = str(prompt or "").strip()
    return text or None


def _answer(record: dict[str, Any]) -> str | None:
    """An AskUserQuestion answer rendered as ``"<question>"="<answer>"`` pairs, or None."""
    if record.get("type") != "user" or not _is_main(record):
        return None
    result = record.get("toolUseResult")
    if not isinstance(result, dict):
        return None
    questions, answers = result.get("questions"), result.get("answers")
    if not isinstance(questions, list) or not isinstance(answers, dict):
        return None
    pairs = [
        f'"{q.get("question")}"="{answers[q.get("question")]}"'
        for q in questions
        if isinstance(q, dict) and q.get("question") in answers
    ]
    return ", ".join(pairs) or None


def _is_interrupt(record: dict[str, Any]) -> bool:
    if record.get("type") != "user":
        return False
    content = _content(record)
    texts = (
        [content]
        if isinstance(content, str)
        else [str(b.get("text") or "") for b in _blocks(record) if b.get("type") == "text"]
    )
    return any(t.strip() in INTERRUPT_MARKERS for t in texts)


# --------------------------------------------------------------------------- scan


@dataclass
class PendingAsk:
    """An AskUserQuestion call without a tool_result."""

    tool_use_id: str
    questions: list[dict[str, Any]]


@dataclass
class Scan:
    """What one pass over the transcript found."""

    last_prompt: str | None = None
    reply_parts: list[str] = field(default_factory=list)
    reply_anchor: str | None = None  # uuid of the record holding the latest reply text
    todo_anchor: str | None = None  # uuid of the record holding the to-do list
    turn_open: bool = False
    pending: list[PendingAsk] = field(default_factory=list)


def _scan_user(s: Scan, record: dict[str, Any], answered: set[str]) -> None:
    """A user record: tool results keep the turn open, an interrupt closes it."""
    for block in _blocks(record):
        if block.get("type") == "tool_result":
            answered.add(str(block.get("tool_use_id")))
            s.turn_open = True
    if _is_interrupt(record):
        s.turn_open = False
        s.pending = []


def _scan_assistant(s: Scan, record: dict[str, Any], answered: set[str]) -> None:
    """An assistant record (one content block): reply text, tool calls, end of turn."""
    # A newer assistant message supersedes any earlier answered ask.
    s.pending = [p for p in s.pending if p.tool_use_id not in answered]
    uuid = str(record.get("uuid"))
    for block in _blocks(record):
        if block.get("type") == "text" and str(block.get("text") or "").strip():
            text = str(block["text"]).strip()
            s.reply_parts.append(text)
            s.reply_anchor = uuid
            if any(TODO_HEADING_RE.match(line) for line in text.splitlines()):
                s.todo_anchor = uuid
        elif block.get("type") == "tool_use":
            if block.get("name") == "AskUserQuestion":
                questions = (block.get("input") or {}).get("questions") or []
                s.pending.append(PendingAsk(str(block.get("id")), list(questions)))
            s.turn_open = True
    message = record.get("message")
    if isinstance(message, dict) and message.get("stop_reason") in END_STOP_REASONS:
        s.turn_open = False


def scan(records: list[dict[str, Any]]) -> Scan:
    """One ordered pass: human inputs, reply text, turn open/closed, pending asks."""
    s = Scan()
    answered: set[str] = set()
    for record in records:
        if record.get("isSidechain"):
            continue
        human = _typed_prompt(record) or _queued_prompt(record) or _answer(record)
        if human is not None:
            s.last_prompt = human
            s.reply_parts = []
            s.reply_anchor = s.todo_anchor = None
            s.turn_open = True
            s.pending = []  # a later human input settles any earlier ask
        rtype = record.get("type")
        if rtype == "user":
            _scan_user(s, record, answered)
        elif rtype == "assistant" and not record.get("isMeta"):
            _scan_assistant(s, record, answered)
        elif rtype == "system" and record.get("subtype") in END_SYSTEM_SUBTYPES:
            s.turn_open = False
    # Only asks of the LAST assistant message can still be pending.
    last_asst_ids = _last_assistant_tool_ids(records)
    s.pending = [
        p for p in s.pending if p.tool_use_id not in answered and p.tool_use_id in last_asst_ids
    ]
    return s


def _last_assistant_tool_ids(records: list[dict[str, Any]]) -> set[str]:
    """tool_use ids of the last assistant API message (one message = several records)."""
    last_id: str | None = None
    ids: set[str] = set()
    for record in records:
        if record.get("type") != "assistant" or record.get("isSidechain"):
            continue
        message = record.get("message")
        msg_id = message.get("id") if isinstance(message, dict) else None
        if msg_id != last_id:
            last_id, ids = msg_id, set()
        ids.update(str(b.get("id")) for b in _blocks(record) if b.get("type") == "tool_use")
    return ids


# --------------------------------------------------------------------------- decisions


@dataclass
class Question:
    """One decision question with its options and (optional) recommendation."""

    text: str
    multi_select: bool
    options: list[dict[str, str | None]]
    recommendation: str | None = None
    reason: str | None = None

    def public(self) -> dict[str, Any]:
        return {"text": self.text, "multi_select": self.multi_select, "options": self.options}


def _strip_recommended(label: str) -> tuple[str, bool]:
    stripped, n = RECOMMENDED_RE.subn("", label)
    return stripped.strip(), n > 0


def ask_questions(pending: PendingAsk) -> list[Question]:
    """Questions of a pending AskUserQuestion call."""
    out: list[Question] = []
    for q in pending.questions:
        if not isinstance(q, dict):
            continue
        options: list[dict[str, str | None]] = []
        rec = reason = None
        for opt in q.get("options") or []:
            if not isinstance(opt, dict):
                continue
            label, recommended = _strip_recommended(str(opt.get("label", "")))
            consequence = str(opt.get("description") or "").strip() or None
            options.append({"label": label, "consequence": consequence})
            if recommended and rec is None:
                rec, reason = label, consequence
        out.append(
            Question(
                text=str(q.get("question", "")).strip(),
                multi_select=bool(q.get("multiSelect")),
                options=options,
                recommendation=rec,
                reason=reason,
            )
        )
    return out


def _top_level_split(text: str, sep: str) -> list[str]:
    """Split *text* on *sep* outside (), [] and {}."""
    parts: list[str] = []
    depth = 0
    start = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0 and text.startswith(sep, i):
            parts.append(text[start:i])
            i += len(sep)
            start = i
            continue
        i += 1
    parts.append(text[start:])
    return parts


def _label(text: str) -> str:
    text = text.strip().rstrip(".?!,;:").strip()
    return text[:1].upper() + text[1:] if text else text


def _explicit_options(body: str) -> tuple[str, list[tuple[str, str | None, bool]]] | None:
    """``(question, [(label, consequence, recommended)])`` from ``a) … b) …`` markers."""
    markers = list(OPTION_MARKER_RE.finditer(body))
    if len(markers) < 2 or markers[0].group(1) != "a":
        return None
    letters = [m.group(1) for m in markers]
    if letters != [chr(ord("a") + i) for i in range(len(letters))]:
        return None
    question = body[: markers[0].start()].strip().rstrip(":").strip()
    options: list[tuple[str, str | None, bool]] = []
    for i, m in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(body)
        seg = body[m.end() : end].strip().rstrip(",;").strip()
        seg = re.sub(r"\s+or$", "", seg)
        seg, recommended = _strip_recommended(seg)
        head, *tail = CONSEQUENCE_SPLIT_RE.split(seg, maxsplit=1)
        consequence = tail[0].strip().rstrip(".") if tail else None
        options.append((_label(head), consequence or None, recommended))
    return question, options


def _or_options(question: str) -> list[tuple[str, str | None, bool]]:
    """Options from a top-level ``X or Y`` split of the question sentence (else [])."""
    core = question.strip().rstrip("?").strip()
    parts = _top_level_split(core, " or ")
    if len(parts) < 2:
        return []
    lead = _top_level_split(parts[0], ": ")
    parts[0] = lead[-1]
    out: list[tuple[str, str | None, bool]] = []
    for part in parts:
        label, recommended = _strip_recommended(part.strip().strip(","))
        out.append((_label(label), None, recommended))
    return out


def todo_questions(reply: str) -> list[Question]:
    """Questions from ``You [decision]:`` lines in the last ``## To-do list`` section."""
    lines = reply.splitlines()
    starts = [i for i, line in enumerate(lines) if TODO_HEADING_RE.match(line)]
    if not starts:
        return []
    section: list[str] = []
    for line in lines[starts[-1] + 1 :]:
        if HEADING_RE.match(line):
            break
        section.append(line)
    out: list[Question] = []
    for line in section:
        m = TODO_DECISION_RE.match(line)
        if not m:
            continue
        body = m.group(1).strip()
        rec: str | None = None
        reason: str | None = None
        rm = RECOMMEND_SENTENCE_RE.search(body)
        if rm:
            rec = rm.group("rec").strip()
            reason = rm.group("why").strip() if rm.group("why") else None
            body = body[: rm.start()].strip()
        explicit = _explicit_options(body)
        if explicit is not None:
            question, opts = explicit
        else:
            question, opts = body, _or_options(body)
        marked = [(label, cons) for label, cons, recommended in opts if recommended]
        if marked:
            rec, reason = marked[0]
        out.append(
            Question(
                text=question,
                multi_select=False,
                options=[{"label": label, "consequence": cons} for label, cons, _r in opts],
                recommendation=rec,
                reason=reason,
            )
        )
    return out


def decision_id(anchor: str, questions: list[Question]) -> str:
    """sha256(anchor, questions, options) per §6."""
    payload = [
        anchor,
        [[q.text, q.multi_select, [o["label"] for o in q.options]] for q in questions],
    ]
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _combine(questions: list[Question], attr: str) -> str | None:
    values = [(i, getattr(q, attr)) for i, q in enumerate(questions, 1) if getattr(q, attr)]
    if not values:
        return None
    if len(questions) == 1:
        return str(values[0][1])
    return "; ".join(f"{i}) {v}" for i, v in values)


def recent_summary(reply: str) -> str:
    """Fallback summary: first 3 sentences of *reply* (headings dropped), ≤ 800 chars."""
    text = " ".join(
        line.strip() for line in reply.splitlines() if line.strip() and not HEADING_RE.match(line)
    )
    summary = " ".join(SENTENCE_SPLIT_RE.split(text)[:3]).strip()
    return _cap_head(summary, SUMMARY_MAX)


# --------------------------------------------------------------------------- inspect


def _cap_head(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _cap_tail(text: str, limit: int) -> str:
    return text if len(text) <= limit else "…" + text[-(limit - 1) :]


def inspect_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The ``data`` object of ``ccc inspect -j`` for *records*."""
    s = scan(records)
    reply = "\n\n".join(s.reply_parts)
    if s.pending:
        state = "waiting"
    elif s.turn_open:
        state = "busy"
    else:
        state = "idle"
    decision: dict[str, Any] | None = None
    questions: list[Question] = []
    source = anchor = None
    if len(s.pending) == 1:
        questions, source, anchor = (
            ask_questions(s.pending[0]),
            "ask_user_question",
            s.pending[0].tool_use_id,
        )
    elif state == "idle" and s.todo_anchor:
        questions, source, anchor = todo_questions(reply), "todo_line", s.todo_anchor
    if questions and source and anchor:
        decision = {
            "decision_id": decision_id(anchor, questions),
            "source": source,
            "questions": [q.public() for q in questions],
            "recommendation": _combine(questions, "recommendation"),
            "recommendation_reason": _combine(questions, "reason"),
            "context": {"aim": None, "ticket": None, "recent_summary": recent_summary(reply)},
        }
    return {
        "state": state,
        "last_reply": _cap_tail(reply, LAST_REPLY_MAX),
        "last_prompt": _cap_head(s.last_prompt or "", LAST_PROMPT_MAX),
        "decision": decision,
    }


def inspect_path(path: Path) -> dict[str, Any]:
    """``ccc inspect`` envelope for one transcript file (malformed → transcript_unknown)."""
    try:
        data = inspect_records(load_records(path))
    except (OSError, ValueError, TypeError) as exc:
        return {
            "schema_version": 1,
            "ok": False,
            "data": None,
            "error": {"code": "transcript_unknown", "message": str(exc)},
        }
    return {"schema_version": 1, "ok": True, "data": data, "error": None}


# --------------------------------------------------------------------------- checks


def check_schema(name: str, records: list[dict[str, Any]]) -> list[str]:
    """Structural invariants every fixture keeps from the real 2.1.295 schema."""
    problems: list[str] = []
    seen: set[str] = set()
    sids = {r.get("sessionId") for r in records if r.get("sessionId")}
    if len(sids) != 1:
        problems.append(f"{name}: sessionId not unique: {sids}")
    last_ts = ""
    for i, r in enumerate(records, 1):
        if "uuid" in r:
            parent = r.get("parentUuid")
            if parent is not None and parent not in seen:
                problems.append(f"{name}:{i}: parentUuid {parent} not an earlier uuid")
            seen.add(r["uuid"])
            if r.get("version") != "2.1.295":
                problems.append(f"{name}:{i}: version {r.get('version')}")
        ts = r.get("timestamp")
        if ts:
            if ts < last_ts:
                problems.append(f"{name}:{i}: timestamp goes backwards")
            last_ts = ts
    return problems


def normalise(data: dict[str, Any]) -> dict[str, Any]:
    """Replace a real decision_id by the placeholder (after checking it is 64 hex)."""
    out = json.loads(json.dumps(data))
    decision = out.get("decision")
    if decision:
        did = decision.get("decision_id", "")
        if not re.fullmatch(r"[0-9a-f]{64}", did):
            raise ValueError(f"decision_id is not a sha256 hex digest: {did!r}")
        decision["decision_id"] = DECISION_ID_PLACEHOLDER
    return out


def check_fixture(directory: Path, name: str, verbose: bool) -> bool:
    records = load_records(directory / f"{name}.jsonl")
    expected = json.loads((directory / f"{name}.expected.json").read_text(encoding="utf-8"))
    got = normalise(inspect_records(records))
    problems = check_schema(name, records)
    if verbose:
        print(json.dumps(got, indent=2, ensure_ascii=False))
    if got != expected:
        problems.append(
            f"{name}: parsed data differs from expected\n--- got\n"
            + json.dumps(got, indent=2, ensure_ascii=False)
            + "\n--- expected\n"
            + json.dumps(expected, indent=2, ensure_ascii=False)
        )
    for p in problems:
        print(f"❌ {p}")
    if not problems:
        print(f"✅ {name}")
    return not problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-d", "--dir", type=Path, default=FIXTURES, help="fixture directory")
    parser.add_argument(
        "-f", "--fixture", action="append", choices=FIXTURE_NAMES, help="check only this one"
    )
    parser.add_argument("-p", "--parse", type=Path, help="print the envelope for any transcript")
    parser.add_argument("-v", "--verbose", action="store_true", help="print the parsed data")
    args = parser.parse_args(argv)
    if args.parse:
        print(json.dumps(inspect_path(args.parse.expanduser()), indent=2, ensure_ascii=False))
        return 0
    names = args.fixture or list(FIXTURE_NAMES)
    results = [check_fixture(args.dir, n, args.verbose) for n in names]  # report every fixture
    ok = all(results)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
