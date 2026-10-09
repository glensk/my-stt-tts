"""Mac operator: free-form requests carried out by ``claude -p`` through the Mac broker.

The ``do_on_mac(request)`` client tool (PLAN_claude-bridge.md 7.4) needs a capability
minted from Albert's authorised transcript; no proposal up front, because the broker
(:mod:`my_stt_tts.mac_broker`) holds every risky call until it is confirmed by code.

One run:

* a fresh 0700 call dir ``~/.local/state/mac-voice/operator/<call-id>`` is the cwd (no
  ``.env`` within reach) and holds ``broker.json`` (the MCP config launching
  ``mac_broker.py`` with this interpreter), the broker's audit and ``finish.json``;
* ``claude -p --model sonnet --tools "" --strict-mcp-config --mcp-config <broker.json>
  --permission-mode bypassPermissions --setting-sources "" --max-turns 30
  --output-format json --json-schema '<inline schema>' --no-session-persistence`` runs
  with the request on stdin, in its own process group, with an environment built from
  scratch (PATH, HOME, LANG, USER, ``CLAUDE_CONFIG_DIR=$HOME/.claude-work``,
  ``AI_NO_AUTOCOMMIT=1`` — nothing else);
* states: running → needs_confirmation | done | failed | cancelled | unknown_partial;
* the tool answers within 10 s (the result, a confirmation request, or ``started``); the
  final result reaches the controller through ``on_result`` and failures are reported
  to ``controller.problems``;
* the run ends as soon as the broker writes ``finish.json`` (no waiting for the
  structured-result turn), at the 5-minute hard limit, on cancel, when the call ends
  (unless the request was explicitly background: "do it even if I hang up") or at daemon
  shutdown — always TERM to the process group, then KILL after 3 s;
* a watcher turns each ``blocked-<sha>.json`` into ``controller.propose("operator_confirm",
  …)`` and says "<summary> — say confirm <code>"; the ``operator_confirm`` executor writes
  the matching ``allow-<sha>.json`` (0600).

Audit (``<root>/audit.jsonl``, 0600): request length + sha256, status, duration — no
content. Screenshots never outlive the broker call that took them; leftovers are swept.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import logging
import os
import pwd
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from . import mac_broker, mac_risk
from .bridge import (
    BridgeController,
    CancellationToken,
    Capability,
    Deadline,
    Mutation,
    PendingAction,
    Problem,
    Refusal,
    before_mutation,
    log_confirm,
    log_refused,
    log_tool,
)

log = logging.getLogger("my_stt_tts.mac_operator")

OPERATOR_FLAG = "MAC_VOICE_OPERATOR"
OPERATOR_ROOT = mac_broker.OPERATOR_ROOT
CLAUDE = Path.home() / ".local" / "bin" / "claude"
SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "operator_result.json"
BROKER_PATH = Path(__file__).resolve().parent / "mac_broker.py"
CONFIRM_KIND = "operator_confirm"

START_WAIT_S = 10.0
HARD_LIMIT_S = 300.0
KILL_GRACE_S = 3.0
STDOUT_CAP = 1024 * 1024
POLL_S = 0.2
REQUEST_CAP = 2000
KEEP_DAYS = 7
MAX_TURNS = 30
MODEL = "sonnet"

RUNNING = "running"
NEEDS_CONFIRMATION = "needs_confirmation"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"
UNKNOWN_PARTIAL = "unknown_partial"
FINAL_STATES = frozenset({DONE, FAILED, CANCELLED, UNKNOWN_PARTIAL, NEEDS_CONFIRMATION})
_RESULT_STATUSES = frozenset({DONE, FAILED, UNKNOWN_PARTIAL, NEEDS_CONFIRMATION})

# Phrases that make a request survive the hang-up (en / de / fr, accent-folded).
_BACKGROUND_RE = re.compile(
    r"hang up|hung up|in the background|auflege|aufgelegt|im hintergrund|raccroche"
    r"|en arriere-plan|en arriere plan"
)

PREAMBLE = """You operate this Mac through the `mac` broker tools only.
Do the request below with as few tool calls as possible: prefer open_url, open_app, key
and menu_select over clicking, and pass observe=true on your last action to get its
screenshot in the same turn. Apps may show leftover state from earlier use.
Risky actions (closing, sending, deleting, moving, buying, installing, settings,
AppleScript) are held until the user confirms by voice; such a call just takes longer.
If a call returns "denied: not confirmed" or "refused: ...", do not retry it and do not
work around it: finish with status needs_confirmation or failed.
After your LAST action, look at a screenshot taken after it, then call
finish(status, summary, evidence_obs_id) with that observation's obs_id. finish is final
and can be called only once. Keep the summary to one short sentence that can be spoken.
Every action tool returns an obs_id when you pass observe=true; your final structured
result MUST name an obs_id taken after your last action. Always call finish before ending.
Your structured result must repeat the status the broker recorded.

Request: """


@dataclass(frozen=True)
class OperatorResult:
    """The final outcome of one operator run (``summary`` is the model's spoken line)."""

    call_id: str
    status: str
    summary: str = field(repr=False)
    duration_s: float
    reason: str = ""
    background: bool = False

    def spoken(self) -> str:
        text = self.summary or self.reason or self.status.replace("_", " ")
        return f"{self.status}: {text}"


@dataclass
class OperatorCall:  # pylint: disable=too-many-instance-attributes  # one run's live state
    """One running operator request."""

    call_id: str
    call_dir: Path
    request_len: int
    request_sha256: str
    background: bool
    token: CancellationToken
    started: float
    state: str = RUNNING
    proc: subprocess.Popen[bytes] | None = None
    result: OperatorResult | None = None
    stdout: bytearray = field(default_factory=bytearray, repr=False)
    stdout_truncated: bool = False
    cancel_reason: str = ""
    proposed: dict[str, PendingAction] = field(default_factory=dict)
    confirm_message: str = ""
    done: threading.Event = field(default_factory=threading.Event)
    changed: threading.Event = field(default_factory=threading.Event)


def operator_enabled(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return env.get(OPERATOR_FLAG, "").strip() == "1"


def operator_env(home: Path | None = None, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The operator's environment, built from scratch (7.4 + the S-OPERATOR deviations)."""
    base = os.environ if base is None else base
    home = home or Path.home()
    user = base.get("USER") or pwd.getpwuid(os.getuid()).pw_name
    lang = base.get("LANG") or "en_US.UTF-8"
    return {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(home),
        "LANG": lang,
        "USER": user,
        "CLAUDE_CONFIG_DIR": str(home / ".claude-work"),
        "AI_NO_AUTOCOMMIT": "1",
    }


def inline_schema(path: Path = SCHEMA_PATH) -> str:
    """The result schema as compact inline JSON (``--json-schema`` takes no file path)."""
    return json.dumps(json.loads(path.read_text()), separators=(",", ":"))


def operator_argv(
    broker_json: Path,
    *,
    claude: Path = CLAUDE,
    schema: str | None = None,
    model: str = MODEL,
    max_turns: int = MAX_TURNS,
) -> list[str]:
    """The exact 7.4 argv (inline schema, ``--no-session-persistence``)."""
    return [
        str(claude),
        "-p",
        "--model",
        model,
        "--tools",
        "",
        "--strict-mcp-config",
        "--mcp-config",
        str(broker_json),
        "--permission-mode",
        "bypassPermissions",
        "--setting-sources",
        "",
        "--max-turns",
        str(max_turns),
        "--output-format",
        "json",
        "--json-schema",
        schema if schema is not None else inline_schema(),
        "--no-session-persistence",
    ]


def broker_config(call_dir: Path, root: Path, python: str) -> dict[str, Any]:
    """MCP config launching the broker by file path with ``python -I`` (no cwd on sys.path)."""
    args = ["-I", str(BROKER_PATH), "-c", str(call_dir), "-r", str(root)]
    return {"mcpServers": {"mac": {"type": "stdio", "command": python, "args": args}}}


def _fold(text: str) -> str:
    return mac_risk.fold(text)


def background_said(text: str) -> bool:
    """True when the transcript asks to keep going after the hang-up."""
    return bool(_BACKGROUND_RE.search(_fold(text)))


def kill_group(proc: subprocess.Popen[Any], grace: float = KILL_GRACE_S) -> bool:
    """TERM the process group, KILL it after ``grace`` s; True when KILL was needed."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = None
    if pgid is not None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=grace)
        exited = True
    except subprocess.TimeoutExpired:
        exited = False
    killed = False
    if pgid is not None:  # the leader may be gone while its children still run
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)
            killed = not exited
    if not exited:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=grace)
        killed = True
    return killed


class MacOperator:  # pylint: disable=too-many-instance-attributes  # the operator's hub
    """Starts, watches and ends operator runs for one :class:`BridgeController`."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        controller: BridgeController,
        *,
        root: Path = OPERATOR_ROOT,
        claude: Path = CLAUDE,
        python: str | None = None,
        env: Mapping[str, str] | None = None,
        notify: Callable[[str], None] | None = None,
        on_result: Callable[[OperatorResult], None] | None = None,
        start_wait: float = START_WAIT_S,
        hard_limit: float = HARD_LIMIT_S,
        kill_grace: float = KILL_GRACE_S,
        stdout_cap: int = STDOUT_CAP,
        poll: float = POLL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.controller = controller
        self.root = root
        self.claude = claude
        self.python = python or sys.executable
        self.env = dict(env) if env is not None else operator_env()
        self.notify = notify
        self.on_result = on_result
        self.start_wait = start_wait
        self.hard_limit = hard_limit
        self.kill_grace = kill_grace
        self.stdout_cap = stdout_cap
        self.poll = poll
        self.clock = clock
        self.results: deque[OperatorResult] = deque(maxlen=20)
        self.current: OperatorCall | None = None
        self._calls: dict[str, OperatorCall] = {}
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._closed = False
        self._starting = False

    @property
    def audit_path(self) -> Path:
        return self.root / "audit.jsonl"

    # -- the client tool --------------------------------------------------------------
    def do_on_mac(self, request: str, background: bool = False) -> str:
        """Client tool: carry out ``request`` on the Mac (answers within ~10 s)."""
        log_tool("do_on_mac", f"{len(request or '')} chars")
        request = " ".join(str(request or "").split())
        if not request:
            return "refused: say what to do"
        if len(request) > REQUEST_CAP:
            return f"refused: request longer than {REQUEST_CAP} characters"
        with self._lock:
            running = self.current is not None and not self.current.done.is_set()
            busy = self._closed or self._starting or running
            self._starting = self._starting or not busy
        if busy:
            log_refused("operator busy")
            return "refused: another Mac task is still running"
        try:
            cap = self.controller.mint()
            if isinstance(cap, Refusal):
                return f"refused: {cap.reason}"
            keep = bool(background) and background_said(cap.text)
            call = self._start(request, cap, keep)
        finally:
            with self._lock:
                self._starting = False
        if isinstance(call, str):
            return call
        return self._first_answer(call)

    def _start(self, request: str, cap: Capability, background: bool) -> OperatorCall | str:
        token = self.controller.cancel
        acquired = self.controller.mutation_lock.acquire(timeout=3.0)
        if not acquired:
            log_refused("another action is running")
            return "refused: another action is running"
        try:
            refusal = before_mutation(
                token,
                Deadline(3.0, self.clock),
                self.controller.caps,
                Mutation("do_on_mac", {}, cap),
            )
            if refusal is not None:
                return f"refused: {refusal.reason}"
            try:
                call = self._spawn(request, token, background)
            except (OSError, ValueError) as exc:
                log.warning("⚠️  operator did not start: %s", type(exc).__name__)
                self._report("did not start", "error")
                return "failed: the Mac operator could not start"
        finally:
            self.controller.mutation_lock.release()
        thread = threading.Thread(
            target=self._supervise, args=(call,), name=f"operator-{call.call_id}", daemon=True
        )
        with self._lock:
            self.current = call
            self._calls[call.call_id] = call
            self._threads = [t for t in self._threads if t.is_alive()] + [thread]
        thread.start()
        return call

    def _first_answer(self, call: OperatorCall) -> str:
        deadline = self.clock() + self.start_wait
        while not call.done.is_set():
            remaining = deadline - self.clock()
            if remaining <= 0:
                break
            if call.confirm_message:
                return f"needs confirmation: {call.confirm_message}"
            call.changed.wait(min(remaining, self.poll))
            call.changed.clear()
        if call.done.is_set() and call.result is not None:
            return call.result.spoken()
        if call.confirm_message:
            return f"needs confirmation: {call.confirm_message}"
        return "started: working on it; the result follows"

    # -- spawning ---------------------------------------------------------------------
    def _new_call_dir(self) -> tuple[str, Path]:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self._prune()
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        call_id = f"op-{stamp}-{secrets.token_hex(3)}"
        call_dir = self.root / call_id
        call_dir.mkdir(mode=0o700)
        call_dir.chmod(0o700)
        return call_id, call_dir

    def _prune(self) -> None:
        """Remove call dirs older than a week (their audits are content-free anyway)."""
        cutoff = time.time() - KEEP_DAYS * 86400
        for child in self.root.glob("op-*"):
            with contextlib.suppress(OSError):
                if child.is_dir() and child.stat().st_mtime < cutoff:
                    shutil.rmtree(child)

    def _spawn(self, request: str, token: CancellationToken, background: bool) -> OperatorCall:
        call_id, call_dir = self._new_call_dir()
        broker_json = call_dir / "broker.json"
        config = broker_config(call_dir, self.root, self.python)
        mac_risk.write_0600(broker_json, json.dumps(config) + "\n")
        argv = operator_argv(broker_json, claude=self.claude)
        call = OperatorCall(
            call_id=call_id,
            call_dir=call_dir,
            request_len=len(request),
            request_sha256=hashlib.sha256(request.encode()).hexdigest(),
            background=background,
            token=token,
            started=self.clock(),
        )
        proc = subprocess.Popen(  # pylint: disable=consider-using-with  # supervised thread
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=call_dir,
            env=dict(self.env),
            start_new_session=True,
            close_fds=True,
        )
        call.proc = proc
        assert proc.stdin is not None
        try:
            proc.stdin.write((PREAMBLE + request + "\n").encode())
            proc.stdin.close()
        except OSError:
            pass  # the child died at once; the supervisor reports it
        log_tool("operator", "started" + (" (background)" if background else ""))
        return call

    # -- supervising ------------------------------------------------------------------
    def _supervise(self, call: OperatorCall) -> None:
        proc = call.proc
        assert proc is not None and proc.stdout is not None
        reader = threading.Thread(
            target=self._read_stdout, args=(call, proc.stdout), name="operator-out", daemon=True
        )
        reader.start()
        status, reason = self._watch(call, proc)
        if proc.poll() is None:
            kill_group(proc, self.kill_grace)
        else:  # leader gone; make sure nothing of its group lingers
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
        reader.join(timeout=2.0)
        summary = ""
        if status is None:
            status, reason, summary = self._from_output(call)
        self._finalise(call, status, reason, summary)

    def _watch(self, call: OperatorCall, proc: subprocess.Popen[bytes]) -> tuple[str | None, str]:
        """Poll until a final state; returns ``(status, reason)`` or ``(None, "")`` on exit."""
        finish_path = call.call_dir / "finish.json"
        while True:
            finish = _read_json(finish_path)
            if finish is not None:
                return _finish_status(finish)
            if proc.poll() is not None:
                finish = _read_json(finish_path)
                return _finish_status(finish) if finish is not None else (None, "")
            if self.clock() - call.started > self.hard_limit:
                return FAILED, "timeout"
            if call.cancel_reason:
                return CANCELLED, call.cancel_reason
            if call.token.cancelled and not call.background:
                return CANCELLED, call.token.reason or "call ended"
            self._scan_blocked(call)
            time.sleep(self.poll)

    def _read_stdout(self, call: OperatorCall, stream: IO[bytes]) -> None:
        """Keep at most ``stdout_cap`` bytes; drain (and drop) the rest; the child never stalls."""
        with contextlib.suppress(OSError, ValueError):
            while chunk := stream.read(65536):
                room = self.stdout_cap - len(call.stdout)
                if room > 0:
                    call.stdout.extend(chunk[:room])
                if len(chunk) > max(room, 0):
                    call.stdout_truncated = True

    def _scan_blocked(self, call: OperatorCall) -> None:
        live: set[str] = set()
        for path in sorted(call.call_dir.glob("blocked-*.json")):
            sha = path.name.removeprefix("blocked-").removesuffix(".json")
            data = _read_json(path)
            if data is None or not mac_risk.SHA_RE.match(sha) or data.get("sha256") != sha:
                continue
            nonce = str(data.get("nonce", ""))
            key = f"{sha}:{nonce}"
            live.add(key)
            if key in call.proposed:
                continue
            summary = mac_risk.clean_summary(str(data.get("summary", "")))
            pending = self.controller.propose(
                CONFIRM_KIND,
                {"call_id": call.call_id, "sha": sha, "summary": summary, "nonce": nonce},
            )
            call.proposed[key] = pending
            message = f"{summary} — say confirm {pending.code}"
            call.confirm_message = message
            self._set_state(call, NEEDS_CONFIRMATION)
            log_confirm(summary)
            if self.notify is not None:
                try:
                    self.notify(message)
                except Exception:  # pylint: disable=broad-exception-caught  # never stop the watch
                    log.warning("⚠️  operator notify failed", exc_info=True)
        for key in [k for k in call.proposed if k not in live]:
            del call.proposed[key]
        if not live and call.state == NEEDS_CONFIRMATION:
            call.confirm_message = ""
            self._set_state(call, RUNNING)

    @staticmethod
    def _set_state(call: OperatorCall, state: str) -> None:
        if call.state != state:
            call.state = state
            call.changed.set()

    def _from_output(self, call: OperatorCall) -> tuple[str, str, str]:
        """``(status, reason, summary)`` from claude's JSON when the broker never saw ``finish``.

        A structured ``done`` counts only when its ``evidence_obs_id`` is an observation the
        broker recorded (``evidence.json``) after its last action; else ``unknown_partial``.
        """
        try:
            out = json.loads(bytes(call.stdout).decode(errors="replace"))
        except ValueError:
            reason = "output truncated" if call.stdout_truncated else "no result"
            return FAILED, reason, ""
        structured = out.get("structured_output") if isinstance(out, dict) else None
        if not isinstance(structured, dict):
            return FAILED, "no result", ""
        status = structured.get("status")
        if status not in _RESULT_STATUSES:
            return FAILED, "no result", ""
        if status != DONE:
            return str(status), "", ""
        evidence = _read_json(call.call_dir / "evidence.json")
        if evidence is None or not _evidence_after_last_action(
            evidence, structured.get("evidence_obs_id")
        ):
            return UNKNOWN_PARTIAL, "unverified", ""
        return DONE, "", _cap_summary(str(structured.get("summary") or ""))

    def _finalise(self, call: OperatorCall, status: str, reason: str, summary: str = "") -> None:
        """``summary`` (verified structured output) is used only when finish.json is absent."""
        duration = self.clock() - call.started
        finish = _read_json(call.call_dir / "finish.json")
        if finish is not None:
            summary = str(finish.get("summary", ""))
        if status == CANCELLED:
            summary = ""
        result = OperatorResult(
            call.call_id, status, summary, round(duration, 1), reason, call.background
        )
        call.result = result
        call.state = status
        self._sweep(call.call_dir)
        self._audit(call, status, reason, duration)
        log.info("🤖 operator: %s (%d s)", status.replace("_", " "), round(duration))
        self.results.append(result)
        self._report_result(result)
        call.done.set()
        call.changed.set()
        if self.on_result is not None:
            try:
                self.on_result(result)
            except Exception:  # pylint: disable=broad-exception-caught
                log.warning("⚠️  operator result callback failed", exc_info=True)

    def _report_result(self, result: OperatorResult) -> None:
        if result.status == DONE:
            return
        if result.status == CANCELLED:
            if result.reason != "daemon shutdown":
                self._report(result.reason or "cancelled", "warning")
            return
        reason = result.reason or result.status.replace("_", " ")
        severity = "warning" if result.status == NEEDS_CONFIRMATION else "error"
        self._report(reason, severity)

    def _report(self, reason: str, severity: str) -> None:
        try:
            self.controller.problems.report(
                Problem(kind="operator", subject="Mac task", reason=reason, severity=severity)
            )
        except Exception:  # pylint: disable=broad-exception-caught
            log.warning("⚠️  problem report failed", exc_info=True)

    @staticmethod
    def _sweep(call_dir: Path) -> None:
        """Delete any screenshot a killed broker left behind."""
        for pattern in ("shot-*", "*.jpg", "*.jpeg", "*.png"):
            for path in call_dir.glob(pattern):
                path.unlink(missing_ok=True)

    def _audit(self, call: OperatorCall, status: str, reason: str, duration: float) -> None:
        record = {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "call_id": call.call_id,
            "request_len": call.request_len,
            "request_sha256": call.request_sha256,
            "status": status,
            "reason": reason,
            "duration_s": round(duration, 2),
            "background": call.background,
            "stdout_truncated": call.stdout_truncated,
        }
        with contextlib.suppress(OSError):
            mac_risk.append_0600(self.audit_path, json.dumps(record) + "\n")

    # -- confirmation executor --------------------------------------------------------
    def confirm_executor(self, pending: PendingAction) -> str:
        """``operator_confirm``: let exactly the blocked broker call run (writes allow)."""
        args = pending.args
        with self._lock:
            call = self._calls.get(str(args.get("call_id", "")))
        if call is None or call.done.is_set():
            log_refused("operator task already over")
            return "refused: that Mac task is no longer running"
        if not mac_risk.write_allow(
            call.call_dir, str(args.get("sha", "")), str(args.get("nonce", ""))
        ):
            log_refused("operator action no longer waiting")
            return "refused: that action is no longer waiting"
        log_tool("operator", "confirmed")
        return "confirmed"

    # -- ending -----------------------------------------------------------------------
    def cancel(self, reason: str = "cancelled") -> None:
        """Stop the running task (if any)."""
        with self._lock:
            call = self.current
        if call is not None and not call.done.is_set():
            call.cancel_reason = reason

    def shutdown(self, timeout: float = 10.0) -> None:
        """Daemon shutdown: end every run (background ones too) and join the watchers."""
        with self._lock:
            self._closed = True
            calls = list(self._calls.values())
            threads = list(self._threads)
        for call in calls:
            if not call.done.is_set():
                call.cancel_reason = "daemon shutdown"
        for thread in threads:
            thread.join(timeout=timeout)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _evidence_after_last_action(evidence: Mapping[str, Any], obs_id: Any) -> bool:
    """True when ``obs_id`` is a recorded observation with a seq after the last action."""
    last = evidence.get("last_action_seq")
    observations = evidence.get("observations")
    if not isinstance(obs_id, str) or not obs_id:
        return False
    if not isinstance(last, int) or isinstance(last, bool) or not isinstance(observations, dict):
        return False
    seq = observations.get(obs_id)
    return isinstance(seq, int) and not isinstance(seq, bool) and seq > last


def _cap_summary(text: str) -> str:
    """Same cleaning and cap as the broker's ``finish.json`` summary."""
    return " ".join(mac_risk.CONTROL_RE.sub(" ", text).split())[: mac_broker.FINISH_SUMMARY_CAP]


def _finish_status(finish: Mapping[str, Any]) -> tuple[str, str]:
    status = str(finish.get("status", ""))
    if status not in _RESULT_STATUSES:
        return FAILED, "invalid finish"
    return status, str(finish.get("reason", ""))


def register(
    controller: BridgeController,
    *,
    env: Mapping[str, str] | None = None,
    notify: Callable[[str], None] | None = None,
    on_result: Callable[[OperatorResult], None] | None = None,
    **options: Any,
) -> MacOperator | None:
    """Add ``do_on_mac`` + the ``operator_confirm`` executor when ``MAC_VOICE_OPERATOR=1``.

    ``notify(text)`` speaks a confirmation request in the call; ``on_result`` receives
    every final :class:`OperatorResult`. ``options`` go to :class:`MacOperator`.
    """
    if not operator_enabled(env):
        return None
    if importlib.util.find_spec("mcp") is None:
        log.warning("⚠️  %s=1 but the 'operator' extra (mcp) is missing", OPERATOR_FLAG)
        return None
    operator = MacOperator(controller, notify=notify, on_result=on_result, **options)
    if not Path(operator.claude).exists():
        log.warning("⚠️  %s not found — Mac operator requests will fail", operator.claude)
    controller.register_tool("do_on_mac", operator.do_on_mac)
    controller.register_executor(CONFIRM_KIND, operator.confirm_executor)
    controller.on_shutdown(operator.shutdown)
    return operator
