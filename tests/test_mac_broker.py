"""Mac broker: risk table, confirmation binding, refusals, evidence ordering, audit.

No screen, no osascript: a fake runner answers every subprocess the broker would start and
records what it was asked to run.
"""

from __future__ import annotations

# pylint: disable=redefined-outer-name  # pytest fixtures
import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from my_stt_tts import mac_broker, mac_risk
from my_stt_tts.mac_broker import DENIED, Broker, Shot, check_call_dir, dom_reply, safari_dom_js
from my_stt_tts.mac_risk import (
    CONFIRM,
    REFUSE,
    SAFE,
    RiskContext,
    args_sha256,
    classify,
    search_or_address_field,
    shell_script_refusal,
    validate_url,
    write_allow,
)

SRC = Path(__file__).resolve().parents[1] / "src"


def fake_jpeg(width: int = 800, height: int = 600) -> bytes:
    sof = b"\xff\xc0\x00\x11\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big")
    return b"\xff\xd8" + sof + b"\x03" + b"\x00" * 16 + b"\xff\xd9"


class FakeRunner:
    """Answers osascript / open / screencapture / sips like a Mac would; records argv."""

    def __init__(self, front: str = "Safari", bounds: str = "100,50,400,300") -> None:
        self.front = front
        self.bounds = bounds
        self.calls: list[list[str]] = []
        self.fail: set[str] = set()

    def __call__(self, argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        del timeout
        self.calls.append(list(argv))
        name = Path(argv[0]).name
        out = ""
        if name in self.fail:
            return subprocess.CompletedProcess(argv, 1, "", "boom")
        if name == "osascript":
            script = argv[2]
            if "frontmost is true" in script and "position of window" not in script:
                out = self.front
            elif "position of window" in script:
                out = f"{self.front}|{self.bounds}"
            elif "bounds of window of desktop" in script:
                out = "0, 0, 1728, 1117"
        elif name == "screencapture":
            Path(argv[-1]).write_bytes(fake_jpeg())
        return subprocess.CompletedProcess(argv, 0, out, "")

    def actions(self) -> list[list[str]]:
        """Calls that act (not the front-app / bounds lookups or screenshots)."""
        reads = ("frontmost is true", "bounds of window of desktop")
        return [
            c
            for c in self.calls
            if Path(c[0]).name not in {"screencapture", "sips"}
            and not (Path(c[0]).name == "osascript" and any(r in c[2] for r in reads))
        ]


@pytest.fixture
def call_dir(tmp_path: Path) -> Path:
    d = tmp_path / "operator" / "op-test"
    d.mkdir(parents=True, mode=0o700)
    d.chmod(0o700)
    return d


def make_broker(call_dir: Path, runner: FakeRunner | None = None, **kw: Any) -> Broker:
    kw.setdefault("confirm_timeout", 0.4)
    kw.setdefault("poll", 0.02)
    kw.setdefault("sleep", lambda s: time.sleep(min(s, 0.02)))
    kw.setdefault("clicker", lambda x, y: None)
    kw.setdefault("hit_test", lambda x, y: {"AXRole": "AXLink", "AXTitle": "Watch video"})
    kw.setdefault("focused", lambda: None)
    return Broker(call_dir, runner=runner or FakeRunner(), **kw)


def text(reply: list[Any]) -> str:
    return "\n".join(p for p in reply if isinstance(p, str))


def ctx(
    front: str = "Safari",
    element: dict[str, str] | None = None,
    focused: dict[str, str] | None = None,
) -> RiskContext:
    return RiskContext(lambda: front, lambda x, y: element, lambda: focused)


# -- risk table ------------------------------------------------------------------------------
SAFE_CASES = [
    ("open_url", {"url": "youtube"}),
    ("open_url", {"url": "https://jellyfin.dom42.space/web/"}),
    ("open_url", {"url": "https://www.youtube.com/results?search_query=" + "a" * 150}),
    ("open_app", {"name": "Calculator"}),
    ("observe_screen", {"app": ""}),
    ("list_ui", {"app": "Calculator"}),
    ("safari_dom", {"selector": "h3", "limit": 10}),
    ("safari_dom", {"selector": 'a[href="/watch"]'}),
    ("safari_dom", {"selector": "'); alert(1); ('"}),  # data only, never code
    ("key", {"combo": "2", "app": "Calculator"}),
    ("key", {"combo": "cmd+l", "app": "Safari"}),
    ("key", {"combo": "ctrl+tab", "app": "Safari"}),
    ("key", {"combo": "down", "app": "Finder"}),
    ("key", {"combo": "cmd+f", "app": "Finder"}),
    ("type_text", {"app": "Safari", "text": "cute cats"}),
    ("type_text", {"app": "Mail", "text": "draft text"}),
    ("menu_select", {"app": "Safari", "path": "View > Enter Full Screen"}),
    ("run_shortcut", {"name": "Focus on"}),
    ("run_shortcut", {"name": "Do Not Disturb"}),
    ("click", {"element": "Play", "app": "Safari"}),
]


@pytest.mark.parametrize(("tool", "args"), SAFE_CASES)
def test_safe_calls(tool: str, args: dict[str, Any]) -> None:
    assert classify(tool, args, ctx()).risk == SAFE


CONFIRM_CASES = [
    ("key", {"combo": "cmd+w", "app": "Safari"}),  # close tab
    ("key", {"combo": "cmd+shift+w", "app": "Safari"}),  # close window
    ("key", {"combo": "cmd+q", "app": "Calculator"}),  # quit
    ("key", {"combo": "cmd+backspace", "app": "Finder"}),  # move to trash
    ("key", {"combo": "cmd+alt+escape", "app": ""}),  # force quit
    ("key", {"combo": "cmd+return", "app": "Mail"}),  # send
    ("key", {"combo": "cmd+shift+d", "app": "Mail"}),  # send
    ("key", {"combo": "return", "app": "Messages"}),  # send
    ("key", {"combo": "return", "app": "Finder"}),  # rename
    ("key", {"combo": "return", "app": "Safari"}),  # focus unknown: may submit a form
    ("key", {"combo": "enter", "app": "Safari"}),
    ("key", {"combo": "shift+return", "app": "Safari"}),
    ("key", {"combo": "cmd+s", "app": "TextEdit"}),  # save
    ("key", {"combo": "cmd+shift+s", "app": "TextEdit"}),  # save as
    ("key", {"combo": "cmd+alt+s", "app": "Xcode"}),  # save all
    ("key", {"combo": "cmd+S", "app": "Safari"}),
    ("open_url", {"url": "https://example.com/?q=" + "x" * 201}),  # long query
    ("open_url", {"url": "https://example.com/#" + "x" * 201}),  # long fragment
    ("key", {"combo": "l", "app": "Terminal"}),  # typing into a shell
    ("key", {"combo": "space", "app": "System Settings"}),  # toggling a setting
    ("type_text", {"app": "Terminal", "text": "ls"}),
    ("type_text", {"app": "iTerm2", "text": "ls"}),
    ("type_text", {"app": "Safari", "text": "hello\n"}),  # submits
    ("type_text", {"app": "Finder", "text": "new name"}),  # rename
    ("menu_select", {"app": "Safari", "path": "File > Close Window"}),
    ("menu_select", {"app": "Safari", "path": "File > Close Other Tabs"}),
    ("menu_select", {"app": "Finder", "path": "File > Move to Trash"}),
    ("menu_select", {"app": "Mail", "path": "Message > Send"}),
    ("menu_select", {"app": "App Store", "path": "Store > Buy"}),
    ("menu_select", {"app": "Finder", "path": "File > Rename"}),
    ("menu_select", {"app": "Calculator", "path": "Calculator > Quit Calculator"}),
    ("menu_select", {"app": "System Settings", "path": "View > Network"}),
    ("click", {"element": "Send", "app": "Mail"}),
    ("click", {"element": "Löschen", "app": "Finder"}),
    ("click", {"element": "Supprimer", "app": "Finder"}),
    ("click", {"element": "Buy now", "app": "Safari"}),
    ("click", {"element": "Install", "app": "Safari"}),
    ("applescript", {"script": 'tell application "Safari" to get URL of document 1'}),
    ("run_shortcut", {"name": "Morning routine"}),
    ("run_shortcut", {"name": "Delete old files"}),
    ("open_app", {"name": "Install macOS Sequoia"}),
    ("teleport", {"where": "mars"}),  # unknown → CONFIRM
]


@pytest.mark.parametrize(("tool", "args"), CONFIRM_CASES)
def test_confirm_calls(tool: str, args: dict[str, Any]) -> None:
    verdict = classify(tool, args, ctx())
    assert verdict.risk == CONFIRM, verdict
    assert verdict.summary and len(verdict.summary) <= mac_risk.SUMMARY_CAP


@pytest.mark.parametrize(
    ("element", "front", "risk"),
    [
        ({"AXRole": "AXButton", "AXTitle": "Do Not Disturb"}, "Control Center", SAFE),
        ({"AXRole": "AXCheckBox", "AXTitle": "Do Not Disturb"}, "System Settings", SAFE),
        ({"AXRole": "AXCheckBox", "AXTitle": "Firewall"}, "System Settings", CONFIRM),
        ({"AXRole": "AXCheckBox", "AXTitle": "Safari"}, "System Settings", CONFIRM),
        ({"AXRole": "AXButton", "AXSubrole": "AXCloseButton"}, "Safari", CONFIRM),
        ({"AXRole": "AXButton"}, "Messages", CONFIRM),  # unlabelled button
        (
            {"AXRole": "AXStaticText", "AXValue": "Send", "parent.AXRole": "AXButton"},
            "Mail",
            CONFIRM,
        ),
        ({"AXRole": "AXLink", "AXTitle": "Watch the video"}, "Safari", SAFE),
        ({"AXRole": "AXGroup"}, "Safari", SAFE),  # unlabelled benign role
        (None, "Safari", CONFIRM),  # hit test failed → unknown → CONFIRM
    ],
)
def test_click_by_point_uses_the_element_under_it(
    element: dict[str, str] | None, front: str, risk: str
) -> None:
    verdict = classify("click", {"x": 10, "y": 10, "_point": (10.0, 10.0)}, ctx(front, element))
    assert verdict.risk == risk, verdict


def test_unknown_front_app_is_confirm() -> None:
    assert classify("key", {"combo": "a", "app": ""}, RiskContext()).risk == CONFIRM
    assert classify("type_text", {"app": "", "text": "x"}, RiskContext()).risk == CONFIRM


def test_summaries_never_contain_typed_text() -> None:
    verdict = classify("type_text", {"app": "Terminal", "text": "rm -rf secret-xyz"}, ctx())
    assert "secret-xyz" not in verdict.summary and "Terminal" in verdict.summary


@pytest.mark.parametrize(
    "script",
    [
        'do shell script "ls"',
        'DO SHELL SCRIPT "ls"',
        'Do   Shell\tScript "ls"',
        'tell me to do\nshell\n  script "id"',
        'do shell script "x" with administrator privileges',
        'run script "do shell" & " script \\"id\\""',
        'load script file "x.scpt"',
        '«event sysoexec» "id"',
    ],
)
def test_do_shell_script_is_refused(script: str) -> None:
    assert shell_script_refusal(script)
    assert classify("applescript", {"script": script}).risk == REFUSE


@pytest.mark.parametrize(
    "url",
    [
        "file:///Users/albert/.env",
        "javascript:alert(1)",
        "ftp://example.com",
        "https://user:pw@example.com/",
        "https://exa mple.com",
        "https://example.com/\x00",
        "https://" + "a" * 2000 + ".com",
        "",
    ],
)
def test_bad_urls_are_refused(url: str) -> None:
    with pytest.raises(ValueError):
        validate_url(url)
    assert classify("open_url", {"url": url}).risk == REFUSE


def test_url_shortcuts_and_bare_hosts() -> None:
    assert validate_url("youtube") == "https://www.youtube.com"
    assert validate_url("Jellyfin") == "https://jellyfin.dom42.space"
    assert validate_url("example.com/a?b=1") == "https://example.com/a?b=1"


@pytest.mark.parametrize("name", ["/Applications/Calculator.app", "-n", "../x", "~/x", "a\nb"])
def test_app_paths_are_refused(name: str) -> None:
    assert classify("open_app", {"name": name}).risk == REFUSE


# -- Return / Enter needs a search field or an address bar -------------------------------------
ADDRESS_BAR = {"AXRole": "AXTextField", "AXDescription": "Smart Search Field"}
CHROME_BAR = {"AXRole": "AXTextField", "AXDescription": "Address and search bar"}
URL_FIELD = {"AXRole": "AXComboBox", "AXIdentifier": "url-field"}
SEARCH = {"AXRole": "AXSearchField"}
SUCHE = {"AXRole": "AXTextField", "AXPlaceholderValue": "Suche"}
RECHERCHE = {"AXRole": "AXTextField", "AXTitle": "Recherche"}
WEB_TEXTAREA = {"AXRole": "AXTextArea", "AXDescription": "Search comment"}
EMAIL_FIELD = {"AXRole": "AXTextField", "AXDescription": "E-mail address"}
PASSWORD = {"AXRole": "AXTextField", "AXSubrole": "AXSecureTextField", "AXTitle": "Search"}
BUTTON = {"AXRole": "AXButton", "AXTitle": "Search"}


@pytest.mark.parametrize(
    ("info", "ok"),
    [
        (ADDRESS_BAR, True),
        (CHROME_BAR, True),
        (URL_FIELD, True),
        (SEARCH, True),
        (SUCHE, True),
        (RECHERCHE, True),
        ({"AXRole": "AXTextField", "AXDescription": "Adresse"}, True),
        (WEB_TEXTAREA, False),
        (EMAIL_FIELD, False),
        (PASSWORD, False),
        (BUTTON, False),
        ({"AXRole": "AXTextField", "AXTitle": "Comment"}, False),
        ({}, False),
        (None, False),
    ],
)
def test_search_or_address_field(info: dict[str, str] | None, ok: bool) -> None:
    assert search_or_address_field(info) is ok


@pytest.mark.parametrize(
    ("focused", "risk"),
    [
        ({**ADDRESS_BAR, "app.AXTitle": "Safari"}, SAFE),  # address bar → SAFE
        ({**SEARCH, "app.AXTitle": "Safari"}, SAFE),  # search field → SAFE
        ({**WEB_TEXTAREA, "app.AXTitle": "Safari"}, CONFIRM),  # web text area
        ({**EMAIL_FIELD, "app.AXTitle": "Safari"}, CONFIRM),  # web form field
        ({**BUTTON, "app.AXTitle": "Safari"}, CONFIRM),
        ({**ADDRESS_BAR, "app.AXTitle": "Google Chrome"}, CONFIRM),  # focus in another app
        (ADDRESS_BAR, CONFIRM),  # the focused app is unknown
        (None, CONFIRM),  # focus lookup failed
    ],
)
def test_return_needs_a_search_or_address_focus(focused: dict[str, str] | None, risk: str) -> None:
    verdict = classify("key", {"combo": "return", "app": "Safari"}, ctx(focused=focused))
    assert verdict.risk == risk, verdict
    if risk == CONFIRM:
        assert verdict.summary == "press Return in Safari"


def test_return_without_app_uses_the_front_apps_focus() -> None:
    assert classify("key", {"combo": "enter"}, ctx(focused=ADDRESS_BAR)).risk == SAFE
    assert classify("key", {"combo": "enter"}, ctx(focused=WEB_TEXTAREA)).risk == CONFIRM


def test_focus_lookup_exception_is_confirm() -> None:
    def boom() -> dict[str, str]:
        raise RuntimeError("AX not permitted")

    risky = RiskContext(lambda: "Safari", None, boom)
    assert risky.focused() is None
    assert classify("key", {"combo": "return", "app": "Safari"}, risky).risk == CONFIRM


def test_return_in_messaging_apps_stays_confirm_even_in_a_search_field() -> None:
    focus = {**SEARCH, "app.AXTitle": "Slack"}
    assert classify("key", {"combo": "return", "app": "Slack"}, ctx(focused=focus)).risk == CONFIRM


def test_focus_is_only_looked_up_for_return() -> None:
    calls: list[int] = []

    def lookup() -> None:
        calls.append(1)

    counting = RiskContext(lambda: "Safari", None, lookup)
    assert classify("key", {"combo": "down", "app": "Safari"}, counting).risk == SAFE
    assert not calls
    assert classify("key", {"combo": "return", "app": "Safari"}, counting).risk == CONFIRM
    assert calls == [1]


def test_saving_is_confirm_in_every_app() -> None:
    for app in ("TextEdit", "Safari", "Preview", "Calculator"):
        for combo in ("cmd+s", "cmd+shift+s", "cmd+alt+s"):
            verdict = classify("key", {"combo": combo, "app": app}, ctx(front=app))
            assert verdict.risk == CONFIRM, (app, combo)


def test_long_url_query_is_confirm_but_url_is_not_refused() -> None:
    at_cap = "https://example.com/search?q=" + "x" * 98 + "#" + "f" * 100  # 100 + 100
    assert classify("open_url", {"url": at_cap}).risk == SAFE
    assert classify("open_url", {"url": at_cap + "f"}).risk == CONFIRM
    long_q = "https://example.com/search?q=" + "x" * 300
    verdict = classify("open_url", {"url": long_q})
    assert verdict.risk == CONFIRM and verdict.summary == "open example.com with a long query"
    assert classify("open_url", {"url": "https://example.com/" + "p" * 500}).risk == SAFE


# -- safari_dom: a fixed read-only script, the selector is data -------------------------------
NASTY_SELECTORS = [
    'a[title="x"]',
    "'); alert(1); ('",
    "\"); fetch('/x'); (\"",
    "`${fetch('/x')}`",
    "</script><script>alert(1)</script>",
    "a\\\"); location.href='x'; //",
    "\u2028alert(1)\u2029",
]


@pytest.mark.parametrize("selector", NASTY_SELECTORS)
def test_safari_dom_selector_is_only_a_json_literal(selector: str) -> None:
    js = safari_dom_js(selector, 20)
    literal = json.dumps(selector, ensure_ascii=True)
    assert js.count(literal) == 1 and json.loads(literal) == selector
    placeholder = "zz-placeholder-zz"
    template = safari_dom_js(placeholder, 20).replace(json.dumps(placeholder), "SEL")
    assert js.replace(literal, "SEL") == template  # nothing else in the script changed
    assert "\u2028" not in js and "\u2029" not in js


def test_safari_dom_limit_and_selector_validation() -> None:
    assert "lim=50," in safari_dom_js("a", 999)
    assert "lim=1," in safari_dom_js("a", 0)
    for bad_sel in ("", "   ", "a" * 501):
        with pytest.raises(ValueError):
            safari_dom_js(bad_sel, 20)
    with pytest.raises(ValueError):
        safari_dom_js("a", "5; alert(1)")  # type: ignore[arg-type]


def test_safari_dom_classification_is_always_safe() -> None:
    for selector in NASTY_SELECTORS:
        assert classify("safari_dom", {"selector": selector}, ctx()).risk == SAFE


def test_safari_dom_never_returns_password_values_and_caps_output() -> None:
    raw = json.dumps(
        {
            "count": 3,
            "items": [
                {"tag": "input", "type": "password", "value": "hunter2"},
                {"tag": "input", "type": "hidden", "value": "csrf-token"},
                {"tag": "input", "type": "text", "value": "visible"},
            ],
        }
    )
    out = dom_reply(raw)
    assert "hunter2" not in out and "csrf-token" not in out and "visible" in out
    big = json.dumps({"count": 50, "items": [{"tag": "p", "text": "x" * 200}] * 50})
    capped = json.loads(dom_reply(big))
    assert capped["truncated"] and len(dom_reply(big)) <= mac_broker.DOM_CAP
    assert dom_reply("error: SyntaxError: bad selector") == "error: SyntaxError: bad selector"


def test_safari_dom_broker_runs_the_template(call_dir: Path) -> None:
    runner = FakeRunner()
    broker = make_broker(call_dir, runner)
    reply = text(broker.call("safari_dom", {"selector": "'); alert(1); ('", "limit": 3}))
    assert reply.startswith("seq=")
    js = runner.calls[-1][4]  # osascript -e <script> -- <js>
    assert js == safari_dom_js("'); alert(1); ('", 3)
    assert not list(call_dir.glob("blocked-*.json"))
    assert text(broker.call("safari_dom", {"selector": ""})).startswith("error:")


_NODE_HARNESS = """
const seen = [];
function el(tag, attrs, props) {
  return Object.assign({
    tagName: tag.toUpperCase(), innerText: props.text || '',
    hasAttribute: n => n in attrs, getAttribute: n => (n in attrs ? attrs[n] : null),
  }, props);
}
const els = [
  el('input', {}, {type: 'password', value: 'hunter2', autocomplete: ''}),
  el('input', {}, {type: 'text', value: 'kept', autocomplete: 'current-password'}),
  el('input', {}, {type: 'search', value: 'cats', autocomplete: ''}),
  el('a', {href: '/w', 'aria-label': 'Watch', role: 'link'}, {href: 'https://x/w', text: ' Go '}),
];
globalThis.document = {querySelectorAll: s => { seen.push(s); return els; }};
globalThis.alert = () => { throw new Error('alert ran'); };
const out = (__JS__);
process.stdout.write(JSON.stringify({seen, out}));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("selector", NASTY_SELECTORS)
def test_safari_dom_script_in_node(tmp_path: Path, selector: str) -> None:
    script = tmp_path / "harness.js"
    script.write_text(_NODE_HARNESS.replace("__JS__", safari_dom_js(selector, 10)))
    node = shutil.which("node") or "node"
    proc = subprocess.run(
        [node, str(script)], capture_output=True, text=True, check=True, timeout=30
    )
    result = json.loads(proc.stdout)
    assert result["seen"] == [selector]  # passed to querySelectorAll verbatim, never run
    reply = json.loads(dom_reply(result["out"]))
    items = reply["items"]
    assert "hunter2" not in proc.stdout and "kept" not in proc.stdout
    assert items[0] == {"tag": "input", "type": "password"}
    assert items[2]["value"] == "cats"
    assert items[3] == {
        "tag": "a",
        "text": "Go",
        "href": "https://x/w",
        "aria-label": "Watch",
        "role": "link",
    }


# -- confirmation binding ------------------------------------------------------------------
def _run_in_thread(
    broker: Broker, tool: str, args: dict[str, Any]
) -> tuple[threading.Thread, list[Any]]:
    out: list[Any] = []
    thread = threading.Thread(target=lambda: out.append(broker.call(tool, args)))
    thread.start()
    return thread, out


def _wait_blocked(call_dir: Path, timeout: float = 2.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        files = list(call_dir.glob("blocked-*.json"))
        if files:
            with_content = [f for f in files if f.stat().st_size]
            if with_content:
                return json.loads(with_content[0].read_text())
        time.sleep(0.01)
    raise AssertionError("no blocked file appeared")


def test_blocked_call_writes_blocked_file_and_runs_once_after_allow(call_dir: Path) -> None:
    runner = FakeRunner()
    broker = make_broker(call_dir, runner, confirm_timeout=3.0)
    args = {"combo": "cmd+w", "app": "Safari"}
    thread, out = _run_in_thread(broker, "key", args)
    blocked = _wait_blocked(call_dir)
    sha = args_sha256("key", args)
    assert blocked["tool"] == "key" and blocked["sha256"] == sha
    assert blocked["summary"] == "press cmd+w in Safari"
    path = call_dir / f"blocked-{sha}.json"
    assert path.stat().st_mode & 0o777 == 0o600
    assert runner.actions() == []  # nothing ran while blocked
    assert write_allow(call_dir, sha, blocked["nonce"])
    assert (call_dir / f"allow-{sha}.json").stat().st_mode & 0o777 == 0o600
    thread.join(timeout=3)
    assert '"ok": true' in text(out[0])
    assert len(runner.actions()) == 1
    assert not path.exists() and not (call_dir / f"allow-{sha}.json").exists()  # consumed


def test_confirm_allows_exactly_one_call(call_dir: Path) -> None:
    runner = FakeRunner()
    broker = make_broker(call_dir, runner, confirm_timeout=3.0)
    args = {"combo": "cmd+w", "app": "Safari"}
    thread, _ = _run_in_thread(broker, "key", args)
    blocked = _wait_blocked(call_dir)
    write_allow(call_dir, blocked["sha256"], blocked["nonce"])
    thread.join(timeout=3)
    assert len(runner.actions()) == 1
    # the identical call again: blocks again and is denied without a new confirmation
    broker.confirm_timeout = 0.3
    assert text(broker.call("key", args)) == DENIED
    # another risky call is not covered either
    assert text(broker.call("key", {"combo": "cmd+q", "app": "Safari"})) == DENIED
    assert len(runner.actions()) == 1


def test_allow_for_one_call_does_not_cover_another(call_dir: Path) -> None:
    runner = FakeRunner()
    broker = make_broker(call_dir, runner, confirm_timeout=0.5)
    thread, out = _run_in_thread(broker, "key", {"combo": "cmd+q", "app": "Safari"})
    _wait_blocked(call_dir)
    other = args_sha256("key", {"combo": "cmd+w", "app": "Safari"})
    mac_risk.write_0600(call_dir / f"allow-{other}.json", json.dumps({"sha256": other}))
    thread.join(timeout=3)
    assert text(out[0]) == DENIED and runner.actions() == []


def test_confirmation_timeout_denies(call_dir: Path) -> None:
    runner = FakeRunner()
    broker = make_broker(call_dir, runner, confirm_timeout=0.2)
    reply = broker.call("menu_select", {"app": "Safari", "path": "File > Close Window"})
    assert text(reply) == DENIED
    assert runner.actions() == []
    assert not list(call_dir.glob("blocked-*.json"))


def test_stale_or_forged_allow_files_are_ignored(call_dir: Path) -> None:
    runner = FakeRunner()
    broker = make_broker(call_dir, runner, confirm_timeout=0.4)
    args = {"script": "beep"}
    sha = args_sha256("applescript", args)
    # an allow written before the block (e.g. a late confirm of an earlier, denied call)
    mac_risk.write_0600(call_dir / f"allow-{sha}.json", json.dumps({"sha256": sha, "nonce": "x"}))
    thread, out = _run_in_thread(broker, "applescript", args)
    _wait_blocked(call_dir)
    mac_risk.write_0600(
        call_dir / f"allow-{sha}.json", json.dumps({"sha256": sha, "nonce": "forged"})
    )
    thread.join(timeout=3)
    assert text(out[0]) == DENIED and runner.actions() == []


def test_write_allow_needs_the_live_blocked_nonce(call_dir: Path) -> None:
    sha = "a" * 64
    assert not write_allow(call_dir, sha, "n")  # nothing blocked
    mac_risk.write_0600(call_dir / f"blocked-{sha}.json", json.dumps({"nonce": "n"}))
    assert not write_allow(call_dir, sha, "other")
    assert not write_allow(call_dir, "../../etc", "n")
    assert write_allow(call_dir, sha, "n")


def test_do_shell_script_never_runs(call_dir: Path) -> None:
    runner = FakeRunner()
    broker = make_broker(call_dir, runner)
    reply = broker.call("applescript", {"script": 'Do  SHELL\nscript "id"'})
    assert text(reply).startswith("refused:")
    assert runner.actions() == [] and not list(call_dir.glob("blocked-*.json"))
    audit = [json.loads(x) for x in broker.audit_path.read_text().splitlines()]
    assert audit[-1]["outcome"] == "refused"


def test_safe_call_runs_without_blocking(call_dir: Path) -> None:
    runner = FakeRunner()
    broker = make_broker(call_dir, runner)
    assert '"ok": true' in text(broker.call("open_url", {"url": "youtube"}))
    assert runner.actions()[-1] == ["/usr/bin/open", "-a", "Safari", "https://www.youtube.com"]


# -- evidence ordering + finish ------------------------------------------------------------
def _obs_id(reply: list[Any]) -> str:
    return json.loads(text(reply).splitlines()[0])["obs_id"]


def test_finish_done_needs_an_observation_after_the_last_action(call_dir: Path) -> None:
    broker = make_broker(call_dir)
    broker.call("open_app", {"name": "Calculator"})
    obs = _obs_id(broker.call("observe_screen", {"app": ""}))
    broker.call("key", {"combo": "2", "app": "Calculator"})  # action AFTER the observation
    reply = json.loads(
        text(broker.call("finish", {"status": "done", "summary": "4", "evidence_obs_id": obs}))
    )
    assert reply["recorded_status"] == "unknown_partial"
    finish = json.loads(broker.finish_path.read_text())
    assert finish["status"] == "unknown_partial" and finish["requested_status"] == "done"
    assert broker.finish_path.stat().st_mode & 0o777 == 0o600


def test_finish_done_with_fresh_evidence(call_dir: Path) -> None:
    broker = make_broker(call_dir)
    broker.call("key", {"combo": "2", "app": "Calculator"})
    obs = _obs_id(broker.call("observe_screen", {"app": ""}))
    broker.call("finish", {"status": "done", "summary": "Shows 4.", "evidence_obs_id": obs})
    finish = json.loads(broker.finish_path.read_text())
    assert finish["status"] == "done" and finish["summary"] == "Shows 4."


@pytest.mark.parametrize("evidence", ["", "obs-999-dead"])
def test_finish_done_without_known_evidence(call_dir: Path, evidence: str) -> None:
    broker = make_broker(call_dir)
    broker.call("finish", {"status": "done", "summary": "s", "evidence_obs_id": evidence})
    assert json.loads(broker.finish_path.read_text())["status"] == "unknown_partial"


def test_failed_action_still_counts_as_an_action(call_dir: Path) -> None:
    runner = FakeRunner()
    broker = make_broker(call_dir, runner)
    obs = _obs_id(broker.call("observe_screen", {"app": ""}))
    runner.fail.add("open")
    assert text(broker.call("open_app", {"name": "Nope"})).startswith("error:")
    assert broker.judge_finish("done", obs)[0] == "unknown_partial"


def test_action_with_observe_returns_valid_evidence(call_dir: Path) -> None:
    broker = make_broker(call_dir)
    reply = broker.call("key", {"combo": "2", "app": "Calculator", "observe": True})
    shots = [p for p in reply if isinstance(p, Shot)]
    assert len(shots) == 1 and shots[0].data.startswith(b"\xff\xd8")
    assert broker.judge_finish("done", shots[0].meta["obs_id"]) == ("done", "")


def test_evidence_json_orders_observe_true_screenshot_after_its_action(call_dir: Path) -> None:
    broker = make_broker(call_dir)
    reply = broker.call("key", {"combo": "2", "app": "Calculator", "observe": True})
    action_seq = json.loads(text(reply).splitlines()[0])["seq"]
    obs_id = next(p for p in reply if isinstance(p, Shot)).meta["obs_id"]
    evidence = json.loads(broker.evidence_path.read_text())
    assert evidence["last_action_seq"] == action_seq
    assert evidence["observations"][obs_id] > action_seq
    assert broker.evidence_path.stat().st_mode & 0o777 == 0o600


def test_evidence_json_tracks_actions_after_observations(call_dir: Path) -> None:
    broker = make_broker(call_dir)
    obs = _obs_id(broker.call("observe_screen", {"app": ""}))
    broker.call("open_app", {"name": "Calculator"})
    evidence = json.loads(broker.evidence_path.read_text())
    assert evidence["observations"][obs] < evidence["last_action_seq"]


def test_evidence_json_keeps_only_the_last_observations(call_dir: Path) -> None:
    broker = make_broker(call_dir)
    ids = [_obs_id(broker.call("observe_screen", {"app": ""})) for _ in range(55)]
    observations = json.loads(broker.evidence_path.read_text())["observations"]
    assert len(observations) == mac_broker.EVIDENCE_KEEP
    assert set(observations) == set(ids[-mac_broker.EVIDENCE_KEEP :])


def test_observe_does_not_leave_screenshots(call_dir: Path) -> None:
    broker = make_broker(call_dir)
    reply = broker.call("observe_screen", {"app": ""})
    meta = json.loads(text(reply))
    assert meta["image_px"] == [800, 600]
    assert {p.name for p in call_dir.iterdir()} == {"broker_audit.jsonl", "evidence.json"}


def test_screenshot_failure_with_exit_status_zero_is_an_error(call_dir: Path) -> None:
    runner = FakeRunner()
    broker = make_broker(call_dir, runner)
    real = runner.__call__

    def silent_fail(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        if Path(argv[0]).name == "screencapture":  # writes nothing, still exits 0
            return subprocess.CompletedProcess(argv, 0, "", "cannot write file")
        return real(argv, timeout)

    broker.runner = silent_fail
    assert text(broker.call("observe_screen", {"app": ""})).startswith("error: screenshot failed")
    assert {p.name for p in call_dir.iterdir()} == {"broker_audit.jsonl"}


def test_capture_target_is_not_a_hidden_file(call_dir: Path) -> None:
    runner = FakeRunner()
    make_broker(call_dir, runner).call("observe_screen", {"app": ""})
    target = next(c[-1] for c in runner.calls if Path(c[0]).name == "screencapture")
    assert not Path(target).name.startswith(".") and Path(target).parent == call_dir


def test_click_maps_observation_pixels_to_screen_points(call_dir: Path) -> None:
    clicks: list[tuple[float, float]] = []
    hits: list[tuple[float, float]] = []

    def hit(x: float, y: float) -> dict[str, str]:
        hits.append((x, y))
        return {"AXRole": "AXButton", "AXTitle": "4"}

    broker = make_broker(call_dir, clicker=lambda x, y: clicks.append((x, y)), hit_test=hit)
    obs = _obs_id(broker.call("observe_screen", {"app": ""}))  # region 100,50 400x300 → 800x600 px
    reply = broker.call("click", {"x": 400, "y": 300, "obs_id": obs, "app": "Calculator"})
    assert '"ok": true' in text(reply)
    assert clicks == [(300.0, 200.0)] and hits == [(300.0, 200.0)]
    assert text(broker.call("click", {"x": 5000, "y": 1, "obs_id": obs})).startswith("error:")


# -- audit, call dir, server -------------------------------------------------------------------
def test_audit_has_no_content_and_mode_0600(call_dir: Path) -> None:
    broker = make_broker(call_dir)
    broker.call("type_text", {"app": "Safari", "text": "my-secret-words"})
    broker.call("safari_dom", {"selector": "h1.secret-heading"})
    broker.call("finish", {"status": "failed", "summary": "nothing to say", "evidence_obs_id": ""})
    raw = broker.audit_path.read_text()
    assert "my-secret-words" not in raw and "secret-heading" not in raw and "nothing" not in raw
    assert broker.audit_path.stat().st_mode & 0o777 == 0o600
    first = json.loads(raw.splitlines()[0])
    assert set(first) == {"tool", "args_sha256", "risk", "outcome", "t_s", "dur_ms"}


def test_args_sha_ignores_observe_and_key_order() -> None:
    a = args_sha256("key", {"combo": "cmd+w", "app": "Safari", "observe": True})
    b = args_sha256("key", {"app": "Safari", "combo": "cmd+w"})
    assert a == b and a != args_sha256("key", {"combo": "cmd+w", "app": "Mail"})


def test_call_dir_must_be_private_and_inside_root(tmp_path: Path) -> None:
    root = tmp_path / "operator"
    inside = root / "op-1"
    inside.mkdir(parents=True, mode=0o700)
    inside.chmod(0o700)
    assert check_call_dir(inside, root) == inside.resolve()
    outside = tmp_path / "elsewhere"
    outside.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="outside"):
        check_call_dir(outside, root)
    with pytest.raises(ValueError, match="outside"):
        check_call_dir(root / ".." / "elsewhere", root)
    inside.chmod(0o755)
    with pytest.raises(ValueError, match="0700"):
        check_call_dir(inside, root)
    assert mac_broker.main(["-c", str(outside), "-r", str(root)]) == 2


def test_server_lists_the_primitives(call_dir: Path) -> None:
    pytest.importorskip("mcp")
    server = mac_broker.build_server(make_broker(call_dir))
    names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert names == {
        "observe_screen",
        "list_ui",
        "safari_dom",
        "click",
        "type_text",
        "key",
        "open_url",
        "open_app",
        "run_shortcut",
        "menu_select",
        "applescript",
        "finish",
    }


def test_server_call_goes_through_the_risk_table(call_dir: Path) -> None:
    pytest.importorskip("mcp")
    runner = FakeRunner()
    server = mac_broker.build_server(make_broker(call_dir, runner))
    result = asyncio.run(server.call_tool("applescript", {"script": 'do shell script "id"'}))
    assert "refused" in result.content[0].text and runner.actions() == []


def test_import_is_light_and_help_works() -> None:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    probe = "import sys, my_stt_tts.mac_broker; print('mcp' in sys.modules, 'numpy' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=True
    )
    assert out.stdout.split() == ["False", "False"]
    broker = Path(mac_broker.__file__)
    helped = subprocess.run(
        [sys.executable, "-I", str(broker), "-h"], capture_output=True, text=True, check=False
    )
    assert helped.returncode == 0 and "--call-dir" in helped.stdout
