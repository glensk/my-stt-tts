"""Risk table of the Mac broker (PLAN_claude-bridge.md 7.4) and its confirmation files.

Pure standard library and side-effect free except :func:`hit_test_ax` and
:func:`focused_ax` (read-only Accessibility lookups of the element under a screen point and
of the focused element). :func:`classify` answers SAFE
(runs), CONFIRM (blocked until the user confirms by code) or REFUSE (never runs) for one
broker call; unknown tools are CONFIRM. Summaries name the action and its target, never
typed text, and are short enough to be spoken.

Confirmation binding: a blocked call is identified by :func:`args_sha256` (tool + its
normalised arguments). The broker writes ``blocked-<sha>.json`` with a fresh nonce; the
daemon answers with :func:`write_allow` (``allow-<sha>.json``, same nonce, 0600) after a
spoken ``confirm_action``; the broker consumes the allow file and runs that call once.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import secrets
import unicodedata
import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUMMARY_CAP = 80
URL_CAP = 2000
URL_QUERY_CAP = 200  # query + fragment longer than this → CONFIRM (screen content in a URL)
SAFE, CONFIRM, REFUSE = "safe", "confirm", "refuse"


def fold(text: str) -> str:
    """Casefold + strip accents, so ``Löschen`` / ``loschen`` and ``Sécurité`` match."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


# Words on a button / menu item / shortcut that make the action CONFIRM (folded, en/de/fr).
_RISKY_WORDS = (
    # closing / quitting
    "close", "closing", "quit", "force quit", "schliessen", "beenden", "fermer", "quitter",
    # deleting / moving / renaming files
    "delete", "deleting", "remove", "trash", "bin", "erase", "wipe", "empty", "move",
    "moving", "rename", "duplicate", "loschen", "entfernen", "papierkorb", "umbenennen",
    "verschieben", "bewegen", "supprimer", "effacer", "corbeille", "renommer", "deplacer",
    # sending / posting / messages
    "send", "sending", "post", "publish", "submit", "share", "reply", "forward", "tweet",
    "call", "senden", "absenden", "schicken", "posten", "veroffentlichen", "teilen",
    "antworten", "weiterleiten", "anrufen", "envoyer", "publier", "partager", "repondre",
    "transferer", "appeler",
    # purchases
    "buy", "purchase", "order", "pay", "payment", "checkout", "subscribe", "donate",
    "kaufen", "bestellen", "bezahlen", "zahlen", "abonnieren", "acheter", "commander",
    "payer", "abonner",
    # installs
    "install", "uninstall", "update", "upgrade", "installieren", "deinstallieren",
    "aktualisieren", "installer", "desinstaller",
    # security / privacy / network / accounts
    "sign out", "log out", "logout", "password", "passkey", "keychain", "privacy",
    "security", "firewall", "network", "wi-fi", "wifi", "vpn", "bluetooth", "permission",
    "allow", "grant", "revoke", "reset", "restart", "shut down", "shutdown", "format",
    "abmelden", "passwort", "datenschutz", "sicherheit", "netzwerk", "zurucksetzen",
    "neustart", "ausschalten", "erlauben", "deconnexion", "mot de passe",
    "confidentialite", "securite", "reseau", "reinitialiser", "redemarrer", "eteindre",
    "autoriser",
)  # fmt: skip
_RISKY_RE = re.compile(r"(?<![\w-])(?:" + "|".join(map(re.escape, _RISKY_WORDS)) + r")(?![\w-])")

# Labels that may be changed even in System Settings / the Control Center (SAFE topics).
_SAFE_TOPICS = (
    "focus", "do not disturb", "nicht storen", "ne pas deranger", "volume", "lautstarke",
    "brightness", "helligkeit", "luminosite", "dark mode", "dunkelmodus", "mode sombre",
    "appearance", "night shift", "sound", "ton", "son", "play", "pause", "next", "previous",
    "media", "music", "musik", "musique",
)  # fmt: skip
_SAFE_TOPIC_RE = re.compile(
    r"(?<![\w-])(?:" + "|".join(map(re.escape, _SAFE_TOPICS)) + r")(?![\w-])"
)

SHELL_APPS = frozenset(
    {"terminal", "iterm", "iterm2", "script editor", "automator", "warp", "alacritty",
     "kitty", "wezterm", "ghostty", "hyper", "console", "skriptprogramm", "editeur de script"}
)  # fmt: skip
MESSAGING_APPS = frozenset(
    {"messages", "nachrichten", "mail", "slack", "whatsapp", "telegram", "signal", "discord",
     "microsoft teams", "teams", "microsoft outlook", "outlook", "spark", "thunderbird",
     "skype", "facetime", "zoom", "zoom.us", "airmail", "mimestream"}
)  # fmt: skip
SETTINGS_APPS = frozenset(
    {"system settings", "system preferences", "systemeinstellungen", "reglages systeme",
     "preferences systeme", "app store", "installer", "installationsprogramm",
     "keychain access", "schlusselbundverwaltung", "trousseau d'acces"}
)  # fmt: skip
FILE_APPS = frozenset({"finder"})

# Roles that are harmless to click when they carry no label at all.
_BENIGN_ROLES = frozenset(
    {"axstatictext", "aximage", "axgroup", "axwebarea", "axscrollarea", "axtextfield",
     "axtextarea", "axsearchfield", "axcombobox", "axtabgroup", "axradiobutton", "axcell",
     "axrow", "axlist", "axoutline", "axtable", "axlink", "axheading", "axsplitgroup",
     "axlayoutarea", "axwindow", "axtoolbar", "axslider", "axscrollbar", "axvalueindicator"}
)  # fmt: skip

NAMED_KEYS = {
    "return": 36, "enter": 76, "tab": 48, "space": 49, "delete": 51, "backspace": 51,
    "forwarddelete": 117, "escape": 53, "esc": 53, "left": 123, "right": 124, "down": 125,
    "up": 126, "home": 115, "end": 119, "pageup": 116, "pagedown": 121,
    "brightnessup": 144, "brightnessdown": 145,
}  # fmt: skip
_MODS = {"cmd": "cmd", "command": "cmd", "shift": "shift", "alt": "alt", "option": "alt",
         "opt": "alt", "ctrl": "ctrl", "control": "ctrl"}  # fmt: skip
_NAV_KEYS = frozenset(
    {"left", "right", "up", "down", "home", "end", "pageup", "pagedown", "escape", "esc",
     "tab", "space"}
)  # fmt: skip
_SUBMIT_KEYS = frozenset({"return", "enter"})
_DELETE_KEYS = frozenset({"delete", "backspace", "forwarddelete"})
# Finder / settings keys that only navigate or search (everything else there is CONFIRM).
_BROWSE_COMBOS = frozenset(
    {("cmd", "f"), ("cmd", "1"), ("cmd", "2"), ("cmd", "3"), ("cmd", "4"), ("cmd", "["),
     ("cmd", "]"), ("cmd", "up"), ("cmd", "down"), ("cmd", "n"), ("cmd", "t")}
)  # fmt: skip

# AppleScript that could run a shell or evaluate hidden source → refused outright.
_SHELL_SCRIPT_RE = re.compile(
    r"do\s+shell\s+script|«\s*event\s+sysoexec|sysoexec|\brun\s+script\b|\bload\s+script\b",
    re.IGNORECASE,
)

# The focused field a plain Return may submit without confirmation: a search field or an
# address / URL bar (folded words of its subrole / identifier / description / title /
# placeholder, en/de/fr). "E-mail address" fields of web forms are not address bars.
_FIELD_SEARCH_RE = re.compile(r"search|suche|recherche")
_FIELD_ADDRESS_RE = re.compile(r"address|adresse|url")
_FIELD_MAIL_RE = re.compile(r"mail|courriel")
_FIELD_KEYS = ("AXSubrole", "AXIdentifier", "AXDescription", "AXTitle", "AXPlaceholderValue")

CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


@dataclass(frozen=True)
class Verdict:
    """The risk table's answer for one call: ``risk`` + a one-line, content-free summary."""

    risk: str
    summary: str
    reason: str = ""


def _label(text: str) -> str:
    """A label safe to speak and log: no control chars, one line, capped."""
    clean = " ".join(CONTROL_RE.sub(" ", str(text)).split())
    return clean[:40]


def clean_summary(text: str) -> str:
    return " ".join(CONTROL_RE.sub(" ", text).split())[:SUMMARY_CAP]


def risky_words(text: str) -> bool:
    """True when ``text`` (a label, menu path, shortcut name) names a risky action."""
    return bool(_RISKY_RE.search(fold(text)))


def safe_topic(text: str) -> bool:
    return bool(_SAFE_TOPIC_RE.search(fold(text)))


def shell_script_refusal(script: str) -> str | None:
    """Why ``script`` is refused outright (``do shell script`` and its bypasses), or None."""
    if _SHELL_SCRIPT_RE.search(unicodedata.normalize("NFKC", script)):
        return "applescript may not run shell commands (do shell script / run script)"
    return None


def search_or_address_field(info: Mapping[str, str] | None) -> bool:
    """True when the focused element ``info`` is a search field or an address / URL bar."""
    if not info:
        return False
    role = fold(info.get("AXRole", ""))
    if role == "axsearchfield":
        return True
    if role not in {"axtextfield", "axcombobox"}:
        return False
    if fold(info.get("AXSubrole", "")) == "axsecuretextfield":
        return False
    words = fold(" ".join(str(info.get(k, "")) for k in _FIELD_KEYS))
    if _FIELD_SEARCH_RE.search(words):
        return True
    return bool(_FIELD_ADDRESS_RE.search(words)) and not _FIELD_MAIL_RE.search(words)


def parse_combo(combo: str) -> tuple[str, list[str]]:
    """``'cmd+shift+a'`` → ``('a', ['cmd', 'shift'])``; named keys lower-cased."""
    text = combo.strip()
    if not text:
        raise ValueError("empty key combo")
    parts = [p.strip() for p in text.split("+")]
    if text.endswith("++"):  # the plus key itself, e.g. "shift++"
        parts = [*parts[:-2], "+"]
    *mod_parts, key = parts
    mods: list[str] = []
    for mod in mod_parts:
        if mod.lower() not in _MODS:
            raise ValueError(f"unknown modifier {mod!r}")
        mods.append(_MODS[mod.lower()])
    low = key.lower().replace("_", "").replace(" ", "")
    if low in NAMED_KEYS:
        return low, sorted(set(mods))
    if len(key) != 1:
        raise ValueError(f"unknown key {key!r} (one character or one of {sorted(NAMED_KEYS)})")
    return key.lower(), sorted(set(mods))


def validate_url(target: str) -> str:
    """An http(s) URL to open, or ValueError (no other schemes, credentials, control chars)."""
    raw = str(target).strip()
    if not raw or len(raw) > URL_CAP:
        raise ValueError("URL empty or too long")
    if CONTROL_RE.search(raw) or " " in raw:
        raise ValueError("URL contains control characters or spaces")
    shortcuts = {"youtube": "https://www.youtube.com", "jellyfin": "https://jellyfin.dom42.space"}
    if raw.casefold() in shortcuts:
        return shortcuts[raw.casefold()]
    match = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*):", raw)
    if match and match.group(1).lower() not in {"http", "https"}:
        raise ValueError(f"only http(s) URLs may be opened, not {match.group(1).lower()}:")
    url = raw if match else f"https://{raw}"
    rest = url.split("://", 1)[1]
    authority = rest.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    if "@" in authority:
        raise ValueError("URLs with credentials are refused")
    if not authority or not re.match(r"^[A-Za-z0-9.-]+(:\d+)?$", authority):
        raise ValueError("URL has no valid host")
    return url


def validate_app_name(name: str) -> str:
    """An app name for ``open -a`` (no paths, options or control chars)."""
    clean = str(name).strip()
    if not clean or len(clean) > 100:
        raise ValueError("app name empty or too long")
    if "/" in clean or clean.startswith("-") or CONTROL_RE.search(clean) or "~" in clean:
        raise ValueError("app name must be a plain name, not a path")
    return clean


class RiskContext:
    """What classification may look up: the front app, the element under a point and the
    focused element (``app.AXTitle`` = the app owning the focus). Failures → unknown."""

    def __init__(
        self,
        front_app: Callable[[], str] | None = None,
        hit_test: Callable[[float, float], Mapping[str, str] | None] | None = None,
        focused: Callable[[], Mapping[str, str] | None] | None = None,
    ) -> None:
        self._front_app = front_app
        self._hit_test = hit_test
        self._focused = focused

    def app(self, given: str) -> str:
        if given.strip():
            return given.strip()
        if self._front_app is None:
            return ""
        try:
            return self._front_app().strip()
        except Exception:  # pylint: disable=broad-exception-caught  # unknown → conservative
            return ""

    def element_at(self, x: float, y: float) -> Mapping[str, str] | None:
        if self._hit_test is None:
            return None
        try:
            return self._hit_test(x, y)
        except Exception:  # pylint: disable=broad-exception-caught  # unknown → conservative
            return None

    def focused(self) -> Mapping[str, str] | None:
        if self._focused is None:
            return None
        try:
            return self._focused()
        except Exception:  # pylint: disable=broad-exception-caught  # unknown → conservative
            return None


def _in(app: str, apps: frozenset[str]) -> bool:
    return fold(app) in apps


def _app_label(app: str) -> str:
    return _label(app) or "the front app"


def _focus_takes_return(given_app: str, ctx: RiskContext) -> bool:
    """True when Return goes to a search field / address bar (of ``given_app`` if named)."""
    info = ctx.focused()
    if info is None or not search_or_address_field(info):
        return False
    if given_app.strip():  # the key is pressed in given_app: the focus must be in it
        return fold(info.get("app.AXTitle", "")) == fold(given_app.strip())
    return True


def _classify_key(args: Mapping[str, Any], ctx: RiskContext) -> Verdict:
    combo = str(args.get("combo", ""))
    try:
        key, mods = parse_combo(combo)
    except ValueError as exc:
        return Verdict(SAFE, f"press {_label(combo)}", f"invalid: {exc}")  # runs as an error
    app = ctx.app(str(args.get("app", "")))
    summary = f"press {'+'.join([*mods, key])} in {_app_label(app)}"
    mod_set = set(mods)
    confirm = (
        ("cmd" in mod_set and key in {"w", "q"})  # close tab / window, quit
        or ("cmd" in mod_set and key in _DELETE_KEYS)  # move to trash / delete
        or ("cmd" in mod_set and key in _SUBMIT_KEYS)  # send in many apps
        or ("cmd" in mod_set and key == "s")  # save / save as / save all: writes files
        or ({"cmd", "alt"} <= mod_set and key in {"escape", "esc"})  # force quit
        or ({"cmd", "shift"} <= mod_set and key in {"d", "q"})  # Mail send, log out
        or ({"cmd", "ctrl"} <= mod_set and key == "q")  # lock screen
        or not app  # unknown target app
    )
    if not confirm and _in(app, SHELL_APPS):
        confirm = key not in _NAV_KEYS or bool(mods)
    if not confirm and _in(app, MESSAGING_APPS):
        confirm = key in _SUBMIT_KEYS
    if not confirm and (_in(app, FILE_APPS) or _in(app, SETTINGS_APPS)):
        browse = (not mods and key in _NAV_KEYS - {"space"}) or (*mods, key) in _BROWSE_COMBOS
        confirm = not browse
    if not confirm and key in _SUBMIT_KEYS:  # Return submits forms: only search / address
        confirm = not _focus_takes_return(str(args.get("app", "")), ctx)
        if confirm and not mods:
            summary = f"press {key.capitalize()} in {_app_label(app)}"
    return Verdict(CONFIRM if confirm else SAFE, summary)


def _classify_type(args: Mapping[str, Any], ctx: RiskContext) -> Verdict:
    text = str(args.get("text", ""))
    app = ctx.app(str(args.get("app", "")))
    summary = f"type text into {_app_label(app)}"
    if "\n" in text or "\r" in text:
        return Verdict(CONFIRM, f"type text with Return into {_app_label(app)}")
    if not app or any(_in(app, apps) for apps in (SHELL_APPS, FILE_APPS, SETTINGS_APPS)):
        return Verdict(CONFIRM, summary)
    return Verdict(SAFE, summary)


def _element_label(info: Mapping[str, str]) -> tuple[str, str]:
    """``(role, label)`` of a hit-tested element (parent button wins over its text)."""
    role = fold(info.get("AXRole", ""))
    parent_role = fold(info.get("parent.AXRole", ""))
    if role in {"axstatictext", "aximage"} and parent_role in {
        "axbutton",
        "axmenuitem",
        "axmenubutton",
        "axpopupbutton",
        "axcheckbox",
    }:
        role = parent_role
    keys = ("AXTitle", "AXDescription", "AXHelp", "AXIdentifier", "AXValue")
    words = [info.get(k, "") for k in keys]
    subrole = info.get("AXSubrole", "").removeprefix("AX")  # AXCloseButton → "Close Button"
    words.append(re.sub(r"(?<=[a-z])(?=[A-Z])", " ", subrole))
    words += [info.get(f"parent.{k}", "") for k in ("AXTitle", "AXDescription")]
    return role, " ".join(w for w in words if w)


def _classify_click(args: Mapping[str, Any], ctx: RiskContext) -> Verdict:
    app = ctx.app(str(args.get("app", "")))
    element = str(args.get("element", "") or "")
    point = args.get("_point")
    if element.strip():
        role, label = "", element
        summary = f"click '{_label(element)}' in {_app_label(app)}"
    elif isinstance(point, tuple):
        info = ctx.element_at(point[0], point[1])
        if info is None:
            return Verdict(CONFIRM, f"click an unidentified spot in {_app_label(app)}")
        role, label = _element_label(info)
        shown = _label(info.get("AXTitle") or info.get("AXDescription") or label)
        what = f"'{shown}'" if shown else (role.removeprefix("ax") or "something")
        summary = f"click {what} in {_app_label(app)}"
    else:
        return Verdict(SAFE, "click (invalid arguments)", "invalid")
    confirm = (
        risky_words(label)
        or (_in(app, SETTINGS_APPS) and not safe_topic(label))
        or (not label.strip() and bool(role) and role not in _BENIGN_ROLES)  # unlabelled button
        or not app
    )
    return Verdict(CONFIRM if confirm else SAFE, summary)


def _classify_menu(args: Mapping[str, Any], ctx: RiskContext) -> Verdict:
    path = str(args.get("path", ""))
    app = ctx.app(str(args.get("app", "")))
    summary = f"choose {_label(path)} in {_app_label(app)}"
    if risky_words(path) or _in(app, SETTINGS_APPS) or not app:
        return Verdict(CONFIRM, summary)
    return Verdict(SAFE, summary)


def _classify_url(args: Mapping[str, Any]) -> Verdict:
    try:
        url = validate_url(str(args.get("url", "")))
    except ValueError as exc:
        return Verdict(REFUSE, "open a URL", str(exc))
    host = url.split("://", 1)[1].split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    parts = urllib.parse.urlsplit(url)
    if len(parts.query) + len(parts.fragment) > URL_QUERY_CAP:
        return Verdict(CONFIRM, f"open {_label(host)} with a long query")
    return Verdict(SAFE, f"open {_label(host)}")


def _classify_app(args: Mapping[str, Any]) -> Verdict:
    try:
        name = validate_app_name(str(args.get("name", "")))
    except ValueError as exc:
        return Verdict(REFUSE, "open an app", str(exc))
    summary = f"open {_label(name)}"
    return Verdict(CONFIRM if risky_words(name) else SAFE, summary)


def _classify_shortcut(args: Mapping[str, Any]) -> Verdict:
    name = str(args.get("name", ""))
    summary = f"run the shortcut {_label(name)}"
    if not name.strip() or CONTROL_RE.search(name):
        return Verdict(REFUSE, summary, "invalid shortcut name")
    if safe_topic(name) and not risky_words(name):
        return Verdict(SAFE, summary)
    return Verdict(CONFIRM, summary)


def classify(tool: str, args: Mapping[str, Any], ctx: RiskContext | None = None) -> Verdict:
    """The broker's risk table (7.4): SAFE runs, CONFIRM blocks, REFUSE never runs.

    Unknown tools are CONFIRM. Summaries name the action and target, never typed text.
    """
    ctx = ctx or RiskContext()
    verdict: Verdict
    if tool in {"observe_screen", "list_ui", "finish"}:
        verdict = Verdict(SAFE, tool.replace("_", " "))
    elif tool == "safari_dom":  # a fixed read-only template; the selector is data only
        verdict = Verdict(SAFE, "read the Safari page")
    elif tool == "applescript":
        refusal = shell_script_refusal(str(args.get("script", "")))
        verdict = Verdict(REFUSE, "run an AppleScript", refusal or "")
        if refusal is None:
            verdict = Verdict(CONFIRM, "run an AppleScript")
    elif tool == "key":
        verdict = _classify_key(args, ctx)
    elif tool == "type_text":
        verdict = _classify_type(args, ctx)
    elif tool == "click":
        verdict = _classify_click(args, ctx)
    elif tool == "menu_select":
        verdict = _classify_menu(args, ctx)
    elif tool == "open_url":
        verdict = _classify_url(args)
    elif tool == "open_app":
        verdict = _classify_app(args)
    elif tool == "run_shortcut":
        verdict = _classify_shortcut(args)
    else:
        verdict = Verdict(CONFIRM, f"use the unknown tool {_label(tool)}")
    return Verdict(verdict.risk, clean_summary(verdict.summary), verdict.reason)


# -- confirmation binding --------------------------------------------------------------------
_NON_ACTION_KEYS = frozenset({"observe"})


def _norm(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        return {str(k): _norm(v) for k, v in value.items() if v is not None}
    if isinstance(value, list | tuple):
        return [_norm(v) for v in value]
    return value


def args_sha256(tool: str, args: Mapping[str, Any]) -> str:
    """sha256 of the tool name + its normalised arguments (NFC, key order, no ``observe``)."""
    clean = {k: v for k, v in args.items() if k not in _NON_ACTION_KEYS and not k.startswith("_")}
    blob = json.dumps(
        {"tool": tool, "args": _norm(clean)},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()


SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def blocked_path(call_dir: Path, sha: str) -> Path:
    return call_dir / f"blocked-{sha}.json"


def allow_path(call_dir: Path, sha: str) -> Path:
    return call_dir / f"allow-{sha}.json"


def write_0600(path: Path, text: str) -> None:
    """Write ``text`` atomically (temp + rename) with mode 0600."""
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, text.encode())
    finally:
        os.close(fd)
    os.replace(tmp, path)


def append_0600(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, text.encode())
    finally:
        os.close(fd)


def write_allow(call_dir: Path, sha: str, nonce: str) -> bool:
    """The daemon's half: allow the blocked call ``sha`` whose blocked file has ``nonce``."""
    if not SHA_RE.match(sha):
        return False
    try:
        blocked = json.loads(blocked_path(call_dir, sha).read_text())
    except (OSError, ValueError):
        return False
    if not secrets.compare_digest(str(blocked.get("nonce", "")), str(nonce)):
        return False
    write_0600(allow_path(call_dir, sha), json.dumps({"sha256": sha, "nonce": nonce}) + "\n")
    return True


def _ax_libs() -> tuple[ctypes.CDLL, ctypes.CDLL] | None:
    """ApplicationServices + CoreFoundation with the prototypes :func:`hit_test_ax` uses."""
    try:
        ax = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
        )
        cf = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
    except OSError:
        return None
    vp, flt = ctypes.c_void_p, ctypes.c_float
    ax.AXUIElementCreateSystemWide.restype = vp
    ax.AXUIElementCopyElementAtPosition.argtypes = [vp, flt, flt, ctypes.POINTER(vp)]
    ax.AXUIElementCopyElementAtPosition.restype = ctypes.c_int32
    ax.AXUIElementCopyAttributeValue.argtypes = [vp, vp, ctypes.POINTER(vp)]
    ax.AXUIElementCopyAttributeValue.restype = ctypes.c_int32
    cf.CFStringCreateWithCString.argtypes = [vp, ctypes.c_char_p, ctypes.c_uint32]
    cf.CFStringCreateWithCString.restype = vp
    cf.CFGetTypeID.argtypes = [vp]
    cf.CFGetTypeID.restype = ctypes.c_ulong
    cf.CFStringGetTypeID.restype = ctypes.c_ulong
    cf.CFStringGetCString.argtypes = [vp, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
    cf.CFStringGetCString.restype = ctypes.c_bool
    cf.CFRelease.argtypes = [vp]
    return ax, cf


_UTF8 = 0x08000100


def _ax_attr(
    libs: tuple[ctypes.CDLL, ctypes.CDLL], element: int, name: str
) -> tuple[str | None, int | None]:
    """``(text, None)`` for a string attribute, ``(None, ref)`` for another CF object (the
    caller releases ``ref``), ``(None, None)`` when the attribute is missing."""
    ax, cf = libs
    key = cf.CFStringCreateWithCString(None, name.encode(), _UTF8)
    val = ctypes.c_void_p()
    try:
        if ax.AXUIElementCopyAttributeValue(element, key, ctypes.byref(val)) or not val.value:
            return None, None
        if cf.CFGetTypeID(val) == cf.CFStringGetTypeID():
            buf = ctypes.create_string_buffer(512)
            ok = cf.CFStringGetCString(val, buf, 512, _UTF8)
            cf.CFRelease(val)
            return (buf.value.decode(errors="replace") if ok else None), None
        return None, val.value
    finally:
        cf.CFRelease(key)


def hit_test_ax(x: float, y: float) -> dict[str, str] | None:
    """Accessibility attributes of the element under screen point (x, y), or None.

    ctypes against ApplicationServices / CoreFoundation (no pyobjc extra needed); the
    first parent's role / title / description come along as ``parent.<attr>``.
    """
    libs = _ax_libs()
    if libs is None:
        return None
    ax, cf = libs
    vp = ctypes.c_void_p

    def attr(element: int, name: str) -> tuple[str | None, int | None]:
        return _ax_attr(libs, element, name)

    system = ax.AXUIElementCreateSystemWide()
    element = vp()
    failed = ax.AXUIElementCopyElementAtPosition(system, float(x), float(y), ctypes.byref(element))
    el = element.value
    if failed or not el:
        cf.CFRelease(system)
        return None
    out: dict[str, str] = {}
    names = ("AXRole", "AXSubrole", "AXTitle", "AXDescription", "AXHelp", "AXIdentifier")
    for name in names:
        text, _ = attr(el, name)
        if text:
            out[name] = text
    if out.get("AXRole") == "AXStaticText":
        text, _ = attr(el, "AXValue")
        if text:
            out["AXValue"] = text
    _, parent = attr(el, "AXParent")
    if parent:
        for name in ("AXRole", "AXTitle", "AXDescription"):
            text, _ = attr(parent, name)
            if text:
                out[f"parent.{name}"] = text
        cf.CFRelease(parent)
    cf.CFRelease(el)
    cf.CFRelease(system)
    return out


def focused_ax() -> dict[str, str] | None:
    """Accessibility attributes of the focused UI element, or None when unknown.

    Role / subrole / identifier / description / title / placeholder of the system-wide
    ``AXFocusedUIElement``, plus ``app.AXTitle`` = the title of ``AXFocusedApplication``.
    """
    libs = _ax_libs()
    if libs is None:
        return None
    ax, cf = libs
    system = ax.AXUIElementCreateSystemWide()
    if not system:
        return None
    try:
        _, element = _ax_attr(libs, system, "AXFocusedUIElement")
        if not element:
            return None
        out: dict[str, str] = {}
        try:
            for name in ("AXRole", *_FIELD_KEYS):
                text, ref = _ax_attr(libs, element, name)
                if ref:
                    cf.CFRelease(ref)
                if text:
                    out[name] = text
        finally:
            cf.CFRelease(element)
        _, app = _ax_attr(libs, system, "AXFocusedApplication")
        if app:
            text, ref = _ax_attr(libs, app, "AXTitle")
            if ref:
                cf.CFRelease(ref)
            if text:
                out["app.AXTitle"] = text
            cf.CFRelease(app)
        return out or None
    finally:
        cf.CFRelease(system)
