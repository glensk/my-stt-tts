"""Safari site adapters for the Mac fast path: YouTube and Jellyfin (spike S-MEDIA).

Selectors, actions and verification expressions come from spike S-MEDIA
(``tests/fixtures/spikes/s_media.json``). Rules the spike settled:

* Safari is driven through ``osascript`` with the JavaScript and the window id as argv
  (never a shell, never string interpolation into AppleScript); the target is the FRONT
  window's id, read once, then every call names ``window id N`` (never an index).
* Every snippet re-checks the page's host, uses DOM methods only (YouTube's Trusted
  Types make ``innerHTML`` throw) and never writes ``video.volume`` (Jellyfin persists it).
* Postconditions: ``play`` → ``paused == false`` re-checked ONCE ~1.5 s after ``play()``
  (a Bluetooth headset can send ``Pause``) and never retried in a loop; ``next`` /
  ``previous`` → the video / item id changed; ``fullscreen`` → ``fullscreenElement`` or
  ``webkitDisplayingFullscreen``. Playlist end / queue start are refused with a reason.
* ``play()`` rejected with ``NotAllowedError`` (autoplay policy) → the optional
  ``real_click`` hook (the Mac operator's broker, Phase 5) clicks the player; without it
  the command is refused with the reason.

Each JS snippet starts with a ``/*mv:<op>*/`` marker so tests (and logs) can tell the
operations apart without parsing JavaScript.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

OSASCRIPT = "/usr/bin/osascript"
YOUTUBE_HOST = "www.youtube.com"
JELLYFIN_HOST = "jellyfin.dom42.space"
JELLYFIN_ID_RE = r"/videos/([0-9a-f-]{32,36})/"
_JS_ID_RE = JELLYFIN_ID_RE.replace("/", "\\/")  # as a JavaScript regex literal
WATCH_URL_RE = re.compile(r"https://www\.youtube\.com/watch\?v=[A-Za-z0-9_-]{6,20}")

PLAY_RECHECK_S = 1.5  # re-check ``paused`` once this long after play()
PROMISE_WAIT_S = 0.6  # how long to wait for play()/requestFullscreen() to settle
PAUSE_WAIT_S = 1.5
SEEK_SETTLE_S = 0.8
SEEK_TOLERANCE_S = 2.0
CHANGE_WAIT_S = 4.0  # next / previous: the id must change within this
NAV_WAIT_S = 8.0  # youtube_play_first: the watch page must load within this
POLL_S = 0.3
CALL_TIMEOUT_S = 2.0  # one osascript call after the action (verification)

MEDIA_COMMANDS = ("play", "pause", "seek", "next", "previous", "fullscreen")


# -- running commands ------------------------------------------------------------------------
@dataclass(frozen=True)
class RunResult:
    """What a subprocess returned (``returncode`` 124 = timeout, 127 = not startable)."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


Runner = Callable[[Sequence[str], float], RunResult]


def run_argv(argv: Sequence[str], timeout: float) -> RunResult:
    """Run ``argv`` (no shell) with a timeout; never raises."""
    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=max(timeout, 0.1),
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return RunResult(124, "", "timeout")
    except OSError as exc:
        return RunResult(127, "", str(exc))
    return RunResult(proc.returncode, proc.stdout, proc.stderr)


def osascript(runner: Runner, script: str, *argv: str, timeout: float) -> RunResult:
    """``osascript -e script argv…`` through ``runner``."""
    return runner([OSASCRIPT, "-e", script, *argv], timeout)


# -- outcomes ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Outcome:
    """``status`` ok | refused | failed; ``message`` short and speakable."""

    status: str
    message: str

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def ok(message: str) -> Outcome:
    return Outcome("ok", message)


def refused(message: str) -> Outcome:
    return Outcome("refused", message)


def failed(message: str) -> Outcome:
    return Outcome("failed", message)


# -- Safari --------------------------------------------------------------------------------
FRONT_WINDOW = """if application "Safari" is running then
tell application "Safari"
if (count of windows) > 0 then return (id of front window) as text
end tell
end if
return ""
"""

RUN_JS = """on run argv
tell application "Safari" to do JavaScript (item 1 of argv) in current tab of window id ((item 2 of argv) as integer)
end run"""

SET_URL = """on run argv
tell application "Safari" to set URL of current tab of window id ((item 2 of argv) as integer) to (item 1 of argv)
end run"""

JS_DISABLED_HINT = "Allow JavaScript from Apple Events"


class SafariError(Exception):
    """A Safari call failed; ``reason`` is short and speakable."""

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


def safari_error(result: RunResult) -> SafariError:
    """Map an osascript failure to a speakable :class:`SafariError`."""
    err = result.stderr
    if result.returncode == 124:
        return SafariError("timeout", "Safari did not answer in time")
    if JS_DISABLED_HINT in err:
        return SafariError(
            "js_disabled", "Safari's 'Allow JavaScript from Apple Events' is off (mac-voice -D)"
        )
    if "-1743" in err or "Not authorized" in err:
        return SafariError("automation_denied", "no permission to control Safari (mac-voice -D)")
    if "-1728" in err or "-1719" in err:
        return SafariError("window_gone", "the Safari window is gone")
    return SafariError("script_error", "Safari did not accept the command")


PRELUDE = (
    "const __vis = e => { if (!e) return false; const r = e.getBoundingClientRect();"
    " return r.width > 0 && r.height > 0 && r.bottom > 0 && r.top < innerHeight; };"
    " const __fs = () => { const fe = document.fullscreenElement"
    " || document.webkitFullscreenElement; const v = document.querySelector('video');"
    " return !!fe || !!(v && v.webkitDisplayingFullscreen); };"
)


def wrap(op: str, body: str) -> str:
    """A page snippet: ``body`` (statements ending in ``return <json-able>``) as JSON text."""
    return (
        f"/*mv:{op}*/(function(){{{PRELUDE}\n"
        f"return JSON.stringify((function(){{\n{body}\n}})());}})()"
    )


class Safari:
    """Window-id-scoped Safari driver (JavaScript + navigation through argv)."""

    def __init__(self, runner: Runner) -> None:
        self.runner = runner

    def front_window(self, timeout: float) -> int | None:
        """The front window's id (Safari not running / no window → None)."""
        result = osascript(self.runner, FRONT_WINDOW, timeout=timeout)
        if result.returncode != 0:
            raise safari_error(result)
        text = result.stdout.strip()
        return int(text) if text.isdigit() else None

    def evaluate(self, win: int, op: str, body: str, timeout: float = CALL_TIMEOUT_S) -> Any:
        """Run ``body`` in the window's current tab and return its JSON value."""
        result = osascript(self.runner, RUN_JS, wrap(op, body), str(win), timeout=timeout)
        if result.returncode != 0:
            raise safari_error(result)
        text = result.stdout.strip()
        if text in ("", "missing value"):
            return None
        try:
            return json.loads(text)
        except ValueError as exc:
            raise SafariError("bad_reply", "Safari sent an unexpected reply") from exc

    def set_url(self, win: int, url: str, timeout: float = CALL_TIMEOUT_S) -> None:
        result = osascript(self.runner, SET_URL, url, str(win), timeout=timeout)
        if result.returncode != 0:
            raise safari_error(result)


# -- adapters ------------------------------------------------------------------------------
@dataclass(frozen=True)
class SiteAdapter:  # pylint: disable=too-many-instance-attributes  # a plain record
    """JavaScript of one site (expressions use ``V`` = video, ``P`` = player, ``S`` = s)."""

    name: str
    host: str
    video: str
    player: str
    item_id: str
    pause: str
    seek: str
    next: str
    previous: str
    fullscreen: str
    previous_in_list: str = ""
    position: str = ""  # YouTube: sync {list, index, length}
    queue: str = ""  # Jellyfin: promise of {queue, now}
    first_tile: str = ""
    play: str = "V.play()"
    labels: dict[str, str] = field(default_factory=dict)


YOUTUBE = SiteAdapter(
    name="youtube",
    host=YOUTUBE_HOST,
    video=(
        "document.querySelector('#movie_player video.html5-main-video')"
        " || document.querySelector('video')"
    ),
    player="document.getElementById('movie_player')",
    item_id="new URL(location.href).searchParams.get('v')",
    pause="P && P.pauseVideo ? P.pauseVideo() : V.pause()",
    seek="P && P.seekTo ? P.seekTo(P.getCurrentTime() + S, true) : (V.currentTime += S)",
    next="P.nextVideo()",
    previous="history.back()",
    previous_in_list="P.previousVideo()",
    fullscreen="(P.requestFullscreen ? P.requestFullscreen() : P.webkitRequestFullscreen())",
    position=(
        "(() => { const u = new URL(location.href);"
        " const pl = P && P.getPlaylist ? P.getPlaylist() : null;"
        " return { list: u.searchParams.get('list'),"
        " index: P && P.getPlaylistIndex ? P.getPlaylistIndex() : -1,"
        " length: pl ? pl.length : 0 }; })()"
    ),
    first_tile=(
        "[...document.querySelectorAll("
        '\'ytd-rich-item-renderer a.ytLockupViewModelContentImage[href^="/watch"],'
        ' ytd-rich-item-renderer a#thumbnail[href^="/watch"]\')]'
        ".find(a => __vis(a) && !a.closest('ytd-ad-slot-renderer'))"
    ),
    labels={"end": "the end of the playlist", "start": "the start of the playlist"},
)

JELLYFIN = SiteAdapter(
    name="jellyfin",
    host=JELLYFIN_HOST,
    video="document.querySelector('video.htmlvideoplayer') || document.querySelector('video')",
    player="document.querySelector('.videoOsdBottom') || document.body",
    item_id=(
        "(() => { const m = V && (V.currentSrc || V.src || '')"
        f".match(/{_JS_ID_RE}/i);"
        " return m ? m[1].replace(/-/g, '')"
        " : new URLSearchParams(location.hash.split('?')[1] || '').get('id'); })()"
    ),
    pause="V.pause()",
    seek="(V.currentTime += S)",
    next="__click('.btnNextTrack')",
    previous="__click('.btnPreviousTrack')",
    fullscreen="V.requestFullscreen()",
    queue=(
        "ApiClient.getJSON(ApiClient.getUrl('Sessions', {DeviceId: ApiClient.deviceId()}))"
        ".then(r => { const s = r[0] || {}; const q = (s.NowPlayingQueue || []).map(x => x.Id);"
        " return { queue: q, now: s.NowPlayingItem ? s.NowPlayingItem.Id : null }; })"
    ),
    labels={"end": "the end of the queue", "start": "the start of the queue"},
)

ADAPTERS = (YOUTUBE, JELLYFIN)
CLICK = (
    "const __click = s => { const b = document.querySelector(s);"
    " if (!b) throw new Error('no button'); b.click(); };"
)


def adapter_for(site: str | None) -> SiteAdapter | None:
    return next((a for a in ADAPTERS if a.name == site), None)


def _bind(adapter: SiteAdapter) -> str:
    """Host guard + ``V`` / ``P`` bindings for one site's snippets."""
    return (
        f"if (location.hostname !== {json.dumps(adapter.host)}) return {{error: 'page_changed'}};"
        f" const V = {adapter.video}; const P = {adapter.player}; {CLICK}"
    )


def state_js() -> str:
    """One snippet that detects the site and reports the player state."""
    parts = ["const H = location.hostname;"]
    for a in ADAPTERS:
        parts.append(
            f"if (H === {json.dumps(a.host)}) {{ const V = {a.video}; const P = {a.player};"
            f" return {{site: {json.dumps(a.name)}, video: !!V, id: V ? {a.item_id} : null,"
            " paused: V ? V.paused : null, t: V ? V.currentTime : null,"
            " duration: V && isFinite(V.duration) ? V.duration : null, fullscreen: __fs()}; }"
        )
    parts.append("return {site: null};")
    return " ".join(parts)


def action_js(adapter: SiteAdapter, statement: str) -> str:
    """Run ``statement`` on the site's video (refuses when the page changed / no video)."""
    return (
        f"{_bind(adapter)} if (!V) return {{error: 'no_video'}};"
        f" try {{ {statement}; }} catch (e) {{ return {{error: e.name || 'error'}}; }}"
        " return {ok: true};"
    )


def promise_js(adapter: SiteAdapter, slot: str, expr: str) -> str:
    """Start the promise ``expr``; its settled state lands in ``window.__mv_<slot>``."""
    store = f"window.__mv_{slot}"
    return (
        f"{_bind(adapter)} if (!V) return {{error: 'no_video'}}; {store} = {{state: 'pending'}};"
        f" try {{ Promise.resolve({expr}).then(() => {store} = {{state: 'resolved'}},"
        f" e => {store} = {{state: 'rejected', name: e && e.name}}); }}"
        f" catch (e) {{ {store} = {{state: 'rejected', name: e && e.name}}; }}"
        " return {ok: true};"
    )


def slot_js(slot: str) -> str:
    return f"return window.__mv_{slot} || null;"


def queue_js(adapter: SiteAdapter) -> str:
    """Jellyfin's queue as a promise stored in ``window.__mv_queue`` (value kept)."""
    store = "window.__mv_queue"
    return (
        f"{_bind(adapter)} {store} = {{state: 'pending'}};"
        f" try {{ Promise.resolve({adapter.queue}).then(v => {store} = "
        f"{{state: 'resolved', value: v}}, e => {store} = {{state: 'rejected'}}); }}"
        f" catch (e) {{ {store} = {{state: 'rejected'}}; }} return {{ok: true}};"
    )


def first_tile_js(adapter: SiteAdapter) -> str:
    return (
        f"{_bind(adapter)} const A = {adapter.first_tile};"
        " return A ? 'https://www.youtube.com/watch?v=' + new URL(A.href).searchParams.get('v')"
        " : null;"
    )


def every_js() -> list[str]:
    """Every snippet the adapters can send (tests check DOM-only, no volume writes)."""
    snippets = [state_js(), slot_js("play")]
    for a in ADAPTERS:
        for statement in (a.pause, a.seek, a.next, a.previous, a.previous_in_list):
            if statement:
                snippets.append(action_js(a, statement))
        snippets.append(promise_js(a, "play", a.play))
        snippets.append(promise_js(a, "fullscreen", a.fullscreen))
        if a.position:
            snippets.append(f"{_bind(a)} return {a.position};")
        if a.queue:
            snippets.append(queue_js(a))
        if a.first_tile:
            snippets.append(first_tile_js(a))
    return [wrap("check", s) for s in snippets]


# -- the media driver ------------------------------------------------------------------------
@dataclass
class Page:
    """The front Safari window, what its page reported and (after a precheck) its position."""

    win: int
    adapter: SiteAdapter
    state: dict[str, Any]
    position: dict[str, int] | None = None

    @property
    def item(self) -> str | None:
        value = self.state.get("id")
        return str(value) if value else None


RealClick = Callable[[int, str], bool]


class MediaDriver:
    """Inspect, precheck and run media commands on the front Safari window."""

    def __init__(
        self,
        safari: Safari,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        real_click: RealClick | None = None,
    ) -> None:
        self.safari = safari
        self.sleep = sleep
        self.clock = clock
        self.real_click = real_click

    # -- reading -----------------------------------------------------------------------
    def inspect(self, timeout: float) -> Page | Outcome:
        """The front window's page, or a refusal (no window, unsupported page, no video)."""
        win = self.safari.front_window(timeout)
        if win is None:
            return refused("no Safari window is open")
        state = self.safari.evaluate(win, "state", state_js(), timeout)
        adapter = adapter_for((state or {}).get("site"))
        if adapter is None or not isinstance(state, dict):
            return refused("unsupported page — media works on YouTube and Jellyfin")
        return Page(win, adapter, state)

    def state(self, page: Page) -> dict[str, Any]:
        value = self.safari.evaluate(page.win, "state", state_js())
        if not isinstance(value, dict) or value.get("site") != page.adapter.name:
            raise SafariError("page_changed", "the page changed")
        return value

    def precheck(self, page: Page, command: str, timeout: float) -> Outcome | None:
        """A refusal / no-op before acting, else None (``timeout`` for the reads)."""
        if not page.state.get("video"):
            return refused("no video on this page")
        if command == "play" and page.state.get("paused") is False:
            return ok("already playing")
        if command == "pause" and page.state.get("paused") is True:
            return ok("already paused")
        if command == "fullscreen" and page.state.get("fullscreen"):
            return ok("already fullscreen")
        if command in ("next", "previous"):
            return self._boundary(page, command, timeout)
        return None

    def position(self, page: Page, timeout: float) -> dict[str, int] | None:
        """``{index, length}`` inside a playlist / queue, None outside one."""
        a = page.adapter
        if a.position:
            pos = self.safari.evaluate(page.win, "position", f"{_bind(a)} return {a.position};")
            if isinstance(pos, dict) and pos.get("list"):
                return {"index": int(pos.get("index", -1)), "length": int(pos.get("length", 0))}
            return None
        if a.queue:
            return self._queue_position(page, timeout)
        return None

    def _queue_position(self, page: Page, timeout: float) -> dict[str, int] | None:
        self.safari.evaluate(page.win, "queue_start", queue_js(page.adapter))
        res = self._poll_slot(page.win, "queue", min(timeout, 1.5))
        value = (res or {}).get("value") or {}
        queue = [str(x).replace("-", "") for x in value.get("queue") or []]
        current = (page.item or "").replace("-", "")
        if not queue or current not in queue:
            return None
        return {"index": queue.index(current), "length": len(queue)}

    def _boundary(self, page: Page, command: str, timeout: float) -> Outcome | None:
        pos = page.position = self.position(page, timeout)
        if pos is None:
            return None
        if command == "next" and pos["index"] >= pos["length"] - 1:
            return refused(f"already at {page.adapter.labels['end']}")
        if command == "previous" and pos["index"] <= 0:
            return refused(f"already at {page.adapter.labels['start']}")
        return None

    def _poll_slot(self, win: int, slot: str, timeout: float) -> dict[str, Any] | None:
        end = self.clock() + timeout
        while True:
            res = self.safari.evaluate(win, f"{slot}_poll", slot_js(slot))
            if isinstance(res, dict) and res.get("state") != "pending":
                return res
            if self.clock() >= end:
                return res if isinstance(res, dict) else None
            self.sleep(0.2)

    def _wait_state(
        self, page: Page, cond: Callable[[dict[str, Any]], bool], timeout: float
    ) -> dict[str, Any]:
        end = self.clock() + timeout
        st = self.state(page)
        while not cond(st) and self.clock() < end:
            self.sleep(POLL_S)
            st = self.state(page)
        return st

    # -- acting --------------------------------------------------------------------------
    def act(self, page: Page, command: str, seconds: float = 10.0) -> Outcome:
        """Run ``command`` and verify its postcondition (SafariError → failed)."""
        try:
            handler = getattr(self, f"_do_{command}")
            return handler(page, seconds) if command == "seek" else handler(page)
        except SafariError as exc:
            return failed(exc.reason)

    def _run(self, page: Page, op: str, statement: str) -> None:
        res = self.safari.evaluate(page.win, op, action_js(page.adapter, statement))
        error = (res or {}).get("error") if isinstance(res, dict) else "no reply"
        if error:
            raise SafariError(str(error), _ACTION_ERRORS.get(str(error), f"{op} failed"))

    def _do_play(self, page: Page) -> Outcome:
        started = self.clock()
        res = self.safari.evaluate(
            page.win, "play", promise_js(page.adapter, "play", page.adapter.play)
        )
        if isinstance(res, dict) and res.get("error"):
            return failed(_ACTION_ERRORS.get(str(res["error"]), "play failed"))
        promise = self._poll_slot(page.win, "play", PROMISE_WAIT_S) or {}
        if promise.get("state") == "rejected":
            outcome = self._rejected_play(page, str(promise.get("name") or ""))
            if outcome is not None:
                return outcome
            started = self.clock()
        return self._recheck_playing(page, started)

    def _rejected_play(self, page: Page, name: str) -> Outcome | None:
        """Autoplay rejection → the real-click hook (None = clicked, go on and verify)."""
        if name != "NotAllowedError":
            return failed(f"play was rejected ({name or 'error'})")
        if self.real_click is None:
            return refused("Safari blocked playback without a real click — click play once")
        if not self.real_click(page.win, page.adapter.name):
            return failed("Safari blocked playback and the click did not start it")
        return None

    def _recheck_playing(self, page: Page, started: float) -> Outcome:
        """One re-check ``PLAY_RECHECK_S`` after play() — never a retry loop."""
        wait = PLAY_RECHECK_S - (self.clock() - started)
        if wait > 0:
            self.sleep(wait)
        if self.state(page).get("paused") is False:
            return ok("playing")
        return failed("playback was paused again (headset or system)")

    def _do_pause(self, page: Page) -> Outcome:
        self._run(page, "pause", page.adapter.pause)
        st = self._wait_state(page, lambda s: s.get("paused") is True, PAUSE_WAIT_S)
        return ok("paused") if st.get("paused") is True else failed("the video did not pause")

    def _do_seek(self, page: Page, seconds: float) -> Outcome:
        before = page.state
        started = self.clock()
        self._run(page, "seek", f"const S = {float(seconds)!r}; {page.adapter.seek}")
        self.sleep(SEEK_SETTLE_S)
        after = self.state(page)
        if _seek_landed(before, after, seconds, self.clock() - started):
            return ok(f"moved {abs(round(seconds))} seconds {'back' if seconds < 0 else 'ahead'}")
        return failed("the video did not move")

    def _do_next(self, page: Page) -> Outcome:
        return self._change(page, "next", page.adapter.next)

    def _do_previous(self, page: Page) -> Outcome:
        a = page.adapter
        in_list = bool(a.previous_in_list) and page.position is not None
        return self._change(page, "previous", a.previous_in_list if in_list else a.previous)

    def _change(self, page: Page, op: str, statement: str) -> Outcome:
        old = page.item
        self._run(page, op, statement)
        st = self._wait_state(
            page, lambda s: bool(s.get("id")) and s.get("id") != old, CHANGE_WAIT_S
        )
        if st.get("id") and st.get("id") != old:
            return ok(f"{op} video" if page.adapter.name == "youtube" else f"{op} item")
        return failed(f"{op}: the video did not change")

    def _do_fullscreen(self, page: Page) -> Outcome:
        expr = page.adapter.fullscreen
        res = self.safari.evaluate(page.win, "fullscreen", promise_js(page.adapter, "fs", expr))
        if isinstance(res, dict) and res.get("error"):
            return failed(_ACTION_ERRORS.get(str(res["error"]), "fullscreen failed"))
        promise = self._poll_slot(page.win, "fs", PROMISE_WAIT_S) or {}
        st = self._wait_state(page, lambda s: bool(s.get("fullscreen")), 1.0)
        if st.get("fullscreen"):
            return ok("fullscreen")
        name = promise.get("name") if promise.get("state") == "rejected" else None
        return failed(f"fullscreen was refused ({name})" if name else "fullscreen did not start")

    # -- youtube_play_first ----------------------------------------------------------------
    def first_tile(self, page: Page) -> str | Outcome:
        """The first visible video's watch URL on a YouTube page, or a refusal."""
        if page.adapter is not YOUTUBE:
            return refused("open YouTube first")
        url = self.safari.evaluate(page.win, "first_tile", first_tile_js(page.adapter))
        if not isinstance(url, str) or not WATCH_URL_RE.fullmatch(url):
            return refused("no video is visible on the page")
        return url

    def play_first(self, page: Page, url: str) -> Outcome:
        """Navigate to ``url``, wait for its player, make sure it plays (one re-check)."""
        try:
            return self._play_first(page, url)
        except SafariError as exc:
            return failed(exc.reason)

    def _play_first(self, page: Page, url: str) -> Outcome:
        want = url.rsplit("v=", 1)[1]
        self.safari.set_url(page.win, url)
        started = self.clock()
        st = self._wait_state(
            page, lambda s: s.get("id") == want and bool(s.get("video")), NAV_WAIT_S
        )
        if st.get("id") != want or not st.get("video"):
            return failed("the video page did not load")
        loaded = Page(page.win, page.adapter, st)
        if st.get("paused") is not False:
            return self._do_play(loaded)
        return self._recheck_playing(loaded, started)


_ACTION_ERRORS = {
    "page_changed": "the page changed",
    "no_video": "no video on this page",
    "NotAllowedError": "Safari blocked it",
    "Error": "the player control is missing",
}


def _seek_landed(
    before: dict[str, Any], after: dict[str, Any], seconds: float, elapsed: float
) -> bool:
    t0, t1 = before.get("t"), after.get("t")
    if not isinstance(t0, int | float) or not isinstance(t1, int | float):
        return False
    expected = seconds + (0.0 if before.get("paused") else elapsed)
    if abs((t1 - t0) - expected) <= SEEK_TOLERANCE_S:
        return True
    duration = after.get("duration")
    at_end = isinstance(duration, int | float) and t1 >= duration - SEEK_TOLERANCE_S
    return (seconds < 0 and t1 <= SEEK_TOLERANCE_S) or (seconds > 0 and at_end)
