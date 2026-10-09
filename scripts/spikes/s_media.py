#!/usr/bin/env python3
"""Spike S-MEDIA: site adapters for YouTube and Jellyfin in Safari.

Holds the adapter JavaScript (selectors, actions, verification), runs one
command against a Safari window this tool opened, and snapshots sanitised HTML
fixtures for Phase 3 tests. Only target windows you opened yourself (see
safari_js.py -n). Test videos are kept at volume 0.03.

Commands (-x): first, play, pause, seek, next, previous, fullscreen.

Examples:
  scripts/spikes/s_media.py -w 6857 -x first            # YouTube home: open first visible video
  scripts/spikes/s_media.py -w 6857 -x seek -a 10       # seek +10 s and verify
  scripts/spikes/s_media.py -w 6919 -x next             # site is detected from the URL
  scripts/spikes/s_media.py -w 6857 -s youtube_watch -o tests/fixtures/spikes
  scripts/spikes/s_media.py -d                          # dump the adapter table as JSON
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from safari_js import js, set_url  # pylint: disable=wrong-import-position

TEST_VOLUME = 0.03

# Common page-side helpers, prepended to every snippet.
PRELUDE = r"""
const __vis = e => { if (!e) return false; const r = e.getBoundingClientRect();
  return r.width > 0 && r.height > 0 && r.bottom > 0 && r.top < innerHeight; };
const __fs = () => { const fe = document.fullscreenElement || document.webkitFullscreenElement;
  const v = document.querySelector('video');
  return { fullscreenElement: fe ? (fe.id || fe.tagName.toLowerCase()) : null,
           webkitDisplayingFullscreen: v ? v.webkitDisplayingFullscreen : null }; };
"""

ADAPTERS: dict[str, dict[str, Any]] = {
    "youtube": {
        "match": "location.hostname === 'www.youtube.com'",
        "video": (
            "document.querySelector('#movie_player video.html5-main-video')"
            " || document.querySelector('video')"
        ),
        "player": "document.getElementById('movie_player')",
        "id": "new URL(location.href).searchParams.get('v')",
        "first_visible": (
            "[...document.querySelectorAll("
            '\'ytd-rich-item-renderer a.ytLockupViewModelContentImage[href^="/watch"],'
            ' ytd-rich-item-renderer a#thumbnail[href^="/watch"]\')]'
            ".find(a => __vis(a) && !a.closest('ytd-ad-slot-renderer'))"
        ),
        "watch_url": "'https://www.youtube.com/watch?v=' + new URL(A.href).searchParams.get('v')",
        "play": "V.play()",
        "pause": "P.pauseVideo ? P.pauseVideo() : V.pause()",
        "seek": "P.seekTo ? P.seekTo(P.getCurrentTime() + S, true) : (V.currentTime += S)",
        "playlist_state": (
            "(() => { const u = new URL(location.href);"
            " const pl = P.getPlaylist ? P.getPlaylist() : null;"
            " return { list: u.searchParams.get('list'),"
            " index: P.getPlaylistIndex ? P.getPlaylistIndex() : -1,"
            " length: pl ? pl.length : 0 }; })()"
        ),
        "next": "P.nextVideo()",
        "next_fallback": "document.querySelector('.ytp-next-button').click()",
        "previous_in_playlist": "P.previousVideo()",
        "previous": "history.back()",
        "fullscreen": "(P.requestFullscreen ? P.requestFullscreen() : P.webkitRequestFullscreen())",
        "fullscreen_alternatives": [
            "document.querySelector('.ytp-fullscreen-button').click()",
            "V.webkitEnterFullscreen()",
        ],
        "exit_fullscreen": "document.exitFullscreen()",
    },
    "jellyfin": {
        "match": "location.hostname === 'jellyfin.dom42.space'",
        "video": (
            "document.querySelector('video.htmlvideoplayer') || document.querySelector('video')"
        ),
        "player": "document.querySelector('.videoOsdBottom') || document.body",
        "id": (
            "(() => { const v = document.querySelector('video.htmlvideoplayer');"
            " const m = v && (v.currentSrc || v.src || '')"
            ".match(/\\/videos\\/([0-9a-f-]{32,36})\\//i);"
            " return m ? m[1].replace(/-/g, '')"
            " : new URLSearchParams(location.hash.split('?')[1] || '').get('id'); })()"
        ),
        "first_visible": (
            "[...document.querySelectorAll('.itemDetailPage:not(.hide) .btnPlay,"
            " .card[data-id] .cardOverlayFab-primary,"
            ' .card[data-id] button[data-action="resume"],'
            ' .card[data-id] button[data-action="play"]\')]'
            ".find(b => __vis(b) || b.classList.contains('btnPlay'))"
        ),
        "watch_url": "null",
        "play_from_details": (
            "document.querySelector('.itemDetailPage:not(.hide) .btnPlay').click()"
        ),
        "play": "V.play()",
        "pause": "V.pause()",
        "seek": "(V.currentTime += S)",
        "queue_state": (
            "ApiClient.getJSON(ApiClient.getUrl('Sessions', {DeviceId: ApiClient.deviceId()}))"
            ".then(r => { const s = r[0] || {}; const q = (s.NowPlayingQueue || []).map(x => x.Id);"
            " return { queue: q, now: s.NowPlayingItem ? s.NowPlayingItem.Id : null }; })"
        ),
        "next": "document.querySelector('.btnNextTrack').click()",
        "previous": "document.querySelector('.btnPreviousTrack').click()",
        "fullscreen": "V.requestFullscreen()",
        "fullscreen_alternatives": [
            "document.querySelector('.btnFullscreen').click()",
            "document.documentElement.requestFullscreen()",
            "V.webkitEnterFullscreen()",
        ],
        "exit_fullscreen": "document.exitFullscreen()",
    },
}


def page(win: int, body: str) -> Any:
    """Evaluate `body` (statements ending in `return <json-able>`) in the page."""
    out = js(
        win, f"(function(){{{PRELUDE}\nreturn JSON.stringify((function(){{\n{body}\n}})());}})()"
    )
    return json.loads(out) if out not in ("", "missing value") else None


def page_async(win: int, expr: str, pre: str = "", timeout: float = 6.0) -> Any:
    """Run statements `pre`, then evaluate a promise expression; poll its settled value."""
    page(
        win,
        "window.__spk = {state: 'pending'}; try { " + pre + " Promise.resolve(" + expr + ")"
        ".then(v => window.__spk = {state: 'resolved', value: v === undefined ? null : v},"
        " e => window.__spk = {state: 'rejected', name: e.name, message: e.message}); }"
        " catch (e) { window.__spk = {state: 'threw', name: e.name, message: e.message}; }"
        " return 1;",
    )
    end = time.time() + timeout
    while time.time() < end:
        res = page(win, "return window.__spk;")
        if res and res.get("state") != "pending":
            return res
        time.sleep(0.2)
    return {"state": "timeout"}


def bind(site: str) -> str:
    """Bind V (video) and P (player) for a site's snippets; clamp volume."""
    a = ADAPTERS[site]
    return (
        f"const V = {a['video']}; const P = {a['player']};"
        f" if (V && V.volume > 0.05) V.volume = {TEST_VOLUME};"
    )


def detect(win: int) -> str | None:
    """Return the adapter name matching the window's page, or None."""
    for name, a in ADAPTERS.items():
        if page(win, f"return {a['match']};"):
            return name
    return None


def state(win: int, site: str) -> dict:
    """Return id, paused, currentTime, volume and fullscreen state."""
    a = ADAPTERS[site]
    return page(
        win,
        bind(site) + f" return Object.assign({{href: location.href, id: {a['id']}, video: !!V,"
        " paused: V ? V.paused : null, t: V ? V.currentTime : null,"
        " volume: V ? V.volume : null}, __fs());",
    )


def wait_for(win: int, site: str, cond, timeout: float = 8.0) -> dict:
    """Poll state() until cond(state) is true or timeout; return last state."""
    end = time.time() + timeout
    st = state(win, site)
    while time.time() < end and not cond(st):
        time.sleep(0.3)
        st = state(win, site)
    return st


def boundary(win: int, site: str, current: str | None) -> dict | None:
    """Return the playlist/queue position, or None when not in one.

    Jellyfin: the server session's NowPlayingItem lags a track change by
    seconds, so the index is taken from the page's current item id.
    """
    if site == "youtube":
        ps = page(win, bind(site) + f" return {ADAPTERS[site]['playlist_state']};")
        return ps if ps and ps.get("list") else None
    res = page_async(win, ADAPTERS[site]["queue_state"])
    if res.get("state") != "resolved" or not res["value"]["queue"]:
        return None
    queue = res["value"]["queue"]
    index = queue.index(current) if current in queue else -1
    return {"now": current, "index": index, "length": len(queue)} if index >= 0 else None


# One flat dispatcher keeps each command's action and postcondition side by side.
# pylint: disable-next=too-many-return-statements,too-many-locals,too-many-statements
def run(win: int, cmd: str, arg: float, site: str | None) -> dict:
    """Run one adapter command with its postcondition check."""
    site = site or detect(win)
    if site is None:
        return {"command": cmd, "result": "refused", "reason": "unsupported page"}
    a = ADAPTERS[site]
    before: dict = state(win, site) if cmd != "first" else {}
    res: dict[str, Any] = {"site": site, "command": cmd, "before": before}
    if cmd == "first":
        target = page(win, f"const A = {a['first_visible']}; return A ? {a['watch_url']} : null;")
        if site == "youtube":
            if not target:
                return res | {"result": "fail", "reason": "no visible video tile"}
            set_url(win, target)
            want = target.split("v=")[1]
            st = wait_for(win, site, lambda s: s and s.get("id") == want and s.get("video"), 12)
        else:
            page(win, a["play_from_details"] + "; return 1;")
            st = wait_for(win, site, lambda s: s and s.get("video") and s.get("t", 0) > 0, 12)
        play = (
            page_async(win, a["play"], pre=bind(site))
            if st.get("paused")
            else {"state": "already playing"}
        )
        time.sleep(1.5)
        after = state(win, site)
        ok = after.get("video") and not after.get("paused")
        return res | {
            "target": target,
            "play_promise": play,
            "after": after,
            "result": "pass" if ok else "fail",
        }
    if cmd == "play":
        promise = page_async(win, a["play"], pre=bind(site))
        time.sleep(1.5)
        after = state(win, site)
        ok = promise.get("state") == "resolved" and not after["paused"]
        return res | {"play_promise": promise, "after": after, "result": "pass" if ok else "fail"}
    if cmd == "pause":
        page(win, bind(site) + f" {a['pause']}; return 1;")
        after = wait_for(win, site, lambda s: s["paused"], 3)
        return res | {"after": after, "result": "pass" if after["paused"] else "fail"}
    if cmd == "seek":
        t0 = time.time()
        page(win, bind(site) + f" const S = {float(arg)}; {a['seek']}; return 1;")
        time.sleep(1.5)
        after = state(win, site)
        elapsed = 0.0 if before["paused"] else time.time() - t0
        delta = after["t"] - before["t"]
        ok = abs(delta - arg - elapsed) <= 2.0
        return res | {
            "after": after,
            "delta": round(delta, 2),
            "expected": round(arg + elapsed, 2),
            "result": "pass" if ok else "fail",
        }
    if cmd in ("next", "previous"):
        pos = boundary(win, site, before.get("id"))
        res["position"] = pos
        if pos and cmd == "next" and pos["index"] >= pos["length"] - 1:
            return res | {"result": "refused", "reason": "playlist end"}
        if pos and cmd == "previous" and pos["index"] <= 0:
            return res | {"result": "refused", "reason": "playlist start"}
        action = (
            a["previous_in_playlist"]
            if (site == "youtube" and cmd == "previous" and pos)
            else a[cmd]
        )
        page(win, bind(site) + f" {action}; return 1;")
        old = before["id"]
        after = wait_for(win, site, lambda s: s and s.get("id") and s["id"] != old, 8)
        ok = after.get("id") not in (None, old)
        return res | {"action": action, "after": after, "result": "pass" if ok else "fail"}
    if cmd == "fullscreen":
        promise = page_async(win, a["fullscreen"], pre=bind(site))
        time.sleep(1.5)
        during = state(win, site)
        ok = bool(during.get("fullscreenElement") or during.get("webkitDisplayingFullscreen"))
        page(win, f"if (document.fullscreenElement) {a['exit_fullscreen']}; return 1;")
        time.sleep(1.5)
        after = state(win, site)
        exited = not after.get("fullscreenElement")
        return res | {
            "promise": promise,
            "during": during,
            "after": after,
            "result": "pass" if ok and exited else "fail",
        }
    raise ValueError(cmd)


SANITIZE = r"""
const KEEP_Q = ['v', 'list', 'index', 'id'];
const fixUrl = s => { try { if (s.startsWith('blob:')) return 'blob:REDACTED';
  const u = new URL(s, location.href); const q = new URLSearchParams();
  for (const k of KEEP_Q) if (u.searchParams.has(k)) q.set(k, u.searchParams.get(k));
  const qs = q.toString();
  const hash = u.hash.startsWith('#/') ? u.hash.split('&')[0] : '';
  const path = u.pathname.startsWith('/@') ? '/@CHANNEL' : u.pathname;
  return u.origin + path + (qs ? '?' + qs : '') + hash; }
  catch (e) { return ''; } };
const KEEP_TEXT = KEEP_TEXT_SEL;
function clean(root) {
  const c = root.cloneNode(true);
  const DROP = 'script,style,link,iframe,noscript,canvas,img,source,track,yt-img-shadow,'
    + 'ytd-thumbnail-overlay-time-status-renderer';
  c.querySelectorAll(DROP).forEach(e => e.remove());
  c.querySelectorAll('svg').forEach(s => s.replaceChildren());
  const walker = document.createTreeWalker(c, NodeFilter.SHOW_TEXT);
  const texts = []; while (walker.nextNode()) texts.push(walker.currentNode);
  for (const t of texts) { if (!t.data.trim()) { t.data = ''; continue; }
    const keep = t.parentElement && KEEP_TEXT && t.parentElement.closest(KEEP_TEXT);
    if (!keep) t.data = 'PLACEHOLDER'; }
  for (const e of [c, ...c.querySelectorAll('*')]) {
    for (const at of [...e.attributes]) {
      const n = at.name;
      if (['style', 'srcset', 'nonce', 'jslog'].includes(n) || n.startsWith('on')) {
        e.removeAttribute(n); continue; }
      if (['href', 'src', 'poster'].includes(n)) { e.setAttribute(n, fixUrl(at.value)); continue; }
      if (['data-tooltip-text', 'data-preview', 'data-tooltip-image'].includes(n)) {
        e.setAttribute(n, 'PLACEHOLDER'); continue; }
      const isControl = e.matches('.ytp-button, .ytp-chrome-bottom button,'
        + ' .videoOsdBottom button, .osdControls button');
      if ((n === 'aria-label' || n === 'title') && !isControl) {
        e.setAttribute(n, 'PLACEHOLDER'); continue; }
      if (at.value.length > 300 && n !== 'class') e.setAttribute(n, at.value.slice(0, 40) + '…');
    }
  }
  return c.outerHTML.replace(/api_key=[^&"]+/g, 'api_key=REDACTED').replace(/\n\s*\n+/g, '\n');
}
"""

SNAPSHOTS = {
    "youtube_home": {
        "roots": "[...document.querySelectorAll('ytd-rich-item-renderer')].slice(0, 6)",
        "keep_text": "null",
    },
    "youtube_watch": {
        "roots": "[document.getElementById('movie_player'),"
        " document.querySelector('ytd-playlist-panel-renderer #header-contents,"
        " ytd-playlist-panel-renderer .index-message')]"
        ".filter(Boolean)",
        "keep_text": "'.ytp-time-display, .index-message, .ytp-time-current, .ytp-time-duration'",
    },
    "jellyfin_player": {
        "roots": "[document.querySelector('.videoPlayerContainer'),"
        " document.querySelector('.videoOsdBottom')]"
        ".filter(Boolean)",
        "keep_text": "'.osdTimeText, .osdPositionText, .osdDurationText'",
    },
}


SNAPSHOTS["youtube_watch_playlist"] = SNAPSHOTS["youtube_watch"]


def snapshot(win: int, name: str, outdir: Path) -> Path:
    """Write a sanitised HTML fixture of the player/tiles region."""
    spec = SNAPSHOTS[name]
    body = SANITIZE.replace("KEEP_TEXT_SEL", spec["keep_text"]) + (
        f"const roots = {spec['roots']};"
        " return { url: fixUrl(location.href), html: roots.map(clean).join('\\n') };"
    )
    res = page(win, body)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.html"
    head = (
        f"<!-- spike S-MEDIA fixture: {name}; page {res['url']};"
        f" captured {time.strftime('%Y-%m-%d')}; sanitised (texts -> PLACEHOLDER,"
        " images/scripts/styles removed, query strings reduced) -->\n"
    )
    path.write_text(f"<!doctype html>\n{head}<html><body>\n{res['html']}\n</body></html>\n")
    return path


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-w", "--window", type=int, help="Safari window id (one you opened)")
    ap.add_argument(
        "-x",
        "--exec",
        choices=["first", "play", "pause", "seek", "next", "previous", "fullscreen"],
        help="run a command",
    )
    ap.add_argument("-a", "--arg", type=float, default=10.0, help="seek seconds (default 10)")
    ap.add_argument(
        "-S",
        "--site",
        choices=sorted(ADAPTERS),
        help="force the adapter (default: detect from URL)",
    )
    ap.add_argument(
        "-s", "--snapshot", choices=sorted(SNAPSHOTS), help="write a sanitised HTML fixture"
    )
    ap.add_argument(
        "-o", "--outdir", type=Path, default=Path("tests/fixtures/spikes"), help="fixture directory"
    )
    ap.add_argument("-d", "--dump", action="store_true", help="print the adapter table as JSON")
    args = ap.parse_args()
    if args.dump:
        print(json.dumps(ADAPTERS, indent=2))
        return 0
    if args.window is None:
        ap.error("-w/--window is required")
    if args.snapshot:
        print(snapshot(args.window, args.snapshot, args.outdir))
    if args.exec:
        out = run(args.window, args.exec, args.arg, args.site)
        print(json.dumps(out))
        return 0 if out.get("result") in ("pass", "refused") else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
