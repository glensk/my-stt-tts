"""Safari media adapters (PLAN_claude-bridge 7.3, spike S-MEDIA): YouTube + Jellyfin.

Postconditions on fake JS replies (:mod:`mac_fakes`) and the adapters' selectors and id
rules against the saved S-MEDIA HTML fixtures.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from typing import Any

import pytest
from mac_fakes import (
    FIXTURES,
    WIN,
    Call,
    jf_state,
    yt_state,
)

from my_stt_tts import mac_sites
from my_stt_tts.bridge_text import derivable
from my_stt_tts.mac_sites import (
    JELLYFIN,
    JELLYFIN_ID_RE,
    SET_URL,
    YOUTUBE,
    RunResult,
    every_js,
)


# -- media: YouTube ----------------------------------------------------------------------------
def test_play_verified_once_after_recheck() -> None:
    call = Call()
    call.fake.js = {
        "state": [yt_state(paused=True), yt_state(paused=False)],
        "play_poll": {"state": "resolved"},
    }
    call.say("play")
    assert call.mac.media("play") == "ok: playing"
    assert call.fake.ops().count("play") == 1
    assert "V.play()" in call.fake.js_of("play")[0]


def test_play_paused_again_by_headset_is_not_retried() -> None:
    call = Call()
    call.fake.js = {"state": [yt_state(paused=True)], "play_poll": {"state": "resolved"}}
    call.say("play")
    out = call.mac.media("play")
    assert out.startswith("failed:") and "paused again" in out
    assert call.fake.ops().count("play") == 1
    assert call.problems[0].subject == "media"


def test_autoplay_rejection_without_real_click_is_refused() -> None:
    call = Call()
    call.fake.js = {
        "state": [yt_state(paused=True)],
        "play_poll": {"state": "rejected", "name": "NotAllowedError"},
    }
    call.say("play")
    out = call.mac.media("play")
    assert out.startswith("refused:") and "real click" in out


def test_autoplay_rejection_uses_the_real_click_hook() -> None:
    clicks: list[tuple[int, str]] = []

    def click(win: int, site: str) -> bool:
        clicks.append((win, site))
        return True

    call = Call(real_click=click)
    call.fake.js = {
        "state": [yt_state(paused=True), yt_state(paused=False)],
        "play_poll": {"state": "rejected", "name": "NotAllowedError"},
    }
    call.say("play")
    assert call.mac.media("play") == "ok: playing"
    assert clicks == [(WIN, "youtube")]


def test_play_when_playing_is_a_noop() -> None:
    call = Call()
    call.fake.js = {"state": yt_state(paused=False)}
    call.say("play")
    assert call.mac.media("play") == "ok: already playing"
    assert "play" not in call.fake.ops()


def test_pause() -> None:
    call = Call()
    call.fake.js = {"state": [yt_state(), yt_state(paused=True)]}
    call.say("pause")
    assert call.mac.media("pause") == "ok: paused"
    assert "pauseVideo" in call.fake.js_of("pause")[0]


def test_seek_forward_verified() -> None:
    call = Call()
    call.fake.js = {"state": [yt_state(t=5.0), yt_state(t=15.8)]}
    call.say("skip ten seconds")
    assert call.mac.media("seek", 10) == "ok: moved 10 seconds ahead"
    assert "const S = 10.0;" in call.fake.js_of("seek")[0]


def test_seek_back_and_failure() -> None:
    call = Call()
    call.fake.js = {"state": [yt_state(t=50.0), yt_state(t=50.5)]}
    call.say("zehn Sekunden zurück")
    assert call.mac.media("seek", -10).startswith("failed:")


def test_seek_without_seconds_needs_a_seek_word() -> None:
    call = Call()
    call.fake.js = {"state": [yt_state(t=5.0), yt_state(t=15.8)]}
    call.say("pause")
    assert "not in what you said" in call.mac.media("seek")
    call.say("skip forward")
    assert call.mac.media("seek").startswith("ok:")


def test_next_outside_playlist_verified_by_id() -> None:
    call = Call()
    call.fake.js = {
        "state": [yt_state(), yt_state(), yt_state(id="Rn4nmFRPe0s")],
        "position": {"list": None, "index": -1, "length": 0},
    }
    call.say("next")
    assert call.mac.media("next") == "ok: next video"
    assert "P.nextVideo()" in call.fake.js_of("next")[0]


def test_next_at_playlist_end_is_refused() -> None:
    call = Call()
    call.fake.js = {
        "state": yt_state(id="TgKwz5Ikpc8"),
        "position": {"list": "PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab", "index": 15, "length": 16},
    }
    call.say("next")
    assert call.mac.media("next") == "refused: already at the end of the playlist"
    assert "next" not in call.fake.ops()


def test_previous_outside_playlist_goes_back() -> None:
    call = Call()
    call.fake.js = {
        "state": [yt_state(id="Rn4nmFRPe0s"), yt_state(id="lDrAZ1wAyVs")],
        "position": {"list": None},
    }
    call.say("previous")
    assert call.mac.media("previous") == "ok: previous video"
    assert "history.back()" in call.fake.js_of("previous")[0]


def test_previous_inside_playlist_and_at_its_start() -> None:
    call = Call()
    call.fake.js = {
        "state": [yt_state(id="k7RM-ot2NWY"), yt_state(id="fNk_zzaMoSs")],
        "position": {"list": "PL", "index": 1, "length": 16},
    }
    call.say("previous")
    assert call.mac.media("previous") == "ok: previous video"
    assert "P.previousVideo()" in call.fake.js_of("previous")[0]
    call.fake.js["position"] = {"list": "PL", "index": 0, "length": 16}
    call.say("zurück")
    assert call.mac.media("previous") == "refused: already at the start of the playlist"


def test_next_that_does_not_change_the_video_fails() -> None:
    call = Call()
    call.fake.js = {"state": yt_state(), "position": {"list": None}}
    call.say("next")
    assert call.mac.media("next").startswith("failed:")


def test_fullscreen_verified() -> None:
    call = Call()
    call.fake.js = {
        "state": [yt_state(), yt_state(fullscreen=True)],
        "fs_poll": {"state": "resolved"},
    }
    call.say("fullscreen")
    assert call.mac.media("fullscreen") == "ok: fullscreen"
    assert "requestFullscreen" in call.fake.js_of("fullscreen")[0]


def test_unsupported_page_is_refused() -> None:
    call = Call()
    call.fake.js = {"state": {"site": None}}
    call.say("pause")
    out = call.mac.media("pause")
    assert out.startswith("refused:") and "unsupported page" in out
    assert call.fake.ops() == ["state"]


def test_no_safari_window() -> None:
    call = Call()
    call.fake.win = None
    call.say("pause")
    assert call.mac.media("pause") == "refused: no Safari window is open"


def test_safari_javascript_off_is_a_problem() -> None:
    call = Call()
    call.fake.js = {
        "state": RunResult(
            1, "", "You must enable the 'Allow JavaScript from Apple Events' option (8)"
        )
    }
    call.say("pause")
    out = call.mac.media("pause")
    assert out.startswith("failed:") and "mac-voice -D" in out
    assert call.problems


def test_unknown_media_command() -> None:
    call = Call()
    call.say("rewind")
    assert call.mac.media("rewind").startswith("refused:")


# -- media: Jellyfin ---------------------------------------------------------------------------
QUEUE = ["aaaa" * 8, "bfa78bb580de320fe4d1dd15c15c4b6a", "cccc" * 8]


def jf_queue(index: int) -> dict[str, Any]:
    return {"state": "resolved", "value": {"queue": QUEUE, "now": QUEUE[index]}}


def test_jellyfin_next_at_queue_end_is_refused() -> None:
    call = Call()
    call.fake.js = {"state": jf_state(id=QUEUE[2]), "queue_poll": jf_queue(2)}
    call.say("nächstes")
    assert call.mac.media("next") == "refused: already at the end of the queue"
    assert "next" not in call.fake.ops()


def test_jellyfin_previous_at_queue_start_is_refused() -> None:
    call = Call()
    call.fake.js = {"state": jf_state(id=QUEUE[0]), "queue_poll": jf_queue(0)}
    call.say("previous")
    assert call.mac.media("previous") == "refused: already at the start of the queue"


def test_jellyfin_next_mid_queue() -> None:
    call = Call()
    call.fake.js = {
        "state": [jf_state(), jf_state(), jf_state(id=QUEUE[2])],
        "queue_poll": [{"state": "pending"}, jf_queue(1)],
    }
    call.say("next")
    assert call.mac.media("next") == "ok: next item"
    assert ".btnNextTrack" in call.fake.js_of("next")[0]


def test_jellyfin_play_and_fullscreen() -> None:
    call = Call()
    call.fake.js = {
        "state": [jf_state(paused=True), jf_state(paused=False)],
        "play_poll": {"state": "resolved"},
    }
    call.say("spiel ab")
    assert call.mac.media("play") == "ok: playing"
    call.fake.js = {"state": [jf_state(), jf_state(fullscreen=True)]}
    call.say("vollbild")
    assert call.mac.media("fullscreen") == "ok: fullscreen"
    assert "V.requestFullscreen()" in call.fake.js_of("fullscreen")[0]


def test_jellyfin_missing_button_fails() -> None:
    call = Call()
    call.fake.js = {
        "state": jf_state(),
        "queue_poll": {"state": "rejected"},
        "next": {"error": "Error"},
    }
    call.say("next")
    assert call.mac.media("next") == "failed: the player control is missing"


# -- youtube_play_first ------------------------------------------------------------------------
def test_youtube_play_first() -> None:
    call = Call()
    url = "https://www.youtube.com/watch?v=lDrAZ1wAyVs"
    call.fake.js = {
        "state": [yt_state(video=False, id=None), yt_state(paused=False)],
        "first_tile": url,
    }
    call.say("play the first video")
    assert call.mac.youtube_play_first() == "ok: playing"
    assert call.fake.scripts(SET_URL) == [[url, str(WIN)]]


def test_youtube_play_first_starts_a_paused_video() -> None:
    call = Call()
    call.fake.js = {
        "state": [
            yt_state(video=False, id=None),
            yt_state(paused=True),
            yt_state(paused=False),
        ],
        "first_tile": "https://www.youtube.com/watch?v=lDrAZ1wAyVs",
        "play_poll": {"state": "resolved"},
    }
    call.say("play the first video")
    assert call.mac.youtube_play_first() == "ok: playing"
    assert "play" in call.fake.ops()


def test_youtube_play_first_needs_youtube() -> None:
    call = Call()
    call.fake.js = {"state": jf_state()}
    call.say("play the first video")
    assert call.mac.youtube_play_first() == "refused: open YouTube first"


def test_youtube_play_first_rejects_a_strange_url() -> None:
    call = Call()
    call.fake.js = {"state": yt_state(video=False), "first_tile": "javascript:alert(1)"}
    call.say("play the first video")
    assert call.mac.youtube_play_first() == "refused: no video is visible on the page"
    assert not call.fake.scripts(SET_URL)


def test_youtube_play_first_page_does_not_load() -> None:
    call = Call()
    call.fake.js = {
        "state": yt_state(video=False, id=None),
        "first_tile": "https://www.youtube.com/watch?v=lDrAZ1wAyVs",
    }
    call.say("play the first video")
    assert call.mac.youtube_play_first() == "failed: the video page did not load"


# -- adapters vs the S-MEDIA fixtures ----------------------------------------------------------
class Tags(HTMLParser):
    """Start tags with attributes and their open ancestors (enough for the selectors)."""

    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[tuple[str, dict[str, str]]] = []
        self.found: list[tuple[str, dict[str, str], list[str]]] = []
        self.text: list[tuple[list[str], str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k: v or "" for k, v in attrs}
        self.found.append((tag, attr, [t for t, _ in self.stack]))
        if tag not in self.VOID:
            self.stack.append((tag, attr))

    def handle_endtag(self, tag: str) -> None:
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def handle_data(self, data: str) -> None:
        if data.strip():
            classes = [c for _, a in self.stack for c in a.get("class", "").split()]
            self.text.append((classes, data.strip()))


def parse(name: str) -> Tags:
    tags = Tags()
    tags.feed((FIXTURES / name).read_text())
    return tags


def has_class(tags: Tags, tag: str, cls: str) -> bool:
    return any(t == tag and cls in a.get("class", "").split() for t, a, _ in tags.found)


def test_youtube_first_tile_skips_the_ad() -> None:
    tags = parse("youtube_home.html")
    tiles = [
        (a["href"], parents)
        for t, a, parents in tags.found
        if t == "a"
        and "ytLockupViewModelContentImage" in a.get("class", "").split()
        and "/watch" in a.get("href", "")
        and "ytd-rich-item-renderer" in parents
    ]
    first = next(href for href, parents in tiles if "ytd-ad-slot-renderer" not in parents)
    assert len(tiles) == 6 and "ytd-ad-slot-renderer" in tiles[0][1]
    spike = json.loads((FIXTURES / "s_media.json").read_text())
    played = spike["sites"]["youtube"]["commands"]["youtube_play_first"]["run"]["target"]
    assert first.split("v=")[1] == played.split("v=")[1] == "lDrAZ1wAyVs"
    assert mac_sites.WATCH_URL_RE.fullmatch(played)
    for part in ("ytd-rich-item-renderer a.ytLockupViewModelContentImage", "ytd-ad-slot-renderer"):
        assert part in YOUTUBE.first_tile


def test_adapter_selectors_match_the_spike() -> None:
    spike = json.loads((FIXTURES / "s_media.json").read_text())["sites"]
    yt, jf = spike["youtube"]["selectors"], spike["jellyfin"]["selectors"]
    assert YOUTUBE.video == yt["video"] and YOUTUBE.player == yt["player"]
    assert YOUTUBE.item_id == yt["video_id"]
    assert YOUTUBE.first_tile == yt["first_visible_tile"]
    assert JELLYFIN.video == jf["video"]
    assert JELLYFIN.queue == jf["queue"].split("   (")[0]


def test_youtube_watch_fixtures() -> None:
    for name in ("youtube_watch.html", "youtube_watch_playlist.html"):
        tags = parse(name)
        assert any(a.get("id") == "movie_player" for _, a, _ in tags.found)
        assert has_class(tags, "video", "html5-main-video")
        assert has_class(tags, "button", "ytp-fullscreen-button")
    index = [
        t for classes, t in parse("youtube_watch_playlist.html").text if "index-message" in classes
    ]
    position, length = (int(x) for x in index[0].split("/"))
    assert position == length == 16  # the playlist-end refusal case


def test_jellyfin_fixture_id_and_buttons() -> None:
    tags = parse("jellyfin_player.html")
    video = next(a for t, a, _ in tags.found if t == "video")
    assert "htmlvideoplayer" in video["class"].split()
    match = re.search(JELLYFIN_ID_RE, video["src"], re.IGNORECASE)
    assert match and match.group(1).replace("-", "") == "bfa78bb580de320fe4d1dd15c15c4b6a"
    for cls in ("btnNextTrack", "btnPreviousTrack", "btnFullscreen", "btnPause"):
        assert has_class(tags, "button", cls), cls
    assert mac_sites._JS_ID_RE in JELLYFIN.item_id


def test_snippets_use_dom_methods_and_never_write_volume() -> None:
    for code in every_js():
        assert "innerHTML" not in code
        assert not re.search(r"\.volume\s*=", code)
        assert code.startswith("/*mv:")


# -- derivability of the media words -----------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "key", "text", "expected"),
    [
        ("seek", "command", "skip forward", True),
        ("seek", "command", "spul vor", True),
        ("10", "seconds", "zehn Sekunden zurück", True),
        ("next", "command", "nächstes Video", True),
        ("fullscreen", "command", "plein écran", True),
        ("seek", "command", "pause", False),
    ],
)
def test_media_words_are_derivable(value: str, key: str, text: str, expected: bool) -> None:
    assert derivable(value, text, key) is expected
