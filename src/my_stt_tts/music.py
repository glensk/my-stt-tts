"""Play music from YouTube (personal-use): search + a stoppable background player.

This is the *local intent router* path for playing music: a turn like "play We
Will Rock You by Queen" is recognised BEFORE the LLM is asked (so it works for
every brain, including ``claude-cli`` which does not do our tool-calling), the
best YouTube match is resolved with **yt-dlp**, and the audio is streamed through
a stoppable background player. The assistant then speaks a short confirmation via
the normal TTS path and skips the LLM for that turn.

Layering / graceful degradation (CORE must import this module without extras):

* ``yt-dlp`` is imported **lazily** inside :func:`search` / :func:`play`, so the
  package imports and the app runs on a core ``uv sync`` (no extras). When it is
  missing, the player returns a clear, spoken reason ("install yt-dlp …") instead
  of crashing the turn.
* Playback prefers **mpv** (``mpv --no-video``: can stream the YouTube *page* URL
  directly and supports stop/pause over its JSON IPC socket), else **ffplay**
  (``ffplay -nodisp``: streams the ``bestaudio`` URL yt-dlp resolves), else a
  final fallback that downloads ``bestaudio`` to a temp file and plays it through
  the project's own playback (:func:`my_stt_tts.audio.play`). mpv/ffmpeg are
  *system* tools; a missing player yields a spoken "install mpv or ffmpeg" reason.
* The player runs **non-blocking** (a background subprocess / thread) so the
  assistant stays responsive, and the process handle is tracked so a later "stop"
  / "pause" intent can act on it.

The intent router (:func:`match_music_intent`) recognises EN / DE / FR play /
stop / pause / resume phrasings (the user is multilingual). It is pure string
work — no imports, no network — so it is cheap to run on every turn and trivially
testable.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("my_stt_tts.music")

# yt-dlp can take a while on a cold network; never hang a turn forever.
_SEARCH_TIMEOUT = 20.0
# How long to wait for mpv's IPC socket to appear before giving up on IPC control
# (mpv still plays; we just lose pause/resume and fall back to process-kill stop).
_IPC_WAIT_S = 3.0


# --- intent router -------------------------------------------------------------

# Action vocabulary the router emits. ``play`` carries a ``query``; the others
# are bare control actions.
ACTIONS = ("play", "stop", "pause", "resume")

# Stop / pause / resume control phrases, case-insensitive, EN/DE/FR. Matched as
# whole-utterance commands (optionally with "the music" / "die Musik" / "la
# musique" trailing) so a normal sentence that merely contains the word "stop"
# does not hijack the turn. Ordered longest-first within each action is not
# needed — we test all and pick the matched action.
_MUSIC_NOUNS = (
    r"(?:the\s+music|the\s+song|music|die\s+musik|der\s+song|das\s+lied|la\s+musique|la\s+chanson)"
)

_STOP_RE = re.compile(
    rf"^\s*(?:stop|halt|stopp|arr[êe]te|arr[êe]tez)(?:\s+{_MUSIC_NOUNS})?\s*[.!]?\s*$",
    re.IGNORECASE,
)
_PAUSE_RE = re.compile(
    rf"^\s*(?:pause|pausiere|pausier|mets?\s+en\s+pause)(?:\s+{_MUSIC_NOUNS})?\s*[.!]?\s*$",
    re.IGNORECASE,
)
_RESUME_RE = re.compile(
    rf"^\s*(?:resume|continue|unpause|weiter|mach\s+weiter|reprends|reprenez|continue?)"
    rf"(?:\s+{_MUSIC_NOUNS})?\s*[.!]?\s*$",
    re.IGNORECASE,
)

# "play some music" / "put on some music" / DE "mach Musik an" / FR "mets de la
# musique" — a play action with NO specific song (router returns query=""), so
# the caller can pick a default/radio query.
_PLAY_GENERIC_RE = re.compile(
    r"^\s*(?:"
    r"play\s+(?:some\s+)?music"
    r"|put\s+on\s+(?:some\s+)?music"
    r"|mach\s+(?:mal\s+)?musik(?:\s+an)?"
    r"|spiel(?:e)?\s+(?:etwas\s+|irgendwelche\s+)?musik"
    r"|mets\s+de\s+la\s+musique"
    r"|joue\s+de\s+la\s+musique"
    r")\s*[.!]?\s*$",
    re.IGNORECASE,
)

# "play <X>" with a concrete song/query. The leading verb is stripped and the
# remainder (minus a trailing "from youtube" / "auf youtube" / "sur youtube" and
# trailing punctuation) becomes the search query. EN: play / put on; DE: spiel(e)
# / mach <X> an (rare) — we accept "spiel(e) <X>"; FR: joue / mets <X>.
_PLAY_QUERY_RE = re.compile(
    r"^\s*(?:please\s+)?(?:can\s+you\s+)?"
    r"(?:play|put\s+on|spiele|spiel|joue|mets)\s+"
    r"(?P<query>.+?)"
    r"\s*[.!?]?\s*$",
    re.IGNORECASE,
)

# Trailing "from/on youtube" in EN/DE/FR, stripped from a play query.
_FROM_YOUTUBE_RE = re.compile(
    r"\s*(?:from|on|über|auf|sur|de|via)\s+youtube\s*$",
    re.IGNORECASE,
)


def _clean_query(raw: str) -> str:
    """Strip a trailing 'from youtube' clause + punctuation from a play query."""
    cleaned = _FROM_YOUTUBE_RE.sub("", raw).strip()
    return cleaned.strip(" .!? ")


def match_music_intent(text: str) -> dict[str, str] | None:
    """Classify ``text`` as a music command, or ``None`` if it is not one.

    Returns ``{"action": "play", "query": "<song>"}`` for a play request (an empty
    ``query`` means "play some music" with no specific song), or
    ``{"action": "stop"|"pause"|"resume"}`` for a control command. Case-insensitive
    across English, German, and French. Pure string work — safe to call on every
    turn before the LLM.
    """
    if not text or not text.strip():
        return None
    stripped = text.strip()
    # Control commands first: they are exact whole-utterance phrases, so a play
    # request like "play stop by …" is not mis-read as a stop command.
    if _STOP_RE.match(stripped):
        return {"action": "stop"}
    if _PAUSE_RE.match(stripped):
        return {"action": "pause"}
    if _RESUME_RE.match(stripped):
        return {"action": "resume"}
    if _PLAY_GENERIC_RE.match(stripped):
        return {"action": "play", "query": ""}
    m = _PLAY_QUERY_RE.match(stripped)
    if m:
        query = _clean_query(m.group("query"))
        if query:
            return {"action": "play", "query": query}
    return None


# --- yt-dlp search -------------------------------------------------------------


@dataclass
class Track:
    """A resolved YouTube result: the page URL + a human title."""

    url: str
    title: str


def yt_dlp_available() -> bool:
    """True when the ``yt_dlp`` module can be imported (the ``music`` extra)."""
    try:
        import yt_dlp  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import
    except Exception:  # noqa: BLE001 — any import failure means "unavailable"
        return False
    return True


def _ytdlp_extract(query: str, *, want_stream: bool) -> dict[str, Any] | None:
    """Run yt-dlp ``ytsearch1:<query>`` and return the first result's info dict.

    ``want_stream`` adds the bestaudio format selection so the caller can read a
    direct media ``url`` (for ffplay). Returns ``None`` when yt-dlp is missing or
    nothing is found. Never raises for an ordinary "no results"/network blip.
    """
    try:
        import yt_dlp  # pylint: disable=import-outside-toplevel
    except Exception:  # noqa: BLE001
        log.warning("yt-dlp not installed; cannot search YouTube")
        return None
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "default_search": "ytsearch1",
        "skip_download": True,
        "socket_timeout": _SEARCH_TIMEOUT,
    }
    if want_stream:
        opts["format"] = "bestaudio/best"
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch1:{query}", download=False)
    except Exception as exc:  # noqa: BLE001 — surface as "no result", never crash
        log.warning("yt-dlp search failed for %r: %s", query, exc)
        return None
    if not info:
        return None
    entries = info.get("entries")
    if entries:
        info = entries[0] if entries else None
    return info if isinstance(info, dict) else None


def search(query: str) -> Track | None:
    """Resolve ``query`` to the best YouTube match (page URL + title), or ``None``.

    Uses yt-dlp ``ytsearch1:`` with no download. Returns ``None`` when yt-dlp is
    not installed or nothing matches — the caller turns that into a spoken reason.
    """
    info = _ytdlp_extract(query, want_stream=False)
    if not info:
        return None
    url = info.get("webpage_url") or info.get("original_url") or info.get("url")
    title = info.get("title") or query
    if not url:
        return None
    return Track(url=str(url), title=str(title))


# --- the player ----------------------------------------------------------------


@dataclass
class PlayResult:
    """Outcome of a :meth:`MusicPlayer.play` call.

    ``ok`` is True when playback started; ``title`` is the played track's title
    (for a spoken confirmation); ``reason`` carries a short, speakable explanation
    when ``ok`` is False (missing dep / no result / no player).
    """

    ok: bool
    title: str = ""
    reason: str = ""


def _which_player(preference: str) -> str | None:
    """Pick a playback tool honouring ``preference`` ('auto'|'mpv'|'ffplay'|'download').

    'auto' prefers mpv (stop + pause/resume via IPC, streams the page URL), then
    ffplay (stop only), then the 'download' fallback (yt-dlp -> temp file ->
    project playback). An explicit preference wins when that tool is present;
    otherwise it degrades to whatever is available.
    """
    have_mpv = shutil.which("mpv") is not None
    have_ffplay = shutil.which("ffplay") is not None
    if preference == "mpv":
        if have_mpv:
            return "mpv"
    elif preference == "ffplay":
        if have_ffplay:
            return "ffplay"
    elif preference == "download":
        return "download"
    # auto (or a preferred tool that is absent): best available, then download.
    if have_mpv:
        return "mpv"
    if have_ffplay:
        return "ffplay"
    return "download"  # always possible via yt-dlp + project playback (needs ffmpeg-less wheels)


@dataclass
class MusicPlayer:
    """A stoppable background YouTube audio player.

    One player owns at most one active playback at a time; :meth:`play` stops any
    current track first. The backing process (mpv/ffplay) or a download-and-play
    thread is tracked so :meth:`stop` / :meth:`pause` / :meth:`resume` can act on
    it. Designed to be safe to construct with no extras installed — the failure
    surfaces only when :meth:`play` is actually called without yt-dlp / a player.
    """

    player: str = "auto"  # cfg.music_player: auto|mpv|ffplay|download
    volume: int | None = None  # 0..100; None = leave the player default
    _proc: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)
    _ipc_path: str | None = field(default=None, init=False, repr=False)
    _backend: str = field(default="", init=False, repr=False)
    _title: str = field(default="", init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    # -- queries --------------------------------------------------------------

    def is_playing(self) -> bool:
        """True when a track is currently playing (or paused but still loaded)."""
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def now_playing(self) -> str:
        """Title of the current track, or '' when nothing is playing."""
        with self._lock:
            return self._title if self.is_playing() else ""

    # -- control --------------------------------------------------------------

    def play(self, query: str) -> PlayResult:
        """Resolve ``query`` on YouTube and start playing it in the background.

        Stops any current track first. Returns a :class:`PlayResult`: on success
        ``title`` names the played track (for the spoken confirmation); on failure
        ``reason`` is a short speakable explanation (missing yt-dlp / no player /
        no result). Never raises.
        """
        if not yt_dlp_available():
            return PlayResult(ok=False, reason="I can't play music yet — please install yt-dlp.")
        backend = _which_player(self.player)
        if backend in (None, "download") and not (shutil.which("mpv") or shutil.which("ffplay")):
            # download backend needs yt-dlp (present) + project playback; that is
            # always possible, so only mpv/ffplay-only preferences can be unmet.
            backend = "download"
        with self._lock:
            self.stop()
            track = search(query)
            if track is None:
                return PlayResult(
                    ok=False,
                    reason=f"I couldn't find {query} on YouTube."
                    if query
                    else "I couldn't find anything to play.",
                )
            try:
                self._start(backend or "download", track)
            except Exception as exc:  # noqa: BLE001 — never crash the turn
                log.warning("failed to start playback for %r: %s", track.title, exc)
                return PlayResult(
                    ok=False,
                    reason="I found the song but couldn't start the player. Install mpv or ffmpeg.",
                )
            self._title = track.title
            return PlayResult(ok=True, title=track.title)

    def stop(self) -> bool:
        """Stop the current track (kill the backing process). True if one was playing."""
        with self._lock:
            proc = self._proc
            if proc is None:
                return False
            was_running = proc.poll() is None
            # Try a graceful mpv 'quit' over IPC first; then terminate/kill.
            if self._backend == "mpv":
                self._mpv_command({"command": ["quit"]})
            with contextlib.suppress(Exception):
                proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except Exception:  # noqa: BLE001 — escalate to kill
                with contextlib.suppress(Exception):
                    proc.kill()
            self._cleanup()
            return was_running

    def pause(self) -> bool:
        """Best-effort pause (mpv IPC only). Returns True when the pause was sent."""
        with self._lock:
            if not self.is_playing() or self._backend != "mpv":
                return False
            return self._mpv_command({"command": ["set_property", "pause", True]})

    def resume(self) -> bool:
        """Best-effort resume (mpv IPC only). Returns True when the resume was sent."""
        with self._lock:
            if not self.is_playing() or self._backend != "mpv":
                return False
            return self._mpv_command({"command": ["set_property", "pause", False]})

    # -- internals ------------------------------------------------------------

    def _start(self, backend: str, track: Track) -> None:
        """Launch the chosen backend for ``track`` (assumes the lock is held)."""
        if backend == "mpv":
            self._start_mpv(track)
        elif backend == "ffplay":
            self._start_ffplay(track)
        else:
            self._start_download(track)
        self._backend = backend

    def _start_mpv(self, track: Track) -> None:
        """Stream the YouTube page URL through ``mpv --no-video`` with an IPC socket.

        mpv handles the yt-dlp resolution itself (it bundles a yt-dlp hook), so we
        hand it the page URL. The JSON IPC socket gives us pause/resume/quit.
        """
        ipc = os.path.join(tempfile.gettempdir(), f"mstt-mpv-{os.getpid()}-{int(time.time())}.sock")
        cmd = [
            "mpv",
            "--no-video",
            "--no-terminal",
            "--really-quiet",
            f"--input-ipc-server={ipc}",
        ]
        if self.volume is not None:
            cmd.append(f"--volume={self.volume}")
        cmd.append(track.url)
        self._proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        self._ipc_path = ipc

    def _start_ffplay(self, track: Track) -> None:
        """Stream the resolved bestaudio URL through ``ffplay -nodisp`` (stop only)."""
        info = _ytdlp_extract(track.title or track.url, want_stream=True)
        media_url = (info or {}).get("url") or track.url
        cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(media_url)]
        self._proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        self._ipc_path = None

    def _start_download(self, track: Track) -> None:
        """Fallback: yt-dlp downloads bestaudio to a temp file, then play it.

        Runs in a daemon thread so the turn stays responsive. The 'process' we
        track is a thin sentinel via a still-running flag on a thread; for stop we
        rely on the project's playback being interruptible. Because the project's
        :func:`audio.play` blocks, this fallback offers stop only by abandoning the
        thread — kept simple as a last resort when neither mpv nor ffplay exists.
        """
        import yt_dlp  # pylint: disable=import-outside-toplevel

        tmpdir = tempfile.mkdtemp(prefix="mstt-music-")
        out_tmpl = os.path.join(tmpdir, "track.%(ext)s")
        opts = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "format": "bestaudio/best",
            "outtmpl": out_tmpl,
        }

        def _worker() -> None:
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.download([track.url])
                files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir)]
                if not files:
                    return
                self._play_file(files[0])
            except Exception as exc:  # noqa: BLE001 — background, never crash the app
                log.warning("download-and-play failed for %r: %s", track.title, exc)

        thread = threading.Thread(target=_worker, name="mstt-music-download", daemon=True)
        thread.start()
        # No real OS process to track for the download path; mark a sentinel so
        # is_playing() reflects the thread liveness via a lightweight wrapper.
        self._proc = _ThreadProc(thread)  # type: ignore[assignment]
        self._ipc_path = None

    @staticmethod
    def _play_file(path: str) -> None:
        """Decode a downloaded audio file to PCM and play it via project playback."""
        try:
            import soundfile as sf  # pylint: disable=import-outside-toplevel

            data, sr = sf.read(path, dtype="float32", always_2d=False)
            from .audio import play  # pylint: disable=import-outside-toplevel

            play(data, int(sr))
        except Exception as exc:  # noqa: BLE001
            log.warning("could not play downloaded file %s: %s", path, exc)

    def _mpv_command(self, payload: dict[str, Any]) -> bool:
        """Send one JSON command to mpv's IPC socket. True on a successful send."""
        ipc = self._ipc_path
        if not ipc:
            return False
        deadline = time.time() + _IPC_WAIT_S
        while not os.path.exists(ipc) and time.time() < deadline:
            time.sleep(0.05)
        if not os.path.exists(ipc):
            return False
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(_IPC_WAIT_S)
                sock.connect(ipc)
                sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            return True
        except OSError as exc:
            log.debug("mpv IPC command failed: %s", exc)
            return False

    def _cleanup(self) -> None:
        """Drop process/IPC references (assumes the lock is held)."""
        if self._ipc_path:
            with contextlib.suppress(Exception):
                os.unlink(self._ipc_path)
        self._proc = None
        self._ipc_path = None
        self._backend = ""
        self._title = ""


class _ThreadProc:
    """A minimal Popen-lookalike wrapping a thread (download-and-play fallback).

    Only the bits the player touches are implemented: ``poll`` (None while the
    thread is alive, 0 once it finishes) and no-op ``terminate``/``kill``/``wait``.
    """

    def __init__(self, thread: threading.Thread) -> None:
        self._thread = thread

    def poll(self) -> int | None:
        """None while the worker thread is alive, 0 once it has finished."""
        return None if self._thread.is_alive() else 0

    def terminate(self) -> None:
        """No-op: a download/playback in flight can't be force-stopped here."""

    def kill(self) -> None:
        """No-op (see :meth:`terminate`)."""

    def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002 — Popen API parity
        """Return immediately; the thread is a daemon and isn't joined."""
        return 0


# Process-wide default player so the intent router (called per turn from the
# stateless turn handler) shares ONE player across turns — so a later "stop" acts
# on the track an earlier "play" started. Built lazily from config on first use.
_DEFAULT_PLAYER: MusicPlayer | None = None


def get_player(*, player: str = "auto", volume: int | None = None) -> MusicPlayer:
    """Return the shared process-wide :class:`MusicPlayer`, building it once.

    The first call fixes the backend preference + volume; later calls reuse the
    same instance (so stop/pause reach the running track) and ignore new args.
    """
    global _DEFAULT_PLAYER  # noqa: PLW0603 — intentional process-wide singleton
    if _DEFAULT_PLAYER is None:
        _DEFAULT_PLAYER = MusicPlayer(player=player, volume=volume)
    return _DEFAULT_PLAYER


def reset_player() -> None:
    """Stop and drop the shared player (used by tests for isolation)."""
    global _DEFAULT_PLAYER  # noqa: PLW0603
    if _DEFAULT_PLAYER is not None:
        with contextlib.suppress(Exception):
            _DEFAULT_PLAYER.stop()
    _DEFAULT_PLAYER = None
