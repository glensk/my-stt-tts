"""Mac fast path (PLAN_claude-bridge 7.3): URLs, apps, volume, brightness, doctor.

Fakes from :mod:`mac_fakes`; no real osascript, open or display access.
"""

from __future__ import annotations

import plistlib
from pathlib import Path
from typing import Any

import pytest
from mac_fakes import (
    OTHER,
    SYSTEM_EVENTS_PROBE,
    Call,
    FakeMac,
    FakeScorer,
    FakeVad,
)

from my_stt_tts import eleven_voice, mac_control
from my_stt_tts.bridge import Authoriser, BridgeController, MemoryProblemSink
from my_stt_tts.mac_control import (
    MDLS,
    OPEN,
    PRESS_KEY,
    READ_VOLUME,
    SAFARI_JS_PROBE,
    SET_MUTE,
    SET_VOLUME,
    TOOL_NAMES,
    Doctor,
    doctor_main,
    installed_apps,
    normalise_url,
    register,
    resolve_app,
)
from my_stt_tts.mac_sites import (
    OSASCRIPT,
    RunResult,
)


# -- URLs ------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "target",
    [
        "javascript:alert(1)",
        "file:///etc/passwd",
        "ftp://example.org",
        "mailto:a@example.org",
        "data:text/html,hi",
        "https://user:pw@evil.example",
        "https://user@evil.example",
        "http://exa\x00mple.com",
        "https://example.com/\x85",
        "https://example.com/\x1b[201~",
        "https://exa mple.com",
        "https://" + "a" * 2000 + ".com",
        "",
        "https://bad_host!.com",
    ],
)
def test_url_rejections(target: str) -> None:
    assert not isinstance(normalise_url(target), str), target


@pytest.mark.parametrize(
    ("target", "url"),
    [
        ("youtube", "https://www.youtube.com"),
        ("YouTube", "https://www.youtube.com"),
        ("You Tube", "https://www.youtube.com"),
        ("jellyfin", "https://jellyfin.dom42.space"),
        ("Jellyfin", "https://jellyfin.dom42.space"),
        ("example.org", "https://example.org"),
        ("example.org/path?q=1", "https://example.org/path?q=1"),
        ("http://example.org", "http://example.org"),
        ("HTTPS://Example.org/x", "HTTPS://Example.org/x"),
        ("wikipedia", "https://wikipedia.com"),
        ("localhost:3000", "https://localhost:3000"),
    ],
)
def test_url_shortcuts_and_bare_hosts(target: str, url: str) -> None:
    assert normalise_url(target) == url


def test_open_url_opens_safari_with_argv() -> None:
    call = Call()
    call.say("open YouTube")
    assert call.mac.open_url("youtube").startswith("ok:")
    assert call.fake.opened() == [[OPEN, "-a", "Safari", "https://www.youtube.com"]]


def test_open_url_bare_host() -> None:
    call = Call()
    call.say("open example dot org please")
    assert call.mac.open_url("example.org").startswith("ok:")
    assert call.fake.opened() == [[OPEN, "-a", "Safari", "https://example.org"]]


def test_open_url_rejected_target_opens_nothing() -> None:
    call = Call()
    call.say("open javascript")
    assert call.mac.open_url("javascript:alert(1)").startswith("refused:")
    assert not call.fake.opened()


def test_open_url_failure_is_reported() -> None:
    call = Call()
    call.fake.open_rc = 1
    call.say("open jellyfin")
    assert call.mac.open_url("jellyfin").startswith("failed:")
    assert [p.kind for p in call.problems] == ["mac_action"]


def test_log_line_names_the_host_only(caplog: pytest.LogCaptureFixture) -> None:
    call = Call()
    call.say("open example dot org")
    with caplog.at_level("INFO", logger="my_stt_tts.bridge"):
        call.mac.open_url("https://example.org/secret/path?token=1")
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("🛠️")]
    assert lines == ["🛠️ open_url example.org"]


# -- capability --------------------------------------------------------------------------------
def test_argument_not_in_transcript_is_refused() -> None:
    call = Call()
    call.say("open YouTube")
    out = call.mac.open_url("example.org")
    assert out.startswith("refused:") and "not in what you said" in out
    assert not call.fake.opened()


def test_no_capability_without_a_transcript() -> None:
    call = Call()
    assert call.mac.open_url("youtube").startswith("refused:")
    assert not call.fake.opened()


def test_other_speaker_is_refused() -> None:
    call = Call()
    call.say("louder", level=OTHER)
    assert call.mac.set_volume(step="up") == "refused: voice not verified"
    assert not call.fake.scripts(SET_VOLUME)


def test_capability_is_single_use() -> None:
    call = Call()
    call.say("open YouTube")
    assert call.mac.open_url("youtube").startswith("ok:")
    assert call.mac.open_url("youtube").startswith("refused:")
    assert len(call.fake.opened()) == 1


def test_cancelled_before_mutation() -> None:
    call = Call()
    call.say("open YouTube")
    call.ctl.cancel.cancel("stop")
    assert call.mac.open_url("youtube") == "refused: stop"
    assert not call.fake.opened()


def test_mutation_lock_busy_is_refused() -> None:
    call = Call()
    call.say("open YouTube")
    with call.ctl.mutation_lock:
        assert "another action" in call.mac.open_url("youtube")
    assert not call.fake.opened()


# -- apps ------------------------------------------------------------------------------------
def make_app(base: Path, name: str, bundle: str | None) -> Path:
    app = base / f"{name}.app"
    (app / "Contents").mkdir(parents=True)
    if bundle is not None:
        with (app / "Contents" / "Info.plist").open("wb") as fh:
            plistlib.dump({"CFBundleIdentifier": bundle}, fh)
    return app


@pytest.fixture(name="apps")
def fixture_apps(tmp_path: Path) -> list[Path]:
    user, system = tmp_path / "Applications", tmp_path / "System"
    make_app(user, "Microsoft Word", "com.microsoft.Word")
    make_app(user, "Microsoft Teams", "com.microsoft.teams2")
    make_app(user, "NoPlist", None)
    make_app(system, "Calculator", "com.apple.calculator")
    make_app(system / "Utilities", "Terminal", "com.apple.Terminal")
    make_app(system, "Microsoft Word", "com.example.shadowed")  # first directory wins
    return [user, system, tmp_path / "missing"]


def test_app_resolution(apps: list[Path]) -> None:
    found = installed_apps(apps)
    assert resolve_app("calculator", found) == apps[1] / "Calculator.app"
    assert resolve_app("Terminal.app", found) == apps[1] / "Utilities" / "Terminal.app"
    assert resolve_app("teams", found) == apps[0] / "Microsoft Teams.app"
    assert resolve_app("Microsoft Word", found) == apps[0] / "Microsoft Word.app"
    assert resolve_app("microsoft", found) == ["Microsoft Teams", "Microsoft Word"]
    assert resolve_app("photoshop", found) is None


def test_open_app_by_bundle_id(apps: list[Path]) -> None:
    call = Call(apps)
    call.say("open the calculator")
    assert call.mac.open_app("Calculator").startswith("ok:")
    assert call.fake.opened() == [[OPEN, "-b", "com.apple.calculator"]]


def test_open_app_ambiguous_names_the_candidates(apps: list[Path]) -> None:
    call = Call(apps)
    call.say("open Microsoft")
    out = call.mac.open_app("Microsoft")
    assert out == "refused: which one: Microsoft Teams, Microsoft Word?"
    assert not call.fake.opened()


def test_open_app_unknown(apps: list[Path]) -> None:
    call = Call(apps)
    call.say("open photoshop")
    assert call.mac.open_app("Photoshop") == "refused: no app called Photoshop"


def test_open_app_bundle_id_from_launch_services(apps: list[Path]) -> None:
    call = Call(apps)
    call.say("open noplist")
    assert call.mac.open_app("NoPlist").startswith("ok:")
    assert call.fake.opened() == [[OPEN, "-b", "com.example.fallback"]]
    assert [a[:3] for a in call.fake.calls if a[0] == MDLS] == [
        [MDLS, "-name", "kMDItemCFBundleIdentifier"]
    ]


# -- volume ----------------------------------------------------------------------------------
def test_volume_level() -> None:
    call = Call()
    call.say("volume thirty")
    assert call.mac.set_volume(level=30) == "ok: volume 30"
    assert call.fake.scripts(SET_VOLUME) == [["30"]]


@pytest.mark.parametrize("level", [101, -1, "loud"])
def test_volume_bounds(level: Any) -> None:
    call = Call()
    call.say(f"volume {level}")
    assert call.mac.set_volume(level=level).startswith("refused:")
    assert not call.fake.scripts(SET_VOLUME)


def test_volume_step_up_default_ten() -> None:
    call = Call()
    call.say("louder")
    assert call.mac.set_volume(step="up") == "ok: volume 65"


def test_volume_step_clamps_at_zero() -> None:
    call = Call()
    call.fake.volume = 4
    call.say("leiser")
    assert call.mac.set_volume(step="down") == "ok: volume 0"


def test_volume_already_at_max_does_nothing() -> None:
    call = Call()
    call.fake.volume = 100
    call.say("lauter")
    assert call.mac.set_volume(step="up") == "ok: volume already 100"
    assert not call.fake.scripts(SET_VOLUME)


def test_volume_numeric_step_must_be_said() -> None:
    call = Call()
    call.say("louder")
    assert "not in what you said" in call.mac.set_volume(step="+20")
    call.say("twenty louder")
    assert call.mac.set_volume(step="+20") == "ok: volume 75"


def test_volume_mute_and_unmute() -> None:
    call = Call()
    call.say("mute")
    assert call.mac.set_volume(mute=True) == "ok: muted"
    call.say("unmute")
    assert call.mac.set_volume(mute=False) == "ok: unmuted"
    assert call.fake.scripts(SET_MUTE) == [["true"], ["false"]]


def test_volume_needs_exactly_one_argument() -> None:
    call = Call()
    call.say("volume thirty louder")
    assert call.mac.set_volume(level=30, step="up").startswith("refused:")
    assert call.mac.set_volume().startswith("refused:")


def test_volume_not_changing_is_a_problem() -> None:
    call = Call()
    call.fake.volume_sticks = True
    call.say("volume twenty")
    assert call.mac.set_volume(level=20) == "failed: volume stayed at 55"
    assert call.problems[0].subject == "set_volume"


# -- brightness --------------------------------------------------------------------------------
def test_brightness_up_on_builtin_display() -> None:
    call = Call()
    call.say("brighter")
    assert call.mac.set_brightness("up") == "ok: brightness 62 percent"
    assert call.fake.scripts(PRESS_KEY) == [["144", "2"]]


def test_brightness_down() -> None:
    call = Call()
    call.display.values = [0.5, 0.375]
    call.say("dunkler")
    assert call.mac.set_brightness("down").startswith("ok:")
    assert call.fake.scripts(PRESS_KEY) == [["145", "2"]]


def test_brightness_external_display_goes_to_operator() -> None:
    call = Call()
    call.display.builtin = False
    call.say("brighter")
    assert "Mac operator" in call.mac.set_brightness("up")
    assert not call.fake.scripts(PRESS_KEY)


def test_brightness_already_max() -> None:
    call = Call()
    call.display.values = [1.0]
    call.say("brighter")
    assert call.mac.set_brightness("up").startswith("ok:")
    assert not call.fake.scripts(PRESS_KEY)


def test_brightness_key_presses_refused() -> None:
    call = Call()
    call.fake.fail[PRESS_KEY] = RunResult(
        1, "", "osascript is not allowed assistive access. (-1719)"
    )
    call.say("brighter")
    out = call.mac.set_brightness("up")
    assert out.startswith("failed:") and "-1719" in out
    assert call.problems


# -- doctor + register -------------------------------------------------------------------------
def green_fake() -> FakeMac:
    return FakeMac()


def test_doctor_all_green(capsys: pytest.CaptureFixture[str]) -> None:
    assert doctor_main(runner=green_fake(), ax=lambda: True) == 0
    out = capsys.readouterr().out
    assert out.count("✅") == 6 and "❌" not in out


def test_doctor_one_red(capsys: pytest.CaptureFixture[str]) -> None:
    fake = green_fake()
    fake.fail[SYSTEM_EVENTS_PROBE] = RunResult(
        1, "", "Not authorized to send Apple events to System Events. (-1743)"
    )
    assert doctor_main(runner=fake, ax=lambda: True) == 1
    out = capsys.readouterr().out
    assert "❌ System Events automation — not authorised (-1743)" in out
    assert "fix: System Settings" in out


def test_doctor_checks_each_permission() -> None:
    fake = green_fake()
    fake.fail[SAFARI_JS_PROBE] = RunResult(
        1, "", "You must enable the 'Allow JavaScript from Apple Events' option. (8)"
    )
    fake.fail[READ_VOLUME] = RunResult(0, "missing value\n")
    checks = {c.name: c for c in Doctor(runner=fake, ax=lambda: False).run()}
    assert checks["Accessibility (key presses)"].status == "fail"
    assert checks["Safari 'Allow JavaScript from Apple Events'"].status == "fail"
    assert "Develop" in checks["Safari 'Allow JavaScript from Apple Events'"].fix
    assert checks["Volume read"].status == "fail"
    assert checks["System Events automation"].status == "ok"


def test_doctor_cli_launches_safari_in_the_background() -> None:
    fake = green_fake()
    fake.safari_windows = "not running"
    checks = Doctor(runner=fake, ax=lambda: True, sleep=lambda _s: None, launch_safari=True).run()
    assert [OPEN, "-g", "-a", "Safari"] in fake.calls
    assert checks[2].status == "skip"  # the fake Safari never starts


def controller() -> BridgeController:
    return BridgeController(Authoriser(FakeScorer(), "albert"), vad_factory=FakeVad)


def test_register_needs_the_flag() -> None:
    ctl = controller()
    assert register(ctl, runner=green_fake(), env={}) is None
    assert set(ctl.tools) == {"confirm_action"}


def test_register_all_green_registers_the_tools() -> None:
    ctl = controller()
    mac = register(ctl, runner=green_fake(), env={"MAC_VOICE_MAC_CONTROL": "1"}, ax=lambda: True)
    assert mac is not None and mac.disabled is None
    assert set(TOOL_NAMES) <= set(ctl.tools)
    assert getattr(ctl.tools["open_url"], "__self__", None) is mac


def test_register_with_a_red_check_disables_every_tool() -> None:
    ctl = controller()
    fake = green_fake()
    fake.fail[READ_VOLUME] = RunResult(1, "", "boom (-1)")
    mac = register(ctl, runner=fake, env={"MAC_VOICE_MAC_CONTROL": "1"}, ax=lambda: True)
    assert mac is not None and mac.disabled == "Volume read"
    for name in TOOL_NAMES:
        assert ctl.tools[name](target="youtube") == "Mac control disabled: Volume read"
    assert not fake.opened()
    sink = ctl.problems
    assert isinstance(sink, MemoryProblemSink) and sink.problems[0].kind == "mac_control"


def test_register_with_safari_closed_keeps_the_tools() -> None:
    ctl = controller()
    fake = green_fake()
    fake.safari_windows = "not running"
    mac = register(ctl, runner=fake, env={"MAC_VOICE_MAC_CONTROL": "1"}, ax=lambda: True)
    assert mac is not None and mac.disabled is None
    assert [OPEN, "-g", "-a", "Safari"] not in fake.calls  # never launches Safari itself


TIMEOUT = RunResult(124, "", "timeout")  # what run_argv returns on subprocess.TimeoutExpired


class TimesOut:
    """Wraps a :class:`FakeMac`: ``script`` times out on its first ``times`` calls."""

    def __init__(self, fake: FakeMac, script: str, times: int) -> None:
        self.fake, self.script, self.left = fake, script, times

    def __call__(self, argv: Any, timeout: float) -> RunResult:
        result = self.fake(argv, timeout)  # recorded in fake.calls either way
        if argv[0] == OSASCRIPT and argv[2] == self.script and self.left > 0:
            self.left -= 1
            return TIMEOUT
        return result


def register_with(runner: Any) -> tuple[BridgeController, Any, list[float]]:
    ctl = controller()
    slept: list[float] = []
    env = {"MAC_VOICE_MAC_CONTROL": "1"}
    mac = register(ctl, runner=runner, env=env, ax=lambda: True, sleep=slept.append)
    return ctl, mac, slept


def test_register_retries_the_doctor_once_after_a_timeout() -> None:
    fake = green_fake()
    ctl, mac, slept = register_with(TimesOut(fake, SAFARI_JS_PROBE, 1))
    assert mac is not None and mac.disabled is None
    assert slept == [1.0]
    assert len(fake.scripts(SAFARI_JS_PROBE)) == 2
    assert getattr(ctl.tools["open_url"], "__self__", None) is mac
    sink = ctl.problems
    assert isinstance(sink, MemoryProblemSink) and not sink.problems


def test_register_disables_after_two_timeouts() -> None:
    fake = green_fake()
    ctl, mac, slept = register_with(TimesOut(fake, SAFARI_JS_PROBE, 2))
    name = "Safari 'Allow JavaScript from Apple Events'"
    assert mac is not None and mac.disabled == name
    assert slept == [1.0]
    assert len(fake.scripts(SAFARI_JS_PROBE)) == 2  # exactly one retry
    for tool in TOOL_NAMES:
        assert ctl.tools[tool](target="youtube") == f"Mac control disabled: {name}"
    sink = ctl.problems
    assert isinstance(sink, MemoryProblemSink) and len(sink.problems) == 1
    assert sink.problems[0].kind == "mac_control"


def test_register_does_not_retry_a_real_failure() -> None:
    fake = green_fake()
    fake.fail[SAFARI_JS_PROBE] = RunResult(
        1, "", "You must enable the 'Allow JavaScript from Apple Events' option. (8)"
    )
    ctl, mac, slept = register_with(fake)
    assert mac is not None and mac.disabled == "Safari 'Allow JavaScript from Apple Events'"
    assert not slept
    assert len(fake.scripts(SAFARI_JS_PROBE)) == 1  # the doctor ran once
    sink = ctl.problems
    assert isinstance(sink, MemoryProblemSink) and len(sink.problems) == 1


def test_mac_voice_doctor_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []

    def doctor() -> int:
        seen.append(True)
        return 1

    monkeypatch.setattr(mac_control, "doctor_main", doctor)
    assert eleven_voice.main(["-D"]) == 1
    assert seen == [True]
