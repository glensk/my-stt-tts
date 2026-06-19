"""_run_browser prints the GUI URL prominently AND auto-opens it (no devices)."""
# pylint: disable=missing-function-docstring,protected-access,too-few-public-methods

from __future__ import annotations

from unittest.mock import MagicMock, patch

from my_stt_tts.config import Config


class _FakeUI:
    """Stand-in WebUI: records construction, returns a fixed URL, never blocks."""

    last_url = "http://127.0.0.1:8765/"
    last_on_action = None

    def __init__(self, _cfg, _on_turn, on_action=None, **_kwargs) -> None:
        type(self).last_on_action = on_action

    def url(self) -> str:
        return self.last_url

    def serve_forever(self) -> None:
        return None


def test_run_browser_prints_and_opens_url(capsys):
    from my_stt_tts import __main__ as main_mod

    cfg = Config(anthropic_api_key="x")
    brain = MagicMock()
    with (
        patch("my_stt_tts.webui.WebUI", _FakeUI),
        patch("webbrowser.open") as wb_open,
    ):
        rc = main_mod._run_browser(
            cfg, brain, MagicMock(), MagicMock(), None, wake=False, port=8765
        )
    assert rc == 0
    out = capsys.readouterr().out
    assert "http://127.0.0.1:8765/" in out  # the link is shown to click
    assert "Open in your browser" in out  # shown prominently
    wb_open.assert_called_once_with("http://127.0.0.1:8765/")  # and auto-opened


def test_mic_test_action_works_without_voice_controller():
    """The mic_test action must run even when voice is off (no controller) — that is
    precisely when the user needs to diagnose the mic. It runs a standalone capture."""
    from my_stt_tts import __main__ as main_mod

    cfg = Config(anthropic_api_key="x")
    with (
        patch("my_stt_tts.webui.WebUI", _FakeUI),
        patch("webbrowser.open"),
        patch.object(main_mod, "_run_mic_test") as run_test,
    ):
        # stt=None → controller is None → the standalone path must still fire.
        main_mod._run_browser(cfg, MagicMock(), MagicMock(), MagicMock(), None, wake=False)
        handler = _FakeUI.last_on_action
        assert callable(handler)
        handler("mic_test", {})  # pylint: disable=not-callable
    # Worker thread runs _run_mic_test; give it a moment then assert it was invoked.
    import time

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not run_test.called:
        time.sleep(0.01)
    run_test.assert_called_once()
