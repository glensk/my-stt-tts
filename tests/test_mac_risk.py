"""Mac risk table (PLAN_claude-bridge 7.4): tab switching in shells, Trash summaries, folders.

The SAFE / CONFIRM tables and the broker's confirmation flow live in ``test_mac_broker``;
these cases cover the rules added after the tp#855 real run (2026-10-15).
"""

from __future__ import annotations

import pytest

from my_stt_tts import mac_risk
from my_stt_tts.mac_risk import CONFIRM, REFUSE, SAFE, RiskContext, classify


def ctx(front: str = "Safari", element: dict[str, str] | None = None) -> RiskContext:
    return RiskContext(lambda: front, lambda x, y: element, lambda: None)


@pytest.mark.parametrize(
    ("element", "risk"),
    [
        (
            {"AXRole": "AXRadioButton", "AXTitle": "rm -rf build", "parent.AXRole": "AXTabGroup"},
            SAFE,
        ),
        ({"AXRole": "AXButton", "AXSubrole": "AXTabButton", "AXTitle": "deploy"}, SAFE),
        ({"AXRole": "AXButton", "AXTitle": "Delete"}, CONFIRM),
        ({"AXRole": "AXTextArea"}, SAFE),
        ({"AXRole": "AXButton"}, CONFIRM),  # unlabeled control in a shell
    ],
)
def test_clicking_a_tab_in_a_shell_runs(element: dict[str, str], risk: str) -> None:
    verdict = classify("click", {"_point": (10.0, 10.0), "app": "iTerm2"}, ctx("iTerm2", element))
    assert verdict.risk == risk, verdict


@pytest.mark.parametrize(
    ("script", "path"),
    [
        ('tell application "Finder" to delete POSIX file "/tmp/bridge-scratch/delete-me.txt"',
         "/tmp/bridge-scratch/delete-me.txt"),
        ('tell application "Finder" to delete (POSIX file "/Users/a/x y.txt" as alias)',
         "/Users/a/x y.txt"),
        ('  TELL APPLICATION "Finder" TO DELETE POSIX FILE "/tmp/a"  ', "/tmp/a"),
    ],
)  # fmt: skip
def test_finder_delete_is_summarised_as_moving_to_the_trash(script: str, path: str) -> None:
    verdict = classify("applescript", {"script": script}, ctx())
    assert verdict.risk == CONFIRM
    assert verdict.summary == f"move {path} to the Trash"
    assert verdict.target == f"trash:{path}"


@pytest.mark.parametrize(
    "script",
    [
        'tell application "Finder" to delete POSIX file "relative/x"',
        'tell application "Finder" to delete POSIX file "/tmp/../etc/x"',
        'tell application "Finder" to delete POSIX file "/tmp/a"\nbeep',
        'tell application "Finder" to delete (POSIX file "/tmp/a"',
        'tell application "Finder" to delete every file of folder "x"',
        'tell application "Finder" to empty trash',
    ],
)
def test_other_finder_scripts_keep_the_generic_summary(script: str) -> None:
    verdict = classify("applescript", {"script": script}, ctx())
    assert verdict.risk == CONFIRM
    assert verdict.summary == "run an AppleScript" and not verdict.target


def test_long_trash_paths_are_cut_from_the_left() -> None:
    path = "/Users/albert/" + "deep/" * 20 + "file.txt"
    verdict = classify(
        "applescript", {"script": f'tell application "Finder" to delete POSIX file "{path}"'}
    )
    assert len(verdict.summary) <= mac_risk.SUMMARY_CAP
    assert verdict.summary.startswith("move …") and verdict.summary.endswith(
        "file.txt to the Trash"
    )
    assert verdict.target == f"trash:{path}"


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ("~/.ssh", "hidden"),
        ("/etc", "only folders under"),
        ("relative/dir", "absolute"),
        ("", "empty"),
    ],
)
def test_list_folder_outside_the_allowed_places_is_refused(path: str, reason: str) -> None:
    verdict = classify("list_folder", {"path": path}, ctx())
    assert verdict.risk == REFUSE and reason in verdict.reason
