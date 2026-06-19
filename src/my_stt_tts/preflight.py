"""Verified first-run bootstrap (R3-8): fetch + checksum models ahead of time.

Smart-Turn is the default endpointer, but its ONNX auto-downloads on first run; a
**silent** download failure used to degrade endpointing to a fixed silence timer
with no warning. This module is the preflight command (``my-stt-tts --preflight``)
that fetches the Smart-Turn model and the configured Piper voices *ahead of time*,
**SHA-256-verifies** the Smart-Turn ONNX against the pinned hash, and reports
clearly what is ready vs. what still needs attention — so the first real
conversation starts with everything verified, not discovered mid-turn.

The :class:`CheckResult` / :func:`run_preflight` core is **pure logic over
injectable fetch + checksum callables**, so it is unit-tested with fakes (no
network, no real download). :func:`preflight_main` wires the real
:func:`~my_stt_tts.turn.ensure_smart_turn_model` + Piper fetch and prints the
report; it is what ``--preflight`` calls.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .turn import ensure_smart_turn_model, verify_checksum

if TYPE_CHECKING:
    from .config import Config

log = logging.getLogger("my_stt_tts.preflight")

# Callables the core depends on, injected so tests can fake the network entirely.
EnsureModel = Callable[[str, str, bool, str], bool]  # (path, url, auto_dl, sha256) -> ok
EnsureVoice = Callable[[str, str], bool]  # (data_dir, voice) -> ok


@dataclass
class CheckResult:
    """The outcome of one preflight check (a model or a voice)."""

    name: str
    ok: bool
    detail: str = ""

    def line(self) -> str:
        """A single ``[ ready ] name — detail`` report row."""
        mark = "ready" if self.ok else "MISSING"
        tail = f" — {self.detail}" if self.detail else ""
        return f"  [ {mark:^7} ] {self.name}{tail}"


@dataclass
class PreflightReport:
    """The aggregate result of all preflight checks."""

    results: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True only when every check passed."""
        return all(r.ok for r in self.results)

    def text(self) -> str:
        """A human-readable multi-line report (header + one row per check)."""
        head = "preflight: all components ready ✓" if self.ok else "preflight: ACTION NEEDED"
        return "\n".join([head, *(r.line() for r in self.results)])


def check_smart_turn(
    cfg: Config,
    *,
    ensure: EnsureModel,
    checksum: Callable[[str, str], bool] = verify_checksum,
) -> CheckResult:
    """Fetch + SHA-256-verify the Smart-Turn ONNX (R3-8).

    When ``turn_analyzer`` is not ``smart`` the model is not needed, so this is a
    pass-through "n/a". Otherwise it ensures the file (downloading if missing) and
    then re-verifies the on-disk file against the pinned hash, so the report
    distinguishes "downloaded but corrupt" from "ready". Pure over ``ensure`` /
    ``checksum`` callables (faked in tests).
    """
    if getattr(cfg, "turn_analyzer", "smart") != "smart":
        return CheckResult("Smart-Turn model", ok=True, detail="not needed (turn_analyzer=silence)")
    path = cfg.smart_turn_model_path
    sha = getattr(cfg, "smart_turn_sha256", "")
    got = ensure(path, cfg.smart_turn_model_url, cfg.smart_turn_auto_download, sha)
    if not got:
        return CheckResult("Smart-Turn model", ok=False, detail=f"download failed -> {path}")
    if sha and not checksum(path, sha):
        return CheckResult(
            "Smart-Turn model", ok=False, detail=f"checksum MISMATCH at {path} (corrupt download)"
        )
    detail = f"{path} (sha256 verified)" if sha else f"{path}"
    return CheckResult("Smart-Turn model", ok=True, detail=detail)


def check_piper_voices(cfg: Config, *, ensure_voice: EnsureVoice) -> list[CheckResult]:
    """Fetch each configured Piper voice ahead of time, one :class:`CheckResult` each.

    Pure over the ``ensure_voice`` callable (the real one downloads via Piper); the
    tests fake it. A voice that already exists / downloads cleanly is ``ready``.
    """
    results: list[CheckResult] = []
    for lang, voice in cfg.tts_voices.items():
        ok = ensure_voice(cfg.piper_data_dir, voice)
        detail = (
            f"{voice}" if ok else f"{voice} (fetch failed; macOS 'say' will be used for {lang})"
        )
        results.append(CheckResult(f"Piper voice [{lang}]", ok=ok, detail=detail))
    return results


def run_preflight(
    cfg: Config,
    *,
    ensure_model: EnsureModel,
    ensure_voice: EnsureVoice,
    checksum: Callable[[str, str], bool] = verify_checksum,
) -> PreflightReport:
    """Run every preflight check and aggregate them into a :class:`PreflightReport`.

    The testable core (R3-8): all network/IO is behind the injected ``ensure_model``
    / ``ensure_voice`` / ``checksum`` callables, so a test drives the full happy
    path, a missing download, and a corrupt (checksum-mismatch) download with fakes.
    """
    report = PreflightReport()
    report.results.append(check_smart_turn(cfg, ensure=ensure_model, checksum=checksum))
    report.results.extend(check_piper_voices(cfg, ensure_voice=ensure_voice))
    return report


def _real_ensure_model(path: str, url: str, auto_dl: bool, sha256: str) -> bool:
    """Adapter from :func:`turn.ensure_smart_turn_model` to the :data:`EnsureModel` shape."""
    return ensure_smart_turn_model(path, url, auto_download=auto_dl, expected_sha256=sha256)


def _real_ensure_voice(data_dir: str, voice: str) -> bool:
    """Adapter to the real Piper voice fetch (lazy import so tests don't need it)."""
    if (Path(data_dir) / f"{voice}.onnx").exists():
        return True
    from .tts import _ensure_piper_voice  # noqa: PLC0415 — pulls subprocess deps lazily

    return _ensure_piper_voice(data_dir, voice)


def preflight_main(cfg: Config) -> int:
    """``--preflight``: fetch + verify the Smart-Turn model and Piper voices; report.

    Wires the real fetch/checksum into :func:`run_preflight`, prints the report, and
    returns ``0`` when everything is ready, ``1`` otherwise — so it doubles as a CI /
    setup gate.
    """
    report = run_preflight(cfg, ensure_model=_real_ensure_model, ensure_voice=_real_ensure_voice)
    print(report.text())
    return 0 if report.ok else 1
