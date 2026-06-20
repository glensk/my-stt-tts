"""Shared pytest fixtures.

Test isolation for the data-driven wake-reliability log: several wake-test paths
call :func:`my_stt_tts.config.record_wake_outcome`, which by default writes the
repo-local ``debug/wake_stats.json``. An autouse fixture redirects that path to a
per-test temp file so the test suite never pollutes (or reads) the real on-disk log
— keeping reliability assertions deterministic regardless of the developer's local
debug/ state.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_wake_stats(tmp_path_factory, monkeypatch):
    """Point ``config.wake_stats_path`` at a unique temp file for every test."""
    target = tmp_path_factory.mktemp("wake_stats") / "wake_stats.json"
    monkeypatch.setattr("my_stt_tts.config.wake_stats_path", lambda: str(target))
    return target
