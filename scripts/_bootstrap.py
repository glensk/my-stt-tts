#!/usr/bin/env python3
"""Stdlib-only venv bootstrap for the ``scripts/`` entrypoints.

So ``./scripts/<name>.py`` works without ``uv run`` or manual venv activation: if
the project package isn't importable (i.e. you're running under system Python and
hit ``ModuleNotFoundError: No module named 'numpy'``), re-exec under the uv-managed
project venv with the optional extras the script needs. This module must stay
**stdlib-only** — it runs before the venv exists.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_GUARD = "_MSTT_SCRIPT_BOOTSTRAP"


def ensure_venv(extras: list[str] | None = None) -> None:
    """Re-exec the calling script under ``uv run`` (+ ``extras``) if the project venv
    isn't active. A no-op when already inside it (probed via importing ``my_stt_tts``),
    and guarded by an env flag so the re-exec can't loop."""
    if os.environ.get(_GUARD) == "1":
        return
    try:
        import my_stt_tts  # noqa: F401  (probe: importable only inside the project venv)

        return
    except ModuleNotFoundError:
        pass
    repo = Path(__file__).resolve().parent.parent
    script = Path(sys.argv[0]).resolve()
    cmd = ["uv", "run"]
    for extra in extras or []:
        cmd += ["--extra", extra]
    cmd += ["python", str(script), *sys.argv[1:]]
    os.environ[_GUARD] = "1"
    os.chdir(repo)
    try:
        os.execvp("uv", cmd)
    except FileNotFoundError:
        sys.exit(
            "This script needs the project venv. Install uv (https://astral.sh/uv), "
            "then re-run — or invoke it as: uv run " + " ".join(sys.argv)
        )
