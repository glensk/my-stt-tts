"""sops_reexec.py — re-exec a Python program through ``sops-env-run`` (tp#1386; vendored template).

Copy this file into the consumer (canonical: mydotfiles ``bin/sops_reexec.py``;
never edit a copy) and call it first thing in ``main()``, AFTER
``parse_args()`` so ``-h`` never decrypts:

    from sops_reexec import ensure_env
    ensure_env(__file__, required=("NOTION_TOKEN",), env_file=".env.sops")

* Every name in *required* already set (non-empty) → nothing happens: an
  exported value (or a parent's sops-env-run) wins.
* Already inside sops-env-run (``SOPS_ENV_RUN_FILE`` set, the loop guard) →
  nothing happens; with ``strict=True`` a still-missing name exits 2
  (names only).
* ``<root>/<env_file>`` missing → nothing happens (``strict=True``: exit 2).
* Else ``os.execv`` into ``sops-env-run <file> -- <sys.orig_argv>``: the same
  interpreter, flags and ``-m module`` / script argv, with the file's keys in
  the environment — never a decrypted file.

``base`` = how many directories above *script*'s directory the env file
lives (0 = the script's own dir; a module in ``src/<pkg>/`` uses 2). Under
pytest (``PYTEST_CURRENT_TEST``) it never re-execs: tests call ``main()``
in-process with their own environment.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Iterable

GUARD_ENV = "SOPS_ENV_RUN_FILE"
TOOL = "sops-env-run"
FALLBACK = "~/obsidian/42-Git/home/mydotfiles/bin/sops-env-run"


def ensure_env(
    script: str,
    required: Iterable[str] = (),
    env_file: str = ".env.sops",
    base: int = 0,
    strict: bool = False,
) -> None:
    """Re-exec through sops-env-run unless the environment is already complete."""
    names = list(required)
    missing = [name for name in names if not os.environ.get(name)]
    if (names and not missing) or os.environ.get("PYTEST_CURRENT_TEST"):
        return
    root = os.path.dirname(os.path.abspath(script))
    for _ in range(base):
        root = os.path.dirname(root)
    path = os.path.join(root, env_file)
    if os.environ.get(GUARD_ENV) or not os.path.isfile(path):
        if strict and missing:
            sys.exit(f"❌ missing {', '.join(missing)} (no {path} or not in it)")
        return
    tool = shutil.which(TOOL) or os.path.expanduser(FALLBACK)
    argv = list(getattr(sys, "orig_argv", [])) or [sys.executable, *sys.argv]
    argv[0] = sys.executable
    os.execv(tool, [tool, path, "--", *argv])
