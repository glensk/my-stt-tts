"""The repo's settings come from ``.env.sops`` through sops-env-run (tp#1386)."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from my_stt_tts import sops_reexec
from my_stt_tts import voice_profiles as vp

CANONICAL = Path.home() / "obsidian/42-Git/home/mydotfiles/bin/sops_reexec.py"


def _fake_sops(tmp_path: Path) -> tuple[Path, Path]:
    log = tmp_path / "sops.log"
    fake = tmp_path / "sops"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"open({str(log)!r}, 'a').write(json.dumps("
        "{'argv': sys.argv[1:], 'stdin': sys.stdin.read()}) + '\\n')\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    return fake, log


def test_default_env_file_is_the_sops_file() -> None:
    assert vp.ENV_FILE.name == ".env.sops"


def test_env_value_never_parses_a_sops_file(tmp_path: Path) -> None:
    sops_file = tmp_path / ".env.sops"
    sops_file.write_text("MAC_VOICE_AUTH_THRESHOLD=ENC[AES256_GCM,data:x]\nsops_version=3\n")
    assert vp.env_value("MAC_VOICE_AUTH_THRESHOLD", sops_file, env={}) == ""
    assert vp.env_float("MAC_VOICE_AUTH_THRESHOLD", 0.35, sops_file, env={}) == 0.35
    env = {"MAC_VOICE_AUTH_THRESHOLD": "0.5"}
    assert vp.env_float("MAC_VOICE_AUTH_THRESHOLD", 0.35, sops_file, env=env) == 0.5


def test_env_set_and_unset_on_sops_use_sops_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, log = _fake_sops(tmp_path)
    monkeypatch.setattr(vp, "SOPS", str(fake))
    target = tmp_path / ".env.sops"
    vp.env_set(target, "MAC_VOICE_AUTH_THRESHOLD", "0.42")
    vp.env_unset(target, "MAC_VOICE_AUTH_MARGIN")
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows[0]["argv"] == [
        "set",
        "--value-stdin",
        "--input-type",
        "dotenv",
        "--output-type",
        "dotenv",
        str(target),
        '["MAC_VOICE_AUTH_THRESHOLD"]',
    ]
    assert rows[0]["stdin"] == json.dumps("0.42")  # the value never on argv
    assert rows[1]["argv"][:1] == ["unset"]
    assert "--value-stdin" not in rows[1]["argv"]


def test_env_set_plaintext_file(tmp_path: Path) -> None:
    target = tmp_path / "plain.env"
    target.write_text("OTHER=x\n")
    vp.env_set(target, "MAC_VOICE_AUTH_THRESHOLD", "0.4")
    assert vp.env_value("MAC_VOICE_AUTH_THRESHOLD", target, env={}) == "0.4"
    vp.env_unset(target, "MAC_VOICE_AUTH_THRESHOLD")
    assert vp.env_value("MAC_VOICE_AUTH_THRESHOLD", target, env={}) == ""


def test_ensure_env_never_reexecs_under_pytest(tmp_path: Path) -> None:
    (tmp_path / ".env.sops").write_text("A=ENC[x]\nsops_version=3\n")
    assert os.environ.get("PYTEST_CURRENT_TEST")
    sops_reexec.ensure_env(str(tmp_path / "x.py"), required=("NOT_SET_ANYWHERE",))


@pytest.mark.skipif(not CANONICAL.is_file(), reason="mydotfiles checkout absent")
def test_vendored_template_is_the_canonical_copy() -> None:
    vendored = Path(sops_reexec.__file__).read_bytes()
    assert vendored == CANONICAL.read_bytes(), "re-copy bin/sops_reexec.py from mydotfiles"
