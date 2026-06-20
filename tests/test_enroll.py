"""enroll.py multi-language coverage: ONE profile spans the household languages.

The enroll script cycles a language HINT across the clips (DE/EN/FR by default,
``--languages`` to override) so a single ``<name>.npy`` centroid covers every
language the person speaks — profiles are NOT split by language. These tests
exercise the pure helpers (parsing + per-clip prompt cycling) without recording.
"""
# pylint: disable=missing-function-docstring,import-outside-toplevel

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_enroll():
    """Import scripts/enroll.py WITHOUT triggering its venv bootstrap re-exec.

    The script calls ``ensure_venv([...])`` at import; in the test venv numpy is
    already importable so the bootstrap is a no-op, but we stub it defensively and
    make ``_bootstrap`` importable from the scripts dir.
    """
    sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location("enroll_under_test", _SCRIPTS / "enroll.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


enroll = _load_enroll()


# --------------------------------------------------------------------------- #
# parse_languages: clean, ordered, de-duplicated; defaults when empty           #
# --------------------------------------------------------------------------- #
def test_parse_languages_default_household_set():
    assert enroll.parse_languages("de,en,fr") == ["de", "en", "fr"]


def test_parse_languages_strips_lowercases_dedups():
    assert enroll.parse_languages(" DE , En, fr , de ") == ["de", "en", "fr"]


def test_parse_languages_empty_falls_back_to_default():
    assert enroll.parse_languages("") == list(enroll.DEFAULT_LANGUAGES)
    assert enroll.parse_languages("  ,, ") == list(enroll.DEFAULT_LANGUAGES)


def test_parse_languages_custom_subset():
    assert enroll.parse_languages("en,it") == ["en", "it"]


# --------------------------------------------------------------------------- #
# clip_prompt: cycles the language hint across the clips                         #
# --------------------------------------------------------------------------- #
def test_clip_prompt_cycles_languages_across_clips():
    langs = ["de", "en", "fr"]
    prompts = [enroll.clip_prompt(i, 6, langs) for i in range(6)]
    # The hint rotates DE, EN, FR, DE, EN, FR over six clips (one .npy spans all).
    assert "German" in prompts[0] and "Clip 1/6" in prompts[0]
    assert "English" in prompts[1]
    assert "French" in prompts[2]
    assert "German" in prompts[3]  # wrapped around
    assert "English" in prompts[4]
    assert "French" in prompts[5]


def test_clip_prompt_more_languages_than_clips():
    langs = ["de", "en", "fr"]
    # Only two clips -> first two languages used, no error.
    assert "German" in enroll.clip_prompt(0, 2, langs)
    assert "English" in enroll.clip_prompt(1, 2, langs)


@pytest.mark.parametrize(
    ("code", "name"),
    [("de", "German"), ("en", "English"), ("fr", "French"), ("xx", "XX")],
)
def test_language_name(code, name):
    assert enroll.language_name(code) == name
