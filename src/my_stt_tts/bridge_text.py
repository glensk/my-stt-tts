"""Text helpers of the Claude/Mac bridge: normalising, number words, hosts, redaction.

* :func:`normalise` — casefold, drop accents (``ß`` → ``ss``, ``é`` → ``e``), keep only
  letters and digits separated by single spaces.
* :func:`numbers_in` — every number a transcript says, as digits or as German, English or
  French number words from 0 to 100 ("dreissig", "thirty", "trente", "fünfunddreißig",
  "quatre-vingt-dix", Swiss "septante"/"huitante"/"nonante").
* :func:`normalise_host` / :func:`host_said` — "YouTube" → ``youtube.com``; a host is
  derivable from a transcript when its main label occurs in it.
* :func:`derivable` — the fast-path rule: an argument must come from what was said.
* :func:`redact` — token / key / JWT / private-key patterns replaced, text capped.
"""

from __future__ import annotations

import re
import unicodedata
from functools import cache
from urllib.parse import urlsplit

REDACTED = "[redacted]"
TEXT_CAP = 1500

SHORTCUTS = {"youtube": "youtube.com", "jellyfin": "jellyfin.dom42.space"}

# Spoken forms of keyword arguments (normalised). A token that STARTS with a form
# counts ("nächstes" → "nachstes" starts with "nachste").
KEYWORDS: dict[str, tuple[str, ...]] = {
    "youtube": ("youtube", "you tube"),
    "jellyfin": ("jellyfin", "jelly fin"),
    "play": ("play", "spiel", "abspiel", "lecture", "joue", "jouer", "lance"),
    "pause": ("pause", "pausier", "anhalten", "stopp", "stop"),
    "next": ("next", "nachste", "weiter", "suivant"),
    "previous": ("previous", "back", "zuruck", "vorherig", "precedent"),
    "fullscreen": ("fullscreen", "full screen", "vollbild", "plein ecran"),
    "seek": (
        "seek",
        "skip",
        "spul",
        "vorspul",
        "zuruckspul",
        "rewind",
        "forward",
        "avance",
        "recule",
    ),
    "mute": ("mute", "stumm", "muet"),
    "unmute": ("unmute", "ton an", "remets le son"),
    "up": ("louder", "up", "lauter", "heller", "brighter", "plus fort", "monte"),
    "down": ("quieter", "down", "leiser", "dunkler", "darker", "moins fort", "baisse"),
}

_SECRETS = (
    re.compile(r"-----BEGIN [^\n-]*-----[\s\S]*?(?:-----END [^\n-]*-----|\Z)"),
    re.compile(r"glpat-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?:ghp_|github_pat_)[A-Za-z0-9_]{16,}"),
    re.compile(r"xox[bp]-[A-Za-z0-9\-]{8,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),
)


def redact(text: str, cap: int = TEXT_CAP) -> str:
    """``text`` with secrets replaced by ``[redacted]`` and cut to ``cap`` characters."""
    for pattern in _SECRETS:
        text = pattern.sub(REDACTED, text)
    return text if len(text) <= cap else text[: max(cap - 1, 0)] + "…"


def normalise(text: str) -> str:
    """Casefolded, accent-free, punctuation-free text with single spaces."""
    folded = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(c for c in folded if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^0-9a-z]+", " ", plain).split())


def compact(text: str) -> str:
    """:func:`normalise` without spaces ("You Tube" → "youtube")."""
    return normalise(text).replace(" ", "")


# -- number words ------------------------------------------------------------------------
_EN_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_EN_TEENS = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
_EN_TENS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_DE_UNITS = ["null", "eins", "zwei", "drei", "vier", "funf", "sechs", "sieben", "acht", "neun"]
_DE_TEENS = "zehn elf zwolf dreizehn vierzehn funfzehn sechzehn siebzehn achtzehn neunzehn"
_DE_TENS = ["zwanzig", "dreissig", "vierzig", "funfzig", "sechzig", "siebzig", "achtzig", "neunzig"]
_FR_UNITS = ["zero", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf"]
_FR_TEENS = "dix onze douze treize quatorze quinze seize dix sept dix huit dix neuf"
_FR_TENS = {2: "vingt", 3: "trente", 4: "quarante", 5: "cinquante", 6: "soixante"}
_FR_SWISS = {7: "septante", 8: ("huitante", "octante"), 9: "nonante"}
_ARTICLES = {"a", "ein", "eine", "un", "une"}  # never a number on their own


def _english() -> dict[str, int]:
    words = {w: i for i, w in enumerate(_EN_UNITS)}
    words |= {w: 10 + i for i, w in enumerate(_EN_TEENS.split())}
    for i, tens in enumerate(_EN_TENS, start=2):
        words[tens] = 10 * i
        words |= {f"{tens} {_EN_UNITS[u]}": 10 * i + u for u in range(1, 10)}
    words |= {"hundred": 100, "one hundred": 100, "a hundred": 100}
    return words


def _german() -> dict[str, int]:
    words = {w: i for i, w in enumerate(_DE_UNITS)}
    words |= {w: 10 + i for i, w in enumerate(_DE_TEENS.split())}
    for i, tens in enumerate(_DE_TENS, start=2):
        words[tens] = 10 * i
        for u in range(1, 10):
            unit = "ein" if u == 1 else _DE_UNITS[u]
            words[f"{unit}und{tens}"] = words[f"{unit} und {tens}"] = 10 * i + u
    words |= {"hundert": 100, "einhundert": 100, "ein hundert": 100}
    variants = {k.replace("funf", "fuenf").replace("zwolf", "zwoelf"): v for k, v in words.items()}
    return words | variants


def _french_below_20() -> dict[str, int]:
    words = {w: i for i, w in enumerate(_FR_UNITS)}
    teens = ["dix", "onze", "douze", "treize", "quatorze", "quinze", "seize"]
    words |= {w: 10 + i for i, w in enumerate(teens)}
    words |= {"dix sept": 17, "dix huit": 18, "dix neuf": 19}
    return words


def _french_tens(base: str, value: int, below: dict[str, int]) -> dict[str, int]:
    """``base`` + 1..9 (e.g. "vingt et un", "vingt deux")."""
    words = {base: value, f"{base} et un": value + 1, f"{base} un": value + 1}
    words |= {f"{base} {w}": value + n for w, n in below.items() if 2 <= n <= 9}
    return words


def _french() -> dict[str, int]:
    below = _french_below_20()
    words = dict(below)
    for i, base in _FR_TENS.items():
        words |= _french_tens(base, 10 * i, below)
    for w, n in below.items():  # 70–79 and 90–99 count on from 60 / 80
        if n >= 10:
            words[f"soixante {w}"] = 60 + n
            words[f"quatre vingt {w}"] = words[f"quatre vingts {w}"] = 80 + n
    words["soixante et onze"] = 71
    words |= {"quatre vingt": 80, "quatre vingts": 80, "cent": 100}
    words |= {f"quatre vingt {w}": 80 + n for w, n in below.items() if 1 <= n <= 9}
    for i, forms in _FR_SWISS.items():
        for base in (forms,) if isinstance(forms, str) else forms:
            words |= _french_tens(base, 10 * i, below)
    return words


@cache
def _number_words() -> dict[tuple[str, ...], int]:
    table = _english() | _german() | _french()
    return {tuple(k.split()): v for k, v in table.items() if k not in _ARTICLES}


def numbers_in(text: str) -> set[int]:
    """Every number ``text`` says (digits, or de/en/fr number words 0–100)."""
    tokens = normalise(text).split()
    words = _number_words()
    longest = max(len(k) for k in words)
    found: set[int] = set()
    i = 0
    while i < len(tokens):
        if tokens[i].isdigit():
            found.add(int(tokens[i]))
            i += 1
            continue
        step = 1
        for size in range(min(longest, len(tokens) - i), 0, -1):
            value = words.get(tuple(tokens[i : i + size]))
            if value is not None:
                found.add(value)
                step = size
                break
        i += step
    return found


# -- hosts and keywords --------------------------------------------------------------------
def normalise_host(target: str) -> str:
    """The bare host a spoken or typed target names ("YouTube" → ``youtube.com``)."""
    raw = target.strip()
    if "://" in raw:
        raw = urlsplit(raw).hostname or ""
    host = raw.split("/", 1)[0].strip().casefold().rstrip(".")
    host = re.sub(r"\s+", "", host)
    host = host.removeprefix("www.")
    if host in SHORTCUTS:
        return SHORTCUTS[host]
    return host if "." in host else f"{compact(host)}.com"


def host_label(host: str) -> str:
    """The label someone says for ``host`` (``jellyfin.dom42.space`` → ``jellyfin``)."""
    host = normalise_host(host)
    for word, target in SHORTCUTS.items():
        if host == target:
            return word
    labels = host.split(".")
    if len(labels) >= 3 and labels[-2] in {"co", "com", "org", "net", "ac", "gov"}:
        return labels[-3]
    return labels[-2] if len(labels) >= 2 else labels[0]


def host_said(host: str, text: str) -> bool:
    """True when the main label of ``host`` occurs in ``text`` (spaces ignored)."""
    label = compact(host_label(host))
    return bool(label) and label in compact(text)


def keyword_said(word: str, text: str) -> bool:
    """True when ``word`` (or one of its de/en/fr forms in :data:`KEYWORDS`) is in ``text``."""
    norm = normalise(text)
    tokens = norm.split()
    forms = KEYWORDS.get(normalise(word), ()) + (normalise(word),)
    for form in forms:
        if not form:
            continue
        if " " in form:
            if f" {form} " in f" {norm} ":
                return True
        elif any(tok == form or (len(form) >= 4 and tok.startswith(form)) for tok in tokens):
            return True
    return compact(word) in compact(text) if len(compact(word)) >= 4 else False


def _looks_like_host(value: str) -> bool:
    plain = value.strip().casefold()
    return (
        "://" in plain
        or plain in SHORTCUTS
        or bool(re.fullmatch(r"[\w\-]+(\.[\w\-]+)+/?\S*", plain))
    )


def derivable(value: object, text: str, key: str = "") -> bool:
    """True when the argument ``value`` (named ``key``) comes from what ``text`` says.

    Numbers must be said (digits or number words), hosts by their main label, any other
    string as a keyword (or one of its spoken forms), a boolean by its key's keyword.
    """
    if isinstance(value, bool):
        return bool(key) and keyword_said(key, text)
    if isinstance(value, int | float):
        return float(value).is_integer() and int(value) in numbers_in(text)
    if not isinstance(value, str) or not value.strip():
        return False
    if value.strip().lstrip("+-").isdigit():
        return int(value.strip().lstrip("+-")) in numbers_in(text)
    if key in {"host", "url", "target"} or _looks_like_host(value):
        return host_said(value, text)
    return keyword_said(value, text)
