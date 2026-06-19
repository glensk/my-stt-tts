"""Pluggable backend registry / factory for STT, TTS and LLM backends (G1).

The pipeline already talks to backends through three seams — the
:class:`~my_stt_tts.stt.Transcriber` protocol, the TTS ``render``/``synth``
surface, and the :class:`~my_stt_tts.brain.Brain` provider switch. Historically
each ``make_*`` function hard-coded an ``if backend == "cloud"`` branch, so
adding a real provider (Deepgram, ElevenLabs, Cartesia, …) meant editing the
selector. This module **formalises** the seam into a small, generic registry so
backends are *registered* (by name + service kind) and *selected* by name, and
new adapters drop in without touching the orchestrator.

Design:

* :class:`ServiceRegistry` — name → builder, namespaced by service kind
  (``"stt"`` / ``"tts"`` / ``"llm"``). Builders take a :class:`Config` and return
  a backend instance. Pure data + lookup, fully unit-tested.
* :data:`registry` — the process-wide default, pre-populated with the built-in
  backends (local + the cloud adapters) at import time.
* Each cloud adapter is **key-gated** and degrades gracefully: it exposes
  ``available()`` (true only when an API key is configured) and the selector
  falls back to the local backend when a cloud backend is selected but unusable.
  Network/SDK clients are lazy-imported from the optional extras so the core
  package stays dependency-light and importable without them.

The real adapters speak the actual provider APIs (Deepgram streaming STT;
ElevenLabs and Cartesia TTS). They are tested against *mocked* SDK/HTTP responses
— no live key is ever required.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import Config
    from .stt import Transcriber

log = logging.getLogger("my_stt_tts.registry")

# The service kinds the registry namespaces builders under.
SERVICE_KINDS = ("stt", "tts", "llm")

Builder = Callable[["Config"], Any]


class ServiceRegistry:
    """Name → builder lookup, namespaced by service kind (G1).

    A backend is :meth:`register`-ed under ``(kind, name)`` with a builder
    ``Config -> backend``; the selector calls :meth:`build` (or :meth:`get`) to
    instantiate the configured backend lazily. Unknown names raise so a typo in
    ``stt_backend`` fails loudly rather than silently picking a default.
    """

    def __init__(self) -> None:
        self._builders: dict[str, dict[str, Builder]] = {k: {} for k in SERVICE_KINDS}

    def register(self, kind: str, name: str, builder: Builder, *, replace: bool = False) -> None:
        """Register ``builder`` for ``(kind, name)``; raise on an unknown kind or clash."""
        if kind not in self._builders:
            raise ValueError(f"unknown service kind {kind!r}; choose from {SERVICE_KINDS}")
        table = self._builders[kind]
        if name in table and not replace:
            raise ValueError(f"{kind} backend {name!r} already registered")
        table[name] = builder

    def names(self, kind: str) -> tuple[str, ...]:
        """Registered backend names for ``kind`` (sorted, for help text / validation)."""
        if kind not in self._builders:
            raise ValueError(f"unknown service kind {kind!r}; choose from {SERVICE_KINDS}")
        return tuple(sorted(self._builders[kind]))

    def has(self, kind: str, name: str) -> bool:
        """Whether a backend ``name`` is registered under ``kind``."""
        return kind in self._builders and name in self._builders[kind]

    def get(self, kind: str, name: str) -> Builder:
        """Return the builder for ``(kind, name)`` or raise ``KeyError``."""
        if not self.has(kind, name):
            available = self.names(kind) if kind in self._builders else ()
            raise KeyError(f"no {kind} backend {name!r}; registered: {available}")
        return self._builders[kind][name]

    def build(self, kind: str, name: str, cfg: Config) -> Any:
        """Instantiate the configured backend via its registered builder."""
        return self.get(kind, name)(cfg)


# --- typed selector ----------------------------------------------------------


class _Selector[T]:
    """A small helper that selects a backend with a graceful local-first fallback.

    Cloud backends expose ``available()`` (key present?). :meth:`select` builds the
    requested backend and, if it is a cloud backend that is *not* usable, builds the
    named ``fallback`` instead — so a missing key never hard-fails the pipeline.
    """

    def __init__(self, kind: str, fallback: str) -> None:
        self.kind = kind
        self.fallback = fallback

    def select(self, cfg: Config, name: str, registry: ServiceRegistry) -> T:
        """Build ``name``; fall back to the local backend if it is unusable."""
        backend = registry.build(self.kind, name, cfg)
        available = getattr(backend, "available", None)
        if callable(available) and not available():
            log.info(
                "%s backend %r selected but unavailable (no key?); using %r.",
                self.kind,
                name,
                self.fallback,
            )
            return registry.build(self.kind, self.fallback, cfg)  # type: ignore[no-any-return]
        return backend  # type: ignore[no-any-return]


_STT_SELECTOR: _Selector[Transcriber] = _Selector("stt", "local")
_TTS_SELECTOR: _Selector[Any] = _Selector("tts", "local")


def select_transcriber(cfg: Config, registry: ServiceRegistry | None = None) -> Transcriber:
    """Select the STT backend named by ``cfg.stt_backend`` (local-first fallback)."""
    return _STT_SELECTOR.select(
        cfg, getattr(cfg, "stt_backend", "local"), registry or globals_reg()
    )


def select_tts_backend(cfg: Config, registry: ServiceRegistry | None = None) -> Any:
    """Select the cloud TTS backend named by ``cfg.tts_backend``; None for local.

    Returns a cloud renderer (``render(text) -> (pcm, sr)``) when a usable cloud
    backend is selected, or ``None`` when the local Piper/``say`` path should be
    used (either ``tts_backend == "local"`` or the cloud backend is key-gated off).
    """
    name = getattr(cfg, "tts_backend", "local")
    if name == "local":
        return None
    backend = registry or globals_reg()
    if not backend.has("tts", name):
        log.info("tts backend %r not registered; using local Piper / say.", name)
        return None
    chosen = backend.build("tts", name, cfg)
    available = getattr(chosen, "available", None)
    if callable(available) and not available():
        log.info("tts backend %r selected but no key set; using local Piper / say.", name)
        return None
    return chosen


# --- module-global default registry ------------------------------------------

_REGISTRY: ServiceRegistry | None = None


def globals_reg() -> ServiceRegistry:
    """Return the process-wide default registry, populating it on first use."""
    global _REGISTRY  # noqa: PLW0603 — intentional lazy singleton
    if _REGISTRY is None:
        _REGISTRY = ServiceRegistry()
        _register_builtins(_REGISTRY)
    return _REGISTRY


def _register_builtins(reg: ServiceRegistry) -> None:
    """Register the built-in STT / TTS / LLM backends (lazy imports inside builders)."""

    # --- STT ---
    def _local_stt(cfg: Config) -> Any:
        from .stt import ParakeetSTT

        return ParakeetSTT(cfg.stt_model)

    def _whispercpp_stt(cfg: Config) -> Any:
        from .stt import WhisperCppSTT

        return WhisperCppSTT(getattr(cfg, "whispercpp_model", "large-v3-turbo"))

    def _faster_whisper_stt(cfg: Config) -> Any:
        from .stt import FasterWhisperSTT

        return FasterWhisperSTT(
            getattr(cfg, "whispercpp_model", "large-v3-turbo"),
            compute_type=getattr(cfg, "faster_whisper_compute", "int8"),
        )

    def _openai_stt(cfg: Config) -> Any:
        from .stt import CloudTranscriber

        return CloudTranscriber(
            cfg.stt_cloud_model, api_key=cfg.stt_cloud_api_key, base_url=cfg.stt_cloud_base_url
        )

    def _deepgram_stt(cfg: Config) -> Any:
        from .stt_cloud import DeepgramSTT

        return DeepgramSTT(
            model=getattr(cfg, "deepgram_model", "nova-3"),
            api_key=getattr(cfg, "deepgram_api_key", None),
            language=getattr(cfg, "deepgram_language", None),
        )

    reg.register("stt", "local", _local_stt)
    reg.register("stt", "whispercpp", _whispercpp_stt)
    reg.register("stt", "faster-whisper", _faster_whisper_stt)
    reg.register("stt", "cloud", _openai_stt)
    reg.register("stt", "openai", _openai_stt)
    reg.register("stt", "deepgram", _deepgram_stt)

    # --- TTS (cloud renderers; "local" handled by the router directly) ---
    def _openai_tts(cfg: Config) -> Any:
        from .tts import CloudTTS

        return CloudTTS(
            cfg.tts_cloud_model,
            voice=cfg.tts_cloud_voice,
            api_key=cfg.tts_cloud_api_key,
            base_url=cfg.tts_cloud_base_url,
        )

    def _elevenlabs_tts(cfg: Config) -> Any:
        from .tts_cloud import ElevenLabsTTS

        return ElevenLabsTTS(
            voice_id=getattr(cfg, "elevenlabs_voice_id", "Rachel"),
            model=getattr(cfg, "elevenlabs_model", "eleven_multilingual_v2"),
            api_key=getattr(cfg, "elevenlabs_api_key", None),
        )

    def _cartesia_tts(cfg: Config) -> Any:
        from .tts_cloud import CartesiaTTS

        return CartesiaTTS(
            voice_id=getattr(cfg, "cartesia_voice_id", ""),
            model=getattr(cfg, "cartesia_model", "sonic-2"),
            api_key=getattr(cfg, "cartesia_api_key", None),
        )

    reg.register("tts", "cloud", _openai_tts)
    reg.register("tts", "openai", _openai_tts)
    reg.register("tts", "elevenlabs", _elevenlabs_tts)
    reg.register("tts", "cartesia", _cartesia_tts)

    # --- LLM (documented for completeness; Brain selects via provider switch) ---
    def _llm(_cfg: Config) -> Any:
        from .brain import Brain

        return Brain

    for provider in ("anthropic", "openai", "openai-compatible", "ollama", "claude-cli"):
        reg.register("llm", provider, _llm)
