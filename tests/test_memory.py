"""Tests for G7 — per-speaker persistent memory + provider-agnostic context.

Covers: the persistent stores (SQLite + JSON), per-speaker isolation (one person's
turns never leak into another's, guests share a bucket), provider-agnostic context
assembly (recall + live, dedup at the seam, budget), the structured-dialogue flow
hook, and the Brain integration (cross-session recall keyed by speaker, persistence
on completion, and commit_spoken amending persistent memory after a barge-in).
Persistence uses tmp_path; the LLM is faked — no network, no real model.
"""
# pylint: disable=missing-function-docstring,protected-access,missing-class-docstring
# pylint: disable=too-few-public-methods,import-outside-toplevel,redefined-outer-name

import numpy as np
import pytest

from my_stt_tts.config import Config
from my_stt_tts.memory import (
    ContextAggregator,
    DialogueFlow,
    FlowState,
    JsonMemoryBackend,
    MemoryStore,
    SqliteMemoryBackend,
    make_memory_store,
    speaker_key,
)

# ---------------------------------------------------------------------------
# speaker_key normalization
# ---------------------------------------------------------------------------


def test_speaker_key_buckets_unknown_and_ambiguous():
    assert speaker_key("alice") == "alice"
    assert speaker_key("unknown") == "_guest"
    assert speaker_key("ambiguous") == "_guest"
    assert speaker_key(None) == "_guest"


# ---------------------------------------------------------------------------
# stores — persistence + isolation (parametrized over both backends)
# ---------------------------------------------------------------------------


@pytest.fixture(params=["sqlite", "json"])
def store(request, tmp_path):
    if request.param == "sqlite":
        return MemoryStore(SqliteMemoryBackend(tmp_path / "mem.db"))
    return MemoryStore(JsonMemoryBackend(tmp_path / "mem.json"))


def test_store_append_and_recent(store):
    store.add_turn("alice", "user", "hi")
    store.add_turn("alice", "assistant", "hello alice")
    hist = store.history("alice", 10)
    assert hist == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello alice"},
    ]


def test_store_per_speaker_isolation(store):
    store.add_turn("alice", "user", "alice secret")
    store.add_turn("bob", "user", "bob secret")
    assert store.history("alice", 10) == [{"role": "user", "content": "alice secret"}]
    assert store.history("bob", 10) == [{"role": "user", "content": "bob secret"}]


def test_store_guest_bucket_shared(store):
    store.add_turn("unknown", "user", "guest 1")
    store.add_turn(None, "user", "guest 2")
    assert store.history("ambiguous", 10) == [
        {"role": "user", "content": "guest 1"},
        {"role": "user", "content": "guest 2"},
    ]


def test_store_recent_limit(store):
    for i in range(10):
        store.add_turn("alice", "user", f"m{i}")
    assert [t["content"] for t in store.history("alice", 3)] == ["m7", "m8", "m9"]


def test_store_forget(store):
    store.add_turn("alice", "user", "x")
    store.forget("alice")
    assert store.history("alice", 10) == []


def test_store_amend_last_assistant(store):
    store.add_turn("alice", "user", "q")
    store.add_turn("alice", "assistant", "long full answer that was cut off")
    store.amend_last_assistant("alice", "long full")
    assert store.history("alice", 10)[-1] == {"role": "assistant", "content": "long full"}


def test_store_amend_drops_when_blank(store):
    store.add_turn("alice", "user", "q")
    store.add_turn("alice", "assistant", "unspoken")
    store.amend_last_assistant("alice", "")
    assert store.history("alice", 10) == [{"role": "user", "content": "q"}]


def test_sqlite_persists_across_reopen(tmp_path):
    path = tmp_path / "mem.db"
    s1 = MemoryStore(SqliteMemoryBackend(path))
    s1.add_turn("alice", "user", "remember this")
    # Re-open a fresh store on the same file -> cross-session recall.
    s2 = MemoryStore(SqliteMemoryBackend(path))
    assert s2.history("alice", 10) == [{"role": "user", "content": "remember this"}]


def test_json_persists_across_reopen(tmp_path):
    path = tmp_path / "mem.json"
    MemoryStore(JsonMemoryBackend(path)).add_turn("bob", "assistant", "noted")
    assert MemoryStore(JsonMemoryBackend(path)).history("bob", 10) == [
        {"role": "assistant", "content": "noted"}
    ]


def test_make_memory_store_picks_backend(tmp_path):
    assert make_memory_store(_cfg(memory_store=None)) is None
    sql = make_memory_store(_cfg(memory_store=str(tmp_path / "m.db")))
    assert isinstance(sql.backend, SqliteMemoryBackend)
    js = make_memory_store(_cfg(memory_store=str(tmp_path / "m.json")))
    assert isinstance(js.backend, JsonMemoryBackend)


def _cfg(**kw):
    cfg = Config()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# ContextAggregator — provider-agnostic assembly
# ---------------------------------------------------------------------------


def test_aggregator_live_only_without_store():
    agg = ContextAggregator(store=None, max_turns=10)
    agg.add_user("hi")
    agg.add_assistant("hello")
    assert agg.assemble() == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_aggregator_prepends_persistent_recall(tmp_path):
    store = MemoryStore(SqliteMemoryBackend(tmp_path / "m.db"))
    store.add_turn("alice", "user", "yesterday")
    store.add_turn("alice", "assistant", "ok")
    agg = ContextAggregator(store=store, max_turns=10)
    agg.add_user("today")
    msgs = agg.assemble("alice")
    assert msgs[0]["content"] == "yesterday"
    assert msgs[-1] == {"role": "user", "content": "today"}


def test_aggregator_dedupes_seam(tmp_path):
    store = MemoryStore(SqliteMemoryBackend(tmp_path / "m.db"))
    # The live turns were already persisted this session (overlap at the seam).
    store.add_turn("alice", "user", "q1")
    store.add_turn("alice", "assistant", "a1")
    agg = ContextAggregator(store=store, max_turns=10)
    agg.live = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    msgs = agg.assemble("alice")
    # Not duplicated: q1/a1 appears exactly once.
    assert msgs == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]


def test_aggregator_budget_trims_oldest(tmp_path):
    store = MemoryStore(SqliteMemoryBackend(tmp_path / "m.db"))
    for i in range(20):
        store.add_turn("alice", "user", f"old{i}")
    agg = ContextAggregator(store=store, max_turns=2)  # budget = 4 messages
    agg.add_user("newest")
    msgs = agg.assemble("alice")
    assert len(msgs) == 4
    assert msgs[-1] == {"role": "user", "content": "newest"}


def test_aggregator_per_speaker(tmp_path):
    store = MemoryStore(SqliteMemoryBackend(tmp_path / "m.db"))
    store.add_turn("alice", "user", "alice ctx")
    store.add_turn("bob", "user", "bob ctx")
    agg = ContextAggregator(store=store, max_turns=10)
    assert agg.assemble("alice")[0]["content"] == "alice ctx"
    assert agg.assemble("bob")[0]["content"] == "bob ctx"


def test_aggregator_persist_and_reset_live(tmp_path):
    store = MemoryStore(SqliteMemoryBackend(tmp_path / "m.db"))
    agg = ContextAggregator(store=store, max_turns=10)
    agg.persist("alice", "q", "a")
    assert store.history("alice", 10) == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    agg.add_user("live")
    agg.reset_live()
    assert agg.live == []


# ---------------------------------------------------------------------------
# DialogueFlow — structured-dialogue hook
# ---------------------------------------------------------------------------


def test_dialogue_flow_default_is_noop():
    flow = DialogueFlow()
    assert flow.augment() == ""
    assert flow.advance("anything") == "default"


def test_dialogue_flow_transitions_and_augments():
    states = {
        "default": FlowState("default", transition=lambda t: "confirm" if "delete" in t else None),
        "confirm": FlowState("confirm", prompt="Ask the user to confirm the deletion."),
    }
    flow = DialogueFlow(states=states)
    assert flow.augment() == ""
    assert flow.advance("please delete the file") == "confirm"
    assert "confirm" in flow.augment().lower()
    flow.reset()
    assert flow.current == "default"


# ---------------------------------------------------------------------------
# Brain integration
# ---------------------------------------------------------------------------


def _make_brain(tmp_path, store_path=None):
    from my_stt_tts.brain import Brain

    cfg = Config(anthropic_api_key="x")
    cfg.tools_enabled = False
    if store_path is not None:
        cfg.memory_store = str(store_path)
    brain = Brain(cfg)
    # Replace the provider stream with an echo so no network is touched.
    brain._stream_anthropic = lambda model: iter(["reply about ", "the topic"])  # type: ignore[assignment]
    return brain


def test_brain_persists_per_speaker(tmp_path):
    path = tmp_path / "brain.db"
    brain = _make_brain(tmp_path, path)
    brain.set_speaker("alice")
    list(brain.stream("what's the weather"))
    # A fresh brain on the same store recalls alice's turn cross-session.
    brain2 = _make_brain(tmp_path, path)
    hist = brain2.context.store.history("alice", 10)
    assert hist[0] == {"role": "user", "content": "what's the weather"}
    assert hist[1]["content"] == "reply about the topic"


def test_brain_speaker_isolation(tmp_path):
    path = tmp_path / "brain.db"
    brain = _make_brain(tmp_path, path)
    brain.set_speaker("alice")
    list(brain.stream("alice question"))
    brain.reset()  # clears LIVE session only
    brain.set_speaker("bob")
    list(brain.stream("bob question"))
    store = brain.context.store
    assert [t["content"] for t in store.history("alice", 10)] == [
        "alice question",
        "reply about the topic",
    ]
    assert store.history("bob", 10)[0]["content"] == "bob question"


def test_brain_reset_keeps_persistent_memory(tmp_path):
    path = tmp_path / "brain.db"
    brain = _make_brain(tmp_path, path)
    brain.set_speaker("alice")
    list(brain.stream("persistent please"))
    brain.reset()
    assert brain.history == []  # live cleared
    assert brain.context.store.history("alice", 10)  # persistent intact


def test_brain_commit_spoken_amends_persistent_memory(tmp_path):
    path = tmp_path / "brain.db"
    brain = _make_brain(tmp_path, path)
    brain.set_speaker("alice")
    list(brain.stream("q"))
    brain.commit_spoken("reply about")  # barge-in truncated the voiced text
    last = brain.context.store.history("alice", 10)[-1]
    assert last == {"role": "assistant", "content": "reply about"}


def test_brain_history_alias_stays_valid_after_trim(tmp_path):
    brain = _make_brain(tmp_path)
    brain.cfg.max_history_turns = 1
    for _ in range(3):
        list(brain.stream("hello"))
    # history is still the SAME object as context.live after in-place trimming
    # (reset_live / _trim both mutate in place, so the alias never goes stale).
    assert brain.history is brain.context.live
    # max_history_turns=1 trims the live session to ~2 messages plus the trailing
    # assistant turn appended after the trim -> bounded, not unbounded growth.
    assert len(brain.history) <= 3


def test_brain_includes_recall_in_assembled_context(tmp_path):
    path = tmp_path / "brain.db"
    brain = _make_brain(tmp_path, path)
    brain.set_speaker("alice")
    list(brain.stream("first turn"))
    # New brain, same store: the assembled context for alice includes the recall.
    brain2 = _make_brain(tmp_path, path)
    brain2.set_speaker("alice")
    assembled = brain2._assembled()
    assert any("first turn" in m["content"] for m in assembled)


def test_speaker_identifier_ties_to_memory():
    from my_stt_tts.speaker_id import SpeakerIdentifier

    alice = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    bob = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    ident = SpeakerIdentifier({"alice": alice, "bob": bob}, threshold=0.5, margin=0.05)
    assert ident.identify(np.array([0.95, 0.05, 0.0], dtype=np.float32)) == "alice"
    assert speaker_key(ident.identify(np.array([0.1, 0.1, 0.9], dtype=np.float32))) == "_guest"
