# AGENTS_REFERENCE.md

Reference text moved verbatim out of `AGENTS.md` (its former "Claude bridge" entry
under "Where things live", tp#1486 B37). `AGENTS.md` keeps the core; read this before
you touch the Claude bridge.

## Rule ledger (D1a)

The moved entry is a code map; its safety-relevant facts are `core` and restated in one
line in `AGENTS.md` § "Where things live" (Claude bridge pointer).

| Rule (short) | Class | Where enforced / restated |
| :-- | :-- | :-- |
| Bridge actions pass `Authoriser` (two risk tiers, call-profile check) and `before_mutation`; ≤ 3 actions per sentence; confirmation via "yes, do it" | core | `bridge.py`; AGENTS.md |
| Mac operator actions are classed SAFE / CONFIRM / REFUSE by `mac_risk.py` (+ blocked/allow files) | core | `mac_risk.py`; AGENTS.md |

## Where things live: Claude bridge

- Claude bridge (tp#855, design in the local `PLAN_claude-bridge.md`; README section
  "Claude bridge"):
  - `bridge.py` → `BridgeController`, `Authoriser` (two risk tiers, 0.35 / ≥ 1.2 s call-profile
    check), `CapabilityStore` (≤ 3 actions per sentence), `PendingActions` ("yes, do it"),
    `before_mutation`; `bridge_problems.py` → `Problem` / sinks / briefing interfaces;
    `bridge_text.py` → normalising, number words, hosts, `derivable`, `spoken_name`
    (misheard names), `redact`; `bridge_wiring.py` → registers the components, each gated
    by its own flag, plus `OperatorNotices`; `turns.py` → `TurnSource` (VAD utterances,
    transcript ↔ utterance binding).
  - `claude_sessions.py` → session tools over `ccc … -j` (list / read / brief / send /
    answer, session in focus, explicit recipients).
  - `mac_control.py` → Mac fast-path tools + `mac-voice -D` doctor; `mac_sites.py` →
    Safari YouTube / Jellyfin adapters.
  - `mac_operator.py` → `do_on_mac` (`claude -p` lifecycle, confirmation watcher, queue);
    `mac_broker.py` → the operator's stdio MCP broker; `mac_risk.py` → risk table
    (SAFE / CONFIRM / REFUSE) + blocked/allow files; schema in `schemas/operator_result.json`.
  - `attention.py` → attention inbox (`attention.db`), `Monitor` (`ccc events`), voice-on
    briefing, `acknowledge_problem`, `mac-voice -K`.
  - `scripts/eleven_agent_config.py` → ElevenLabs agent tools + prompt section (backup,
    apply, rollback); `scripts/enroll_call.py` → call-domain voice profile;
    `scripts/acceptance_setup.sh` + `scripts/ACCEPTANCE_RUNBOOK.md` → attended acceptance
    fixtures and steps; `scripts/spikes/` → Phase-0 spike scripts.
