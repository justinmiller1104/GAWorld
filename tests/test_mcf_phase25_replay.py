"""Phase 2.5 RED tests — LLM replay wrapper (TASK-3590).

The module ``benchmark/mcf/replay.py`` does not exist yet: these tests pin
the contract ``install_llm_replay`` must satisfy BEFORE it is written,
mirroring the proven canary seam (``benchmark.mcf.launch.install_canary``:
module-attribute rebind of ``gaworld.llm.providers.call_llm``, restorable).

Contracts pinned:

  1. ``install_llm_replay(recorded_responses, stop_predicate)`` rebinds
     ``providers.call_llm`` canary-style and returns ``(replay, restore)``.
  2. Cache HIT: an identical request — same (task, agent_id, system,
     temperature, prompt) — returns the recorded response WITHOUT calling
     the live base llm.
  3. Key sensitivity: any differing key component (task, agent_id,
     system, temperature, prompt) must NOT hit another call's record.
  4. Cache MISS pre-stop is LOUD: raises (never a silent fall-through to
     live sampling that would silently diverge the pre-trigger history).
  5. Stop predicate: once the predicate says the replay window is over
     (e.g. past the intervention trigger tick), calls fall through to the
     LIVE base llm and fresh sampling resumes.
  6. Restorable: ``restore()`` puts the original ``providers.call_llm``
     back — no global state leaks (same guarantee as install_canary).

Everything uses an in-memory stub base llm. No LLM, no network.
"""

import pytest

import gaworld.llm.providers as providers
from benchmark.mcf.replay import ReplayCacheMiss, install_llm_replay


def _stub_llm(response="LIVE"):
    calls = []

    def llm(prompt, task=None, agent_id=None, system=None, temperature=None, **kwargs):
        calls.append(
            {
                "prompt": prompt,
                "task": task,
                "agent_id": agent_id,
                "system": system,
                "temperature": temperature,
            }
        )
        return response

    llm.calls = calls
    return llm


# ---------------------------------------------------------------------------
# Contract 1 — canary-style installation and restorability
# ---------------------------------------------------------------------------


def test_install_rebinds_module_attribute_and_restores():
    original = providers.call_llm
    _replay, restore = install_llm_replay({})
    try:
        assert providers.call_llm is not original, "install must rebind providers.call_llm (canary seam)"
    finally:
        restore()
    assert providers.call_llm is original, "restore() must put the original back"


# ---------------------------------------------------------------------------
# Contract 2 — cache hit on identical request
# ---------------------------------------------------------------------------


def test_cache_hit_returns_recorded_response_without_live_call():
    base = _stub_llm("LIVE")
    key = ("act", 4, "sys", 0.7, "prompt-1")
    recorded = {key: "RECORDED"}
    replay, restore = install_llm_replay(recorded, base_llm=base)
    try:
        result = replay(key[4], task=key[0], agent_id=key[1], system=key[2], temperature=key[3])
        assert result == "RECORDED"
        assert base.calls == [], "a cache hit must never reach the live llm"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 3 — every key component participates in the lookup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value", [("task", "memory"), ("agent_id", 5), ("system", "other"), ("temperature", 0.0)]
)
def test_differing_key_component_is_not_a_hit(field, value):
    """A request differing in ONE component must not be answered from
    another call's record (it is a miss → loud, contract 4)."""
    key = ("act", 4, "sys", 0.7, "prompt-1")
    recorded = {key: "RECORDED"}
    kwargs = {"task": key[0], "agent_id": key[1], "system": key[2], "temperature": key[3]}
    kwargs[field] = value
    replay, restore = install_llm_replay(recorded, base_llm=_stub_llm("LIVE"))
    try:
        with pytest.raises(ReplayCacheMiss):
            replay(key[4], **kwargs)
    finally:
        restore()


def test_differing_prompt_is_not_a_hit():
    key = ("act", 4, "sys", 0.7, "prompt-1")
    recorded = {key: "RECORDED"}
    replay, restore = install_llm_replay(recorded, base_llm=_stub_llm("LIVE"))
    try:
        with pytest.raises(ReplayCacheMiss):
            replay("a DIFFERENT prompt", task=key[0], agent_id=key[1], system=key[2], temperature=key[3])
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 4 — cache miss pre-stop is loud, never a silent fall-through
# ---------------------------------------------------------------------------


def test_cache_miss_raises_loudly_before_stop():
    replay, restore = install_llm_replay({}, base_llm=_stub_llm("LIVE"))
    try:
        with pytest.raises(ReplayCacheMiss):
            replay("unrecorded prompt", task="act", agent_id=4, system="sys", temperature=0.7)
    finally:
        restore()


def test_rebound_module_attribute_also_misses_loudly():
    """After installation the MODULE attribute itself must enforce the
    miss (that is the seam every kernel path dispatches through)."""
    _original = providers.call_llm
    base = _stub_llm("LIVE")
    _replay, restore = install_llm_replay({}, base_llm=base)
    try:
        with pytest.raises(ReplayCacheMiss):
            providers.call_llm("unrecorded prompt", task="act", agent_id=4, system="sys", temperature=0.7)
        assert base.calls == [], "a pre-stop miss must not silently fall through to live sampling"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 5 — stop predicate: past the trigger, live sampling resumes
# ---------------------------------------------------------------------------


def test_past_stop_predicate_falls_through_to_live_llm():
    """The replay window ends at the intervention trigger: calls after the
    stop predicate fires bypass the cache and hit the LIVE base llm."""
    base = _stub_llm("LIVE")
    key = ("act", 4, "sys", 0.7, "prompt-1")
    recorded = {key: "RECORDED"}

    def stop_predicate(*args, **kwargs):
        # e.g. "past the trigger tick": everything at/after 10:00 stops replay
        return kwargs.get("trigger_tick") is not None

    replay, restore = install_llm_replay(recorded, base_llm=base, stop_predicate=stop_predicate)
    try:
        # pre-stop: recorded
        assert replay(key[4], task=key[0], agent_id=key[1], system=key[2], temperature=key[3]) == "RECORDED"
        assert base.calls == []
        # past the trigger: live, even for an identical (recorded) request
        result = providers.call_llm(
            key[4], task=key[0], agent_id=key[1], system=key[2], temperature=key[3], trigger_tick=600
        )
        assert result == "LIVE"
        assert len(base.calls) == 1
    finally:
        restore()


def test_default_no_stop_predicate_replays_forever_is_allowed_but_predicates_are_supported():
    """Without a stop predicate every call replays (documented baseline
    behaviour); predicates are the mechanism that ends the window."""
    base = _stub_llm("LIVE")
    key = ("act", 4, None, None, "p")
    replay, restore = install_llm_replay({key: "R"}, base_llm=base)
    try:
        assert replay("p", task="act", agent_id=4) == "R"
        assert replay("p", task="act", agent_id=4) == "R"
        assert base.calls == []
    finally:
        restore()
