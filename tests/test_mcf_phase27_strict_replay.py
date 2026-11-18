"""Phase 2.7 preflight RED tests — STRICT sequence replay + sim-time stop
boundary (TASK-3611, spec Parts A & C).

Phase 2.6 finding: ``SequenceReplay`` (``benchmark/mcf/replay.py``, the
prompt-comparison branch at ~line 94) TOLERATES a completely unexpected
pre-trigger prompt — the comment reads "the request shape still identifies the
call". For the MCF protocol that is a silent-divergence hole: any call whose
prompt was never recorded slips through and is answered from the WRONG event's
recorded response (or advances the cursor past an event that never happened),
corrupting the "identical pre-trigger history" guarantee the fork protocol
rests on.

Contracts pinned here (all must raise ``ReplayMismatch`` before the stop
boundary, naming the event index and a kind):

  A1. Unexpected call — a brand-new prompt that matches NO recorded event —
      raises ReplayMismatch (today it is tolerated: RED).
  A2. Duplicated call — the same prompt replayed twice while events remain —
      raises ReplayMismatch (today it silently consumes the NEXT event: RED).
  A3. Reordered call raises ReplayMismatch with kind "reordered" at the
      current index (already strict — regression pin, stays green).
  A4. Missing trailing calls — the window closing (stop predicate firing)
      while recorded events were never consumed — raises ReplayMismatch
      naming the first unconsumed index, kind "missing" (today it is
      silent: RED).
  A5. Ordered replay of legitimately repeated identical prompts keeps
      working (Phase 2.6 sequence contract — regression pin, stays green).

Part C — the stop boundary is AUTHORITATIVE SIM TIME:

  C1. ``sim_time_stop_predicate(day, time)`` — a factory that evaluates the
      stop predicate against the sim's own clock carried in the prompt's
      clock line (``第 {day} 天 ... 现在是 {time}``) — exists. With a
      day-1-12:00 trigger and recorded events spanning 11:00–13:00, replay
      consumes EXACTLY the pre-12:00 events and the first 12:00-or-later
      call goes live (RED: the factory does not exist).
  C2. Boundary exactness: a 11:59 call replays; the 12:00 call goes live —
      the window can end neither before nor after the trigger (RED).
  C3. A PRE-trigger event arriving AFTER the stop fired is a mismatch —
      ReplayMismatch (kind "late"), never a silent live fall-through
      (RED: latched calls bypass all validation today).

Deterministic in-memory stubs; no LLM, no network, no real simulation.
"""

from __future__ import annotations

import pytest

from benchmark.mcf.replay import ReplayMismatch, install_sequence_replay


def _event(prompt, response, *, task="t", agent_id=1, system="s", temperature=0.7):
    return {
        "task": task,
        "agent_id": agent_id,
        "system": system,
        "temperature": temperature,
        "prompt": prompt,
        "response": response,
    }


def _call(replay, prompt, **over):
    kwargs = {"task": "t", "agent_id": 1, "system": "s", "temperature": 0.7}
    kwargs.update(over)
    return replay(prompt, **kwargs)


# ---------------------------------------------------------------------------
# Part A — every pre-trigger mismatch raises, including a brand-new prompt
# ---------------------------------------------------------------------------


def test_unexpected_new_prompt_pre_stop_raises_replay_mismatch():
    """A1. Recorded event has prompt 'A'; the sim issues a prompt that was
    never recorded. The request shape does NOT identify the call — replaying
    event 0's response for a different request silently diverges the shared
    history. Must raise ReplayMismatch naming index 0."""
    replay, restore = install_sequence_replay([_event("A", "RA")])
    try:
        with pytest.raises(ReplayMismatch) as exc:
            _call(replay, "从未记录过的全新提示")
        msg = str(exc.value)
        assert "0" in msg, f"mismatch message must name the index: {msg!r}"
        assert replay.first_mismatch is not None
        assert replay.first_mismatch["kind"] == "unexpected"
        assert replay.first_mismatch["index"] == 0
    finally:
        restore()


def test_duplicated_call_mid_sequence_raises_replay_mismatch():
    """A2. Recorded A then B. The sim re-issues A (a retry, a duplicated
    dispatch). Today the duplicate silently consumes event B's response —
    the wrong answer for the wrong call. Must raise ReplayMismatch at
    index 1."""
    replay, restore = install_sequence_replay([_event("A", "RA"), _event("B", "RB")])
    try:
        assert _call(replay, "A") == "RA"
        with pytest.raises(ReplayMismatch) as exc:
            _call(replay, "A")  # duplicated: B was expected here
        msg = str(exc.value)
        assert "1" in msg, f"duplicate must be located at index 1: {msg!r}"
        assert replay.first_mismatch["index"] == 1
    finally:
        restore()


def test_reordered_call_reports_kind_reordered_at_current_index():
    """A3 (regression pin — already strict in Phase 2.6)."""
    replay, restore = install_sequence_replay([_event("A", "RA"), _event("B", "RB")])
    try:
        with pytest.raises(ReplayMismatch) as exc:
            _call(replay, "B")
        assert "0" in str(exc.value)
        assert replay.first_mismatch["kind"] == "reordered"
    finally:
        restore()


def test_window_closing_with_unconsumed_events_raises_missing():
    """A4. Missing trailing calls: the stop predicate fires at the trigger
    while recorded pre-trigger events were never replayed (the recording
    overshot the window). The protocol must not end silently with an
    unverified history — ReplayMismatch naming the first unconsumed index,
    kind 'missing'."""
    replay, restore = install_sequence_replay(
        [_event("p1", "R1"), _event("p2", "R2"), _event("p3", "R3")],
        base_llm=lambda prompt, *a, **k: "LIVE",
        stop_predicate=lambda prompt, *a, **k: "触发" in prompt,
    )
    try:
        assert _call(replay, "p1") == "R1"
        with pytest.raises(ReplayMismatch) as exc:
            _call(replay, "触发时刻的感知")  # stop fires with p2/p3 unconsumed
        msg = str(exc.value)
        assert "1" in msg, f"first unconsumed index must be named: {msg!r}"
        assert replay.first_mismatch is not None
        assert replay.first_mismatch["kind"] == "missing"
    finally:
        restore()


def test_legitimately_repeated_identical_prompts_still_replay_in_order():
    """A5 (regression pin — the Phase 2.6 sequence contract stays intact
    under strictness)."""
    replay, restore = install_sequence_replay([_event("P", "R1"), _event("P", "R2")])
    try:
        assert _call(replay, "P") == "R1"
        assert _call(replay, "P") == "R2"
        assert replay.coverage() == {"hits": 2, "total": 2}
    finally:
        restore()


# ---------------------------------------------------------------------------
# Part C — the stop boundary is authoritative simulation time
# ---------------------------------------------------------------------------


def _clock_prompt(day, time_str, tail="的感知"):
    """The sim's own clock line, as the perception stages stamp it into
    every request the replay sees (phase-2.6 marker convention)."""
    return f"第 {day} 天 现在是 {time_str} {tail}"


def _sim_time_stop(day, time_str):
    from benchmark.mcf.replay import sim_time_stop_predicate

    return sim_time_stop_predicate(day=day, time=time_str)


def test_replay_consumes_exactly_the_pre_trigger_events_for_day1_12_00():
    """C1. Day-1-12:00 trigger, recorded events spanning 11:00–13:00:
    replay consumes exactly the 11:00 and 11:30 events; the first
    12:00-or-later call (12:00 itself) goes live; 12:30/13:00 stay live."""
    events = [
        _event(_clock_prompt(1, "11:00"), "R1100"),
        _event(_clock_prompt(1, "11:30"), "R1130"),
        _event(_clock_prompt(1, "12:00"), "STALE1200"),
        _event(_clock_prompt(1, "12:30"), "STALE1230"),
        _event(_clock_prompt(1, "13:00"), "STALE1300"),
    ]
    base_calls = []

    def base(prompt, *a, **k):
        base_calls.append(prompt)
        return f"LIVE@{prompt}"

    replay, restore = install_sequence_replay(
        events, base_llm=base, stop_predicate=_sim_time_stop(1, "12:00")
    )
    try:
        assert _call(replay, _clock_prompt(1, "11:00")) == "R1100"
        assert _call(replay, _clock_prompt(1, "11:30")) == "R1130"
        assert _call(replay, _clock_prompt(1, "12:00")) == "LIVE@" + _clock_prompt(1, "12:00")
        assert _call(replay, _clock_prompt(1, "12:30")).startswith("LIVE@")
        assert _call(replay, _clock_prompt(1, "13:00")).startswith("LIVE@")
        assert base_calls == [_clock_prompt(1, "12:00"), _clock_prompt(1, "12:30"), _clock_prompt(1, "13:00")]
        assert replay.coverage() == {"hits": 2, "total": 5}, (
            "exactly the pre-12:00 events consumed; the 12:00-or-later events "
            "must never be replayed past the trigger"
        )
    finally:
        restore()


def test_stop_boundary_is_exact_neither_early_nor_late():
    """C2. 11:59 still replays; 12:00 goes live — the window cannot end
    before the trigger (silently replaying past it) or after it (silently
    sampling live before it)."""
    events = [
        _event(_clock_prompt(1, "11:59"), "R1159"),
        _event(_clock_prompt(1, "12:00"), "STALE"),
    ]
    live = []

    def base(prompt, *a, **k):
        live.append(prompt)
        return "LIVE"

    replay, restore = install_sequence_replay(
        events, base_llm=base, stop_predicate=_sim_time_stop(1, "12:00")
    )
    try:
        assert _call(replay, _clock_prompt(1, "11:59")) == "R1159", "11:59 is pre-trigger: must replay"
        assert _call(replay, _clock_prompt(1, "12:00")) == "LIVE", "12:00 IS the trigger: must go live"
        assert live == [_clock_prompt(1, "12:00")]
    finally:
        restore()


def test_pre_trigger_event_arriving_after_stop_fired_is_a_mismatch():
    """C3. After the window latched at 12:00, a call stamped 11:30 (a
    late-arriving pre-trigger event, e.g. an out-of-order dispatch) must
    raise ReplayMismatch — a silent live fall-through would sample a fresh
    answer for an event the protocol promised to replay."""
    events = [
        _event(_clock_prompt(1, "11:30"), "R1130"),
        _event(_clock_prompt(1, "12:00"), "STALE"),
    ]
    replay, restore = install_sequence_replay(
        events, base_llm=lambda p, *a, **k: "LIVE", stop_predicate=_sim_time_stop(1, "12:00")
    )
    try:
        assert _call(replay, _clock_prompt(1, "12:00")) == "LIVE"  # boundary: window latches
        with pytest.raises(ReplayMismatch) as exc:
            _call(replay, _clock_prompt(1, "11:30"))  # pre-trigger event, post-stop
        assert replay.first_mismatch is not None
        assert replay.first_mismatch["kind"] == "late"
        assert "11:30" in str(exc.value) or "index" in str(exc.value), str(exc.value)
    finally:
        restore()
