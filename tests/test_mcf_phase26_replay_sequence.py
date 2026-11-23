"""Phase 2.6 RED tests — sequence-aware ordered LLM replay (TASK-3597).

Phase 2.5 recorded 1,558 calls but only 1,099 unique keys: the dict replay
(``benchmark/mcf/replay.py`` ``install_llm_replay``) keys on the full request
tuple, so a repeated identical prompt overwrote its earlier record — two calls
with the SAME prompt but DIFFERENT responses cannot both be replayed
faithfully. These tests pin the ordered, sequence-aware replay contract
BEFORE it is written, on top of the proven canary seam.

Contracts pinned (module ``benchmark.mcf.replay``):

  1. ``install_sequence_replay(recorded_events, base_llm=None,
     stop_predicate=None)`` → ``(replay, restore)``; ``recorded_events`` is an
     ORDERED list of ``{"task", "agent_id", "system", "temperature",
     "prompt", "response"}`` dicts — every recorded call is an event, repeats
     included.
  2. Ordered responses: each replayed call returns ITS OWN event's response,
     in order — two identical prompts recorded with different responses
     replay R1 then R2 (the dict replay loses R2).
  3. Request validation: each incoming request is checked against its
     expected recorded counterpart (event at the current position); a
     task/agent/system/prompt mismatch raises ``ReplayMismatch`` naming the
     exact index/location.
  4. Unexpected calls (more calls than recorded events, pre-stop) raise
     ``ReplayMismatch`` too.
  5. Reordered calls (the sequence's later event arrives first) raise
     ``ReplayMismatch`` at the current index.
  6. Coverage report: ``replay.coverage()`` → ``{"hits": <int>,
     "total": <int recorded events>}``; ``replay.first_mismatch`` → ``None``
     or ``{"index", "kind", "expected", "got"}``. A window that ends before
     every event was replayed shows hits < total (missing calls visible).
  7. Stop predicate: past the intervention boundary tick the call goes to the
     live base llm, NO event is consumed, coverage unchanged.
  8. Restorable + canary composition: install over a canary-style wrapper —
     replay hits never touch the wrapper, stop-passed calls fall THROUGH to
     it, and restore() unwinds one layer (wrapper's own restore brings the
     original back). Response type/content preserved on hits.
  9. All-arms protocol: ``fork_arms_from_checkpoint`` must install replay in
     the BASELINE and NEUTRAL arms too — every arm carrying recorded
     responses replays the same recorded sequence until its trigger, not
     only the intervention arm (today the baseline arm is excluded:
     ``run_replay = arm_name != "baseline"``).

Deterministic in-memory stubs; no LLM, no network, no real simulation.
"""

from pathlib import Path

import pytest

import gaworld.llm.providers as providers
from benchmark.mcf.experiment import fork_arms_from_checkpoint
from benchmark.mcf.replay import ReplayMismatch, install_sequence_replay

TRIGGER = {"day": 2, "time": "10:00"}


def _event(prompt, response, *, task="t", agent_id=1, system="s", temperature=0.7):
    return {
        "task": task,
        "agent_id": agent_id,
        "system": system,
        "temperature": temperature,
        "prompt": prompt,
        "response": response,
    }


def _stub_llm(response="LIVE"):
    calls = []

    def llm(prompt, *args, **kwargs):
        calls.append({"prompt": prompt, "args": args, "kwargs": kwargs})
        return response

    llm.calls = calls
    return llm


# ---------------------------------------------------------------------------
# Contract 1+2 — ordered events, repeated prompts keep their own responses
# ---------------------------------------------------------------------------


def test_repeated_identical_prompts_replay_their_own_responses_in_order():
    """The Phase 2.5 defect: 1,558 calls / 1,099 unique keys. The same prompt
    twice with different responses must replay R1 then R2, in call order."""
    replay, restore = install_sequence_replay([_event("同样的提示", "R1"), _event("同样的提示", "R2")])
    try:
        assert replay("同样的提示", task="t", agent_id=1, system="s", temperature=0.7) == "R1"
        assert replay("同样的提示", task="t", agent_id=1, system="s", temperature=0.7) == "R2"
    finally:
        restore()


def test_events_are_kept_in_recorded_order_even_when_responses_differ_by_type():
    """Response type/content preserved exactly as recorded (dict response)."""
    replay, restore = install_sequence_replay(
        [_event("p1", {"plan": ["去商店"], "why": "crowded"}), _event("p2", "plain text")]
    )
    try:
        first = replay("p1", task="t", agent_id=1, system="s", temperature=0.7)
        second = replay("p2", task="t", agent_id=1, system="s", temperature=0.7)
        assert first == {"plan": ["去商店"], "why": "crowded"}
        assert second == "plain text"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 3 — request validated against the expected recorded counterpart
# ---------------------------------------------------------------------------


def test_request_mismatch_raises_replay_mismatch_with_exact_index():
    """Event 0 expects agent_id=1; a call with agent_id=2 must fail loudly,
    naming the exact location (index 0) and the differing field."""
    replay, restore = install_sequence_replay([_event("p", "R", agent_id=1)])
    try:
        with pytest.raises(ReplayMismatch) as exc:
            replay("p", task="t", agent_id=2, system="s", temperature=0.7)
        msg = str(exc.value)
        assert "0" in msg, f"mismatch message must name the index: {msg!r}"
        assert "agent_id" in msg, f"mismatch message must name the field: {msg!r}"
    finally:
        restore()


def test_task_mismatch_raises_replay_mismatch_at_current_index():
    events = [_event("p1", "R1"), _event("p2", "R2")]
    replay, restore = install_sequence_replay(events)
    try:
        replay("p1", task="t", agent_id=1, system="s", temperature=0.7)  # ok, advances to index 1
        with pytest.raises(ReplayMismatch) as exc:
            replay("p2", task="OTHER", agent_id=1, system="s", temperature=0.7)
        assert "1" in str(exc.value), "second event mismatch must be located at index 1"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 4 — unexpected (extra) calls pre-stop are loud
# ---------------------------------------------------------------------------


def test_extra_call_beyond_the_recorded_sequence_raises():
    replay, restore = install_sequence_replay([_event("p", "R")])
    try:
        replay("p", task="t", agent_id=1, system="s", temperature=0.7)  # the only recorded call
        with pytest.raises(ReplayMismatch):
            replay("p", task="t", agent_id=1, system="s", temperature=0.7)  # duplicated/unexpected
    finally:
        restore()


def test_duplicated_call_reported_as_unexpected_with_location():
    replay, restore = install_sequence_replay([_event("p", "R")])
    try:
        replay("p", task="t", agent_id=1, system="s", temperature=0.7)
        with pytest.raises(ReplayMismatch):
            replay("p", task="t", agent_id=1, system="s", temperature=0.7)
        report = replay.coverage()
        assert report["hits"] == 1, "the failing call must not count as a hit"
        assert report["total"] == 1
        mm = replay.first_mismatch
        assert mm is not None, "first_mismatch must record the unexpected call"
        assert mm["index"] == 1, f"unexpected call located at index 1: {mm!r}"
        assert mm["kind"] == "unexpected"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 5 — reordered calls are detected at the current index
# ---------------------------------------------------------------------------


def test_reordered_call_raises_at_the_current_index():
    """Recorded A then B; calling with B's prompt first is a mismatch at
    index 0, even though B exists later in the recording."""
    replay, restore = install_sequence_replay([_event("A", "RA"), _event("B", "RB")])
    try:
        with pytest.raises(ReplayMismatch) as exc:
            replay("B", task="t", agent_id=1, system="s", temperature=0.7)
        assert "0" in str(exc.value)
        assert replay.first_mismatch is not None
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 6 — exact coverage report; missing calls visible
# ---------------------------------------------------------------------------


def test_coverage_reports_hits_over_total_recorded_events():
    replay, restore = install_sequence_replay([_event("p1", "R1"), _event("p2", "R2"), _event("p3", "R3")])
    try:
        for p in ("p1", "p2", "p3"):
            replay(p, task="t", agent_id=1, system="s", temperature=0.7)
        assert replay.coverage() == {"hits": 3, "total": 3}
        assert replay.first_mismatch is None
    finally:
        restore()


def test_window_ending_early_shows_missing_calls_in_coverage():
    """Stop predicate fires after the first call: the remaining recorded
    events were never replayed — coverage must expose hits < total."""
    replay, restore = install_sequence_replay(
        [_event("p1", "R1"), _event("p2", "R2")],
        stop_predicate=lambda prompt, *a, **k: True,  # replay window already over
    )
    try:
        live = replay("fresh", task="t", agent_id=1, system="s", temperature=0.7)
        assert live == "LIVE"
        report = replay.coverage()
        assert report == {"hits": 0, "total": 2}, report
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 7 — stop predicate ends replay exactly at the boundary
# ---------------------------------------------------------------------------


def test_stop_predicate_switches_to_live_and_consumes_nothing():
    events = [_event("p1", "R1"), _event("p2", "R2")]
    base = _stub_llm("LIVE-AT-TRIGGER")
    # the Phase 2.5 latching content marker: replay ends at the perception
    # prompt that carries the trigger tick's clock line
    replay, restore = install_sequence_replay(
        events, base_llm=base, stop_predicate=lambda prompt, *a, **k: "现在是 10:00" in prompt
    )
    try:
        assert replay("pre-trigger", task="t", agent_id=1, system="s", temperature=0.7) == "R1"
        assert (
            replay("现在是 10:00 的感知", task="t", agent_id=1, system="s", temperature=0.7)
            == "LIVE-AT-TRIGGER"
        )
        assert len(base.calls) == 1, "stop-passed call must reach the live base llm"
        assert replay.coverage() == {"hits": 1, "total": 2}, "stop-passed call consumes no event"
    finally:
        restore()


def test_stop_predicate_latches_for_subsequent_calls():
    """Once past the boundary every call is live — the window does not
    re-open for a prompt that happens to match an unused event."""
    events = [_event("p1", "R1"), _event("past", "STALE")]
    base = _stub_llm("LIVE")
    state = {"open": True}

    def stop(prompt, *a, **k):
        if state["open"]:
            state["open"] = False
            return False  # this one still replays
        return True

    replay, restore = install_sequence_replay(events, base_llm=base, stop_predicate=stop)
    try:
        assert replay("p1", task="t", agent_id=1, system="s", temperature=0.7) == "R1"
        assert replay("p1", task="t", agent_id=1, system="s", temperature=0.7) == "LIVE"
        assert replay("p1", task="t", agent_id=1, system="s", temperature=0.7) == "LIVE"
        assert replay.coverage()["hits"] == 1
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 8 — restore + composition with the canary seam
# ---------------------------------------------------------------------------


def test_restore_puts_the_original_provider_back():
    original = providers.call_llm
    replay, restore = install_sequence_replay([_event("p", "R")])
    try:
        assert providers.call_llm is replay, "install must rebind providers.call_llm"
    finally:
        restore()
    assert providers.call_llm is original


def test_composes_over_a_canary_style_wrapper():
    """canary installed first (module-attribute rebind), replay second:
    hits bypass the canary, stop-passed calls fall THROUGH to it, and the
    restores unwind in order back to the original."""
    original = providers.call_llm
    canary_calls = []

    def canary(prompt, *args, **kwargs):
        canary_calls.append(prompt)
        return f"CANARY:{prompt}"

    providers.call_llm = canary  # what install_canary does
    try:
        replay, restore_replay = install_sequence_replay(
            [_event("p1", "R1")],
            stop_predicate=lambda prompt, *a, **k: "现在是 10:00" in prompt,
        )
        try:
            assert replay("p1", task="t", agent_id=1, system="s", temperature=0.7) == "R1"
            assert canary_calls == [], "replay hit must not reach the canary wrapper"
            assert (
                replay("现在是 10:00 的感知", task="t", agent_id=1, system="s", temperature=0.7)
                == "CANARY:现在是 10:00 的感知"
            ), "stop-passed call must fall through to the canary wrapper"
        finally:
            restore_replay()
        assert providers.call_llm is canary, "replay restore unwinds one layer"
    finally:
        providers.call_llm = original


# ---------------------------------------------------------------------------
# Contract 9 — all-arms replay protocol in fork_arms_from_checkpoint
# ---------------------------------------------------------------------------


def _fake_checkpoint(tmp_path: Path) -> Path:
    ckpt = tmp_path / "checkpoints" / "day_0001"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "manifest.json").write_text('{"day": 1}', encoding="utf-8")
    (ckpt / "news_cache.json").write_text('{"headlines": ["frozen"]}', encoding="utf-8")
    return ckpt


def _recording_harness(log: list, tag: str):
    def harness(config, run_dir):
        log.append({"tag": tag, "config": dict(config), "run_dir": str(run_dir)})
        (Path(run_dir) / "agent_1.jsonl").write_text("", encoding="utf-8")
        return {"ok": True}

    return harness


def test_baseline_arm_replays_too(tmp_path, monkeypatch):
    """All-arms protocol: an arm named ``baseline`` carrying recorded
    responses must ALSO run under the replay wrapper until its trigger —
    today only non-baseline arms do (experiment.py: ``arm_name != "baseline"``)."""
    log = []
    ckpt = _fake_checkpoint(tmp_path)
    stop = {"armed": True}

    def stop_predicate(prompt, *a, **k):
        return not stop["armed"]

    arms = {
        "baseline": {
            "config": {},
            "harness": _recording_harness(log, "baseline"),
            "trigger": dict(TRIGGER),
            "recorded_responses": {("t", 1, "s", 0.7, "p"): "R"},
            "stop_predicate": stop_predicate,
        },
        "neutral_control": {
            "config": {},
            "harness": _recording_harness(log, "neutral_control"),
            "trigger": dict(TRIGGER),
            "recorded_responses": {("t", 1, "s", 0.7, "p"): "R"},
            "stop_predicate": stop_predicate,
        },
    }
    result = fork_arms_from_checkpoint(ckpt, arms, run_root=tmp_path / "fork")
    for arm_name in ("baseline", "neutral_control"):
        assert "replay" in result[arm_name], (
            f"arm {arm_name!r} must report a replay window (all-arms replay protocol)"
        )
        assert result[arm_name]["replay"]["stop"] == TRIGGER
    manifest = (tmp_path / "fork" / "baseline" / "run_manifest.json").read_text(encoding="utf-8")
    assert "replay_window" in manifest, "baseline manifest must carry fork provenance incl. replay window"
