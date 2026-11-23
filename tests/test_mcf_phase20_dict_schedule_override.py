"""Phase 2.0 RED tests — dict-format schedules in apply_schedule_override.

Defect (RESULT-3502): the pipeline's real call site (gaworld/sim/_stages.py:445)
passes ``scope.schedule_map[agent_id]`` whose entries are **dicts**
(``{"time": t, "activity": a}``, see _stages.py:419/509) — not tuple lists.
``apply_schedule_override`` (and ``get_activity_for_time``) iterate
``for t, a in schedule``, so on a dict schedule they unpack each entry's
**keys** as ``(t, a)``: the sleep slot is never seen as sleep
(``is_sleep_activity(current)`` is always False), the duration_minutes
expiry branch never fires, and the original slots are destroyed — the
override owns the whole night. The existing
``test_mcf_phase20_sleep_override_expiry`` tests pass because they only pin
tuple schedules.

Contracts pinned here (mechanism only):
  1. With a dict-format schedule containing a sleep slot, a duration-bounded
     override expires: sleep is re-asserted at time+N and the original slot
     survives.
  2. Sleep detection sees the dict entry's ACTIVITY value, not its keys —
     ``get_activity_for_time`` / ``is_sleep_activity`` work on dict schedules
     with 睡前/睡觉 labels.
  3. Non-sleep dict slots keep persistent semantics; the legacy tuple form
     and the legacy 2-arg call are unchanged (covered by the existing suite).

All tests are deterministic unit-level tests on gaworld/sim/_schedule.py; no
LLM, no network.
"""

from gaworld.sim._schedule import apply_schedule_override, get_activity_for_time, is_sleep_activity

# The dict shape the pipeline actually stores in scope.schedule_map
# (gaworld/sim/_stages.py:419 and :509 build it this way).
NIGHT_SCHEDULE_DICT = [
    {"time": "00:00", "activity": "睡前"},
    {"time": "08:00", "activity": "上午工作"},
    {"time": "12:30", "activity": "午餐"},
    {"time": "14:00", "activity": "下午工作"},
    {"time": "19:00", "activity": "个人时间"},
    {"time": "23:30", "activity": "睡觉"},
]

INTERRUPT_TIME = "00:05"
INTERRUPT_ACTIVITY = "找地方避雨"
INTERRUPT_DURATION_MINUTES = 20


def _dict_sched(entries):
    return [{"time": t, "activity": a} for t, a in entries]


# ---------------------------------------------------------------------------
# Contract 1 — dict-format sleep override is duration-bounded and expires.
# ---------------------------------------------------------------------------


def test_dict_schedule_get_activity_for_time_reads_activity_value():
    """Contract 2: sleep detection works on dict-format sleep slots — the
    activity VALUE is seen, not the dict keys."""
    current = get_activity_for_time(NIGHT_SCHEDULE_DICT, "00:02")
    assert current == "睡前", f"00:02 on a dict schedule must resolve to the sleep slot, got {current!r}"
    assert is_sleep_activity(current), "the resolved 睡前 slot must be detected as sleep"


def test_dict_schedule_sleep_override_expires_and_sleep_re_asserts():
    """Contract 1a: with a dict-format schedule, a duration-bounded interrupt on
    a sleep slot is active only during its window — sleep re-asserts at expiry."""
    sched = apply_schedule_override(
        NIGHT_SCHEDULE_DICT, INTERRUPT_TIME, INTERRUPT_ACTIVITY, duration_minutes=INTERRUPT_DURATION_MINUTES
    )
    # 00:05 up to (but excluding) expiry is the interrupt…
    assert get_activity_for_time(sched, INTERRUPT_TIME) == INTERRUPT_ACTIVITY
    assert get_activity_for_time(sched, "00:24") == INTERRUPT_ACTIVITY
    # …and from expiry onward sleep is re-asserted.
    assert is_sleep_activity(get_activity_for_time(sched, "00:25")), (
        "sleep must resume at the override's expiry (00:05 + 20 min) even when the schedule is dict-format"
    )
    assert is_sleep_activity(get_activity_for_time(sched, "06:59")), (
        "sleep must hold for the remaining night after expiry on a dict schedule"
    )


def test_dict_schedule_original_sleep_slots_survive_override():
    """Contract 1b: the original sleep slot must not be destroyed by the
    override — pre-interrupt minutes stay sleep and the original slots are
    still present in the returned schedule."""
    sched = apply_schedule_override(
        NIGHT_SCHEDULE_DICT, INTERRUPT_TIME, INTERRUPT_ACTIVITY, duration_minutes=INTERRUPT_DURATION_MINUTES
    )
    assert is_sleep_activity(get_activity_for_time(sched, "00:00")), "pre-interrupt minutes must stay sleep"
    times = (
        {entry["time"] for entry in sched}
        if all(isinstance(e, dict) for e in sched)
        else {t for t, _ in sched}
    )
    for original_time in ("00:00", "08:00", "12:30", "19:00", "23:30"):
        assert original_time in times, f"original slot {original_time} must survive the dict-format override"


def test_dict_schedule_night_ticks_after_expiry_are_all_sleep():
    """Contract 1c: the interrupt must not own the night — every night tick
    after expiry resolves to sleep (00:25 through 06:59)."""
    sched = apply_schedule_override(
        NIGHT_SCHEDULE_DICT, INTERRUPT_TIME, INTERRUPT_ACTIVITY, duration_minutes=INTERRUPT_DURATION_MINUTES
    )
    minute = 25
    while minute < 7 * 60:
        hour, m = divmod(minute, 60)
        tick = f"{hour:02d}:{m:02d}"
        assert is_sleep_activity(get_activity_for_time(sched, tick)), (
            f"night tick {tick} after expiry must resolve to sleep on a dict schedule"
        )
        minute += 35


# ---------------------------------------------------------------------------
# Contract 3 — non-sleep dict slots keep persistent semantics.
# ---------------------------------------------------------------------------


def test_dict_schedule_non_sleep_override_stays_persistent():
    """Contract 3a: overriding a NON-sleep dict slot keeps the legacy
    persistent semantics (no expiry slot is invented)."""
    sched = apply_schedule_override(NIGHT_SCHEDULE_DICT, "12:35", "临时外出办事", duration_minutes=30)
    expiry = "13:05"
    assert get_activity_for_time(sched, "12:40") == "临时外出办事"
    assert get_activity_for_time(sched, "13:10") != "临时外出办事", (
        "a non-sleep override must persist past its duration (legacy semantics)"
    )
    times = (
        {entry["time"] for entry in sched}
        if all(isinstance(e, dict) for e in sched)
        else {t for t, _ in sched}
    )
    assert expiry not in times, "no expiry re-assertion slot may be added for a non-sleep override"


def test_dict_schedule_legacy_two_arg_call_unchanged():
    """Contract 3b: the legacy 2-arg call on a dict schedule behaves as a
    persistent override (no duration plumbing involved)."""
    sched = apply_schedule_override(NIGHT_SCHEDULE_DICT, "10:00", "临时会议")
    assert get_activity_for_time(sched, "10:00") == "临时会议"
    assert get_activity_for_time(sched, "12:00") == "临时会议", (
        "a 2-arg dict override must persist until the next schedule slot"
    )
    assert get_activity_for_time(sched, "12:35") == "午餐", (
        "the following dict slot must take over after its own time, as before"
    )
