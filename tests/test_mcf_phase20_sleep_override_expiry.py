"""Phase 2.0 RED tests — transient interrupt must not clobber sleep for the whole night.

Defect (RESULT-3497): at 00:00 a 梅雨 weather interrupt ("找地方避雨") replaces the
00:00 睡前 slot via ``apply_schedule_override`` and the override persists for all
subsequent night ticks — agents travel ~4000 night-travel minutes and sleep 10 min.

Contract pinned here (mechanism only, no sleep-duration tuning):
  * A transient activity interrupt on a sleep slot is duration-bounded: after the
    interrupt's own window the schedule re-asserts sleep for the remaining night.
  * The original sleep slot is not destroyed by the override.
  * Overriding a NON-sleep slot keeps the legacy (persistent) semantics, and the
    legacy call form (no duration) is unchanged — only the sleep path is protected.

All tests are deterministic unit-level tests on gaworld/sim/_schedule.py; no LLM.
"""

from gaworld.sim._schedule import apply_schedule_override, get_activity_for_time, is_sleep_activity

# Synthetic night schedule shaped like the real ones (RESULT-3497):
# a 00:00 睡前 head and a 23:30 睡前 tail around a working day.
NIGHT_SCHEDULE = [
    ("00:00", "睡前"),
    ("08:00", "上午工作"),
    ("12:30", "午餐"),
    ("14:00", "下午工作"),
    ("19:00", "个人时间"),
    ("23:30", "睡前"),
]

INTERRUPT_TIME = "00:05"
INTERRUPT_ACTIVITY = "找地方避雨"
INTERRUPT_DURATION_MINUTES = 20


def _minutes(h: int, m: int) -> int:
    return h * 60 + m


def test_transient_sleep_override_is_active_only_during_its_window():
    """Contract 1a: during the interrupt's own duration the override applies."""
    sched = apply_schedule_override(
        NIGHT_SCHEDULE, INTERRUPT_TIME, INTERRUPT_ACTIVITY, duration_minutes=INTERRUPT_DURATION_MINUTES
    )
    # 00:00–00:04 still sleep (override starts at 00:05)…
    assert is_sleep_activity(get_activity_for_time(sched, "00:00")), "pre-interrupt minutes must stay sleep"
    # …00:05 up to (but excluding) expiry is the interrupt…
    assert get_activity_for_time(sched, "00:05") == INTERRUPT_ACTIVITY
    assert get_activity_for_time(sched, "00:24") == INTERRUPT_ACTIVITY
    # …and from expiry onward sleep is re-asserted.
    assert is_sleep_activity(get_activity_for_time(sched, "00:25")), (
        "sleep must resume at the override's expiry (00:05 + 20 min)"
    )


def test_sleep_is_re_asserted_for_every_remaining_night_tick():
    """Contract 1b: the interrupt must not own the night — every night tick after
    expiry resolves to sleep (00:25 through 06:59, the 720-min window's body)."""
    sched = apply_schedule_override(
        NIGHT_SCHEDULE, INTERRUPT_TIME, INTERRUPT_ACTIVITY, duration_minutes=INTERRUPT_DURATION_MINUTES
    )
    start = _minutes(0, 25)  # expiry
    end = _minutes(6, 59)  # last night tick before morning slots
    non_sleep = [
        f"{m // 60:02d}:{m % 60:02d}"
        for m in range(start, end + 1)
        if not is_sleep_activity(get_activity_for_time(sched, f"{m // 60:02d}:{m % 60:02d}"))
    ]
    assert not non_sleep, f"non-sleep night ticks after expiry: {non_sleep[:10]} ({len(non_sleep)} total)"


def test_original_sleep_slot_is_not_destroyed():
    """Contract 2a: the 00:00 睡前 slot survives the override (expiry can restore it)."""
    sched = apply_schedule_override(
        NIGHT_SCHEDULE, INTERRUPT_TIME, INTERRUPT_ACTIVITY, duration_minutes=INTERRUPT_DURATION_MINUTES
    )
    assert ("00:00", "睡前") in sched, "the original sleep slot must remain in the schedule"


def test_sleep_tail_is_untouched():
    """Contract 2b: the late-night 睡前 tail is unaffected by a mid-night interrupt."""
    sched = apply_schedule_override(
        NIGHT_SCHEDULE, INTERRUPT_TIME, INTERRUPT_ACTIVITY, duration_minutes=INTERRUPT_DURATION_MINUTES
    )
    assert ("23:30", "睡前") in sched
    assert is_sleep_activity(get_activity_for_time(sched, "23:30"))


def test_non_sleep_override_keeps_legacy_persistent_semantics():
    """Contract 3: only sleep slots get the expiry guard — overriding a daytime
    slot with an explicit duration still replaces it persistently."""
    sched = apply_schedule_override(
        NIGHT_SCHEDULE, "14:00", "临时外出办事", duration_minutes=INTERRUPT_DURATION_MINUTES
    )
    assert get_activity_for_time(sched, "14:00") == "临时外出办事"
    assert get_activity_for_time(sched, "15:00") == "临时外出办事", (
        "non-sleep slots keep the legacy persistent-override behaviour"
    )


def test_legacy_call_form_is_unchanged():
    """Contract 3b: the old two-argument call form keeps today's semantics
    (callers other than the interrupt path must not change behaviour)."""
    sched = apply_schedule_override(NIGHT_SCHEDULE, "00:05", INTERRUPT_ACTIVITY)
    assert get_activity_for_time(sched, "00:05") == INTERRUPT_ACTIVITY
    assert get_activity_for_time(sched, "01:00") == INTERRUPT_ACTIVITY
