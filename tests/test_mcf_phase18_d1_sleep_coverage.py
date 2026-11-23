"""Phase 1.8 Part D, defect D1 — RED tests: head-of-day schedule coverage hole.

Root cause (RESULT-nighttime.md, D1): the day grid starts at 00:00
(``gaworld/sim/_utils.py`` ``_build_time_grid``) but ``ensure_sleep_in_schedule``
(``gaworld/sim/_schedule.py``) only *appends* a 睡前 tail slot (23:30 for the
default profile). A schedule whose first slot is 08:00 leaves 00:00–07:50
uncovered, and ``get_activity_for_time`` fills that hole with 个人时间 — so
agents "do personal time" (and then travel) all night instead of sleeping.

Contract pinned here (fix must satisfy all of it, without implementing a
hardcoded sleep timetable):

* After ``ensure_sleep_in_schedule``, ``get_activity_for_time`` over every
  minute of the day must NEVER return the 个人时间 fallback: the inserted
  sleep/rest window must stretch back over midnight up to the first morning
  slot.
* Every head-of-day minute between 00:00 and the first scheduled slot must
  resolve to a sleep-tagged activity (``is_sleep_activity``).
* Existing behaviour preserved: the 睡前 tail slot remains, its derived time
  per profile (default 23:30, retired 22:30, student 00:30, late 01:00) is
  unchanged, and schedules that already cover the head of day are returned
  unchanged.
* No arbitrary timetable: the head coverage must derive from the schedule's
  own slots (the inserted sleep slot extended across the day boundary), not
  from fixed new hours — probed by varying the profile-derived sleep time and
  the first morning slot.

Deterministic only: no LLM, no network, no simulation run.
"""

from __future__ import annotations

import pytest

from gaworld.sim._schedule import (
    ensure_sleep_in_schedule,
    get_activity_for_time,
    is_sleep_activity,
)
from gaworld.sim._utils import _build_time_grid


# Gated (GREEN phase 1, task 3878): head-of-day coverage is an MCF-realism
# behavior behind mcf_realism_enabled(); these tests exercise the gate-ON
# path, default runs keep the pre-change schedule.
@pytest.fixture(autouse=True)
def _mcf_realism_on(monkeypatch):
    monkeypatch.setenv("GAWORLD_MCF_REALISM", "1")


# The pilot archetype template (pilot16 agent_4_schedule.json) minus the
# 睡前 tail: exactly the shape that produced the 00:00–07:50 hole.
PILOT_DAY_SCHEDULE: list[tuple[str, str]] = [
    ("08:00", "吃早饭"),
    ("09:00", "通勤"),
    ("10:00", "上午工作"),
    ("12:00", "午饭"),
    ("14:00", "下午工作"),
    ("18:30", "下班"),
    ("21:00", "个人时间"),
]

DEFAULT_AGENT: dict = {"job": "职员", "personality": "", "daily_life": "", "values": "", "work_style": ""}
RETIRED_AGENT: dict = {"job": "", "personality": "已退休", "daily_life": "", "values": "", "work_style": ""}
STUDENT_AGENT: dict = {"job": "学生", "personality": "", "daily_life": "", "values": "", "work_style": ""}
LATE_AGENT: dict = {"job": "", "personality": "晚睡", "daily_life": "", "values": "", "work_style": ""}


def _sleep_slot(schedule: list[tuple[str, str]]) -> tuple[str, str] | None:
    """The (single) sleep-tagged slot the fix must have inserted/kept."""
    found = [(t, a) for t, a in schedule if is_sleep_activity(a)]
    return found[-1] if found else None


def _head_minutes(schedule: list[tuple[str, str]]) -> list[int]:
    """Minutes of the day strictly before the schedule's first slot."""
    first = min(int(t.split(":")[0]) * 60 + int(t.split(":")[1]) for t, _ in schedule)
    return list(range(first))


# ---------------------------------------------------------------------------
# 1. The hole is gone: no 个人时间 fallback anywhere on the day grid.
# ---------------------------------------------------------------------------


def test_no_personal_time_fallback_on_full_day_grid():
    schedule = ensure_sleep_in_schedule(DEFAULT_AGENT, list(PILOT_DAY_SCHEDULE))
    for time_str in _build_time_grid(10):
        activity = get_activity_for_time(schedule, time_str)
        assert activity != "个人时间" or time_str >= "21:00", (
            f"{time_str}: 个人时间 fallback — head-of-day hole still present (schedule={schedule})"
        )


def test_every_head_of_day_minute_is_sleep():
    schedule = ensure_sleep_in_schedule(DEFAULT_AGENT, list(PILOT_DAY_SCHEDULE))
    for minutes in _head_minutes(schedule):
        hh, mm = divmod(minutes, 60)
        time_str = f"{hh:02d}:{mm:02d}"
        activity = get_activity_for_time(schedule, time_str)
        assert is_sleep_activity(activity), (
            f"{time_str}: expected sleep coverage before the first slot ({schedule[0][0]}), got {activity!r}"
        )


# ---------------------------------------------------------------------------
# 2. Coverage derives from the schedule, not a hardcoded timetable.
# ---------------------------------------------------------------------------


def test_coverage_extends_over_midnight_for_late_profile():
    # A 晚睡 agent sleeps at 01:00; the inserted rest must still cover
    # 00:00–01:00 by wrapping the day boundary, not by shifting to a
    # hardcoded hour.
    schedule = ensure_sleep_in_schedule(LATE_AGENT, list(PILOT_DAY_SCHEDULE))
    sleep = _sleep_slot(schedule)
    assert sleep is not None, f"no sleep slot at all: {schedule}"
    assert is_sleep_activity(get_activity_for_time(schedule, "00:00")), (
        f"00:00 not sleep for late-schedule agent (sleep slot {sleep}): {schedule}"
    )


def test_first_morning_slot_bounds_the_sleep_window():
    # With a later first slot (09:00), sleep coverage must reach it — the
    # window ends where the schedule's own day begins, not at a fixed hour.
    later = [("09:00", "吃早饭"), *list(PILOT_DAY_SCHEDULE[1:])]
    schedule = ensure_sleep_in_schedule(DEFAULT_AGENT, later)
    for minutes in range(9 * 60):
        hh, mm = divmod(minutes, 60)
        activity = get_activity_for_time(schedule, f"{hh:02d}:{mm:02d}")
        assert is_sleep_activity(activity), (
            f"{hh:02d}:{mm:02d}: not sleep though first slot is 09:00: {schedule}"
        )
    assert not is_sleep_activity(get_activity_for_time(schedule, "09:00")), (
        "09:00 itself must still be the morning slot, not swallowed by sleep"
    )


def test_profile_derived_tail_time_preserved():
    # The derivation of *when* sleep starts must not change. Default 23:30,
    # retired 22:30; the student 00:30 target is clamped by today's
    # ``min(max_minutes + 60, 23:59)`` rule (last slot 21:00 → 22:00), and
    # that clamp is existing behaviour the fix must keep.
    assert _sleep_slot(ensure_sleep_in_schedule(DEFAULT_AGENT, list(PILOT_DAY_SCHEDULE)))[0] == "23:30"
    assert _sleep_slot(ensure_sleep_in_schedule(RETIRED_AGENT, list(PILOT_DAY_SCHEDULE)))[0] == "22:30"
    assert _sleep_slot(ensure_sleep_in_schedule(STUDENT_AGENT, list(PILOT_DAY_SCHEDULE)))[0] == "22:00"


# ---------------------------------------------------------------------------
# 3. Existing behaviour preserved.
# ---------------------------------------------------------------------------


def test_tail_sleep_slot_still_present_and_last_of_day():
    schedule = ensure_sleep_in_schedule(DEFAULT_AGENT, list(PILOT_DAY_SCHEDULE))
    sleep = _sleep_slot(schedule)
    assert sleep is not None, f"睡前 tail slot lost: {schedule}"
    tail_activities = [a for t, a in schedule if t >= "21:00"]
    assert any(is_sleep_activity(a) for a in tail_activities), (
        f"sleep must remain in the evening tail: {schedule}"
    )


def test_schedule_already_covering_midnight_is_unchanged():
    full_day = [("00:00", "睡觉"), ("08:00", "吃早饭"), ("23:30", "睡前")]
    schedule = ensure_sleep_in_schedule(DEFAULT_AGENT, list(full_day))
    assert schedule == full_day


def test_schedule_with_existing_sleep_slot_untouched():
    has_sleep = [("08:00", "吃早饭"), ("22:00", "睡觉")]
    schedule = ensure_sleep_in_schedule(DEFAULT_AGENT, list(has_sleep))
    assert schedule == has_sleep


def test_schedule_remains_sorted_and_unique_times():
    schedule = ensure_sleep_in_schedule(DEFAULT_AGENT, list(PILOT_DAY_SCHEDULE))
    times = [t for t, _ in schedule]
    assert times == sorted(times), f"schedule not sorted after fix: {schedule}"
    assert len(times) == len(set(times)), f"duplicate slot times: {schedule}"
