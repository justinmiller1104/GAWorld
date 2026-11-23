"""Phase 1.9 — RED tests: D1 guard widening (non-sleep 00:00 head slot).

PILOT19 (PILOT19-REPORT.md §2, RESULT-3475.md) proved the Phase 1.8 D1 fix
is a no-op for exactly the agents it was meant to save: ``ensure_sleep_in_schedule``
(``gaworld/sim/_schedule.py``) only appends a 00:00 head slot when the
schedule has NO 00:00 slot. Both pilot agents (4 and 15) already had a
non-sleep 00:00 head slot — ``个人时间`` — so the guard never fired, sleep
was 0 minutes, and the unscheduled night filled with travel.

Contract pinned here (the fix must satisfy all of it, without a hardcoded
sleep timetable):

* A NON-sleep 00:00 head slot (e.g. ``个人时间``) must be converted into
  sleep coverage — the guard may not stay silent just because a 00:00 slot
  already exists.
* Both real pilot night-schedule head shapes (agent 4 and agent 15 from
  ``pilot16/cities/绍兴柯桥/memory/``, read-only) must end up with
  sleep-resolvable head-of-day coverage: ``get_activity_for_time`` at 00:00
  resolves to a sleep-tagged activity, and the pre-morning grid resolves no
  ``个人时间`` fallback.
* Times derive from the schedule's own slots — no invented hours: the
  widened guard may only reuse times already present in the schedule (plus
  the existing profile-derived tail slot).
* Existing preservation behaviour is unchanged: schedules already carrying
  a sleep slot (including a sleep 00:00 head slot) come back untouched, and
  the Phase 1.8 tail-slot derivation still holds.

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

DEFAULT_AGENT: dict = {"job": "职员", "personality": "", "daily_life": "", "values": "", "work_style": ""}


# Gated (GREEN phase 1, task 3878): the head-slot conversion this module pins
# is an MCF-realism behavior now behind mcf_realism_enabled(); these tests
# exercise the gate-ON path, default runs keep the pre-change schedule.
@pytest.fixture(autouse=True)
def _mcf_realism_on(monkeypatch):
    monkeypatch.setenv("GAWORLD_MCF_REALISM", "1")


# The Phase 1.8 fixture shape (no 00:00 slot at all) — control group.
PILOT_DAY_SCHEDULE: list[tuple[str, str]] = [
    ("08:00", "吃早饭"),
    ("09:00", "通勤"),
    ("10:00", "上午工作"),
    ("12:00", "午饭"),
    ("14:00", "下午工作"),
    ("18:30", "下班"),
    ("21:00", "个人时间"),
]

# PILOT19 agent 4 night head (PILOT19-REPORT.md §2): the 00:00 slot EXISTS
# and is 个人时间 — the exact shape that defeated the Phase 1.8 guard.
AGENT4_NIGHT_HEAD: list[tuple[str, str]] = [
    ("00:00", "个人时间"),
    ("00:10", "技能提升"),
    ("00:20", "聊邻居停车"),
    ("08:00", "吃早饭"),
    ("09:00", "通勤"),
    ("10:00", "上午工作"),
    ("12:00", "午饭"),
    ("14:00", "下午工作"),
    ("18:30", "下班"),
]

# PILOT19 agent 15 night head (PILOT19-REPORT.md §2): same defect, a
# non-sleep 00:00 head slot followed by unscheduled night activity slots.
# (蜷缩被窝 at 00:20 is sleep-adjacent wording but NOT matched by
# SLEEP_KEYWORDS — the guard must not depend on keyword luck here.)
AGENT15_NIGHT_HEAD: list[tuple[str, str]] = [
    ("00:00", "个人时间"),
    ("00:10", "工作"),
    ("00:20", "蜷缩被窝"),
    ("00:30", "找地方避雨"),
    ("08:00", "吃早饭"),
    ("09:00", "工作"),
    ("12:00", "午饭"),
    ("18:30", "下班"),
]


def _grid_minutes(schedule: list[tuple[str, str]]) -> list[str]:
    """The sim's own day grid, at its own tick granularity."""
    return list(_build_time_grid(10))


def _head_minutes(schedule: list[tuple[str, str]]) -> list[int]:
    """Minutes of the day strictly before the schedule's last head-of-day
    slot at/after 07:00 — i.e. the night region the guard must cover."""
    morning = [
        int(t.split(":")[0]) * 60 + int(t.split(":")[1]) for t, _ in schedule if int(t.split(":")[0]) >= 7
    ]
    return list(range(min(morning)) if morning else [])


# ---------------------------------------------------------------------------
# 1. The guard widens: a non-sleep 00:00 head slot is converted to sleep.
# ---------------------------------------------------------------------------


def test_non_sleep_head_slot_becomes_sleep_coverage():
    schedule = ensure_sleep_in_schedule(
        DEFAULT_AGENT, [("00:00", "个人时间"), ("08:00", "吃早饭"), ("12:00", "午饭")]
    )
    activity = get_activity_for_time(schedule, "00:00")
    assert is_sleep_activity(activity), (
        f"00:00 head slot 个人时间 was not converted into sleep coverage: {schedule}"
    )


def test_non_sleep_head_slot_covers_whole_pre_morning_grid():
    # With only the head slot + morning slots, sleep must stretch from 00:00
    # to the first morning slot — every grid minute in between is sleep.
    schedule = ensure_sleep_in_schedule(
        DEFAULT_AGENT, [("00:00", "个人时间"), ("08:00", "吃早饭"), ("12:00", "午饭")]
    )
    for minutes in _head_minutes(schedule):
        hh, mm = divmod(minutes, 60)
        time_str = f"{hh:02d}:{mm:02d}"
        activity = get_activity_for_time(schedule, time_str)
        assert is_sleep_activity(activity), (
            f"{time_str}: expected sleep before the first morning slot, got {activity!r} "
            f"(schedule={schedule})"
        )


# ---------------------------------------------------------------------------
# 2. Both PILOT19 agents end up sleep-resolvable at the head of day.
# ---------------------------------------------------------------------------


def test_pilot_agent4_head_is_sleep_resolvable():
    schedule = ensure_sleep_in_schedule(DEFAULT_AGENT, list(AGENT4_NIGHT_HEAD))
    assert is_sleep_activity(get_activity_for_time(schedule, "00:00")), (
        f"agent 4 shape: 00:00 still resolves non-sleep: {schedule}"
    )
    for minutes in _head_minutes(schedule):
        hh, mm = divmod(minutes, 60)
        activity = get_activity_for_time(schedule, f"{hh:02d}:{mm:02d}")
        assert activity != "个人时间", f"agent 4 shape: {hh:02d}:{mm:02d} fell back to 个人时间: {schedule}"


def test_pilot_agent15_head_is_sleep_resolvable():
    schedule = ensure_sleep_in_schedule(DEFAULT_AGENT, list(AGENT15_NIGHT_HEAD))
    assert is_sleep_activity(get_activity_for_time(schedule, "00:00")), (
        f"agent 15 shape: 00:00 still resolves non-sleep: {schedule}"
    )
    for minutes in _head_minutes(schedule):
        hh, mm = divmod(minutes, 60)
        activity = get_activity_for_time(schedule, f"{hh:02d}:{mm:02d}")
        assert activity != "个人时间", f"agent 15 shape: {hh:02d}:{mm:02d} fell back to 个人时间: {schedule}"


# ---------------------------------------------------------------------------
# 3. Times derive from existing slots — no invented timetable.
# ---------------------------------------------------------------------------


def test_no_invented_slot_times():
    schedule = ensure_sleep_in_schedule(DEFAULT_AGENT, list(AGENT4_NIGHT_HEAD))
    input_times = {t for t, _ in AGENT4_NIGHT_HEAD}
    # The existing Phase 1.8 tail derivation may add its own time; every
    # other output time must already exist in the input schedule.
    for t in {t for t, _ in schedule} - input_times:
        assert is_sleep_activity(dict(schedule)[t]), f"a new non-sleep slot time {t} was invented: {schedule}"
    assert "00:00" in {t for t, _ in schedule}, (
        f"the existing 00:00 head time was dropped instead of converted: {schedule}"
    )


def test_widening_varies_with_the_schedule_not_a_fixed_hour():
    # A later first morning slot must get the same coverage — the window
    # follows the schedule's own slots, not a fixed wake hour.
    late = [("00:00", "个人时间"), ("09:00", "吃早饭"), ("12:00", "午饭")]
    schedule = ensure_sleep_in_schedule(DEFAULT_AGENT, late)
    for minutes in range(0, 9 * 60, 10):
        hh, mm = divmod(minutes, 60)
        assert is_sleep_activity(get_activity_for_time(schedule, f"{hh:02d}:{mm:02d}")), (
            f"{hh:02d}:{mm:02d}: not sleep though the first slot is 09:00: {schedule}"
        )
    assert not is_sleep_activity(get_activity_for_time(schedule, "09:00")), (
        "09:00 itself must still be the morning slot, not swallowed by sleep"
    )


# ---------------------------------------------------------------------------
# 4. Existing preservation behaviour unchanged (Phase 1.8 still holds).
# ---------------------------------------------------------------------------


def test_schedule_with_sleep_head_slot_untouched():
    full_day = [("00:00", "睡觉"), ("08:00", "吃早饭"), ("23:30", "睡前")]
    assert ensure_sleep_in_schedule(DEFAULT_AGENT, list(full_day)) == full_day


def test_schedule_with_existing_sleep_slot_untouched():
    has_sleep = [("08:00", "吃早饭"), ("22:00", "睡觉")]
    assert ensure_sleep_in_schedule(DEFAULT_AGENT, list(has_sleep)) == has_sleep


def test_schedule_remains_sorted_with_unique_times():
    schedule = ensure_sleep_in_schedule(DEFAULT_AGENT, list(AGENT15_NIGHT_HEAD))
    times = [t for t, _ in schedule]
    assert times == sorted(times), f"schedule not sorted after widening: {schedule}"
    assert len(times) == len(set(times)), f"duplicate slot times: {schedule}"
