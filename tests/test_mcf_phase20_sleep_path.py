"""Phase 2.0 Part A — RED tests: the sleep path is still broken at bbfa70f.

Offline verification (zero inference) over the complete sleep path —
schedule generation → midnight slot → activity selection → routine
overrides → movement → recorded activity — on the Phase 1.9 dry harness
(stubbed LLM, canary ``dry_run=True``) with agents 4 and 15 of record
(/home/justin/Documents/gaworld-agent-work/pilot19 profiles, 绍兴柯桥):

* recorded activity is NOT sleep for any tick of 00:00–07:00 for either
  agent — both wander all night (agent_4: bus/bike trips across the city;
  agent_15: taxi/car work trips). 0 sleep minutes each.
* Root cause pinned here (file:line at bbfa70f): the widened D1 guard in
  ``ensure_sleep_in_schedule`` (gaworld/sim/_schedule.py:322-345) is
  DEAD CODE whenever the schedule already carries a sleep-tagged slot:
  the early return at gaworld/sim/_schedule.py:315-316
  (``if any(is_sleep_activity(activity) ...)``) fires first, because real
  cached schedules (output/cities/*/memory/agent_4_schedule.json) already
  end with a 23:30 睡前 tail — and still have no 00:00 head slot. The
  00:00–07:50 window then falls into the ``get_activity_for_time``
  个人时间 fallback, and the night is free for unplanned travel.

Contract pinned (fix must satisfy all of it):

* A schedule whose ONLY sleep evidence is a late-evening 睡前 tail and
  which has no 00:00 head slot must, after ``ensure_sleep_in_schedule``,
  resolve every minute of 00:00 up to the first morning slot to a
  sleep-tagged activity — never the 个人时间 fallback.
* The exact cached pilot schedule shape (below, from the bbfa70f dry run
  of record) must be covered.
* Existing behaviour preserved: schedules already covering the head of
  day are returned unchanged; the tail slot's profile-derived time is
  unchanged.

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


@pytest.fixture(autouse=True)
def _mcf_realism_opt_in(monkeypatch):
    """These contracts pin the Phase-2.0/2.1 realism-engine dynamics, which
    are now gated behind behavior.mcf_realism (default off restores the
    pre-change comparability baseline). Opt in for the whole module."""
    monkeypatch.setenv("GAWORLD_MCF_REALISM", "1")


@pytest.fixture(autouse=True)
def _mcf_realism_opt_in(monkeypatch):
    """These contracts pin the Phase-2.0/2.1 realism-engine dynamics, which
    are now gated behind behavior.mcf_realism (default off restores the
    pre-change comparability baseline). Opt in for the whole module."""
    monkeypatch.setenv("GAWORLD_MCF_REALISM", "1")


DEFAULT_AGENT: dict = {"job": "职员", "personality": "", "daily_life": "", "values": "", "work_style": ""}

# The cached schedule of record for agent 4, read from the Phase 2.0
# dry-run output (output/cities/绍兴柯桥/memory/agent_4_schedule.json).
# Shape: a 23:30 睡前 tail, NO 00:00 head slot — the input that defeats
# the widened guard via the early return.
CACHED_SCHEDULE_OF_RECORD: list[tuple[str, str]] = [
    ("08:00", "吃早饭"),
    ("09:00", "通勤"),
    ("10:00", "上午工作"),
    ("12:00", "午饭"),
    ("14:00", "下午工作"),
    ("18:30", "下班"),
    ("21:00", "个人时间"),
    ("23:30", "睡前"),
]


def _night_fallback_minutes(schedule: list[tuple[str, str]]) -> list[str]:
    """Grid minutes from 00:00 up to the first morning slot that resolve to
    the 个人时间 fallback instead of a sleep-tagged activity."""
    first = min(int(t.split(":")[0]) * 60 + int(t.split(":")[1]) for t, _ in schedule)
    return [
        t
        for t in _build_time_grid(10)
        if int(t.split(":")[0]) * 60 + int(t.split(":")[1]) < first
        and get_activity_for_time(schedule, t) == "个人时间"
    ]


def test_cached_schedule_with_sleep_tail_still_gets_head_coverage():
    """The widened D1 guard must cover the head of day EVEN WHEN the schedule
    already contains a sleep-tagged (睡前) tail slot — the early return at
    _schedule.py:315-316 currently makes the guard unreachable for exactly
    the real cached schedules that produced PILOT19's 0-sleep defect."""
    schedule = ensure_sleep_in_schedule(DEFAULT_AGENT, list(CACHED_SCHEDULE_OF_RECORD))
    bad = _night_fallback_minutes(schedule)
    assert not bad, (
        f"{len(bad)} night grid minutes (00:00–first slot) still resolve to "
        f"个人时间 after ensure_sleep_in_schedule (first: {bad[:3]}): the early "
        "return on any pre-existing sleep-tagged slot defeats head coverage — "
        "this is why both pilot agents still record 0 sleep minutes at bbfa70f"
    )


def test_every_head_minute_is_sleep_tagged_after_ensure():
    schedule = ensure_sleep_in_schedule(DEFAULT_AGENT, list(CACHED_SCHEDULE_OF_RECORD))
    first = min(int(t.split(":")[0]) * 60 + int(t.split(":")[1]) for t, _ in schedule)
    for t in _build_time_grid(10):
        minute = int(t.split(":")[0]) * 60 + int(t.split(":")[1])
        if minute >= first:
            break
        activity = get_activity_for_time(schedule, t)
        assert is_sleep_activity(activity), (
            f"{t}: head-of-day minute resolves to {activity!r}, not a "
            "sleep-tagged activity — night wandering persists"
        )


def test_schedule_already_covering_head_is_unchanged():
    original = [("00:00", "睡觉"), ("08:00", "吃早饭"), ("23:30", "睡前")]
    assert ensure_sleep_in_schedule(DEFAULT_AGENT, list(original)) == original
