"""TDD RED tests — Phase 1.8 Part B.1: observed-only time accounting.

Contract (TASK-3453-SPEC, Contract 1):
- A simulation ending at 12:20 must NOT attribute activity minutes beyond
  the last observed tick/record (no invented time to midnight).
- The scorer reports separately: observed minutes, classified observed
  minutes, unknown observed minutes, unobserved minutes; observed-span
  coverage and full-day coverage are reported as separate numbers.
- A record spanning 23:30->00:30 accounts correctly across the day boundary.
- A full-day (complete observation window) day accounts exactly: minutes
  sum exactly to the window length.

The new API imported below does NOT exist yet — the ImportError is the RED
state by design. No LLM calls, no network, no simulation.
"""

from __future__ import annotations

import pytest

from benchmark.mcf.time_use import (
    ObservedBreakdown,  # noqa: F401  (RED: missing API)
    compute_observed_minutes,  # noqa: F401  (RED: missing API)
    coverage_breakdown,  # noqa: F401  (RED: missing API)
)


TICK = 10


def _ep(agent_id: int, day: int, time: str, activity: str) -> dict[str, object]:
    return {"agent_id": agent_id, "day": day, "time": time, "final_activity": activity}


# ---------------------------------------------------------------------------
# 1. No invented time beyond the last observed tick
# ---------------------------------------------------------------------------


def test_no_activity_minutes_after_last_observed_tick():
    """A day whose last record is 12:20 must not credit leisure/work to 12:20-24:00."""
    episodes = [
        _ep(1, 1, "08:00", "在家休闲，看电视"),
        _ep(1, 1, "12:20", "在家休闲，看电视"),
    ]
    breakdown = compute_observed_minutes(episodes, tick_minutes=TICK)
    minutes = breakdown[1][1]
    classified = sum(v for k, v in minutes.items() if k != "unknown")
    # At most the observed span (00:00->12:20 = 740 min) can be classified;
    # the evening tail (12:20->24:00 = 220 min... rather 700 min) is unobserved.
    assert classified <= 740, f"classified {classified} min exceeds the observed span (740)"


def test_time_after_last_record_is_reported_unobserved_not_classified():
    episodes = [
        _ep(1, 1, "09:00", "在公司处理工作事务"),
        _ep(1, 1, "12:20", "在公司处理工作事务"),
    ]
    breakdown = compute_observed_minutes(episodes, tick_minutes=TICK)
    minutes = breakdown[1][1]
    assert minutes["paid_work"] <= 200, "paid_work credited past the last record at 12:20"
    unobserved = minutes_total(minutes) - observed_span_minutes(minutes)
    assert unobserved >= 700, (
        f"the 12:20->24:00 span ({1440 - 740} min) must land in a reported "
        f"unobserved bucket, not in a category (got breakdown {minutes})"
    )


def minutes_total(minutes: dict[str, int]) -> int:
    return sum(minutes.values())


def observed_span_minutes(minutes: dict[str, int]) -> int:
    observed = minutes.get("observed_minutes")
    if observed is not None:
        return int(observed)
    return sum(v for k, v in minutes.items() if k not in ("unknown", "unobserved_minutes"))


# ---------------------------------------------------------------------------
# 2. Separate accounting: observed / classified / unknown / unobserved
# ---------------------------------------------------------------------------


def test_reports_observed_classified_unknown_unobserved_separately():
    episodes = [
        _ep(1, 1, "08:00", "在家休闲，看电视"),
        _ep(1, 1, "12:20", "在逛街，不知道在做什么"),  # unclassifiable -> unknown
    ]
    breakdown = compute_observed_minutes(episodes, tick_minutes=TICK)
    minutes = breakdown[1][1]
    for key in ("observed_minutes", "classified_minutes", "unknown_minutes", "unobserved_minutes"):
        assert key in minutes, f"breakdown must report {key} separately (got keys {sorted(minutes)})"
    assert minutes["observed_minutes"] == 740  # 00:00 -> 12:20
    assert minutes["unobserved_minutes"] == 700  # 12:20 -> 24:00
    assert minutes["classified_minutes"] + minutes["unknown_minutes"] == 740
    assert minutes["observed_minutes"] + minutes["unobserved_minutes"] == 1440


def test_observed_coverage_and_full_day_coverage_are_reported_separately():
    episodes = [
        _ep(1, 1, "08:00", "在公司处理工作事务"),
        _ep(1, 1, "12:20", "在公司处理工作事务"),
    ]
    report = coverage_breakdown(episodes, tick_minutes=TICK)
    assert "observed_coverage" in report, "observed-span coverage must be a separate field"
    assert "full_day_coverage" in report, "full-day coverage must be a separate field"
    # The observed span (09:00->12:20, plus a corroborated head/tail per the
    # duration contract) is fully classified; the unobserved evening is not.
    assert report["observed_coverage"] > report["full_day_coverage"], (
        "observed coverage must not be diluted into full-day coverage"
    )


# ---------------------------------------------------------------------------
# 3. Cross-midnight records
# ---------------------------------------------------------------------------


def test_record_spanning_midnight_accounts_across_day_boundary():
    """A 23:30->00:30 sleep span credits both days, splitting at midnight."""
    episodes = [
        # Day 1: record that starts 23:30 and is observed to run past midnight.
        _ep(1, 1, "23:30", "睡觉"),
        # Day 2: first record 00:30 — the 00:00->00:30 head belongs to the
        # same sleep span.
        _ep(1, 2, "00:30", "睡觉"),
    ]
    breakdown = compute_observed_minutes(episodes, tick_minutes=TICK)
    day1 = breakdown[1][1]
    day2 = breakdown[1][2]
    # The 30 minutes after midnight are observed on day 2, in sleep.
    assert day2["observed_minutes"] >= 30, "the 00:00->00:30 span must be observed on day 2"
    assert day2["sleep"] >= 30, "the 23:30->00:30 sleep span must credit day 2's head"
    # Day 1 accounts the span up to midnight.
    assert day1["sleep"] >= 30, "day 1 must credit the pre-midnight part of the sleep span"
    assert day1["observed_minutes"] + day1["unobserved_minutes"] == 1440
    assert day2["observed_minutes"] + day2["unobserved_minutes"] == 1440


# ---------------------------------------------------------------------------
# 4. Full observation window: exact accounting
# ---------------------------------------------------------------------------


def test_full_day_minutes_sum_exactly_to_window_length():
    """A complete 24h observation window sums exactly to 1440 minutes."""
    episodes = [
        _ep(1, 1, "00:00", "睡觉"),
        _ep(1, 1, "07:30", "吃早餐"),
        _ep(1, 1, "08:00", "在公司处理工作事务"),
        _ep(1, 1, "12:00", "吃午餐"),
        _ep(1, 1, "12:30", "在公司处理工作事务"),
        _ep(1, 1, "18:00", "在家休闲，看电视"),
        _ep(1, 1, "22:00", "睡觉"),
        _ep(1, 1, "23:50", "睡觉"),
    ]
    breakdown = compute_observed_minutes(episodes, tick_minutes=TICK)
    minutes = breakdown[1][1]
    total = minutes_total(minutes)
    assert total == 1440, f"a full-day window must account exactly 1440 min (got {total})"


def test_partial_window_sums_to_window_not_day():
    """An observation window of 08:00->12:20 accounts exactly 260 minutes."""
    episodes = [
        _ep(1, 1, "08:00", "在公司处理工作事务"),
        _ep(1, 1, "12:20", "在公司处理工作事务"),
    ]
    breakdown = compute_observed_minutes(episodes, tick_minutes=TICK, window_start=480, window_end=740)
    minutes = breakdown[1][1]
    observed = minutes.get("observed_minutes", minutes_total(minutes) - minutes.get("unobserved_minutes", 0))
    assert observed == 260, (
        f"an explicit 08:00->12:20 window must report exactly 260 observed min (got {observed})"
    )


# ---------------------------------------------------------------------------
# 5. ObservedBreakdown structure
# ---------------------------------------------------------------------------


def test_observed_breakdown_is_a_structured_type():
    assert hasattr(ObservedBreakdown, "__dataclass_fields__") or callable(ObservedBreakdown), (
        "ObservedBreakdown must be a structured result type exposing the four "
        "observed-only buckets"
    )
