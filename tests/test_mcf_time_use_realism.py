"""TDD RED tests (Phase 0.8 audit, TASK-3201): realism corrections for the
MCF time-use scorer.

These tests pin the corrections identified by the Task A audit of
``benchmark/mcf/time_use.py`` against the REAL episode writer
(``gaworld/sim/_stages.py`` episode compose, ``gaworld/memory/experience.py``
JSONL store):

1. Cross-midnight sleep: real data emits one episode per agent per tick with
   ``day`` incremented at rollover, so a day's recorded span starts at its
   first episode (~07:30) — the overnight tail of sleep currently vanishes.
2. Incomplete days (last episode well before 24:00) must reduce coverage and
   raise an uncertainty warning, not inflate the last category.
3. Low classified coverage (< 0.85) must raise an uncertainty warning.
4. TARGETS must carry explicit survey-basis metadata (2024 = weekly average;
   2018 = workday/restday split) so weekday-only sims are not silently
   compared to weekly averages; the scorer accepts an explicit day-type flag.
5. Out-of-order timestamps must be handled deterministically (sort by time
   within a day), not silently clamped to zero.
6. Per-agent-day per-category minutes must be exposed on the score result so
   an external paid_work-vs-leisure correlation (24h-budget coupling) is
   possible; per-category entries must report how many agent-days were scored
   (participants vs excluded distinction for NILF agents).

All episodes are synthetic dicts in the REAL schema (``day``, ``time``,
``final_activity``). No LLM calls, no network, no simulation.
"""

from __future__ import annotations

import pytest

from benchmark.mcf.time_use import (
    TARGETS_2018,
    TARGETS_2024,
    compute_daily_minutes,
    score_against_targets,
)


def _ep(agent_id: int, day: int, time: str, activity: str) -> dict[str, object]:
    return {
        "agent_id": agent_id,
        "day": day,
        "time": time,
        "final_activity": activity,
        "scheduled_activity": activity,
    }


# ---------------------------------------------------------------------------
# 1. Cross-midnight accounting: the head of a day (00:00 -> first episode)
#    belongs to the previous day's last category (typically sleep). Currently
#    the scorer starts counting at the first episode, so ~7.5h of sleep per
#    day vanishes and sleep systematically undercounts vs the 586-min target.
# ---------------------------------------------------------------------------


def test_overnight_head_of_day_is_attributed_not_dropped():
    daily = {
        1: {
            1: [
                _ep(1, 1, "09:00", "上班工作"),
                _ep(1, 1, "18:00", "回家休闲放松"),
                _ep(1, 1, "23:40", "上床睡觉"),
            ],
            2: [_ep(1, 2, "07:30", "吃早饭"), _ep(1, 2, "08:00", "上班工作")],
        }
    }
    daily = compute_daily_minutes(
        [ep for days in daily.values() for eps in days.values() for ep in eps], tick_minutes=10
    )
    result = score_against_targets(daily, employed_agents=[1])
    day2 = result.per_agent_day[1][2]
    # 00:00 -> 07:30 is the tail of night sleep: 450 minutes of sleep.
    assert day2["sleep"] >= 440, (
        f"overnight sleep on day 2 should be carried into day 2 accounting, got {day2['sleep']} minutes"
    )
    # And the day must still sum to the full 24h span.
    total = sum(day2.values())
    assert total == 1440, f"day 2 accounts for {total} minutes, expected 1440"


def test_cross_midnight_sleep_reaches_sleep_target_scale():
    """Two agents with realistic schedules: sleep must land near ~8h/day."""
    daily: dict[int, dict[int, list[dict[str, object]]]] = {}
    for agent in (1, 2):
        daily[agent] = {
            1: [
                _ep(agent, 1, "08:00", "上班工作"),
                _ep(agent, 1, "18:00", "看电视放松"),
                _ep(agent, 1, "23:00", "睡觉"),
            ],
            2: [
                _ep(agent, 2, "07:30", "洗漱吃早饭"),
                _ep(agent, 2, "08:00", "上班工作"),
                _ep(agent, 2, "22:30", "睡觉"),
            ],
        }
    result = score_against_targets(
        compute_daily_minutes([ep for a in daily.values() for d in a.values() for ep in d], tick_minutes=10),
        employed_agents=[1, 2],
    )
    assert result.per_category["sleep"]["sim"] >= 480, (
        f"sleep mean {result.per_category['sleep']['sim']:.0f} min is far below the "
        "~586-min target; the overnight head of each day is being dropped"
    )


# ---------------------------------------------------------------------------
# 2. Incomplete days: a day whose last episode is at 20:00 has an observed
#    span of 720 min, not 1440. It must reduce coverage and raise an
#    uncertainty warning — not silently pad the last category to 24:00.
# ---------------------------------------------------------------------------


def test_incomplete_day_triggers_uncertainty_warning_and_short_span():
    daily = {
        1: {
            1: [_ep(1, 1, "08:00", "上班工作"), _ep(1, 1, "20:00", "看电视放松")],
        }
    }
    result = score_against_targets(compute_daily_minutes(daily[1][1], tick_minutes=10), employed_agents=[1])
    # 08:00 -> 20:00 observed; 20:00 -> 24:00 is unobserved tail, not leisure.
    day1 = result.per_agent_day[1][1]
    assert day1["leisure"] <= 60, (
        f"unobserved 20:00-24:00 tail padded leisure to {day1['leisure']} min; "
        "an incomplete day must not inflate its last category"
    )
    assert any("coverage" in w or "incomplete" in w for w in result.uncertainty_warnings), (
        "incomplete day must trigger an uncertainty warning"
    )
    assert result.coverage < 0.85, f"coverage {result.coverage:.2f} ignores that half the day was unobserved"


# ---------------------------------------------------------------------------
# 3. Low classified coverage (< 0.85) must surface an uncertainty warning.
# ---------------------------------------------------------------------------


def test_low_coverage_triggers_uncertainty_warning():
    daily = {
        1: {
            1: [
                _ep(1, 1, "08:00", "在房间里待着"),  # unclassifiable -> unknown
                _ep(1, 1, "20:00", "继续待着"),  # unknown
                _ep(1, 1, "23:00", "睡觉"),
            ],
        }
    }
    result = score_against_targets(compute_daily_minutes(daily[1][1], tick_minutes=10), employed_agents=[1])
    assert result.coverage < 0.85
    assert any("coverage" in w for w in result.uncertainty_warnings), (
        "coverage below 0.85 must raise an uncertainty warning so a bad mapper cannot look deceptively clean"
    )


# ---------------------------------------------------------------------------
# 4. Survey-basis metadata + explicit weekday/weekend flag.
#    2024 targets are a weekly average; 2018 is reported as workday/restday.
#    A weekday-only simulation must be checkable against the declared basis.
# ---------------------------------------------------------------------------


def test_targets_carry_survey_basis_metadata():
    assert TARGETS_2024["sleep"].get("survey_basis") == "weekly_average", (
        "TARGETS_2024 must declare it is a weekly average so weekday-only sims "
        "are not silently compared against it"
    )
    assert TARGETS_2018["sleep"].get("survey_basis") == "workday_restday_split", (
        "TARGETS_2018 must declare the workday/restday split basis"
    )


def test_score_against_targets_accepts_explicit_day_types():
    daily = {
        1: {
            1: [_ep(1, 1, "09:00", "上班工作"), _ep(1, 1, "19:00", "看电视放松")],
            2: [_ep(1, 2, "10:00", "看电视放松"), _ep(1, 2, "15:00", "访友聚会")],
        }
    }
    # A workday/restday flag must be accepted; with it, paid_work is scored
    # only over workdays (weekend day 2 must not dilute the participants mean).
    result = score_against_targets(
        compute_daily_minutes(
            [ep for d in daily.values() for eps in d.values() for ep in eps], tick_minutes=10
        ),
        employed_agents=[1],
        day_types={1: {1: "workday", 2: "restday"}},
    )
    day1 = result.per_agent_day[1][1]
    day2 = result.per_agent_day[1][2]
    assert day1["paid_work"] > 0
    assert day2["paid_work"] == 0
    # The workday-only paid_work mean must reflect the workday minutes.
    assert result.per_category["paid_work"]["sim"] == pytest.approx(day1["paid_work"])


# ---------------------------------------------------------------------------
# 5. Irregular timestamps: out-of-order entries within a day are handled
#    deterministically (sorted by time), not silently clamped to zero.
# ---------------------------------------------------------------------------


def test_out_of_order_timestamps_sorted_deterministically():
    episodes = [
        _ep(1, 1, "13:00", "看电视放松"),  # out of order
        _ep(1, 1, "09:00", "上班工作"),
        _ep(1, 1, "23:00", "睡觉"),
        _ep(1, 1, "18:00", "做饭家务"),
    ]
    daily = compute_daily_minutes(episodes, tick_minutes=10)
    result = score_against_targets(daily, employed_agents=[1])
    day1 = result.per_agent_day[1][1]
    # Sorted: 09:00 work -> 13:00 leisure -> 18:00 housework -> 23:00 sleep.
    assert day1["paid_work"] == 240, f"expected sorted 09:00-13:00 work, got {day1['paid_work']}"
    assert day1["leisure"] == 300, f"expected sorted 13:00-18:00 leisure, got {day1['leisure']}"
    assert day1["housework"] == 300, f"expected sorted 18:00-23:00 housework, got {day1['housework']}"
    total = sum(day1.values())
    assert total == 1440, f"durations must account for the full day, got {total}"


# ---------------------------------------------------------------------------
# 6. Per-agent-day outputs for external correlation analysis (24h-budget
#    coupling between paid_work and leisure), and per-category n reporting
#    (participants basis: employed vs NILF distinction must be checkable).
# ---------------------------------------------------------------------------


def test_per_agent_day_outputs_exposed_for_correlation():
    mix_a = [
        _ep(1, 1, "08:00", "上班工作"),
        _ep(1, 1, "21:00", "看电视放松"),
        _ep(1, 1, "23:00", "睡觉"),
    ]
    mix_b = [
        _ep(2, 1, "10:00", "散步健身"),
        _ep(2, 1, "12:00", "做饭家务"),
        _ep(2, 1, "22:00", "看书学习"),
        _ep(2, 1, "23:00", "睡觉"),
    ]
    result = score_against_targets({1: {1: mix_a}, 2: {1: mix_b}}, employed_agents=[1])
    # per_agent_day gives the per-agent per-category minutes an external
    # paid_work-vs-leisure correlation needs (constant 1440 budget => co-move).
    agent1 = result.per_agent_day[1][1]
    agent2 = result.per_agent_day[2][1]
    assert agent1["paid_work"] > agent2["paid_work"]
    assert agent1["leisure"] != agent2["leisure"]
    # The 24h-budget coupling (paid_work and leisure co-move by construction)
    # is analysable because per-category errors are exposed separately.
    assert set(agent1) >= {"paid_work", "leisure", "sleep"}


def test_paid_work_category_reports_number_of_scored_agent_days():
    """Participants basis: NILF agents are excluded, not counted as zero —
    and the result must say how many agent-days were actually scored."""
    daily = {
        1: {1: [_ep(1, 1, "09:00", "上班工作"), _ep(1, 1, "19:00", "看电视放松")]},
        2: {1: [_ep(2, 1, "10:00", "散步健身"), _ep(2, 1, "22:00", "看电视放松")]},
    }
    daily = compute_daily_minutes(
        [ep for days in daily.values() for eps in days.values() for ep in eps], tick_minutes=10
    )
    result = score_against_targets(daily, employed_agents=[1])
    spec = result.per_category["paid_work"]
    assert spec["n_agent_days"] == 1, (
        "paid_work mean must be over employed agent-days only and report that count "
        f"so NILF exclusion (vs present-but-zero) is checkable, got {spec.get('n_agent_days')}"
    )
