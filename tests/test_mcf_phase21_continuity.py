"""Phase 2.1 RED tests — activity continuity (RESULT-3515 / TASK-3516).

Defect (RESULT-phase21-investigation): the dynamic-behaviour path in
``gaworld/behavior/dynamic.py`` re-fires interrupts on every tick with no
continue-vs-replace distinction, no (source, kind) cooldown, and a commitment
table that scores bare sleep tokens (睡前/睡觉/…) at the 0.35 default — so
background weather and time-pressure candidates displace sleep (and work,
meals, leisure identically) tick after tick.

Ten contracts pinned here (the investigation's RED table), all deterministic
pure-Python unit tests on stubbed state — no LLM, no network:

  1.  bare sleep tokens are high-commitment activities
  2.  uninterrupted sleep survives a whole night's low-priority churn
  3.  a genuine emergency (fire 0.95) DOES displace sleep, and the override
      stays duration-bounded (Phase 2.0 semantics preserved)
  4.  a non-urgent night suggestion (梅雨 weather_rain) over 睡前 is rejected;
      the same candidate over a daytime activity is allowed
  5.  scheduled waking (睡前 → 吃早饭) needs no interrupt at all
  6.  repeat time_pressure candidates are cooldown-suppressed: ≤1 accept
      per window while working
  7.  a hunger candidate over an already-eating agent is a continue, not a
      re-insertion every tick
  8.  spontaneous leisure switching is allowed once, then suppressed until
      the accepted switch's window expires
  9.  the same weather event on 5 consecutive ticks is accepted exactly once
 10.  cross-midnight continuity: the 00:00 head slot survives the night and
      compute_observed_minutes credits sleep, not night travel
"""

from __future__ import annotations

import random

import pytest

from benchmark.mcf.time_use import compute_observed_minutes
from gaworld.behavior.dynamic import (
    activity_commitment,
    evaluate_step_dynamics,
)
from gaworld.sim._schedule import (
    apply_schedule_override,
    get_activity_for_time,
    is_sleep_activity,
)


@pytest.fixture(autouse=True)
def _mcf_realism_opt_in(monkeypatch):
    """These contracts pin the Phase-2.0/2.1 realism-engine dynamics, which
    are now gated behind behavior.mcf_realism (default off restores the
    pre-change comparability baseline). Opt in for the whole module."""
    monkeypatch.setenv("GAWORLD_MCF_REALISM", "1")


# ---------------------------------------------------------------------------
# Deterministic stubs
# ---------------------------------------------------------------------------


def _stub_agent(**state):
    """A neutral agent: no traits (trait_modifier → 1.0), quiet needs."""
    base = {
        "self_control": 0.6,
        "risk_preference": 0.5,
        "hunger": 0.20,
        "energy": 0.90,
        "fatigue_debt": 0.10,
        "time_pressure": 0.10,
        "stress": 0.30,
        "emotion": 0.60,
        "social_need": 0.50,
    }
    base.update(state)
    return {"agent_id": 4, "locations": {"current": "家"}, "state": base}


# The 梅雨 background condition (gaworld/city/environment.py) — the per-tick
# Loop A candidate from the investigation.
MEIYU_EVENT = {
    "type": "natural",
    "topic": "weather",
    "description": "梅雨持续，空气潮湿",
    "severity": 0.3,
}

# A genuine emergency — must always be able to displace sleep.
FIRE_EVENT = {
    "type": "emergency",
    "description": "附近发生火灾",
    "severity": 1.0,
}

NIGHT_SCHEDULE = [
    ("00:00", "睡前"),
    ("08:00", "上午工作"),
    ("12:30", "午饭"),
    ("14:00", "下午工作"),
    ("19:00", "个人时间"),
    ("23:30", "睡前"),
]


def _step(agent, time_str, activity, env_events=()):
    return evaluate_step_dynamics(agent, time_str, activity, list(env_events), [], {}, config={})


def _tick_time(tick):
    """tick 0 → '00:00', tick 1 → '00:10', … (10-min ticks like the sim)."""
    minutes = tick * 10
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# ---------------------------------------------------------------------------
# Case 1 — commitment classifier
# ---------------------------------------------------------------------------


def test_case1_bare_sleep_tokens_are_high_commitment():
    """睡眠-class keywords belong in COMMITMENT_KEYWORDS at ≥0.85 (the same
    level as 开会/上课), not at the 0.35 fall-through default."""
    for token in ("睡觉", "睡眠", "入睡", "就寝", "睡前"):
        commitment = activity_commitment(token)
        assert commitment >= 0.85, (
            f"activity_commitment({token!r}) = {commitment}, expected ≥ 0.85 "
            "(bare sleep tokens currently fall through to the 0.35 default)"
        )


# ---------------------------------------------------------------------------
# Case 2 — uninterrupted sleep
# ---------------------------------------------------------------------------


def test_case2_sleep_survives_a_full_night_of_low_priority_churn():
    """A sleeping agent under the persistent 梅雨 background condition must
    accumulate ≥60 consecutive uninterrupted sleep ticks (00:00–09:50)."""
    agent = _stub_agent()
    consecutive = 0
    for tick in range(60):
        random.seed(tick)  # deterministic per tick
        result = _step(agent, _tick_time(tick), "睡前", [MEIYU_EVENT])
        if result["changed"]:
            consecutive = 0
        else:
            assert is_sleep_activity(result["activity"]) or result["activity"] == "睡前", (
                f"tick {tick}: activity drifted off sleep without an override ({result['activity']!r})"
            )
            consecutive += 1
        assert not result["changed"], (
            f"tick {tick} ({_tick_time(tick)}): low-priority churn displaced sleep "
            f"(interrupt={result['interrupt']})"
        )
    assert consecutive >= 60, f"only {consecutive} consecutive sleep ticks accumulated"


# ---------------------------------------------------------------------------
# Case 3 — nighttime emergency still gets through
# ---------------------------------------------------------------------------


def test_case3_fire_displaces_sleep_with_bounded_override():
    """A 0.95 emergency_fire at 03:00 DOES displace 睡前, and the schedule
    override stays duration-bounded (Phase 2.0 expiry semantics)."""
    random.seed(3)
    agent = _stub_agent()
    result = _step(agent, "03:00", "睡前", [FIRE_EVENT])
    assert result["changed"], "a genuine emergency must be able to displace sleep"
    assert result["interrupt"] is not None
    assert result["interrupt"]["priority"] >= 0.85

    sched = apply_schedule_override(
        NIGHT_SCHEDULE, "03:00", result["activity"], duration_minutes=result["interrupt"]["duration_minutes"]
    )
    expiry = (3 * 60 + result["interrupt"]["duration_minutes"]) % (24 * 60)
    after = f"{expiry // 60:02d}:{expiry % 60:02d}"
    assert not is_sleep_activity(get_activity_for_time(sched, "03:05")), (
        "during the emergency window the override applies"
    )
    assert is_sleep_activity(get_activity_for_time(sched, after)), (
        "after the emergency window sleep must be re-asserted"
    )


# ---------------------------------------------------------------------------
# Case 4 — non-urgent night suggestion rejected; same candidate allowed by day
# ---------------------------------------------------------------------------


def test_case4_night_weather_suggestion_over_sleep_is_rejected():
    """梅雨 weather_rain (base 0.55) over 睡前 must not clear the raised
    night acceptance threshold (~0.74 with high sleep commitment)."""
    for tick in range(12, 42):  # 02:00–06:50
        random.seed(tick)
        result = _step(_stub_agent(), _tick_time(tick), "睡前", [MEIYU_EVENT])
        assert not result["changed"], (
            f"tick {_tick_time(tick)}: non-urgent weather displaced sleep (interrupt={result['interrupt']})"
        )


def test_case4b_same_weather_candidate_over_daytime_activity_is_allowed():
    """The identical candidate over a daytime 通勤 slot is acceptable — the
    night rejection must come from sleep commitment, not a blanket weather ban."""
    random.seed(11)
    result = _step(_stub_agent(), "14:00", "通勤", [MEIYU_EVENT])
    assert result["changed"], (
        f"the same weather candidate must remain acceptable over a daytime slot "
        f"(activity={result['activity']!r})"
    )


# ---------------------------------------------------------------------------
# Case 5 — scheduled waking needs no interrupt
# ---------------------------------------------------------------------------


def test_case5_scheduled_waking_transition_needs_no_interrupt():
    """The 睡前 → 吃早饭 slot boundary resolves through the clock alone: with
    no interrupt candidates at all, the step must not disturb it."""
    assert get_activity_for_time(NIGHT_SCHEDULE, "07:50") == "睡前"
    assert get_activity_for_time(NIGHT_SCHEDULE, "08:00") == "上午工作"
    random.seed(5)
    result = _step(_stub_agent(), "08:00", get_activity_for_time(NIGHT_SCHEDULE, "08:00"))
    assert result["activity"] == "上午工作" and not result["changed"], (
        "a scheduled transition must survive an empty candidate pool untouched"
    )


# ---------------------------------------------------------------------------
# Case 6 — sustained work: repeat time_pressure suppressed by cooldown
# ---------------------------------------------------------------------------


def test_case6_repeat_time_pressure_candidates_are_cooldown_suppressed():
    """Working under high time_pressure, the (task, time_pressure) candidate
    may be accepted once, then must be suppressed for the interrupt's own
    window — ≤1 accept per cooldown window across the work block."""
    agent = _stub_agent(self_control=0.4, time_pressure=0.95)
    accepted = 0
    # 09:50–12:00 work block, 10-min ticks (agent state held fixed).
    for tick in range(60, 72):
        random.seed(100 + tick)
        result = _step(agent, _tick_time(tick), "上午工作")
        if result["changed"] and result["interrupt"]["kind"] == "time_pressure":
            accepted += 1
    assert accepted <= 1, (
        f"{accepted} time_pressure interrupts accepted in a 2-hour window "
        f"(≤1 per cooldown window expected); per-tick re-firing not suppressed"
    )


# ---------------------------------------------------------------------------
# Case 7 — meal interruption: continue, not replace
# ---------------------------------------------------------------------------


def test_case7_hunger_candidate_over_the_same_activity_is_a_continue():
    """An agent already eating (previous tick accepted 找点吃的) must not have
    找点吃的 re-inserted every further tick: a candidate whose activity equals
    the current effective activity is a continue, not a change."""
    agent = _stub_agent(hunger=0.80)
    random.seed(7)
    result = _step(agent, "15:00", "找点吃的")
    assert not result["changed"], (
        f"hunger candidate re-inserted over the ongoing 找点吃的 "
        f"(interrupt={result['interrupt']}) — continue-vs-replace missing"
    )


# ---------------------------------------------------------------------------
# Case 8 — leisure switching: once, then suppressed until window expiry
# ---------------------------------------------------------------------------


def test_case8_spontaneous_leisure_switch_happens_at_most_once_per_window():
    """A bored agent in 个人时间 may make ONE spontaneous switch; further
    spontaneous switches are suppressed until the accepted switch's window
    expires (cooldown keyed on the interrupt's duration)."""
    agent = _stub_agent(self_control=0.2, stress=0.90, emotion=0.20)
    accepted = 0
    # Five 10-min ticks of 个人时间 — shorter than any spontaneous switch's
    # duration, so at most one accept may land inside the window.
    for tick in range(5):
        random.seed(200 + tick)
        result = _step(agent, _tick_time(tick), "个人时间")
        if result["changed"] and result["interrupt"]["source"] == "spontaneous":
            accepted += 1
    assert accepted <= 1, (
        f"{accepted} spontaneous switches accepted within one cooldown window; "
        f"leisure churn is not suppressed"
    )


# ---------------------------------------------------------------------------
# Case 9 — repeated minor events: exactly one accept in five ticks
# ---------------------------------------------------------------------------


def test_case9_same_weather_event_accepted_exactly_once_in_five_ticks():
    """The same (source, kind) minor event on 5 consecutive ticks yields
    exactly 1 accepted interrupt and 4 no-change results."""
    accepted = 0
    for tick in range(5):
        random.seed(300 + tick)
        result = _step(_stub_agent(), _tick_time(tick), "个人时间", [MEIYU_EVENT])
        if result["changed"]:
            accepted += 1
    assert accepted == 1, (
        f"the same weather event was accepted {accepted} times in 5 ticks "
        f"(expected exactly 1 — cooldown must suppress the repeats)"
    )


# ---------------------------------------------------------------------------
# Case 10 — override expiry (Phase 2.0 stays green) + cross-midnight continuity
# ---------------------------------------------------------------------------


def test_case10a_bounded_override_expires_and_re_asserts_the_displaced_slot():
    """Phase 2.0 regression pin: a duration-bounded interrupt on the 00:00
    睡前 slot re-asserts sleep at expiry, and the original slot survives."""
    sched = apply_schedule_override(NIGHT_SCHEDULE, "00:05", "找地方避雨", duration_minutes=20)
    assert get_activity_for_time(sched, "00:05") == "找地方避雨"
    assert is_sleep_activity(get_activity_for_time(sched, "00:25"))
    assert ("00:00", "睡前") in sched, "the original sleep head slot must survive"


def test_case10b_cross_midnight_head_slot_survives_and_scores_sleep_not_travel():
    """With the 00:00 head slot intact through the night, the observed-minute
    window 00:00–07:00 reports sleep > 300 min and night travel ≈ 0."""
    # Every 10-min tick of the night resolves to the uninterrupted 睡前 head.
    for minutes in range(0, 420, 10):
        t = f"{minutes // 60:02d}:{minutes % 60:02d}"
        assert get_activity_for_time(NIGHT_SCHEDULE, t) == "睡前", (
            f"{t}: the 00:00 sleep head slot must survive the night"
        )

    episodes = [
        {
            "agent_id": 4,
            "day": 1,
            "time": _tick_time(tick),
            "final_activity": "睡前",
            "scheduled_activity": "睡前",
        }
        for tick in range(42)  # 00:00–06:50, one episode per tick
    ]
    breakdown = compute_observed_minutes(episodes, tick_minutes=10, window_start=0, window_end=420)[4][1]
    assert breakdown["sleep"] > 300, (
        f"cross-midnight night sleep credited as {breakdown['sleep']} min, expected > 300"
    )
    travel = breakdown.get("commuting", 0)
    assert travel < 30, f"night travel credited as {travel} min, expected ≈ 0"
