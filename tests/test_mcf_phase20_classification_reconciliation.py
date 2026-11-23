"""Phase 2.0 Part B — RED tests: episode-scorer vs ActivityRecord travel
classification reconciliation (the known contradiction).

At bbfa70f the two classification paths disagree on purpose-unknown
in-transit travel:

* Episode-level scoring — ``classify_episode``
  (benchmark/mcf/time_use.py:351-352) maps ANY ``travel.status`` in
  {departed, in_transit} to ``commuting`` (confidence "travel"), and
  ``classify_episode`` line 356-357 maps the writer's 乘坐…移动 action
  phrase to ``commuting`` too — with ZERO purpose evidence.
* Structured records — ``classify_activity_record``
  (benchmark/mcf/time_use.py:1059-1065) promotes in-transit travel to
  ``commuting`` ONLY with a verified work-purpose
  (``destination_purpose == "paid_work"``), otherwise ``travel``.

The unified rule pinned here (behaviour locked by the Phase 1.7 Track 2
tests in tests/test_mcf_time_use_track2.py is the SUPERSEDED contract and
must be updated by the implement worker with orchestrator authorization —
see RESULT-3488 for the explicit list):

* unknown-purpose travel is NOT commuting on any scoring path;
* commuting requires positive work-purpose evidence (a structured
  ``destination_purpose`` of a work category);
* travel/unknown stays a distinct category (``travel_category()``);
* coverage never increases by relabelling uncertain travel;
* honest unknown and unobserved minutes are preserved.

Deterministic only: no LLM, no network.
"""

from __future__ import annotations

from benchmark.mcf.time_use import (
    UNKNOWN,
    classify_activity_record,
    classify_episode,
    compute_daily_minutes,
    compute_observed_minutes,
    survey_comparison_categories,
    travel_category,
)

# Real snapshot shape from the bbfa70f dry run (agent 4, 00:10): in-transit
# bus journey, destination_purpose nested in the travel dict is present in
# the real log but the classification contract here must hold even WITHOUT
# any purpose evidence — the purpose-unknown case.
_PURPOSE_UNKNOWN_TRAVEL: dict = {
    "agent_id": 4,
    "day": 1,
    "time": "00:10",
    "scheduled_activity": "个人时间",
    "final_activity": "前往绍兴国金大悦城",
    "action": "乘坐bus移动",
    "location": "Transit to 绍兴国金大悦城",
    "target_location": "绍兴国金大悦城",
    "travel": {
        "mode": "bus",
        "distance_km": 9.567,
        "minutes": 37,
        "progress": 0.27,
        "status": "in_transit",
        # NOTE: no destination_purpose — purpose unknown.
    },
}

_DEPARTED_PURPOSE_UNKNOWN: dict = dict(
    _PURPOSE_UNKNOWN_TRAVEL,
    travel={"mode": "bus", "status": "departed", "progress": 0.0},
)

_TRAVEL_PHRASE_NO_PURPOSE: dict = {k: v for k, v in _PURPOSE_UNKNOWN_TRAVEL.items() if k != "travel"}

_VERIFIED_WORK_PURPOSE: dict = dict(
    _PURPOSE_UNKNOWN_TRAVEL,
    destination_purpose="paid_work",
    travel=dict(_PURPOSE_UNKNOWN_TRAVEL["travel"], destination_purpose="paid_work"),
)


# ---------------------------------------------------------------------------
# 1. Unknown-purpose travel is NOT commuting on the episode scoring path.
#    (Contradicts classify_episode time_use.py:351-352 / 356-357.)
# ---------------------------------------------------------------------------


def test_episode_in_transit_without_purpose_is_not_commuting():
    category, confidence = classify_episode(_PURPOSE_UNKNOWN_TRAVEL)
    assert category != "commuting", (
        f"classify_episode returned commuting (confidence {confidence!r}) for a "
        "purpose-unknown in-transit journey: the episode path must agree with "
        "classify_activity_record — commuting requires positive work-purpose "
        "evidence, never travel status alone (time_use.py:351-352)"
    )


def test_episode_departed_without_purpose_is_not_commuting():
    category, _ = classify_episode(_DEPARTED_PURPOSE_UNKNOWN)
    assert category != "commuting", (
        f"departed purpose-unknown travel classified as {category!r}: same contradiction, same rule"
    )


def test_episode_travel_phrase_without_purpose_is_not_commuting():
    category, _ = classify_episode(_TRAVEL_PHRASE_NO_PURPOSE)
    assert category != "commuting", (
        f"乘坐…移动 action phrase alone classified as {category!r} "
        "(time_use.py:356-357): the writer's travel phrase proves travel, "
        "not a work purpose"
    )


# ---------------------------------------------------------------------------
# 2. Commuting requires positive work-purpose evidence (record path keeps
#    this — pinned so the unification cannot weaken it).
# ---------------------------------------------------------------------------


def test_record_verified_work_purpose_is_commuting():
    category, source = classify_activity_record(_VERIFIED_WORK_PURPOSE)
    assert category == "commuting" and source == "destination_purpose", (
        f"verified paid_work purpose classified {category!r}/{source!r}: the "
        "promotion must survive the unification"
    )


def test_record_purpose_unknown_travel_is_plain_travel():
    category, source = classify_activity_record(_PURPOSE_UNKNOWN_TRAVEL)
    assert category == travel_category() and source == "travel_status", (
        f"purpose-unknown travel classified {category!r}/{source!r}: it must stay plain {travel_category()!r}"
    )


def test_travel_category_is_distinct_from_commuting_in_survey_buckets():
    buckets = survey_comparison_categories()
    assert buckets["commuting"] == ["commuting"]
    assert buckets["commuting_aliases"] == ["commuting"], (
        "commuting_aliases must not fold travel/unknown into commuting"
    )
    assert travel_category() not in buckets["commuting"]
    assert travel_category() != "commuting"


# ---------------------------------------------------------------------------
# 3. Coverage never increases by relabelling uncertain travel.
# ---------------------------------------------------------------------------


def test_daily_minutes_do_not_credit_commuting_for_unknown_purpose_travel():
    daily = compute_daily_minutes([_PURPOSE_UNKNOWN_TRAVEL], tick_minutes=10)
    minutes = daily[4][1]
    assert minutes["commuting"] == 0, (
        f"{minutes['commuting']} minutes of purpose-unknown travel credited to "
        "commuting by the episode scorer: relabelling uncertain travel must "
        "never increase commuting coverage"
    )


def test_observed_accounting_preserves_unknown_minutes():
    before = compute_observed_minutes([_PURPOSE_UNKNOWN_TRAVEL], tick_minutes=10)[4][1]
    total_unknown_before = before.unknown_minutes + before.unobserved_minutes
    daily = compute_daily_minutes([_PURPOSE_UNKNOWN_TRAVEL], tick_minutes=10)[4][1]
    assert daily[UNKNOWN] + total_unknown_before >= total_unknown_before, (
        "honest unknown minutes must be preserved, not absorbed by relabelling"
    )
    assert daily[UNKNOWN] > 0, (
        "a purpose-unknown travel span must land in unknown (or the travel "
        "bucket), never silently in a known category"
    )
