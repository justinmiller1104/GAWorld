"""TDD RED tests for the MCF time-use scorer contract.

Pins the API of a future ``benchmark/mcf/time_use.py`` module (which does NOT
exist yet — the plain import below is the RED state by design).

Contract summary (from BRIEF-MCF-phase07-tests.md + TASK-3191):
- 9-category Chinese keyword classifier over GAWorld episode JSONL.
- Durations inferred by timestamp differencing with an explicit tick_minutes.
- Gap/overlap/day-boundary/unknown rules; unknown time is never discarded.
- Coverage accounting; 2024 national time-use survey targets (paid_work on a
  participants basis; study flagged child_inflated and excluded by default).

All test episodes are synthetic hand-written dicts / JSONL in tmp_path.
No LLM calls, no network, no simulation.
"""

import json

import pytest

from benchmark.mcf.time_use import (  # noqa: F401  (RED: module missing)
    CATEGORIES,
    TARGETS_2024,
    classify_activity,
    compute_daily_minutes,
    coverage,
    load_episodes,
    score_against_targets,
    score_directory,
)


# ---------------------------------------------------------------------------
# 1. Categories
# ---------------------------------------------------------------------------


def test_categories_is_the_nine_survey_categories():
    expected = {
        "sleep",
        "paid_work",
        "housework",
        "personal_care",
        "meals",
        "leisure",
        "socializing",
        "commuting",
        "study",
    }
    assert set(CATEGORIES) == expected


# ---------------------------------------------------------------------------
# 2. Targets (verified 2024 national time-use survey values, RESULT-3188 §5)
# ---------------------------------------------------------------------------


def test_targets_2024_exact_minutes_and_basis():
    expected_minutes = {
        "sleep": 586,
        "personal_care": 56,
        "meals": 103,
        "paid_work": 383,
        "housework": 77,
        "leisure": 153,
        "socializing": 18,
        "commuting": 50,
        "study": 107,
    }
    for cat, minutes in expected_minutes.items():
        assert TARGETS_2024[cat]["minutes"] == minutes, f"target for {cat}"
    for cat in expected_minutes:
        basis = TARGETS_2024[cat]["basis"]
        if cat == "paid_work":
            assert basis == "participants", "paid_work is a participants-basis target"
        elif cat == "study":
            assert basis == "child_inflated", "study must be flagged child_inflated"
        else:
            assert basis == "all_persons", f"{cat} should be an all-persons target"


def test_study_is_excluded_from_scoring_by_default():
    # The study target is inflated by 6–17-year-olds in the survey sample, so
    # the default scoring pass must not treat it as comparable.
    comparable = {
        cat
        for cat, spec in TARGETS_2024.items()
        if spec.get("comparable_by_default", True) and spec["basis"] != "child_inflated"
    }
    assert "study" not in comparable
    assert "paid_work" in comparable  # scored, but only over employed agents


# ---------------------------------------------------------------------------
# 3. classify_activity — deterministic Chinese keyword mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("上班工作，处理邮件", "paid_work"),
        ("加班写报告", "paid_work"),
        ("工作", "paid_work"),
        ("通勤去公司", "commuting"),
        ("睡觉", "sleep"),
        ("午休", "sleep"),
        ("睡前刷手机", "sleep"),
        ("吃午饭", "meals"),
        ("用餐", "meals"),
        ("吃饭", "meals"),
        ("洗漱", "personal_care"),
        ("洗澡", "personal_care"),
        ("做家务", "housework"),
        ("做饭", "housework"),
        ("买菜", "housework"),
        ("打扫房间", "housework"),
        ("散步", "leisure"),
        ("娱乐一下", "leisure"),
        ("健身", "leisure"),
        ("看电视", "leisure"),
        ("休闲", "leisure"),
        ("社交活动", "socializing"),
        ("聚会", "socializing"),
        ("访友", "socializing"),
        ("学习", "study"),
        ("上课", "study"),
        ("读书", "study"),
    ],
)
def test_classify_activity_keyword_mapping(text, expected):
    category, confidence = classify_activity(text)
    assert category == expected
    assert confidence == "keyword"


def test_classify_activity_unknown_and_empty():
    assert classify_activity("量子修仙") == (None, "unknown")
    assert classify_activity("") == (None, "unknown")
    assert classify_activity("   ") == (None, "unknown")


# ---------------------------------------------------------------------------
# 4. load_episodes
# ---------------------------------------------------------------------------


def _write_episodes(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_load_episodes_parses_jsonl_and_keeps_agent_id(tmp_path):
    path = tmp_path / "agent_31_episodes.jsonl"
    _write_episodes(
        path,
        [
            {"day": 1, "time": "00:00", "final_activity": "睡觉", "scheduled_activity": "睡觉"},
            {"day": 1, "time": "07:00", "final_activity": "吃早饭", "scheduled_activity": "吃饭"},
        ],
    )
    episodes = load_episodes(path)
    assert len(episodes) == 2
    assert all(e.agent_id == 31 for e in episodes)
    assert episodes[0].day == 1
    assert episodes[0].time == "00:00"
    assert episodes[0].final_activity == "睡觉"


def test_load_episodes_empty_file_is_safe(tmp_path):
    path = tmp_path / "agent_1_episodes.jsonl"
    path.write_text("", encoding="utf-8")
    assert load_episodes(path) == []


# ---------------------------------------------------------------------------
# 5. compute_daily_minutes — differencing, gaps, overlaps, day boundaries
# ---------------------------------------------------------------------------


def _eps(agent_id, day, *pairs):
    """Build synthetic episode dicts from (time, final_activity) pairs."""
    return [
        {
            "agent_id": agent_id,
            "day": day,
            "time": t,
            "final_activity": a,
            "scheduled_activity": a,
        }
        for t, a in pairs
    ]


def test_hand_calculated_full_day_exact_minutes(tmp_path):
    """A full 00:00–24:00 day with tick_minutes=30; every category by hand.

    00:00 睡觉      -> 06:30 = 390 sleep
    06:30 洗漱      -> 06:50 = 20 personal_care; 06:50→07:00 gap 10 ≤ 2×30
                       attributed to previous → personal_care 30 total
    07:00 通勤      -> 08:00 = 60 commuting
    08:00 上班工作  -> 12:00 = 240 paid_work
    12:00 吃午饭    -> 12:30 = 30 meals
    12:30 上班工作  -> 18:00 = 330 paid_work; 18:00→18:30 gap 30 ≤ 60
                       attributed to previous → paid_work 600 total
    18:30 做晚饭家务 -> 19:30 = 60 housework
    19:30 看电视休闲 -> 22:00 = 150 leisure
    22:00 睡觉      -> clamped at 24:00 = 120; sleep total 510
    """
    episodes = _eps(
        31,
        1,
        ("00:00", "睡觉"),
        ("06:30", "洗漱"),
        ("06:50", "洗漱"),
        ("07:00", "通勤去公司"),
        ("08:00", "上班工作"),
        ("12:00", "吃午饭"),
        ("12:30", "上班工作"),
        ("18:00", "上班工作"),
        ("18:30", "做晚饭家务"),
        ("19:30", "看电视休闲"),
        ("22:00", "睡觉"),
    )
    result = compute_daily_minutes(episodes, tick_minutes=30)
    minutes = result[31][1]

    assert minutes["sleep"] == 510
    assert minutes["personal_care"] == 30
    assert minutes["commuting"] == 60
    assert minutes["paid_work"] == 600
    assert minutes["meals"] == 30
    assert minutes["housework"] == 60
    assert minutes["leisure"] == 150
    assert minutes["socializing"] == 0
    assert minutes["study"] == 0
    assert minutes["unknown"] == 0
    # Sum-checked against a full 24h day.
    assert sum(minutes.values()) == 1440
    assert coverage(result) == pytest.approx(1.0)


def test_gap_over_two_ticks_is_attributed_to_unknown():
    # tick_minutes=10; 08:00 工作 -> 09:00 娱乐: gap 60 min > 2×10.
    # Previous category (paid_work) gets min(gap, 2×tick) = 20; the remaining
    # 40 min go to "unknown" — never silently dropped.
    episodes = _eps(
        7,
        1,
        ("08:00", "上班工作"),
        ("09:00", "娱乐"),
        ("10:00", "睡觉"),
    )
    minutes = compute_daily_minutes(episodes, tick_minutes=10)[7][1]
    assert minutes["paid_work"] == 20
    assert minutes["unknown"] == 40
    assert minutes["leisure"] == 60  # 09:00 -> 10:00


def test_gap_within_two_ticks_is_attributed_to_previous_category():
    # tick_minutes=10; 08:00 工作 -> 08:15 娱乐: gap 15 ≤ 2×10 → all paid_work.
    episodes = _eps(7, 1, ("08:00", "上班工作"), ("08:15", "娱乐"), ("09:00", "睡觉"))
    minutes = compute_daily_minutes(episodes, tick_minutes=10)[7][1]
    assert minutes["paid_work"] == 15


def test_overlapping_timestamps_produce_no_negative_durations():
    # Out-of-order timestamps: 08:00 工作 followed by 07:30 娱乐 gives a
    # negative raw difference, clamped to 0 for the previous category.
    episodes = _eps(3, 1, ("08:00", "上班工作"), ("07:30", "娱乐"), ("10:00", "睡觉"))
    minutes = compute_daily_minutes(episodes, tick_minutes=10)[3][1]
    assert minutes["paid_work"] == 0
    assert minutes["leisure"] == 150  # 07:30 -> 10:00
    assert all(v >= 0 for v in minutes.values())


def test_day_boundary_split_last_episode_clamped_at_24h():
    episodes = (
        _eps(5, 1, ("22:00", "睡觉"))
        + _eps(5, 2, ("00:00", "睡觉"), ("07:00", "吃早饭"), ("08:00", "上班工作"))
    )
    result = compute_daily_minutes(episodes, tick_minutes=10)
    assert result[5][1]["sleep"] == 120  # 22:00 -> 24:00, does not bleed into day 2
    assert result[5][2]["sleep"] == 420  # 00:00 -> 07:00
    assert result[5][2]["meals"] == 60
    assert result[5][2]["paid_work"] == 16 * 60  # 08:00 -> 24:00


def test_missing_days_are_absent_and_deterministic(tmp_path):
    episodes = (
        _eps(9, 1, ("08:00", "上班工作"), ("18:00", "睡觉"))
        + _eps(9, 3, ("08:00", "上班工作"), ("18:00", "睡觉"))
    )  # day 2 has no episodes at all
    result = compute_daily_minutes(episodes, tick_minutes=10)
    assert 1 in result[9]
    assert 2 not in result[9]
    assert 3 in result[9]
    # Deterministic: two calls produce identical output.
    assert compute_daily_minutes(episodes, tick_minutes=10) == result


# ---------------------------------------------------------------------------
# 6. coverage
# ---------------------------------------------------------------------------


def test_coverage_counts_unknown_time_against_full_score():
    episodes = _eps(
        4,
        1,
        ("00:00", "睡觉"),
        ("08:00", "上班工作"),
        ("09:00", "量子修仙"),  # unclassifiable
        ("10:00", "睡觉"),
    )
    # tick 10: sleep 480 (00:00-08:00), paid_work 60, unknown: 10 (09:00->10:00
    # is a 60-min gap > 2×10 → 20 to previous-unknown-category... this episode
    # itself IS unknown) + last-episode clamp 14h to unknown-activity? No — the
    # final episode extends to 24:00 in its own (unknown) category.
    result = compute_daily_minutes(episodes, tick_minutes=10)
    minutes = result[4][1]
    assert minutes["unknown"] > 0
    classified = sum(minutes[c] for c in CATEGORIES if c != "unknown")
    assert minutes["unknown"] == 1440 - classified
    assert coverage(result) < 1.0


# ---------------------------------------------------------------------------
# 7. score_against_targets
# ---------------------------------------------------------------------------


def _score_result_shape(result):
    """Contract check on the ScoreResult object itself."""
    assert hasattr(result, "per_category")
    assert hasattr(result, "overall_error")
    assert hasattr(result, "coverage")


def test_score_against_targets_paid_work_only_over_employed_agents():
    daily = {
        1: {1: {"paid_work": 400, "sleep": 500, "leisure": 200, "unknown": 0}},
        2: {1: {"paid_work": 600, "sleep": 500, "leisure": 200, "unknown": 0}},
    }
    employed = [1]  # agent 2 is not employed; their 600 min must be ignored
    result = score_against_targets(daily, employed_agents=employed)
    _score_result_shape(result)
    assert result.per_category["paid_work"]["sim"] == 400
    assert result.per_category["paid_work"]["target"] == 383
    assert result.per_category["paid_work"]["error"] == 17
    assert result.per_category["sleep"]["target"] == 586
    assert result.per_category["sleep"]["sim"] == 500  # mean over ALL agents
    assert result.per_category["sleep"]["error"] == 86


def test_score_against_targets_overall_error_and_zero_division_safety():
    daily = {
        1: {1: {"paid_work": 383, "sleep": 586, "personal_care": 56, "meals": 103,
                "housework": 77, "leisure": 153, "socializing": 18, "commuting": 50,
                "unknown": 0}},
    }
    result = score_against_targets(daily, employed_agents=[1])
    assert result.overall_error == pytest.approx(0.0)
    # A perfect-day leftover is exactly 0 minutes (sum == 1440).
    assert result.leftover_minutes == pytest.approx(0.0)
    # Empty input must not raise and must yield zeroed output.
    empty = score_against_targets({}, employed_agents=[])
    _score_result_shape(empty)
    assert empty.overall_error == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 8. score_directory
# ---------------------------------------------------------------------------


def test_score_directory_scores_every_agent_file(tmp_path):
    _write_episodes(
        tmp_path / "agent_31_episodes.jsonl",
        [
            {"day": 1, "time": "00:00", "final_activity": "睡觉", "scheduled_activity": "睡觉"},
            {"day": 1, "time": "08:00", "final_activity": "上班工作", "scheduled_activity": "工作"},
            {"day": 1, "time": "09:00", "final_activity": "睡觉", "scheduled_activity": "睡觉"},
        ],
    )
    _write_episodes(
        tmp_path / "agent_32_episodes.jsonl",
        [
            {"day": 1, "time": "00:00", "final_activity": "睡觉", "scheduled_activity": "睡觉"},
            {"day": 1, "time": "08:00", "final_activity": "上班工作", "scheduled_activity": "工作"},
            {"day": 1, "time": "09:00", "final_activity": "睡觉", "scheduled_activity": "睡觉"},
        ],
    )
    result = score_directory(tmp_path, employed_agent_ids=[31], tick_minutes=10)
    _score_result_shape(result)
    # Both agents contributed; sleep 00:00–08:00 = 480 plus the 09:00→24:00
    # evening tail (900) now carried as sleep — the sleep-after-work artifact
    # was removed in the Phase 0.9 preflight, so sleep sim = 480 + 900 = 1380;
    # paid_work (08:00–09:00 = 60) counts only agent 31's day → sim 60,
    # error |60 − 383| = 323.
    assert result.per_category["sleep"]["sim"] == 1380
    assert result.per_category["paid_work"]["sim"] == 60
    assert result.per_category["paid_work"]["error"] == 323
