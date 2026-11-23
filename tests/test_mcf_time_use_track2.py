"""Phase 1.7 Track 2 regression tests (TASK-3439).

Pins the structured-field classification and contamination-segmentation
contract of ``benchmark/mcf/time_use.py`` against the verbatim fixtures from
the Track-1 log investigation (RESULT-3432, pilot16 agent 4 / agent 15):

- Fixture A: travel snapshot at departure (travel.status="departed").
- Fixture B: stationary 「处理待办」 errand — unclassifiable without an
  errand category; purpose-unknown stays unknown.
- Fixture C: travel snapshot mid-trip (taxi, in_transit, progress < 1).
- Fixture D: contaminated duplicate tick (two runs appended to one file).

All episodes are hand-written dicts in the REAL episode schema. No LLM
calls, no network, no simulation.
"""

from __future__ import annotations

import json

import pytest

from benchmark.mcf.time_use import (
    classify_activity,
    classify_episode,
    compute_daily_minutes,
    coverage,
    load_episodes,
    split_run_segments,
    travel_category,
)

# ---------------------------------------------------------------------------
# Verbatim fixtures (Track-1 report §7, quotes preserved)
# ---------------------------------------------------------------------------


def _fixture_a() -> dict[str, object]:
    """Agent 4, tick 00:00 — travel snapshot at departure (bus)."""
    return {
        "episode_id": "acc0afd9-6840-4677-80a5-a8c8f2d43dcd",
        "day": 1,
        "time": "00:00",
        "scheduled_activity": "个人时间",
        "final_activity": "前往绍兴国金大悦城",
        "action": "乘坐bus移动",
        "location": "Transit to 绍兴国金大悦城",
        "target_location": "绍兴国金大悦城",
        "travel": {
            "mode": "bus",
            "mode_reason": "alternative",
            "distance_km": 9.567,
            "minutes": 37,
            "progress": 0.2702702702702703,
            "route": ["联合村", "金庄村", "城南街道", "塔山街道", "府山街道", "绍兴国金大悦城"],
            "cost": 2.0,
            "rush_hour": False,
            "congestion": 1.0,
            "status": "departed",
        },
        "change_reason": "梅雨持续，空气潮湿——找地方避雨",
        "tags": ["routine"],
    }


def _fixture_b() -> dict[str, object]:
    """Agent 4, tick 13:20 — stationary 「处理待办」 errand."""
    return {
        "episode_id": "e755e1e1-45bf-4793-a7d3-2c85b8ca4117",
        "day": 1,
        "time": "13:20",
        "scheduled_activity": "办事逛街",
        "final_activity": "处理待办",
        "action": "按原节奏继续当前安排",
        "location": "供销超市黄贤店",
        "target_location": "供销超市黄贤店",
        "travel": {
            "mode": "",
            "distance_km": 0.0,
            "minutes": 0,
            "progress": 1.0,
            "route": ["供销超市黄贤店"],
            "congestion": 1.0,
            "status": "stationary",
        },
        "change_reason": "时间压力突然冒出来，想先把最急的事处理掉。(tp=0.94)",
        "tags": ["routine"],
    }


def _fixture_c() -> dict[str, object]:
    """Agent 15, tick 00:00 — travel snapshot mid-trip (taxi, in transit)."""
    return {
        "episode_id": "883bf32d-62c1-46aa-9706-2fdb6ce361ca",
        "day": 1,
        "time": "00:00",
        "scheduled_activity": "个人时间",
        "final_activity": "前往百福园社区党群服务中心",
        "action": "乘坐taxi移动",
        "location": "Transit to 百福园社区党群服务中心",
        "target_location": "百福园社区党群服务中心",
        "travel": {
            "mode": "taxi",
            "mode_reason": "alternative",
            "distance_km": 13.594,
            "minutes": 27,
            "progress": 0.37037037037037035,
            "route": ["低田畈村", "安昌街道", "华舍街道", "柯桥街道", "百福园社区党群服务中心"],
            "cost": 39.48,
            "rush_hour": False,
            "congestion": 1.0,
            "status": "departed",
        },
        "change_reason": "wealth_pursuit_income_seek",
        "tags": ["routine"],
    }


# ---------------------------------------------------------------------------
# 1. Classification — structured fields (fixtures A–C)
# ---------------------------------------------------------------------------


def test_fixture_a_departed_travel_is_travel():
    # Phase 2.0 §2B repin: departed status alone is purpose-unknown travel,
    # never commuting (supersedes the pre-reconciliation contract).
    category, confidence = classify_episode(_fixture_a())
    assert category == travel_category()
    assert confidence == "travel_status"


def test_fixture_c_in_transit_travel_is_travel():
    # Phase 2.0 §2B repin: in-transit status without a verified work purpose
    # is plain travel (supersedes the pre-reconciliation contract).
    category, confidence = classify_episode(_fixture_c())
    assert category == travel_category()
    assert confidence == "travel_status"


def test_fixture_b_stationary_todo_stays_unknown():
    # 「处理待办」 at a supermarket: the location hints at errands but the
    # purpose is not logged — conservative rule keeps it unknown. Its
    # scheduled_activity (办事逛街) matches no category keyword either.
    category, confidence = classify_episode(_fixture_b())
    assert category is None
    assert confidence == "unknown"


def test_stale_travel_label_after_arrival_is_not_commuting():
    # 「前往X」 persists past arrival (status arrived/stationary): the label
    # is stale, the agent is dwelling at X doing something unlogged — never
    # commuting, and no POI-type guessing from the destination name.
    ep = _fixture_a() | {
        "time": "08:30",
        "travel": dict(_fixture_a()["travel"], status="arrived", progress=1.0),  # type: ignore[arg-type]
        "action": "在商场里看看",
    }
    category, confidence = classify_episode(ep)
    assert category is None
    assert confidence == "unknown"


def test_arrived_tick_with_classifiable_scheduled_activity_uses_it_as_weak_evidence():
    ep = _fixture_a() | {
        "scheduled_activity": "上午工作",
        "travel": dict(_fixture_a()["travel"], status="arrived", progress=1.0),  # type: ignore[arg-type]
        "action": "把手里的事推进一下",
    }
    category, confidence = classify_episode(ep)
    assert category == "paid_work"
    assert confidence == "scheduled_keyword"


def test_action_travel_phrase_covers_missing_travel_block():
    # Same 「前往X」 snapshot without the travel dict: the writer's own
    # 乘坐…移动 action phrase evidences travel — Phase 2.0 §2B repin, it
    # proves travel, never a work purpose (supersedes the pre-
    # reconciliation contract that promoted it to commuting).
    ep = {k: v for k, v in _fixture_a().items() if k != "travel"}
    category, confidence = classify_episode(ep)
    assert category == travel_category()
    assert confidence == "action_keyword"


def test_plain_keyword_episodes_are_unchanged():
    # A dict with no structured fields classifies exactly as before
    # (backward compatibility with the locked Phase 0.7 contract).
    ep = {
        "agent_id": 1,
        "day": 1,
        "time": "08:00",
        "final_activity": "上班工作",
        "scheduled_activity": "工作",
    }
    assert classify_episode(ep) == ("paid_work", "keyword")
    assert classify_activity("上班工作") == ("paid_work", "keyword")


def test_no_sleep_is_ever_invented():
    # Nothing in the pilot logs evidences sleep; a travel-labelled tick must
    # not flip to sleep even at night, and unknown stays unknown.
    for fixture in (_fixture_a(), _fixture_b(), _fixture_c()):
        assert classify_episode(fixture)[0] != "sleep"


# ---------------------------------------------------------------------------
# 2. Duration attribution with structured categories (conservative, and the
#    existing gap rules — never stretched, unknown never dropped)
# ---------------------------------------------------------------------------


def test_travel_ticks_attribute_commuting_minutes_not_the_stale_label():
    # 00:00 departed (fixture A) -> 00:20 in transit -> 00:30 arrived.
    # In-transit ticks are plain travel (Phase 2.0 §2B repin —
    # purpose-unknown travel is never commuting); the arrived tick's
    # dwell is unknown.
    eps = [
        _fixture_a(),
        _fixture_a()
        | {"time": "00:20", "travel": dict(_fixture_a()["travel"], status="in_transit", progress=0.54)},  # type: ignore[arg-type]
        _fixture_a()
        | {
            "time": "00:30",
            "travel": dict(_fixture_a()["travel"], status="arrived", progress=1.0),
            "action": "在商场里看看",
        },  # type: ignore[arg-type]
        _fixture_a()
        | {
            "time": "00:50",
            "travel": dict(_fixture_a()["travel"], status="stationary", progress=1.0),
            "action": "在商场里看看",
        },  # type: ignore[arg-type]
    ]
    minutes = compute_daily_minutes(eps, tick_minutes=10)[0][1]
    # 00:00->00:20 same category: gap kept (20); 00:20->00:30 within 2*tick
    # (10); the arrived/stationary dwell from 00:30 on is unlogged unknown.
    assert minutes["travel"] == 30
    assert minutes["unknown"] == 1410  # dwell 00:30->00:50 + tail to 24:00
    assert minutes["commuting"] == 0
    assert sum(minutes.values()) == 1440


def test_large_gap_is_still_split_not_stretched_by_structured_fields():
    # The Phase 0.7 gap rules are untouched: a 60-min gap > 2*tick keeps only
    # 2*tick in the previous category, the remainder stays unknown. The 01:00
    # tick is a stationary 「处理待办」 (different, unknown category) so the
    # equal-category full-gap carry does not apply.
    eps = [
        _fixture_a(),
        _fixture_b() | {"time": "01:00"},
    ]
    minutes = compute_daily_minutes(eps, tick_minutes=10)[0][1]
    # Phase 2.0 §2B repin: the departed tick is purpose-unknown travel, so
    # the ≤2*tick gap carry lands in travel, not commuting.
    assert minutes["travel"] == 20
    assert minutes["unknown"] == 40 + 1440 - 60  # gap remainder + unobserved tail


# ---------------------------------------------------------------------------
# 3. Midnight boundaries and missing observations
# ---------------------------------------------------------------------------


def test_midnight_boundary_day_split_with_structured_fields():
    day1 = [_fixture_a(), _fixture_a() | {"time": "00:10"}]
    _dwell = dict(_fixture_a()["travel"], status="stationary", progress=1.0)  # type: ignore[arg-type]
    day2 = [
        _fixture_a()
        | {
            "agent_id": 0,
            "day": 2,
            "time": "07:30",
            "final_activity": "吃早饭",
            "scheduled_activity": "吃饭",
            "action": "吃早饭",
            "travel": _dwell,
        }
    ]
    result = compute_daily_minutes(day1 + day2, tick_minutes=10)
    # Day 1 ends early (00:10) on a non-sleep category: per the locked
    # early-stop rule the recorded activity persists to 24:00 — this is
    # exactly the stale-label persistence artifact on sparse logs.
    assert result[0][1]["travel"] == 1440  # Phase 2.0 §2B repin: travel, not commuting
    # Day 2: the head 00:00->07:30 carries into sleep (previous day recorded),
    # the 07:30 breakfast tick persists to 24:00 (early-stop rule again).
    assert result[0][2]["sleep"] == 450
    assert result[0][2]["meals"] == 990
    assert sum(result[0][1].values()) == 1440
    assert sum(result[0][2].values()) == 1440


def test_missing_observation_windows_land_in_unknown():
    # Ticks at 08:00 (paid_work) and 10:00 (leisure) with a 2-hour hole in
    # between: only 2*tick of the hole stays with paid_work, the rest is
    # unknown; the head 00:00->08:00 is unobserved (first day, no sleep
    # evidence -> unknown); the early 10:00 leisure tick persists to 24:00
    # (locked early-stop rule).
    eps = [
        _fixture_a()
        | {
            "time": "08:00",
            "final_activity": "上班工作",
            "scheduled_activity": "上午工作",
            "action": "推进工作",
            "travel": dict(_fixture_a()["travel"], status="stationary", progress=1.0),  # type: ignore[arg-type]
        },
        _fixture_a()
        | {
            "time": "10:00",
            "final_activity": "散步休闲",
            "scheduled_activity": "个人时间",
            "action": "出门走走",
            "travel": dict(_fixture_a()["travel"], status="stationary", progress=1.0),  # type: ignore[arg-type]
        },
    ]
    minutes = compute_daily_minutes(eps, tick_minutes=10)[0][1]
    assert minutes["paid_work"] == 20
    assert minutes["unknown"] == 480 + 100
    assert minutes["leisure"] == 840
    assert sum(minutes.values()) == 1440


def test_head_of_first_day_is_unknown_without_sleep_evidence():
    # The pilot logs open at 00:00 with travel, and show no sleep anywhere —
    # the first day's head must be unknown (never assumed sleep).
    eps = [_fixture_a() | {"time": "06:00"}]
    minutes = compute_daily_minutes(eps, tick_minutes=10)[0][1]
    assert minutes["sleep"] == 0
    assert minutes["unknown"] == 360
    assert minutes["travel"] == 1440 - 360  # Phase 2.0 §2B repin: travel, not commuting


# ---------------------------------------------------------------------------
# 4. Duplicate ticks and out-of-order records
# ---------------------------------------------------------------------------


def test_duplicate_tick_does_not_double_count_minutes():
    # Two records at the same tick (fixture D content, both travel here):
    # differencing must not produce negative or double-counted minutes. The
    # single 00:00 stamp persists to 24:00 under the locked early-stop rule
    # (travel 1440 — Phase 2.0 §2B repin: purpose-unknown travel, not
    # commuting), and duplicates must not push it past that.
    dup_a = _fixture_a()
    dup_b = _fixture_c()  # same tick 00:00, different destination
    minutes = compute_daily_minutes([dup_a, dup_b], tick_minutes=10)[0][1]
    assert minutes["travel"] == 1440
    assert minutes["commuting"] == 0
    assert sum(minutes.values()) == 1440
    # Deterministic across calls.
    assert compute_daily_minutes([dup_a, dup_b], tick_minutes=10) == compute_daily_minutes(
        [dup_a, dup_b], tick_minutes=10
    )


def test_small_out_of_order_jitter_still_clamps_to_zero():
    eps = [
        _fixture_a() | {"time": "08:00", "final_activity": "上班工作", "scheduled_activity": "上午工作"},
        _fixture_a() | {"time": "07:30", "final_activity": "上班工作", "scheduled_activity": "上午工作"},
    ]
    minutes = compute_daily_minutes(eps, tick_minutes=10)[0][1]
    assert all(v >= 0 for v in minutes.values())


# ---------------------------------------------------------------------------
# 5. Contamination segmentation (fixture D scenario: two runs in one file)
# ---------------------------------------------------------------------------


def _contaminated_stream() -> list[dict[str, object]]:
    """Dry-run ticks 00:00 -> 12:20, then a live run restarting at 00:00."""
    stream: list[dict[str, object]] = []
    for minute in range(0, 12 * 60 + 20 + 1, 10):
        hh, mm = divmod(minute, 60)
        stream.append(_fixture_a() | {"time": f"{hh:02d}:{mm:02d}"})
    for minute in range(0, 21 * 60 + 40 + 1, 10):
        hh, mm = divmod(minute, 60)
        stream.append(_fixture_b() | {"time": f"{hh:02d}:{mm:02d}"})
    return stream


def test_split_run_segments_detects_clock_reset():
    segments = split_run_segments(_contaminated_stream())
    assert len(segments) == 2
    assert len(segments[0]) == 75  # 00:00..12:20 inclusive
    assert segments[0][0].time == "00:00"
    assert segments[0][-1].time == "12:20"
    assert len(segments[1]) == 131  # 00:00..21:40 inclusive
    assert segments[1][0].time == "00:00"


def test_clean_stream_is_one_segment():
    eps = [_fixture_a() | {"time": f"{h:02d}:00"} for h in range(8)]
    assert len(split_run_segments(eps)) == 1


def test_small_backward_jitter_does_not_split():
    eps = [
        _fixture_a() | {"time": "10:00"},
        _fixture_a() | {"time": "09:50"},  # 10-min write jitter
        _fixture_a() | {"time": "10:30"},
    ]
    assert len(split_run_segments(eps)) == 1


def test_segments_are_scored_separately_never_merged():
    stream = _contaminated_stream()
    segments = split_run_segments(stream)
    per_segment = [compute_daily_minutes(seg, tick_minutes=10) for seg in segments]
    # Each segment accounts for its own full days; no cross-run bleed.
    for result in per_segment:
        for days in result.values():
            for minutes in days.values():
                assert sum(minutes.values()) == 1440
    # Segments keep different content: segment 0 is all travel ticks
    # (Phase 2.0 §2B repin: purpose-unknown travel, not commuting).
    assert per_segment[0][0][1]["travel"] > 0
    assert per_segment[0][0][1]["commuting"] == 0
    assert per_segment[1][0][1]["commuting"] == 0


def test_load_episodes_reads_structured_fields(tmp_path):
    path = tmp_path / "agent_4_episodes.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(_fixture_a(), ensure_ascii=False) + "\n")
        fh.write(json.dumps(_fixture_b(), ensure_ascii=False) + "\n")
    episodes = load_episodes(path)
    assert episodes[0].travel is not None and episodes[0].travel["status"] == "departed"  # type: ignore[index]
    assert episodes[0].action == "乘坐bus移动"
    assert episodes[0].target_location == "绍兴国金大悦城"
    assert episodes[1].location == "供销超市黄贤店"
    assert episodes[1].travel is not None and episodes[1].travel["status"] == "stationary"  # type: ignore[index]


# ---------------------------------------------------------------------------
# 6. Full 1440-minute accounting
# ---------------------------------------------------------------------------


def test_every_day_sums_to_exactly_1440_minutes():
    stream = _contaminated_stream()
    for segment in split_run_segments(stream):
        result = compute_daily_minutes(segment, tick_minutes=10)
        for days in result.values():
            for minutes in days.values():
                assert sum(minutes.values()) == 1440


def test_unknown_time_never_silently_dropped_with_structured_rules():
    # A pilot-realistic day: mostly stale 前往X arrivals and 处理待办 —
    # nearly everything must land in unknown, and nothing may vanish.
    eps = []
    for minute in range(0, 24 * 60, 10):
        hh, mm = divmod(minute, 60)
        eps.append(_fixture_b() | {"agent_id": 0, "time": f"{hh:02d}:{mm:02d}"})
    result = compute_daily_minutes(eps, tick_minutes=10)
    minutes = result[0][1]
    assert minutes["unknown"] == 1440
    assert coverage(result) == pytest.approx(0.0)
