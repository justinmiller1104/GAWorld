"""TDD RED tests — Phase 1.8 Part B.2/B.3: structured activity records
and travel-vs-commuting classification.

Contract (TASK-3453-SPEC, Contract 2):
- Machine-readable activity records with fields run_id, agent_id, category,
  start_min, end_min, evidence_source; an explicit 'unknown' category is
  allowed; no fabricated sleep/work/leisure.
- Overlapping activity spans are not double-counted (merged / deduplicated
  deterministically).
- travel.status == in_transit is classified as travel, NOT commuting;
  commuting only with a verified work purpose; unknown-purpose travel is
  counted as travel/unknown and excluded from commuting in survey
  comparisons.
- When authoritative structured tick-level state exists it takes precedence
  over keyword-inferred categories.

The new API imported below does NOT exist yet — the ImportError is the RED
state by design. No LLM calls, no network, no simulation.
"""

from __future__ import annotations

from benchmark.mcf.time_use import (
    ActivityRecord,
    dedupe_activity_records,
    travel_category,
)

# ---------------------------------------------------------------------------
# 1. ActivityRecord schema
# ---------------------------------------------------------------------------


def test_activity_record_has_the_contract_fields():
    record = ActivityRecord(
        run_id="r1",
        agent_id=4,
        category="sleep",
        start_min=0,
        end_min=450,
        evidence_source="schedule_slot",
    )
    assert record.run_id == "r1"
    assert record.agent_id == 4
    assert record.category == "sleep"
    assert record.start_min == 0
    assert record.end_min == 450
    assert record.evidence_source == "schedule_slot"


def test_activity_record_allows_explicit_unknown_category():
    record = ActivityRecord(
        run_id="r1",
        agent_id=4,
        category="unknown",
        start_min=600,
        end_min=660,
        evidence_source="unclassifiable",
    )
    assert record.category == "unknown"


# ---------------------------------------------------------------------------
# 2. Overlap handling: no double counting
# ---------------------------------------------------------------------------


def _rec(agent_id: int, category: str, start: int, end: int, source: str = "schedule_slot") -> ActivityRecord:
    return ActivityRecord(
        run_id="r1",
        agent_id=agent_id,
        category=category,
        start_min=start,
        end_min=end,
        evidence_source=source,
    )


def test_identical_overlapping_spans_are_deduplicated():
    records = [
        _rec(4, "sleep", 0, 450),
        _rec(4, "sleep", 0, 450),
        _rec(4, "sleep", 0, 450),
    ]
    merged = dedupe_activity_records(records)
    total = sum(r.end_min - r.start_min for r in merged)
    assert total == 450, f"three identical 450-min spans must merge to 450 min (got {total})"


def test_partially_overlapping_spans_are_not_double_counted():
    records = [
        _rec(4, "leisure", 720, 800),
        _rec(4, "leisure", 780, 860),  # overlaps 780..800
    ]
    merged = dedupe_activity_records(records)
    total = sum(r.end_min - r.start_min for r in merged)
    assert total == 140, f"overlapping leisure spans must union to 140 min (got {total})"


def test_merge_is_deterministic():
    records = [_rec(4, "sleep", 0, 450), _rec(4, "meals", 450, 550), _rec(4, "sleep", 0, 450)]
    first = dedupe_activity_records(list(reversed(records)))
    second = dedupe_activity_records(records)
    assert [(r.category, r.start_min, r.end_min) for r in first] == [
        (r.category, r.start_min, r.end_min) for r in second
    ], "merging must be deterministic regardless of input order"


# ---------------------------------------------------------------------------
# 3. travel vs commuting
# ---------------------------------------------------------------------------


def test_travel_is_a_distinct_category_from_commuting():
    assert travel_category() == "travel"
    assert travel_category() != "commuting"


def test_in_transit_travel_is_classified_as_travel_not_commuting():
    from benchmark.mcf.time_use import classify_activity_record

    record = {
        "run_id": "r1",
        "agent_id": 4,
        "start_min": 480,
        "end_min": 540,
        "travel": {"status": "in_transit", "mode": "taxi", "progress": 0.5},
        "scheduled_activity": "前往公司",
    }
    category, _ = classify_activity_record(record)
    assert category == "travel", f"in_transit travel must be 'travel', got {category!r}"


def test_commuting_requires_verified_work_purpose():
    from benchmark.mcf.time_use import classify_activity_record

    # Work-purpose travel: the destination schedule slot is paid_work.
    work_travel = {
        "run_id": "r1",
        "agent_id": 4,
        "start_min": 480,
        "end_min": 540,
        "travel": {"status": "in_transit", "mode": "bus", "progress": 0.3},
        "scheduled_activity": "前往公司",
        "destination_purpose": "paid_work",
    }
    category, source = classify_activity_record(work_travel)
    assert category == "commuting", "travel with a verified work purpose is commuting"
    assert source != "keyword", "the verified purpose must come from structured evidence"

    # The same in-transit trip without a verified purpose is plain travel.
    purposeless = dict(work_travel)
    purposeless.pop("destination_purpose")
    category, _ = classify_activity_record(purposeless)
    assert category == "travel", "unknown-purpose travel must stay travel, never commuting"


def test_unknown_purpose_travel_is_excluded_from_commuting_in_survey_comparison():
    """Survey-comparison aggregation never maps travel into the commuting category."""
    from benchmark.mcf.time_use import survey_comparison_categories

    categories = survey_comparison_categories()
    assert "travel" in categories, "travel must be a first-class survey-comparison bucket"
    assert "commuting" in categories
    assert "travel" not in categories.get("commuting_aliases", ["commuting"]), (
        "travel/unknown must be excluded from commuting in survey comparisons"
    )


# ---------------------------------------------------------------------------
# 4. Structured state precedence over keywords
# ---------------------------------------------------------------------------


def test_structured_tick_state_takes_precedence_over_keywords():
    from benchmark.mcf.time_use import classify_activity_record

    # The keyword says leisure, but the authoritative structured slot says study.
    record = {
        "run_id": "r1",
        "agent_id": 15,
        "start_min": 900,
        "end_min": 960,
        "scheduled_activity": "在家休闲，看电视",  # keyword -> leisure
        "activity_category": "study",  # authoritative structured field
    }
    category, source = classify_activity_record(record)
    assert category == "study", f"structured state must win over keywords (got {category!r})"
    assert source == "structured", f"the winning evidence must be marked structured (got {source!r})"


def test_no_fabricated_categories_from_empty_records():
    from benchmark.mcf.time_use import classify_activity_record

    empty = {"run_id": "r1", "agent_id": 4, "start_min": 600, "end_min": 660}
    category, _source = classify_activity_record(empty)
    assert category == "unknown", "nothing classifiable means unknown, never invented sleep/work/leisure"
