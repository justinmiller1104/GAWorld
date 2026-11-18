"""Phase 2.0 Part C — RED tests: destination_purpose must reach the REAL
activity logger's classification path (offline, stub harness evidence).

bbfa70f state: the move path derives ``destination_purpose``
deterministically (gaworld/sim/_location.py:59, emitted at
gaworld/sim/_location.py:462 and :486) and the real episode logger writes
it NESTED INSIDE the ``travel`` dict. Evidence from the zero-inference
dry run of record (output/cities/绍兴柯桥/memory/agent_4_episodes.jsonl,
132 episodes): ``destination_purpose`` present nested in ``travel`` for
124/132 episodes; present at TOP LEVEL for 0/132.

But the consumer, ``classify_activity_record``
(benchmark/mcf/time_use.py:1061), reads ``record.get("destination_purpose")``
— TOP LEVEL ONLY. So on real logger output:

* a verified work journey (travel.destination_purpose == "paid_work")
  NEVER promotes to commuting — it degrades to plain ``travel``;
* verified commuting is unreachable end-to-end, even when the log states
  the purpose machine-readably.

Contract pinned here:

* the classifier must discover ``destination_purpose`` wherever the real
  logger writes it (nested in ``travel`` AND top level);
* verified work purpose (paid_work) + in transit -> commuting, evidence
  "destination_purpose";
* personal / leisure purpose + in transit -> travel, NEVER commuting;
* unknown purpose + in transit -> travel;
* commuting is never inferred from employment status or destination
  alone (a workplace-shaped target_location and an employed agent prove
  nothing without the structured purpose).

Deterministic only: no LLM, no network.
"""

from __future__ import annotations

from benchmark.mcf.time_use import travel_category, classify_activity_record

# Real episode shape from the dry run of record (agent 15, 00:10 — a work
# trip with a machine-readable purpose nested in travel).
_NESTED_WORK_PURPOSE: dict = {
    "agent_id": 15,
    "day": 1,
    "time": "00:10",
    "scheduled_activity": "上午工作",
    "final_activity": "前往百福园社区党群服务中心",
    "action": "乘坐taxi移动",
    "location": "Transit to 百福园社区党群服务中心",
    "target_location": "百福园社区党群服务中心",
    "travel": {
        "mode": "taxi",
        "distance_km": 2.1,
        "minutes": 8,
        "progress": 0.5,
        "status": "in_transit",
        "destination_purpose": "paid_work",
    },
    # NOTE: NO top-level destination_purpose — this is what the real
    # episode logger actually emits (0/132 top-level in the dry run).
}

_NESTED_PERSONAL_PURPOSE: dict = dict(
    _NESTED_WORK_PURPOSE,
    time="06:20",
    final_activity="前往府山公园",
    target_location="府山公园",
    travel=dict(_NESTED_WORK_PURPOSE["travel"], destination_purpose="leisure"),
)

_NESTED_UNKNOWN_PURPOSE: dict = dict(
    _NESTED_WORK_PURPOSE,
    travel={k: v for k, v in _NESTED_WORK_PURPOSE["travel"].items() if k != "destination_purpose"},
)

# Commuting must never be inferred from employment status or destination
# alone: a workplace-shaped destination and an employed agent, no purpose.
_DESTINATION_ALONE: dict = dict(
    _NESTED_UNKNOWN_PURPOSE,
    target_location="兰亭镇人民政府",  # the agent's workplace-shaped POI
    employed=True,
)


def test_classifier_reads_purpose_nested_in_travel_dict():
    category, source = classify_activity_record(_NESTED_WORK_PURPOSE)
    assert category == "commuting" and source == "destination_purpose", (
        f"verified paid_work purpose (nested in travel, the real logger's "
        f"placement) classified {category!r}/{source!r}: classify_activity_record "
        "(time_use.py:1061) reads only a TOP-LEVEL destination_purpose, so "
        "verified commuting is unreachable on real logger output "
        "(0/132 episodes carry it top-level in the dry run of record)"
    )


def test_personal_purpose_is_never_commuting():
    category, source = classify_activity_record(_NESTED_PERSONAL_PURPOSE)
    assert category == travel_category() and source == "travel_status", (
        f"leisure-purpose journey classified {category!r}/{source!r}: a "
        "non-work purpose must never promote to commuting"
    )


def test_unknown_purpose_is_never_commuting():
    category, _ = classify_activity_record(_NESTED_UNKNOWN_PURPOSE)
    assert category == travel_category(), (
        f"purpose-unknown journey classified {category!r}: stays plain travel"
    )


def test_commuting_never_inferred_from_destination_or_employment_alone():
    category, source = classify_activity_record(_DESTINATION_ALONE)
    assert category != "commuting", (
        f"{category!r}/{source!r}: a workplace-shaped destination plus an "
        "employed agent is not work-purpose evidence — only the structured "
        "destination_purpose may promote to commuting"
    )
