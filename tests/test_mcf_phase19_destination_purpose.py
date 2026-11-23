"""Phase 1.9 — RED tests: destination_purpose logging on the move path.

PILOT19 (PILOT19-REPORT.md §5, §8.3): the move path logs ``travel`` blocks
(status/mode/progress/…) but NEVER a ``destination_purpose``, so verified
commuting is permanently unestimable — 172 purpose-unknown travel records,
0 verified commuting. The classification side already exists
(``benchmark/mcf/time_use.py`` ``classify_activity_record`` promotes an
in-transit record to ``commuting`` only on a structured purpose), but:

* the sim's move path (``gaworld/sim/_location.py`` ``move_agent`` — the
  function whose ``travel`` dict is what episodes and records carry) does
  not produce the field at all, and
* the classifier currently promotes ANY snake_case purpose to commuting,
  even a leisure one — commuting must be marked only when the purpose is
  work-related.

Contract pinned here:

* Records produced by the move path (both the fresh-departure and the
  in-transit branches of ``move_agent``) carry a structured
  ``destination_purpose`` field, derived from the agent's stated
  scheduled activity for that movement (the ``activity`` argument) — no
  extra LLM inference, no network.
* Absent a determinable purpose the field is explicitly ``None`` /
  unknown — the key is always present, never guessed.
* Purposes are snake_case category names the classifier accepts, and the
  classifier marks verified commuting only for work-related purposes.

Deterministic only: no LLM, no network, no simulation run.
"""

from __future__ import annotations

import os
import tempfile

from gaworld.sim._location import move_agent
from gaworld.world import city_map as cm

WORK_PURPOSE = "paid_work"
NON_WORK_PURPOSES = ("leisure", "socializing", "personal_care")


def _mini_map():
    """Three nodes in a line: A -arterial- B -local- C."""
    content = (
        "# City Map\n"
        "@node: Alpha | kind=hub | category=residential | x=0.0 | y=0.0 | capacity=100\n"
        "@node: Beta | kind=hub | category=commerce | x=3.0 | y=0.0 | capacity=100\n"
        "@node: Gamma | kind=hub | category=industry | x=6.0 | y=0.0 | capacity=100\n"
        "@road: Alpha -> Beta | type=arterial\n"
        "@road: Beta -> Gamma | type=local\n"
        "\n- City: Demo\n  - Hub: Alpha\n  - Hub: Beta\n  - Hub: Gamma\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return cm.load_city_map(path)


def _agent(home: str = "Alpha") -> dict:
    return {"id": 4, "home": home, "locations": {"current": home, "home": home}}


# ---------------------------------------------------------------------------
# 1. The move path produces destination_purpose.
# ---------------------------------------------------------------------------


def test_fresh_departure_travel_dict_carries_destination_purpose():
    agent = _agent()
    movement = move_agent(agent, "Gamma", "工作", "08:00", 1, _mini_map())
    travel = movement["travel"]
    assert travel.get("status") in {"departed", "in_transit"}, (
        f"fixture expected a real trip, got status {travel.get('status')!r}"
    )
    assert "destination_purpose" in travel, (
        f"move-path travel dict has no destination_purpose field: keys={sorted(travel)}"
    )


def test_purpose_is_derived_from_the_scheduled_activity():
    agent = _agent()
    movement = move_agent(agent, "Gamma", "工作", "08:00", 1, _mini_map())
    purpose = movement["travel"]["destination_purpose"]
    assert purpose == WORK_PURPOSE, (
        f"a 工作 movement must log the work purpose {WORK_PURPOSE!r}, got {purpose!r}"
    )


def test_purpose_is_snake_case_category_the_classifier_accepts():
    import re

    from benchmark.mcf.time_use import CATEGORIES

    agent = _agent()
    movement = move_agent(agent, "Gamma", "上班", "08:00", 1, _mini_map())
    purpose = movement["travel"]["destination_purpose"]
    assert re.match(r"^[a-z_]+$", str(purpose)), f"purpose {purpose!r} is not a snake_case category token"
    assert purpose in CATEGORIES, f"purpose {purpose!r} is not a known activity category"


def test_in_transit_records_carry_the_field_too():
    # The in-transit branch (a second call while a trip is under way) is
    # what PILOT19's 172 in_transit snapshots came from — it must carry the
    # purpose logged at departure, not drop it.
    agent = _agent()
    move_agent(agent, "Gamma", "工作", "08:00", 1, _mini_map())
    movement = move_agent(agent, "Gamma", "工作", "08:05", 1, _mini_map())
    travel = movement["travel"]
    assert travel.get("status") == "in_transit", (
        f"fixture expected an in_transit trip, got {travel.get('status')!r}"
    )
    assert "destination_purpose" in travel, (
        f"in-transit travel dict dropped destination_purpose: keys={sorted(travel)}"
    )
    assert travel["destination_purpose"] == WORK_PURPOSE, (
        f"in-transit purpose must match the departure purpose, got {travel['destination_purpose']!r}"
    )


# ---------------------------------------------------------------------------
# 2. Absent a determinable purpose: explicitly null, never guessed.
# ---------------------------------------------------------------------------


def test_undeterminable_purpose_is_explicit_null_not_guessed():
    # A movement whose activity names no purpose must still carry the field,
    # explicitly None — no keyword guessing, no silent omission.
    agent = _agent()
    movement = move_agent(agent, "Beta", "随便走走", "10:00", 1, _mini_map())
    travel = movement["travel"]
    assert travel.get("status") in {"departed", "in_transit"}, (
        f"fixture expected a real trip, got status {travel.get('status')!r}"
    )
    assert "destination_purpose" in travel, (
        f"field missing entirely instead of explicit null: keys={sorted(travel)}"
    )
    assert travel["destination_purpose"] is None, (
        f"undeterminable purpose must be None, got {travel['destination_purpose']!r} "
        "(guessed purposes are forbidden)"
    )


# ---------------------------------------------------------------------------
# 3. Classification: verified commuting ONLY for work-related purposes.
# ---------------------------------------------------------------------------


def _in_transit_record(purpose) -> dict:
    return {
        "run_id": "r1",
        "agent_id": 4,
        "start_min": 480,
        "end_min": 540,
        "travel": {"status": "in_transit", "mode": "bus", "progress": 0.3},
        "scheduled_activity": "前往公司",
        "destination_purpose": purpose,
    }


def test_work_purpose_in_transit_is_commuting():
    from benchmark.mcf.time_use import classify_activity_record

    category, source = classify_activity_record(_in_transit_record(WORK_PURPOSE))
    assert category == "commuting", "work-purpose in-transit travel is verified commuting"
    assert source == "destination_purpose", f"evidence must name the purpose, got {source!r}"


def test_non_work_purpose_in_transit_is_travel_not_commuting():
    from benchmark.mcf.time_use import classify_activity_record

    for purpose in NON_WORK_PURPOSES:
        category, _ = classify_activity_record(_in_transit_record(purpose))
        assert category == "travel", (
            f"purpose {purpose!r} is not work-related; in-transit travel must stay 'travel', got {category!r}"
        )


def test_null_purpose_in_transit_is_travel_not_commuting():
    from benchmark.mcf.time_use import classify_activity_record

    category, _ = classify_activity_record(_in_transit_record(None))
    assert category == "travel", f"explicit-null purpose must never promote to commuting, got {category!r}"
