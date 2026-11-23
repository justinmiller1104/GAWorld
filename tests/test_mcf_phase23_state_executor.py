"""Phase 2.3 RED tests — the set_agent_state policy_events executor (TASK-3558).

RESULT-3556 limitation 1: ``policy_events`` entries only feed description text
to the intervention feed (``gaworld/policy/plugin.py``); the state overwrite
never happens. Phase 2.3 (§3 / §8.1 of ``docs/MCF_GAWORLD.md``) authorizes
building exactly this missing integration: a declarative ``policy_events``
entry carrying an ``operation`` must translate into an ACTUAL supported
simulation operation at the matched tick, via the existing audited kernel path
(``Controller.intervene("set_agent_state", ...)``).

These tests pin that contract BEFORE the executor exists. Deterministic
pure-Python tests on a hand-built kernel context — no LLM, no network, no
full simulation.

Natural home of the implementation: ``gaworld/kernel/policy_events.py`` —
``execute_policy_events(ctx, events, day, time_str)`` is called by the tick
loop (which already matches entries per day/time at generative_city_sim.py:3354)
with ``events`` from ``CONFIG["policy_events"]``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gaworld.kernel import build_kernel
from gaworld.kernel.policy_events import execute_policy_events

# ---------------------------------------------------------------------------
# Fixtures — a deterministic kernel context with two agents
# ---------------------------------------------------------------------------

EVENT = {
    "day": 1,
    "time": "14:00",
    "name": "mood boost",
    "description": "平台发放消费补贴，居民情绪提振。",
    "operation": "set_agent_state",
    "targets": [4],
    "kwargs": {"emotion": 0.95},
}


def _make_ctx(tmp_path: Path):
    """A minimal kernel context: agents 4 and 15 with known initial states."""
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    ctx = build_kernel({"records": {"output_dir": str(tmp_path / "records")}}, load_entry_points=False)
    ctx.set_agents(
        [
            {
                "id": 4,
                "state": {"emotion": 0.42, "stress": 0.51, "policy_sensitivity": 0.53},
            },
            {
                "id": 15,
                "state": {"emotion": 0.31, "stress": 0.60, "policy_sensitivity": 0.52},
            },
        ]
    )
    yield ctx
    ctx.recorder.close()
    os.chdir(old_cwd)


@pytest.fixture
def ctx(tmp_path):
    yield from _make_ctx(tmp_path)


def _records(tmp_path: Path):
    """Read back every record the recorder wrote (all tables)."""
    rows = []
    rec_dir = tmp_path / "records"
    if not rec_dir.is_dir():
        return rows
    for fh in sorted(rec_dir.glob("*.jsonl")):
        for line in fh.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Contract 1 — the matched tick OVERWRITES the targeted agent's state
# ---------------------------------------------------------------------------


def test_matched_tick_overwrites_target_state(ctx, tmp_path):
    """At day=1 14:00 the entry fires: agent 4's emotion becomes exactly 0.95."""
    execute_policy_events(ctx, [dict(EVENT)], day=1, time_str="14:00")
    assert ctx.agents_by_id[4]["state"]["emotion"] == pytest.approx(0.95), (
        "policy_events entry with operation=set_agent_state must overwrite the "
        "targeted agent's state field at the trigger tick"
    )


def test_before_differs_from_after(ctx, tmp_path):
    """The write is a real overwrite: 0.42 -> 0.95, not a no-op."""
    before = ctx.agents_by_id[4]["state"]["emotion"]
    assert before != 0.95
    execute_policy_events(ctx, [dict(EVENT)], day=1, time_str="14:00")
    assert ctx.agents_by_id[4]["state"]["emotion"] == pytest.approx(0.95)
    assert before == pytest.approx(0.42)


def test_unmatched_tick_is_inert(ctx, tmp_path):
    """The same entry at any other tick must not touch any state."""
    for day, time_str in ((1, "13:45"), (1, "14:15"), (2, "14:00")):
        execute_policy_events(ctx, [dict(EVENT)], day=day, time_str=time_str)
    assert ctx.agents_by_id[4]["state"]["emotion"] == pytest.approx(0.42)
    assert ctx.agents_by_id[15]["state"]["emotion"] == pytest.approx(0.31)


# ---------------------------------------------------------------------------
# Contract 2 — collateral: only the targeted field of the targeted agent moves
# ---------------------------------------------------------------------------


def test_only_targeted_field_and_agent_change(ctx, tmp_path):
    execute_policy_events(ctx, [dict(EVENT)], day=1, time_str="14:00")
    agent4 = ctx.agents_by_id[4]["state"]
    agent15 = ctx.agents_by_id[15]["state"]
    # Untargeted fields of the targeted agent untouched.
    assert agent4["stress"] == pytest.approx(0.51)
    assert agent4["policy_sensitivity"] == pytest.approx(0.53)
    # Untargeted agent untouched entirely.
    assert agent15 == {"emotion": 0.31, "stress": 0.60, "policy_sensitivity": 0.52}


# ---------------------------------------------------------------------------
# Contract 3 — audit record with true before/after from the actual state dict
# ---------------------------------------------------------------------------


def test_audit_record_carries_before_after(ctx, tmp_path):
    """Every fired entry leaves an audited record whose before/after come from
    the actual state dict (before=0.42, after=0.95), via Controller.intervene."""
    execute_policy_events(ctx, [dict(EVENT)], day=1, time_str="14:00")
    rows = _records(tmp_path)
    # The kernel intervention itself is audited (existing K5 contract).
    kernel_rows = [r for r in rows if r.get("name") == "set_agent_state"]
    assert kernel_rows, "Controller.intervene('set_agent_state') must be audited"
    # And the execution is auditable with the true before/after values.
    audit = [r for r in rows if "before" in r and "after" in r]
    assert audit, (
        "the executor must produce an audit record with before/after taken from the agent's actual state dict"
    )
    rec = audit[0]
    assert rec["before"] == pytest.approx(0.42), rec
    assert rec["after"] == pytest.approx(0.95), rec


# ---------------------------------------------------------------------------
# Contract 4 — unknown operations are still rejected; no silent no-op
# ---------------------------------------------------------------------------


def test_unknown_operation_rejected(ctx, tmp_path):
    bad = dict(EVENT, operation="teleport_agent")
    with pytest.raises(Exception):
        execute_policy_events(ctx, [bad], day=1, time_str="14:00")
    assert ctx.agents_by_id[4]["state"]["emotion"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Contract 5 — entries without an operation keep the legacy text-feed behaviour
# ---------------------------------------------------------------------------


def test_entry_without_operation_stays_text_feed(ctx, tmp_path):
    """Backward compatible: a plain description-only event executes nothing."""
    legacy = {
        "day": 1,
        "time": "14:00",
        "name": "Platform worker protection policy",
        "description": "Increase social security coverage and wage transparency.",
    }
    # Must not raise and must not touch state.
    execute_policy_events(ctx, [legacy], day=1, time_str="14:00")
    assert ctx.agents_by_id[4]["state"]["emotion"] == pytest.approx(0.42)
    assert ctx.agents_by_id[15]["state"]["emotion"] == pytest.approx(0.31)
    assert not [r for r in _records(tmp_path) if "before" in r and "after" in r]
