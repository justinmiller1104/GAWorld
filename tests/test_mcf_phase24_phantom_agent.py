"""Phase 2.4 RED test — phantom "agent 0" in MCF scoring.

Reproduced defect (task 3568, demo dir
``~/work/crew/test/OUTBOX/phase23-demo2/exec1/baseline/20261112_141423_3c6041ff/``):
``score_directory`` reported a cohort of **3** agents — 4, 15 and a phantom
``agent 0`` with 1440 unattributed minutes — although only agents 4 and 15
exist.

Root cause (pinned, ``benchmark/mcf/time_use.py``):
- ``score_directory`` globs ``*.jsonl`` indiscriminately, so non-episode
  files such as ``intervention_executions.jsonl`` in a run dir are scored
  as episodes;
- ``load_episodes`` (line ~413) falls back to ``fallback_id = 0`` when the
  filename has no ``agent_<N>`` match, and rows without an ``agent_id``
  key (line ~421 ``row.get("agent_id", fallback_id)``) inherit it —
  every row of ``intervention_executions.jsonl`` becomes "agent 0";
- ``score_against_targets`` then counts agent 0 in the cohort (warning:
  "cohort of 3 agent(s)") and reports its fabricated day (1440
  unattributed minutes) as if it were a real resident.

Fix contract pinned here:
1. A scored cohort must contain only agents with real identifiers — no
   agent 0 may appear in ``per_agent_day``, and the cohort-size warning
   must not count it.
2. Records that cannot be attributed to an agent must be reported
   explicitly in ``uncertainty_warnings`` (an unattributed-records
   count), never silently bucketed as a fake agent.

Deterministic, zero-inference, network-free (synthetic JSONL in tmp_path).
"""

from __future__ import annotations

import json

from benchmark.mcf.time_use import score_directory


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _episode(agent_id, day, time, activity):
    return {
        "agent_id": agent_id,
        "day": day,
        "time": time,
        "final_activity": activity,
        "scheduled_activity": activity,
        "action": activity,
        "location": "联合村",
        "target_location": "联合村",
        "travel": {"mode": "", "minutes": 0, "status": "stationary"},
    }


def _run_dir(tmp_path):
    """A run dir shaped like a Phase 2.3 demo run: one episode file per
    agent plus an ``intervention_executions.jsonl`` audit file whose rows
    carry no ``agent_id`` key."""
    d = tmp_path / "run1"
    d.mkdir()
    _write_jsonl(
        d / "agent_4.jsonl",
        [
            _episode(4, 1, "00:00", "睡觉"),
            _episode(4, 1, "12:00", "吃午饭"),
            _episode(4, 1, "13:00", "上班工作"),
        ],
    )
    _write_jsonl(
        d / "agent_15.jsonl",
        [
            _episode(15, 1, "00:00", "睡觉"),
            _episode(15, 1, "12:00", "吃午饭"),
            _episode(15, 1, "13:00", "上班工作"),
        ],
    )
    _write_jsonl(
        d / "intervention_executions.jsonl",
        [
            {
                "name": "trigger_tick_episode_snapshot",
                "day": 1,
                "time": "14:00",
                "targets": [4],
                "applied": False,
                "before": {"emotion": 0.32},
            },
            {
                "name": "trigger_tick_episode_snapshot",
                "day": 1,
                "time": "14:00",
                "targets": [15],
                "applied": False,
                "before": {"emotion": 0.31},
            },
        ],
    )
    return d


def test_score_directory_never_reports_phantom_agent_0(tmp_path):
    """The cohort must contain only agents with real identifiers."""
    result = score_directory(_run_dir(tmp_path), employed_agent_ids=[4, 15])
    per_agent_day = result.per_agent_day or {}
    assert 0 not in per_agent_day, (
        "phantom agent 0 in the cohort: records without a real agent "
        f"identifier were bucketed as agent 0 (got agents {sorted(per_agent_day)})"
    )
    assert sorted(per_agent_day) == [4, 15], (
        f"cohort must be exactly the agents with real identifiers, got {sorted(per_agent_day)}"
    )


def test_unattributable_records_reported_in_warnings_not_as_an_agent(tmp_path):
    """Records that cannot be attributed to an agent must surface as an
    explicit unattributed count in warnings — never as a fake agent."""
    result = score_directory(_run_dir(tmp_path), employed_agent_ids=[4, 15])
    warnings = result.uncertainty_warnings or []
    joined = "\n".join(warnings)
    assert "unattributed" in joined.lower(), (
        "records dropped for lacking a real agent identifier must be reported "
        f"explicitly (e.g. an unattributed-records count); warnings were: {joined!r}"
    )
    # no warning may name agent 0 as if it were a real cohort member
    for warning in warnings:
        assert "agent 0" not in warning.lower(), (
            f"warning implies agent 0 is a real cohort member: {warning!r}"
        )


def test_cohort_size_warning_counts_only_real_agents(tmp_path):
    """The population-gate warning must count real agents only."""
    result = score_directory(_run_dir(tmp_path), employed_agent_ids=[4, 15])
    joined = "\n".join(result.uncertainty_warnings or [])
    assert "cohort of 2 agent(s)" in joined, (
        "cohort-size warning must count only agents with real identifiers "
        f"(expected 'cohort of 2 agent(s)'); warnings were: {joined!r}"
    )
    assert "cohort of 3 agent(s)" not in joined, f"cohort-size warning counted the phantom agent: {joined!r}"


def test_blank_agent_id_records_do_not_become_agent_0(tmp_path):
    """Rows whose ``agent_id`` is null/blank inside a file whose name does
    not match ``agent_<N>`` must never widen the cohort with agent 0."""
    d = tmp_path / "run2"
    d.mkdir()
    _write_jsonl(
        d / "agent_4.jsonl",
        [_episode(4, 1, "00:00", "睡觉"), _episode(4, 1, "12:00", "吃午饭")],
    )
    _write_jsonl(
        d / "intervention_executions.jsonl",
        [
            {
                "name": "set_agent_state",
                "day": 1,
                "time": "14:00",
                "targets": [4],
                "applied": True,
                "agent_id": None,
                "before": {},
                "after": {},
            },
            {
                "name": "set_agent_state",
                "day": 1,
                "time": "15:00",
                "targets": [4],
                "applied": True,
                "before": {},
                "after": {},
            },
        ],
    )
    result = score_directory(d, employed_agent_ids=[4])
    per_agent_day = result.per_agent_day or {}
    assert 0 not in per_agent_day, (
        f"null/missing agent_id rows became phantom agent 0 (got agents {sorted(per_agent_day)})"
    )
    assert sorted(per_agent_day) == [4]
