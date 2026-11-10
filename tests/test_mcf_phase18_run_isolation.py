"""TDD RED tests — Phase 1.8 Part C: run isolation.

Contract (TASK-3453-SPEC, Contract 3):
- Two consecutive runs in the same process/directory: unique run_id per
  run, isolated per-run output directories, no appending to a previous
  run's episode files, manifest correctly associated with its own run's
  outputs.
- Interrupted run (partial write / no final manifest): files stay in the
  interrupted run's own directory and a subsequent run cannot contaminate
  them.
- Schedule consistency: episode timestamps fall within the run's
  observation window; no records outside it.

The new API imported below does NOT exist yet — the ImportError/AttributeError
is the RED state by design. No LLM calls, no network, no simulation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.mcf.launch import (
    create_run,  # noqa: F401  (RED: missing API)
    finalize_run,  # noqa: F401  (RED: missing API)
    validate_observation_window,  # noqa: F401  (RED: missing API)
)


# Regression fixture: the real interrupted-run manifest from the Phase 1.6
# pilot16 run (20261109_133955, outcome in_progress, finished_at "").
PARTIAL_MANIFEST = Path(__file__).parent / "fixtures" / "pilot16" / "run_manifest_20261109_133955.partial.json"


def _fake_llm():
    calls = {"n": 0}

    def llm(prompt, task=None, agent_id=None, **kwargs):
        calls["n"] += 1
        return "看电视"

    llm.calls = calls
    return llm


# ---------------------------------------------------------------------------
# 1. Two consecutive runs: unique ids, isolated directories
# ---------------------------------------------------------------------------


def test_two_consecutive_runs_get_unique_run_ids(tmp_path: Path):
    with create_run(tmp_path, config={"sim_days": 1}) as run_a:
        pass
    with create_run(tmp_path, config={"sim_days": 1}) as run_b:
        pass
    assert run_a.run_id != run_b.run_id, "each run must get a unique run_id"


def test_consecutive_runs_write_to_isolated_directories(tmp_path: Path):
    with create_run(tmp_path, config={"sim_days": 1}) as run_a:
        episodes_a = run_a.run_dir / "episodes_agent_4.jsonl"
        episodes_a.write_text(json.dumps({"day": 1, "time": "08:00"}) + "\n", encoding="utf-8")
        dir_a = run_a.run_dir
    with create_run(tmp_path, config={"sim_days": 1}) as run_b:
        dir_b = run_b.run_dir
        episodes_b = dir_b / "episodes_agent_4.jsonl"
        episodes_b.write_text(json.dumps({"day": 1, "time": "08:00"}) + "\n", encoding="utf-8")
    assert dir_a != dir_b, "runs must use separate output directories"
    assert episodes_a.read_text(encoding="utf-8").count("\n") == 1, "run A's episode file untouched"


def test_second_run_does_not_append_to_first_runs_episode_files(tmp_path: Path):
    with create_run(tmp_path, config={"sim_days": 1}) as run_a:
        (run_a.run_dir / "episodes_agent_4.jsonl").write_text(
            json.dumps({"day": 1, "time": "08:00"}) + "\n", encoding="utf-8"
        )
    with create_run(tmp_path, config={"sim_days": 1}) as run_b:
        (run_b.run_dir / "episodes_agent_4.jsonl").write_text(
            json.dumps({"day": 1, "time": "09:00"}) + "\n", encoding="utf-8"
        )
        pass
    first = (run_a.run_dir / "episodes_agent_4.jsonl").read_text(encoding="utf-8")
    assert first.count("\n") == 1, "run B must not append into run A's episode file"


# ---------------------------------------------------------------------------
# 2. Manifest association
# ---------------------------------------------------------------------------


def test_manifest_is_associated_with_its_own_run(tmp_path: Path):
    with create_run(tmp_path, config={"sim_days": 1}) as run_a:
        (run_a.run_dir / "episodes_agent_4.jsonl").write_text("{}\n", encoding="utf-8")
    with create_run(tmp_path, config={"sim_days": 1}) as run_b:
        (run_b.run_dir / "episodes_agent_4.jsonl").write_text("{}\n", encoding="utf-8")
        manifest_b = run_b.manifest_path

    payload = json.loads(manifest_b.read_text(encoding="utf-8"))
    assert payload["run_id"] == run_b.run_id
    assert payload["run_id"] != run_a.run_id
    outputs = payload.get("outputs") or payload.get("output_dir") or ""
    assert str(run_b.run_id) in str(outputs), "the manifest must reference its own run's outputs"


def test_finalize_run_writes_final_manifest_into_own_dir(tmp_path: Path):
    with create_run(tmp_path, config={"sim_days": 1}) as run:
        finalize_run(run, outcome="completed")
    payload = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert payload["outcome"] == "completed"
    assert payload["finished_at"], "a finalized manifest must carry a finish timestamp"


# ---------------------------------------------------------------------------
# 3. Interrupted run: partial manifest, no contamination
# ---------------------------------------------------------------------------


def test_interrupted_run_leaves_partial_manifest_format(tmp_path: Path):
    """The real pilot16 interrupted run used a .partial.json with empty finished_at."""
    payload = json.loads(PARTIAL_MANIFEST.read_text(encoding="utf-8"))
    assert payload["outcome"] == "in_progress"
    assert payload["finished_at"] == ""


def test_subsequent_run_cannot_contaminate_interrupted_runs_directory(tmp_path: Path):
    with create_run(tmp_path, config={"sim_days": 1}) as interrupted:
        (interrupted.run_dir / "episodes_agent_4.jsonl").write_text(
            json.dumps({"day": 1, "time": "08:00"}) + "\n", encoding="utf-8"
        )
        # Simulate an interruption: no finalize_run, context exited early.
    with create_run(tmp_path, config={"sim_days": 1}) as following:
        (following.run_dir / "episodes_agent_4.jsonl").write_text(
            json.dumps({"day": 1, "time": "08:00"}) + "\n"
            + json.dumps({"day": 1, "time": "08:10"}) + "\n",
            encoding="utf-8",
        )
    still = (interrupted.run_dir / "episodes_agent_4.jsonl").read_text(encoding="utf-8")
    assert still.count("\n") == 1, "the interrupted run's file must stay untouched"
    assert interrupted.run_dir != following.run_dir


# ---------------------------------------------------------------------------
# 4. Schedule consistency: observation window
# ---------------------------------------------------------------------------


def test_episode_inside_window_validates():
    records = [{"agent_id": 4, "day": 1, "time": "08:00"}, {"agent_id": 4, "day": 1, "time": "12:20"}]
    problems = validate_observation_window(records, window_start_min=480, window_end_min=740)
    assert problems == [], "records inside the window must validate clean"


def test_episode_outside_window_is_flagged():
    records = [
        {"agent_id": 4, "day": 1, "time": "08:00"},  # in window
        {"agent_id": 4, "day": 1, "time": "23:50"},  # outside 08:00->12:20
    ]
    problems = validate_observation_window(records, window_start_min=480, window_end_min=740)
    assert problems, "records outside the observation window must be flagged"
    assert any("23:50" in str(p) or 1430 in [getattr(p, "minute_of_day", -1)] for p in problems), (
        "the flagged problem must identify the offending record"
    )


def test_records_from_another_day_are_flagged():
    records = [{"agent_id": 4, "day": 2, "time": "09:00"}]
    problems = validate_observation_window(records, window_start_min=480, window_end_min=740, days=(1,))
    assert problems, "records outside the run's days must be flagged"
