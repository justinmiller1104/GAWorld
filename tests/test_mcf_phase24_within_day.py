"""Phase 2.4 RED tests — within-day comparison windows (TASK-3577).

These tests pin the within-day intervention-window contract that
``benchmark/mcf/compare.py`` must satisfy BEFORE it is implemented:

  1. ``compare_arms`` accepts an intervention trigger with a day AND a time
     of day (e.g. day 1, "14:00" = minute 840) and slices the outcome series
     at tick resolution, not just at day granularity (where day 1 would be a
     single bucket).
  2. The trigger tick itself belongs to the POST window.
  3. For arms identical before the trigger time and diverging after it,
     pre diff == 0 and post diff > 0 at tick granularity.
  4. Day-granular behaviour stays backward compatible (the existing Phase
     2.2 call signature must keep producing the same day-window report).

Everything is deterministic: a tiny mocked-run generator writes synthetic
episode JSONL + config per arm in exactly the shapes
``benchmark/mcf/time_use.score_directory`` consumes. No LLM, no network, no
real simulation.

These tests are EXPECTED TO FAIL until the within-day window is implemented.
"""

import json
from pathlib import Path

import pytest

from benchmark.mcf.compare import compare_arms

# ---------------------------------------------------------------------------
# Deterministic mocked-run generator (score_directory-compatible shapes)
# ---------------------------------------------------------------------------

TRIGGER_DAY = 1
TRIGGER_TIME = "14:00"  # minute 840
TICK_MINUTES = 10
N_DAYS = 8  # >= MIN_COMPARISON_DAYS so the arms are data_status "ok"
N_AGENTS = 25  # >= MIN_COMPARISON_POPULATION

# Intraday episode slots at 10-minute resolution between 13:00 and 16:00 —
# the window straddling the trigger tick. The rest of the day is boilerplate.
STRADDLE_TIMES = (
    [f"13:{m:02d}" for m in (0, 10, 20, 30, 40, 50)]
    + [f"14:{m:02d}" for m in (0, 10, 20, 30, 40, 50)]
    + [f"15:{m:02d}" for m in (0, 10, 20, 30, 40, 50)]
    + ["16:00"]
)

DAY_SCHEDULE = (
    [("07:00", "吃早饭"), ("08:00", "上班工作"), ("12:00", "吃午饭")]
    + [(t, "上班工作") for t in STRADDLE_TIMES]
    + [("22:00", "睡觉")]
)


def _write_arm(directory: Path, *, seed: int, diverge_after_trigger: bool) -> Path:
    """Write one mocked run directory.

    With ``diverge_after_trigger=True`` the arm's episodes at or after the
    trigger tick (day >= TRIGGER_DAY, time >= "14:00") read "社区志愿服务"
    instead of "上班工作"; everything before the trigger tick is byte-identical
    to the baseline arm.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for agent_id in range(1, N_AGENTS + 1):
        path = directory / f"agent_{agent_id}.jsonl"
        lines = []
        for day in range(1, N_DAYS + 1):
            for time, activity in DAY_SCHEDULE:
                final = activity
                if (
                    diverge_after_trigger
                    and day >= TRIGGER_DAY
                    and time >= TRIGGER_TIME
                    and activity == "上班工作"
                    and (agent_id + day) % 2 == 0
                ):
                    final = "社区志愿服务"
                lines.append(
                    json.dumps(
                        {
                            "agent_id": agent_id,
                            "day": day,
                            "time": time,
                            "final_activity": final,
                            "scheduled_activity": final,
                            "action": "",
                            "location": "家",
                            "target_location": "",
                        },
                        ensure_ascii=False,
                    )
                )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "config.json").write_text(
        json.dumps({"random_seed": seed, "fos_fast_mode": True}), encoding="utf-8"
    )
    return directory


@pytest.fixture()
def arms(tmp_path: Path):
    baseline = _write_arm(tmp_path / "baseline", seed=42, diverge_after_trigger=False)
    intervention = _write_arm(tmp_path / "intervention", seed=42, diverge_after_trigger=True)
    return baseline, intervention


# ---------------------------------------------------------------------------
# Contract 1+2+3 — within-day trigger, tick-resolution slicing
# ---------------------------------------------------------------------------


def test_trigger_time_accepted_and_reported(arms):
    """compare_arms accepts (day, time_str) and records the within-day trigger."""
    baseline, intervention = arms
    report = compare_arms(
        baseline,
        intervention,
        intervention_day=TRIGGER_DAY,
        intervention_time=TRIGGER_TIME,
        tick_minutes=TICK_MINUTES,
    )
    assert report["intervention_day"] == TRIGGER_DAY
    assert report["intervention_time"] == TRIGGER_TIME


def test_pre_window_zero_post_window_positive(arms):
    """Arms identical before 14:00 and diverging after: pre == 0, post > 0."""
    baseline, intervention = arms
    report = compare_arms(
        baseline,
        intervention,
        intervention_day=TRIGGER_DAY,
        intervention_time=TRIGGER_TIME,
        tick_minutes=TICK_MINUTES,
    )
    pre = report["windows"]["pre"]["max_abs_diff"]
    post = report["windows"]["post"]["max_abs_diff"]
    assert pre == 0.0, f"pre window must be 0 (identical before trigger), got {pre}"
    assert post > 0.0, f"post window must be > 0 (divergence after trigger), got {post}"


def test_slicing_at_tick_resolution(arms):
    """Pre/post slicing happens at episode-time resolution, not day buckets.

    With a day-granular window (the current implementation), day 1 is one
    bucket, so day 1 diffs — which include pre-trigger identical episodes —
    leak into the post window. A tick-resolution window keeps 13:00–13:50
    clean (0 diff) and puts the divergence from 14:00 onward in post.
    """
    baseline, intervention = arms
    report = compare_arms(
        baseline,
        intervention,
        intervention_day=TRIGGER_DAY,
        intervention_time=TRIGGER_TIME,
        tick_minutes=TICK_MINUTES,
    )
    per_time = report.get("per_time")
    assert per_time, "report must carry a per-time (tick-resolution) section"
    for time, entry in per_time.items():
        if time < TRIGGER_TIME:
            assert entry["max_abs_diff"] == 0.0, (
                f"{time} is before the trigger; diff must be 0, got {entry['max_abs_diff']}"
            )
        else:
            assert entry["max_abs_diff"] > 0.0, (
                f"{time} is at/after the trigger; diff must be > 0, got {entry['max_abs_diff']}"
            )


def test_trigger_tick_belongs_to_post_window(arms):
    """The trigger tick itself (14:00, minute 840) is inside the POST window."""
    baseline, intervention = arms
    report = compare_arms(
        baseline,
        intervention,
        intervention_day=TRIGGER_DAY,
        intervention_time=TRIGGER_TIME,
        tick_minutes=TICK_MINUTES,
    )
    trigger_tick = report["per_time"][TRIGGER_TIME]
    assert trigger_tick["window"] == "post"


def test_minute_conversion_helper():
    """The documented tick arithmetic: day 1, 14:00 == minute 840."""
    from benchmark.mcf.compare import time_to_minute  # noqa: missing until implemented

    assert time_to_minute("14:00") == 840


# ---------------------------------------------------------------------------
# Contract 4 — day-granular behaviour stays available (backward compatible)
# ---------------------------------------------------------------------------


def test_day_granular_signature_unchanged(arms):
    """Calling with only ``intervention_day`` (the Phase 2.2 signature) still
    produces the day-window report: day 1 diffs land in the post window."""
    baseline, intervention = arms
    report = compare_arms(baseline, intervention, intervention_day=TRIGGER_DAY)
    assert report.get("intervention_time") in (None, "")
    # Day-granular: day 1 now carries both identical and diverging episodes,
    # so its diff is > 0 and sits in the post window.
    assert report["per_day"][TRIGGER_DAY]["max_abs_diff"] > 0.0
    assert report["windows"]["pre"]["max_abs_diff"] == 0.0
    assert report["windows"]["post"]["max_abs_diff"] > 0.0
