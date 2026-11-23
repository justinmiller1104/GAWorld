"""Phase 2.5 RED tests — corrected comparison metrics (TASK-3590).

The current ``benchmark/mcf/compare.py`` normalizes per-slot / per-day
mismatches by the baseline arm's WHOLE-RUN episode count, so a 100%
episode disagreement for a slot collapses to 2/288 ≈ 0.00694 (see the
Phase 2.5 investigation, Part 3). These tests pin the corrected contract:

  1. Episode disagreement rate per window with an EXPLICIT denominator:
     mismatches / |common keys in window| — not the whole-run count.
     Pinned with a constructed pair where 2 slots (2 records) disagree
     inside a 576-record pre window → rate = 2/576, never 2/864.
  2. ``first_divergence_tick`` in the report: the earliest differing
     (agent, day, time).
  3. Named per-variable absolute diffs per window (max |Δ| per variable,
     the variable named) — not one folded scalar.
  4. ``max_abs_diff`` demoted: retained in the report JSON as a raw
     diagnostic, but NEVER the headline verdict field (``effect`` carries
     no max_abs_diff; the verdict is judged on the disagreement rate).
  5. Day-granular path stays backward compatible (existing phase 2.2/2.4
     tests keep passing — verified against the full suite).

Everything is deterministic: tiny mocked run directories in the
``agent_<id>.jsonl`` shape ``_load_run`` consumes. No LLM, no network.
"""

import json
from pathlib import Path

import pytest

from benchmark.mcf.compare import NEUTRAL_VERDICTS, compare_arms

# ---------------------------------------------------------------------------
# Deterministic mocked-run generator (the exact shape _load_run reads)
# ---------------------------------------------------------------------------

AGENTS = (1, 2)
PRE_DAYS = (1, 2)
POST_DAYS = (3,)
# 10-minute tick grid (required by the MCF attach's score_directory):
# 144 slots/day → 2 agents × 2 days × 144 = 576 records in the pre window
SLOTS_PER_DAY = 144  # 576 = 2 agents × 2 days × 144 = 576

# post-window records: 2 agents × 1 day × 144 = 288 (whole run = 864)


def _slot_times(n):
    """n deterministic HH:MM slot strings across the day."""
    step = 1440 // n
    return [f"{m // 60:02d}:{m % 60:02d}" for m in range(0, 1440, step)][:n]


SLOT_TIMES = _slot_times(SLOTS_PER_DAY)


def _write_arm(directory: Path, *, diverge: bool) -> Path:
    """One mocked run dir: agent_<id>.jsonl episodes with final_activity and
    a per-episode ``state`` dict (numeric vars the diff layer can name)."""
    directory.mkdir(parents=True, exist_ok=True)
    for day in (*PRE_DAYS, *POST_DAYS):
        for agent_id in AGENTS:
            path = directory / f"agent_{agent_id}.jsonl"
            with open(path, "a", encoding="utf-8") as fh:
                for i, time in enumerate(SLOT_TIMES):
                    activity = "上班工作"
                    state = {"stress": 0.10, "emotion": 0.50}
                    if diverge and day == 2 and time == SLOT_TIMES[7]:
                        # exactly ONE slot (both agents) diverges pre-trigger:
                        # 2 mismatched records inside the 576-record window
                        activity = "在线课程"
                        state = {"stress": 0.50, "emotion": 0.50}  # |Δstress| = 0.4
                    fh.write(
                        json.dumps(
                            {
                                "agent_id": agent_id,
                                "day": day,
                                "time": time,
                                "final_activity": activity,
                                "state": state,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    (directory / "config.json").write_text("{}", encoding="utf-8")
    return directory


def _make_pair(tmp_path: Path):
    baseline = _write_arm(tmp_path / "baseline", diverge=False)
    intervention = _write_arm(tmp_path / "intervention", diverge=True)
    return baseline, intervention


# ---------------------------------------------------------------------------
# Contract 1 — per-window episode disagreement rate, explicit denominator
# ---------------------------------------------------------------------------


def test_window_episode_disagreement_rate_uses_window_denominator(tmp_path):
    """2 mismatched records in a 576-record pre window → 2/576 ≈ 0.00347,
    NOT 2/864 (the whole-run baseline count the old code divided by)."""
    baseline, intervention = _make_pair(tmp_path)
    report = compare_arms(baseline, intervention, intervention_day=3)
    expected = 2 / 576
    assert report["windows"]["pre"]["episode_disagreement_rate"] == pytest.approx(expected), (
        "pre-window rate must be mismatches / |common keys in the window| "
        f"(2/{576}), not a whole-run denominator (2/864 = {2 / 864:.5f})"
    )


def test_zero_post_window_disagreement_rate(tmp_path):
    """The post window is identical here → rate exactly 0.0, not MISSING."""
    baseline, intervention = _make_pair(tmp_path)
    report = compare_arms(baseline, intervention, intervention_day=3)
    assert report["windows"]["post"]["episode_disagreement_rate"] == 0.0


# ---------------------------------------------------------------------------
# Contract 2 — first_divergence_tick
# ---------------------------------------------------------------------------


def test_first_divergence_tick_reported(tmp_path):
    """The earliest differing (agent, day, time) is in the report."""
    baseline, intervention = _make_pair(tmp_path)
    report = compare_arms(baseline, intervention, intervention_day=3)
    tick = report["first_divergence_tick"]
    assert tick is not None, "a diverging pair must carry first_divergence_tick"
    assert tick["day"] == 2
    assert tick["time"] == SLOT_TIMES[7]
    assert set(tick["agent_id"] if isinstance(tick["agent_id"], (set, list)) else [tick["agent_id"]]) <= {1, 2}


def test_first_divergence_tick_none_when_identical(tmp_path):
    baseline = _write_arm(tmp_path / "b2", diverge=False)
    intervention = _write_arm(tmp_path / "i2", diverge=False)
    report = compare_arms(baseline, intervention, intervention_day=3)
    assert report["first_divergence_tick"] is None


# ---------------------------------------------------------------------------
# Contract 3 — named per-variable absolute diffs per window
# ---------------------------------------------------------------------------


def test_named_per_variable_diffs_per_window(tmp_path):
    """Per-variable max |Δ| inside the pre window, variable named — stress
    0.4 at the diverging slot; not one folded scalar."""
    baseline, intervention = _make_pair(tmp_path)
    report = compare_arms(baseline, intervention, intervention_day=3)
    pre_vars = report["windows"]["pre"]["variable_diffs"]
    assert "stress" in pre_vars, "the differing variable must be named"
    assert pre_vars["stress"] == pytest.approx(0.4)
    # emotion is identical everywhere → no diff attributed to it
    assert pre_vars.get("emotion", 0.0) == pytest.approx(0.0)
    assert report["windows"]["post"]["variable_diffs"]["stress"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Contract 4 — max_abs_diff demoted from headline verdict
# ---------------------------------------------------------------------------


def test_effect_verdict_not_max_abs_diff_headline(tmp_path):
    """The effect section is judged on the disagreement rate; max_abs_diff
    never appears as (or in) the headline verdict — it survives only as a
    raw diagnostic in windows."""
    baseline, intervention = _make_pair(tmp_path)
    report = compare_arms(baseline, intervention, intervention_day=3)
    # raw diagnostic retained
    assert "max_abs_diff" in report["windows"]["pre"]
    # but never the headline: no max_abs_diff key leaks into the verdict
    for key in report["effect"]:
        assert "max_abs_diff" not in key, f"effect[{key!r}] keeps max_abs_diff as headline"
    assert "post_max_abs_diff" not in report["effect"]
    # here post disagreement is exactly 0 → a neutral verdict (the fixture
    # is 2 agents × 3 days, below MIN_COMPARISON_DAYS/POPULATION, so the
    # M2 invariant maps UNRELIABLE to insufficient_data — still headline-safe).
    assert report["effect"]["verdict"] in (*NEUTRAL_VERDICTS, "insufficient_data")


# ---------------------------------------------------------------------------
# Contract 5 — day-granular path backward compatible
# ---------------------------------------------------------------------------


def test_day_granular_path_backward_compatible(tmp_path):
    """Without intervention_time, the pre/post windows still slice by day
    and carry the classic fields the phase 2.2/2.4 consumers read."""
    baseline, intervention = _make_pair(tmp_path)
    report = compare_arms(baseline, intervention, intervention_day=3)
    assert set(report["windows"]) == {"pre", "post"}
    assert "max_abs_diff" in report["windows"]["pre"]
    assert set(report["per_day"]) == {1, 2, 3}
    assert "max_abs_diff" in report["per_day"][2]
    assert "effect" in report and "data_status" in report
