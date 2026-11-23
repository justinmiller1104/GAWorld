"""Phase 2.2 RED tests — outcome comparison framework (TASK-3542).

The comparison module does not exist yet: these tests pin the
baseline-vs-intervention comparison contract that
``benchmark/mcf/compare.py`` must satisfy BEFORE it is written. Everything is
deterministic: a tiny mocked-run generator writes synthetic episode JSONL +
config per arm in exactly the shapes ``benchmark/mcf/time_use.score_directory``
consumes (``agent_<id>.jsonl`` episode files, real run-directory layout). No
LLM, no network, no real simulation.

Contracts pinned here (TASK-PH22-RED-SPEC):

  1. Reproducibility — identical scenarios (same seed + config) produce an
     identical comparison result.
  3. No cross-scenario state leak — arm directories are independent; running
     arms in any order, or repeatedly, yields the same result, and comparing
     never mutates the run directories.
  4. Outcome-calculation consistency — same inputs give an identical diff
     report (pure functions).
  5. Repeated-run variation — spread across repeated runs is reported
     alongside the mean differences, never collapsed into one number.
  6. Neutral intervention — a zero-effect delta produces ~0 differences and
     the report never labels it beneficial.
  7. Honest missing-data reporting — absent episode files / below-minimum
     cohorts are reported as MISSING / UNRELIABLE markers, not zeros.
  8. Pre/post window slicing — outcome differences are measured inside the
     intervention window vs before it, tracked over simulation days.
 10. MCF attach separation — the report carries a validation section (per-arm
     MCF scores) SEPARATE from the effect-difference section, with limitations
     text (baseline realism ≠ intervention-response evidence).
"""

import json
from pathlib import Path

import pytest

from benchmark.mcf.compare import (
    NEUTRAL_VERDICTS,
    compare_arms,
    compare_repeated_runs,
)

# ---------------------------------------------------------------------------
# Deterministic mocked-run generator (score_directory-compatible shapes)
# ---------------------------------------------------------------------------

INTERVENTION_DAY = 5
N_DAYS = 8
N_AGENTS = 25  # >= MIN_COMPARISON_POPULATION so MCF scoring is frontier-valid
TICK_MINUTES = 10

# Daily episode template: (time, final_activity). The 18:00 slot is where an
# intervention effect (extra leisure) is injected, post-intervention-day only.
DAY_SCHEDULE_BASE = [
    ("07:00", "吃早饭"),
    ("08:00", "上班工作"),
    ("12:00", "吃午饭"),
    ("13:00", "上班工作"),
    ("18:00", "上班工作"),
    ("22:00", "睡觉"),
]


def _write_arm(
    directory: Path,
    *,
    seed: int,
    leisure_shift: float = 0.0,
    days: int = N_DAYS,
    agents: int = N_AGENTS,
    write_config: bool = True,
    from_day: int = 1,
) -> Path:
    """Write one mocked run directory the way a real run lays episodes out.

    Deterministic in (seed, leisure_shift, days, agents, from_day): the same
    arguments always produce byte-identical directories, and every call is
    independent (no shared state — that is what contract 3 checks).
    """
    directory.mkdir(parents=True, exist_ok=True)
    threshold = seed % 100  # deterministic per-arm evening mood, no RNG object
    for agent_id in range(1, agents + 1):
        path = directory / f"agent_{agent_id}.jsonl"
        lines = []
        for day in range(from_day, days + 1):
            schedule = list(DAY_SCHEDULE_BASE)
            if leisure_shift and day >= INTERVENTION_DAY and (agent_id + day + threshold) % 2 == 0:
                schedule[4] = ("18:00", "散步放松娱乐")
            for time, activity in schedule:
                lines.append(
                    json.dumps(
                        {
                            "agent_id": agent_id,
                            "day": day,
                            "time": time,
                            "final_activity": activity,
                            "scheduled_activity": activity,
                            "action": "",
                            "location": "家",
                            "target_location": "",
                        },
                        ensure_ascii=False,
                    )
                )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if write_config:
        (directory / "config.json").write_text(
            json.dumps({"random_seed": seed, "fos_fast_mode": True}), encoding="utf-8"
        )
    return directory


def _baseline_dir(root: Path, name: str = "baseline", seed: int = 42) -> Path:
    return _write_arm(root / name, seed=seed, leisure_shift=0.0)


def _intervention_dir(
    root: Path,
    name: str = "intervention",
    seed: int = 42,
    shift: float = 1.0,
    days: int = N_DAYS,
    agents: int = N_AGENTS,
) -> Path:
    return _write_arm(root / name, seed=seed, leisure_shift=shift, days=days, agents=agents)


# ---------------------------------------------------------------------------
# Contracts 1 & 4 — reproducibility and pure-function consistency
# ---------------------------------------------------------------------------


def test_identical_scenarios_reproduce_identical_results(tmp_path):
    first = _baseline_dir(tmp_path / "a")
    second = _baseline_dir(tmp_path / "b")
    intervention = _intervention_dir(tmp_path / "c")
    r1 = compare_arms(first, intervention, intervention_day=INTERVENTION_DAY)
    r2 = compare_arms(second, intervention, intervention_day=INTERVENTION_DAY)
    assert r1 == r2


def test_compare_is_a_pure_function(tmp_path):
    baseline = _baseline_dir(tmp_path / "base")
    intervention = _intervention_dir(tmp_path / "arm")
    r1 = compare_arms(baseline, intervention, intervention_day=INTERVENTION_DAY)
    r2 = compare_arms(baseline, intervention, intervention_day=INTERVENTION_DAY)
    assert r1 == r2


# ---------------------------------------------------------------------------
# Contract 3 — no cross-scenario state leak
# ---------------------------------------------------------------------------


def test_arm_execution_order_does_not_matter(tmp_path):
    baseline = _baseline_dir(tmp_path / "base")
    neutral = _intervention_dir(tmp_path / "neutral", shift=0.0)
    real = _intervention_dir(tmp_path / "real", shift=1.0)
    r_neutral_first = compare_arms(baseline, neutral, intervention_day=INTERVENTION_DAY)
    _r_real_then_neutral = compare_arms(baseline, real, intervention_day=INTERVENTION_DAY)
    r_neutral_again = compare_arms(baseline, neutral, intervention_day=INTERVENTION_DAY)
    assert r_neutral_again == r_neutral_first


def test_compare_never_mutates_run_directories(tmp_path):
    baseline = _baseline_dir(tmp_path / "base")
    intervention = _intervention_dir(tmp_path / "arm")
    before = {
        p: p.read_bytes()
        for p in sorted(baseline.rglob("*")) + sorted(intervention.rglob("*"))
        if p.is_file()
    }
    compare_arms(baseline, intervention, intervention_day=INTERVENTION_DAY)
    after = {
        p: p.read_bytes()
        for p in sorted(baseline.rglob("*")) + sorted(intervention.rglob("*"))
        if p.is_file()
    }
    assert before == after


def test_separate_directories_stay_independent(tmp_path):
    # Two interventions built with different shifts, each compared against its
    # own baseline copy: neither comparison may influence the other.
    b1 = _baseline_dir(tmp_path / "pair1" / "baseline")
    i1 = _intervention_dir(tmp_path / "pair1" / "arm", shift=1.0)
    b2 = _baseline_dir(tmp_path / "pair2" / "baseline")
    i2 = _intervention_dir(tmp_path / "pair2" / "arm", shift=2.0)
    r1 = compare_arms(b1, i1, intervention_day=INTERVENTION_DAY)
    r2 = compare_arms(b2, i2, intervention_day=INTERVENTION_DAY)
    assert r1 != r2  # different deltas must yield different reports


# ---------------------------------------------------------------------------
# Contract 8 — pre/post window slicing
# ---------------------------------------------------------------------------


def test_effect_is_localized_to_the_post_window(tmp_path):
    baseline = _baseline_dir(tmp_path / "base")
    intervention = _intervention_dir(tmp_path / "arm", shift=1.0)
    report = compare_arms(baseline, intervention, intervention_day=INTERVENTION_DAY)
    windows = report["windows"]
    assert set(windows) >= {"pre", "post"}
    # Arms are byte-identical before the intervention day: no pre effect.
    assert windows["pre"]["max_abs_diff"] == pytest.approx(0.0)
    # Leisure shifted to extra leisure after the intervention day.
    assert windows["post"]["max_abs_diff"] > 0.0


def test_effects_are_tracked_per_day_over_simulation_time(tmp_path):
    baseline = _baseline_dir(tmp_path / "base")
    intervention = _intervention_dir(tmp_path / "arm", shift=1.0)
    report = compare_arms(baseline, intervention, intervention_day=INTERVENTION_DAY)
    per_day = report["per_day"]
    assert set(per_day) == set(range(1, N_DAYS + 1))
    pre_days = [per_day[d] for d in range(1, INTERVENTION_DAY)]
    post_days = [per_day[d] for d in range(INTERVENTION_DAY, N_DAYS + 1)]
    assert all(day["max_abs_diff"] == pytest.approx(0.0) for day in pre_days)
    assert any(day["max_abs_diff"] > 0.0 for day in post_days)


# ---------------------------------------------------------------------------
# Contract 6 — neutral intervention
# ---------------------------------------------------------------------------


def test_neutral_intervention_shows_no_artificial_improvement(tmp_path):
    baseline = _baseline_dir(tmp_path / "base")
    neutral = _intervention_dir(tmp_path / "neutral", shift=0.0)
    report = compare_arms(baseline, neutral, intervention_day=INTERVENTION_DAY)
    assert report["windows"]["pre"]["max_abs_diff"] == pytest.approx(0.0)
    assert report["windows"]["post"]["max_abs_diff"] == pytest.approx(0.0)
    assert report["effect"]["verdict"] in NEUTRAL_VERDICTS
    assert not report["effect"].get("beneficial", False)


# ---------------------------------------------------------------------------
# Contract 5 — repeated-run variation
# ---------------------------------------------------------------------------


def test_repeated_runs_report_spread_alongside_mean(tmp_path):
    baseline_dirs = [_baseline_dir(tmp_path / f"rep{k}" / "baseline", seed=40 + k) for k in range(3)]
    intervention_dirs = [
        _intervention_dir(tmp_path / f"rep{k}" / "arm", seed=40 + k, shift=1.0 + 0.5 * k) for k in range(3)
    ]
    report = compare_repeated_runs(baseline_dirs, intervention_dirs, intervention_day=INTERVENTION_DAY)
    variation = report["variation"]
    assert len(variation["per_run"]) == 3
    assert variation["spread"] >= 0.0
    # The mean differences sit next to the spread, not instead of it.
    assert "mean" in variation
    assert variation["per_run"] != [variation["mean"]]


# ---------------------------------------------------------------------------
# Contract 7 — honest MISSING / UNRELIABLE data reporting
# ---------------------------------------------------------------------------


def test_absent_episode_files_reported_as_missing(tmp_path):
    baseline = _baseline_dir(tmp_path / "base")
    empty = tmp_path / "empty_arm"
    empty.mkdir()
    report = compare_arms(baseline, empty, intervention_day=INTERVENTION_DAY)
    assert report["data_status"]["intervention"] == "MISSING"
    assert "MISSING" in json.dumps(report)


def test_below_minimum_cohort_reported_as_unreliable(tmp_path):
    baseline = _baseline_dir(tmp_path / "base")
    # 2 agents x 3 days — far below MIN_COMPARISON_DAYS/POPULATION.
    small = _intervention_dir(tmp_path / "small", shift=1.0, days=3, agents=2)
    report = compare_arms(baseline, small, intervention_day=2)
    assert report["data_status"]["intervention"] == "UNRELIABLE"


def test_missing_data_is_never_reported_as_zero_effect(tmp_path):
    baseline = _baseline_dir(tmp_path / "base")
    empty = tmp_path / "empty_arm"
    empty.mkdir()
    report = compare_arms(baseline, empty, intervention_day=INTERVENTION_DAY)
    # A missing arm must not silently read as "no effect".
    assert report["effect"]["verdict"] not in NEUTRAL_VERDICTS


# ---------------------------------------------------------------------------
# Contract 10 — MCF validation attach, separate from effect differences
# ---------------------------------------------------------------------------


def test_report_separates_mcf_validation_from_effect_section(tmp_path):
    baseline = _baseline_dir(tmp_path / "base")
    intervention = _intervention_dir(tmp_path / "arm", shift=1.0)
    report = compare_arms(baseline, intervention, intervention_day=INTERVENTION_DAY)
    assert "mcf_validation" in report
    assert "effect" in report
    assert set(report["mcf_validation"]) >= {"baseline", "intervention"}
    for arm_score in report["mcf_validation"].values():
        # per-arm MCF scores carry the score_against_targets essentials
        assert "mean_category_error_not_for_frontier" in arm_score
    # Validation must not live inside the effect section.
    assert "mcf" not in report["effect"]


def test_validation_carries_limitations_text(tmp_path):
    baseline = _baseline_dir(tmp_path / "base")
    intervention = _intervention_dir(tmp_path / "arm", shift=1.0)
    report = compare_arms(baseline, intervention, intervention_day=INTERVENTION_DAY)
    limitations = report["mcf_validation"]["limitations"]
    assert isinstance(limitations, (list, str)) and limitations
    text = " ".join(limitations) if isinstance(limitations, list) else limitations
    # Baseline realism is not evidence of intervention response.
    assert "realism" in text
