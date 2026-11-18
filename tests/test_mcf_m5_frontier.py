"""M5 RED tests — offline frontier module ``benchmark/mcf/frontier.py``.

Spec source: docs/MCF_GAWORLD.md (§3 bands/coverage, §4 A(M1|M0), §5 gates,
§8 report determinism) + MCF_COMPLETION_PLAN.md milestone M5.

All inputs are synthetic fixtures, clearly labeled. No inference, no network.
RED phase: every test below fails with ModuleNotFoundError until
``benchmark/mcf/frontier.py`` exists.
"""

import json
from importlib import import_module

import pytest

from benchmark.mcf.time_use import (
    MIN_COMPARISON_DAYS,
    MIN_COMPARISON_POPULATION,
    ScoreResult,
)


def _frontier():
    return import_module("benchmark.mcf.frontier")


# ---------------------------------------------------------------------------
# Synthetic fixture builders (fixture=True is a required label downstream)
# ---------------------------------------------------------------------------


def make_score_result(
    *,
    paid_work_error=20.0,
    leisure_error=15.0,
    socializing_error=5.0,
    frontier_valid=True,
    n_agents=None,
    n_days=None,
    coverage=0.95,
    overall_error=None,
):
    """A synthetic ScoreResult with the two validated frontier dimensions.

    ``leisure_social`` error is the mean of the leisure and socializing
    category errors (Phase 0.8-verified pair). Bounds failing the coverage /
    population / duration gates are marked ``frontier_valid=False`` exactly as
    ``score_against_targets`` does.
    """
    per_category = {
        "sleep": {"sim": 590.0, "target": 586.0, "error": 4.0, "n_agent_days": 140.0},
        "paid_work": {
            "sim": 383.0 + paid_work_error,
            "target": 383.0,
            "error": paid_work_error,
            "n_agent_days": 100.0,
        },
        "leisure": {
            "sim": 153.0 + leisure_error,
            "target": 153.0,
            "error": leisure_error,
            "n_agent_days": 140.0,
        },
        "socializing": {
            "sim": 18.0 + socializing_error,
            "target": 18.0,
            "error": socializing_error,
            "n_agent_days": 140.0,
        },
    }
    if overall_error is None:
        errs = [v["error"] for v in per_category.values()]
        overall_error = sum(errs) / len(errs)
    return ScoreResult(
        per_category=per_category,
        overall_error=overall_error,
        coverage=coverage,
        leftover_minutes=3.0,
        per_agent_day={},
        uncertainty_warnings=[],
        frontier_valid=frontier_valid,
    )


def make_vector(pw, ls, *, frontier_valid=True):
    """A synthetic already-assembled 2-D error vector."""
    return {
        "paid_work_error": pw,
        "leisure_social_error": ls,
        "frontier_valid": frontier_valid,
        "fixture": True,
    }


def make_seeded_result(pw, ls, *, frontier_valid=True):
    """(score_result, expected leisure/social error) for one synthetic leg."""
    sr = make_score_result(
        paid_work_error=pw,
        leisure_error=ls - 2.0,
        socializing_error=ls + 2.0,  # mean = ls
        frontier_valid=frontier_valid,
    )
    return sr, ls


# ---------------------------------------------------------------------------
# 1. assemble_error_vector — error-vector assembly from ScoreResults
# ---------------------------------------------------------------------------


def test_assemble_error_vector_extracts_both_dimensions():
    frontier = _frontier()
    sr = make_score_result(paid_work_error=25.0, leisure_error=12.0, socializing_error=8.0)
    vec = frontier.assemble_error_vector(sr, config="c1", seed=1, fixture=True)
    assert vec["paid_work_error"] == pytest.approx(25.0)
    assert vec["leisure_social_error"] == pytest.approx(10.0)  # mean(12, 8)


def test_assemble_error_vector_is_labeled_synthetic():
    frontier = _frontier()
    vec = frontier.assemble_error_vector(make_score_result(), config="c", seed=1, fixture=True)
    assert vec.get("fixture") is True, "every assembled vector must carry the synthetic label"


def test_assemble_error_vector_refuses_missing_dimension():
    frontier = _frontier()
    sr = make_score_result()
    broken = dict(sr.per_category)
    broken.pop("socializing")  # leisure/social pair incomplete
    sr2 = ScoreResult(
        per_category=broken,
        overall_error=sr.overall_error,
        coverage=sr.coverage,
        leftover_minutes=0.0,
        frontier_valid=True,
    )
    with pytest.raises((ValueError, KeyError)):
        frontier.assemble_error_vector(sr2, config="c", seed=1, fixture=True)


def test_assemble_error_vector_excludes_gate_failed_with_reason():
    frontier = _frontier()
    sr = make_score_result(frontier_valid=False)
    vec = frontier.assemble_error_vector(sr, config="c", seed=1, fixture=True)
    assert vec.get("excluded") is True, "a frontier_valid=False leg must be excluded"
    assert vec.get("reason"), "the exclusion must say why, never silently"
    assert vec.get("frontier_valid") is False


def test_gate_constants_are_the_documented_floors():
    assert MIN_COMPARISON_DAYS == 7
    assert MIN_COMPARISON_POPULATION == 20


# ---------------------------------------------------------------------------
# 2. average_across_seeds — complete + incomplete-loud
# ---------------------------------------------------------------------------


def test_average_across_seeds_means_and_spread():
    frontier = _frontier()
    per_seed = {
        1: make_vector(20.0, 10.0),
        2: make_vector(30.0, 20.0),
    }
    out = frontier.average_across_seeds("c1", per_seed)
    assert out["paid_work_error"] == pytest.approx(25.0)
    assert out["leisure_social_error"] == pytest.approx(15.0)
    assert out["paid_work_spread"] == pytest.approx(5.0)  # half-range SD-ish spread
    assert out["complete"] is True


def test_average_across_seeds_single_seed_spread_is_zero():
    frontier = _frontier()
    out = frontier.average_across_seeds("c1", {1: make_vector(10.0, 5.0)})
    assert out["paid_work_spread"] == 0.0


def test_average_across_seeds_missing_seed_is_loud():
    frontier = _frontier()
    per_seed = {1: make_vector(20.0, 10.0)}  # seed 2 never ran
    out = frontier.average_across_seeds("c1", per_seed, expected_seeds=[1, 2])
    assert out["complete"] is False, "a missing seed must be marked incomplete"
    assert 2 in out.get("missing_seeds", []), "the missing seed must be named"
    assert out.get("reason") or out.get("warning"), "incompleteness must be loud, not silent"


def test_average_across_seeds_refuses_expected_seeds_omitted():
    """Without expected_seeds the caller cannot know a leg is missing.

    Passing expected_seeds explicitly is the loud contract; omitting them
    entirely must not fabricate a 'complete' verdict either — the result must
    at minimum not claim complete=True.
    """
    frontier = _frontier()
    out = frontier.average_across_seeds("c1", {1: make_vector(20.0, 10.0)}, expected_seeds=[1, 2, 3])
    assert out["complete"] is False
    assert set(out["missing_seeds"]) == {2, 3}


# ---------------------------------------------------------------------------
# 3. pareto_frontier — componentwise minimization, ties/duplicates
# ---------------------------------------------------------------------------


def test_pareto_known_grid():
    frontier = _frontier()
    pts = [(1.0, 9.0), (2.0, 5.0), (4.0, 4.0), (9.0, 1.0), (5.0, 5.0), (6.0, 6.0)]
    front = frontier.pareto_frontier(pts)
    assert set(front) == {(1.0, 9.0), (2.0, 5.0), (4.0, 4.0), (9.0, 1.0)}
    assert (5.0, 5.0) not in front and (6.0, 6.0) not in front


def test_pareto_dominated_point_not_on_front():
    frontier = _frontier()
    front = frontier.pareto_frontier([(3.0, 3.0), (1.0, 1.0)])
    assert front == [(1.0, 1.0)]


def test_pareto_ties_on_one_axis_both_survive():
    frontier = _frontier()
    front = frontier.pareto_frontier([(2.0, 5.0), (2.0, 7.0)])
    assert sorted(front) == [(2.0, 5.0), (2.0, 7.0)]


def test_pareto_duplicates_collapsed_deterministically():
    frontier = _frontier()
    front = frontier.pareto_frontier([(2.0, 5.0), (2.0, 5.0), (2.0, 5.0)])
    assert front == [(2.0, 5.0)]


def test_pareto_output_deterministic_order():
    frontier = _frontier()
    pts = [(1.0, 9.0), (9.0, 1.0), (2.0, 5.0)]
    a = frontier.pareto_frontier(list(reversed(pts)))
    b = frontier.pareto_frontier(pts)
    assert a == b, "frontier extraction must not depend on input order"


def test_pareto_single_point():
    frontier = _frontier()
    assert frontier.pareto_frontier([(4.0, 4.0)]) == [(4.0, 4.0)]


# ---------------------------------------------------------------------------
# 4. arch_compare — A(M1|M0) newly-attainable set
# ---------------------------------------------------------------------------


def test_arch_compare_newly_attainable_points():
    frontier = _frontier()
    m0 = frontier.pareto_frontier([(4.0, 4.0), (9.0, 1.0)])
    m1 = frontier.pareto_frontier([(4.0, 4.0), (9.0, 1.0), (1.5, 1.5)])
    gain = frontier.arch_compare(m1, m0)
    assert (1.5, 1.5) in gain, "a point M1 reaches that M0 cannot is newly attainable"
    assert all(p in m1 for p in gain)


def test_arch_compare_identical_frontiers_empty_set():
    frontier = _frontier()
    m = frontier.pareto_frontier([(4.0, 4.0), (9.0, 1.0)])
    gain = frontier.arch_compare(m, m)
    assert gain == [], "identical architectures gain nothing — explicit empty set"


def test_arch_compare_m0_dominates_m1_empty_set():
    frontier = _frontier()
    m0 = frontier.pareto_frontier([(1.0, 1.0)])
    m1 = frontier.pareto_frontier([(4.0, 4.0)])
    gain = frontier.arch_compare(m1, m0)
    assert gain == [], "a strictly worse architecture must report an explicit empty set"


# ---------------------------------------------------------------------------
# 5. Diagnostics — tuning-limit and noise-domination
# ---------------------------------------------------------------------------


def test_tuning_limit_flagged_at_band_boundary_without_improvement():
    frontier = _frontier()
    flagged = frontier.tuning_limit(
        config_value=0.40, band=(0.10, 0.40), improved=False
    )
    assert flagged is True, "at the band edge with no improvement the config is tuning-limited"


def test_no_tuning_limit_when_improved():
    frontier = _frontier()
    assert frontier.tuning_limit(config_value=0.40, band=(0.10, 0.40), improved=True) is False


def test_no_tuning_limit_inside_band():
    frontier = _frontier()
    assert frontier.tuning_limit(config_value=0.25, band=(0.10, 0.40), improved=False) is False


def test_noise_domination_flagged_when_spread_covers_range():
    frontier = _frontier()
    flagged = frontier.noise_domination(seed_spread=0.5, config_range=0.3)
    assert flagged is True, "between-seed spread >= between-config range: effect indistinguishable from noise"


def test_no_noise_domination_when_config_range_dominates():
    frontier = _frontier()
    assert frontier.noise_domination(seed_spread=0.2, config_range=0.8) is False


def test_noise_domination_boundary_is_inclusive():
    frontier = _frontier()
    assert frontier.noise_domination(seed_spread=0.5, config_range=0.5) is True


# ---------------------------------------------------------------------------
# 6+7. Report — deterministic JSON + Markdown, gates, ASCII viz, labeling
# ---------------------------------------------------------------------------


def _report_inputs():
    rows = []
    for cfg, seeds in (("c_lo", [11, 22]), ("c_hi", [11, 22])):
        for seed in seeds:
            sr, ls = make_seeded_result(20.0 if cfg == "c_lo" else 8.0, ls=10.0)
            rows.append({"config": cfg, "seed": seed, "score_result": sr})
    return rows


def test_report_json_and_markdown_deterministic():
    frontier = _frontier()
    rows = _report_inputs()
    r1 = frontier.render_report(rows, fixture=True)
    r2 = frontier.render_report(rows, fixture=True)
    assert r1["json"] == r2["json"], "doc §8: byte-identical JSON for identical inputs"
    assert r1["markdown"] == r2["markdown"], "doc §8: byte-identical Markdown for identical inputs"


def test_report_json_is_parseable_and_carves_all_sections():
    frontier = _frontier()
    rep = frontier.render_report(_report_inputs(), fixture=True)
    doc = json.loads(rep["json"])
    for key in ("configs", "seeds", "frontier", "arch_gain", "diagnostics", "gates", "fixture"):
        assert key in doc, f"report JSON must carry section: {key}"


def test_report_markdown_has_ascii_visualization():
    frontier = _frontier()
    md = frontier.render_report(_report_inputs(), fixture=True)["markdown"]
    lines = [ln for ln in md.splitlines() if ln.strip()]
    # A crude ASCII scatter: axis characters or point markers, no plotting libs.
    viz_lines = [
        ln
        for ln in lines
        if any(ch in ln for ch in ("+", "|", "-", "*")) and len(ln) >= 10
    ]
    assert viz_lines, "Markdown report must include an ASCII frontier visualization"


def test_report_marks_gated_out_data_frontier_invalid():
    frontier = _frontier()
    rows = _report_inputs()
    for row in rows:  # 12-agent / 3-day leg: under MIN_COMPARISON_* floors
        row["score_result"] = make_score_result(frontier_valid=False)
        row["n_agents"] = 12
        row["n_days"] = 3
    rep = frontier.render_report(rows, fixture=True)
    doc = json.loads(rep["json"])
    assert doc["gates"]["frontier_valid"] is False, "gated-out data must never read as a valid frontier"
    assert doc["gates"].get("reason"), "the gate verdict must say why"
    assert "frontier_valid: False" in rep["markdown"] or "frontier_valid=False" in rep["markdown"]


def test_report_carries_synthetic_fixture_label_in_both_outputs():
    frontier = _frontier()
    rep = frontier.render_report(_report_inputs(), fixture=True)
    doc = json.loads(rep["json"])
    assert doc.get("fixture") is True, "JSON must label its data synthetic"
    assert "synthetic" in rep["markdown"].lower() or "fixture" in rep["markdown"].lower(), (
        "Markdown must label its data synthetic"
    )


def test_report_includes_incomplete_config_loudly():
    frontier = _frontier()
    rows = [
        {"config": "c_lo", "seed": 11, "score_result": make_seeded_result(20.0, 10.0)[0]},
        {"config": "c_hi", "seed": 11, "score_result": make_seeded_result(8.0, 10.0)[0]},
        # c_hi seed 22 missing entirely
    ]
    rep = frontier.render_report(rows, fixture=True, expected_seeds=[11, 22])
    assert "incomplete" in rep["markdown"].lower(), "a missing leg must be loud in the report"
    doc = json.loads(rep["json"])
    assert any(not cfg.get("complete", True) for cfg in doc["configs"].values()), (
        "JSON must mark the incomplete config loudly too"
    )
