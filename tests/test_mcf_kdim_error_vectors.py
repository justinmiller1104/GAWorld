"""Step 9 — k-dimensional error vectors (axes registry, ≥3 objectives).

``frontier`` must not be hard-wired to the two time-use axes: an axes
registry maps objective names to extractors over ``ScoreResult.per_category``,
assembly/averaging/dominance/comparison all work on k dimensions (with ties
and missing-data exclusions), and the default 2-D time-use behavior is
unchanged.
"""

from __future__ import annotations

import pytest

from benchmark.mcf import frontier

THREE_AXES = {
    "paid_work_error": lambda pc: float(pc["paid_work"]["error"]),
    "leisure_social_error": lambda pc: (float(pc["leisure"]["error"]) + float(pc["socializing"]["error"])) / 2.0,
    "sleep_error": lambda pc: float(pc["sleep"]["error"]),
}


def score_result(paid, leisure, social, sleep, *, valid=True, fixture=True):
    return type(
        "ScoreResult",
        (),
        {
            "per_category": {
                "paid_work": {"error": paid},
                "leisure": {"error": leisure},
                "socializing": {"error": social},
                "sleep": {"error": sleep},
            },
            "frontier_valid": valid,
            "fixture": fixture,
        },
    )()


# ---------------------------------------------------------------- assembly


def test_three_objective_assembly():
    v = frontier.assemble_error_vector(
        score_result(0.20, 0.10, 0.14, 0.30), config="c1", seed=7, fixture=True, axes=THREE_AXES
    )
    assert v["paid_work_error"] == pytest.approx(0.20)
    assert v["leisure_social_error"] == pytest.approx(0.12)
    assert v["sleep_error"] == pytest.approx(0.30)
    assert v["excluded"] is False


def test_missing_dimension_raises():
    sr = score_result(0.20, 0.10, 0.14, 0.30)
    del sr.per_category["sleep"]
    with pytest.raises(KeyError):
        frontier.assemble_error_vector(sr, config="c1", seed=7, fixture=True, axes=THREE_AXES)


def test_gated_leg_is_excluded_with_reason():
    v = frontier.assemble_error_vector(
        score_result(0.2, 0.1, 0.1, 0.3, valid=False), config="c1", seed=7, fixture=True, axes=THREE_AXES
    )
    assert v["excluded"] is True
    assert "frontier_valid=False" in v["reason"]


def test_default_two_dimensional_assembly_unchanged():
    v = frontier.assemble_error_vector(score_result(0.20, 0.10, 0.14, 0.30), config="c", seed=1, fixture=True)
    assert set(v) == {
        "config",
        "seed",
        "paid_work_error",
        "leisure_social_error",
        "frontier_valid",
        "fixture",
        "excluded",
    }


def test_average_across_seeds_k_dimensions_and_missing_seeds():
    seeds = {
        7: frontier.assemble_error_vector(
            score_result(0.20, 0.10, 0.14, 0.30), config="c1", seed=7, fixture=True, axes=THREE_AXES
        ),
        11: frontier.assemble_error_vector(
            score_result(0.24, 0.12, 0.16, 0.34), config="c1", seed=11, fixture=True, axes=THREE_AXES
        ),
    }
    avg = frontier.average_across_seeds("c1", seeds, expected_seeds=[7, 11, 13], axes=list(THREE_AXES))
    assert avg["paid_work_error"] == pytest.approx(0.22)
    assert avg["sleep_error"] == pytest.approx(0.32)
    assert avg["complete"] is False
    assert avg["missing_seeds"] == [13]
    assert avg["paid_work_spread"] == pytest.approx(0.02)
    assert avg["sleep_spread"] == pytest.approx(0.02)


# ---------------------------------------------------------- dominance / A


def test_pareto_frontier_three_dimensions_with_ties():
    pts = [(1.0, 1.0, 3.0), (1.0, 1.0, 2.0), (2.0, 2.0, 1.0), (3.0, 3.0, 3.0)]
    front = frontier.pareto_frontier(pts)
    # weak domination: points tied on one axis neither dominate nor exclude
    # each other, so both survive; strictly worse points drop off
    assert (1.0, 1.0, 2.0) in front
    assert (1.0, 1.0, 3.0) in front
    assert (2.0, 2.0, 1.0) in front
    assert (3.0, 3.0, 3.0) not in front


def test_pareto_frontier_exact_ties_both_survive():
    pts = [(1.0, 2.0, 1.0), (1.0, 2.0, 1.0), (1.5, 1.5, 1.5)]
    front = frontier.pareto_frontier(pts)
    assert front == [(1.0, 2.0, 1.0), (1.5, 1.5, 1.5)]


def test_arch_compare_three_dimensions():
    m1 = [(1.0, 1.0, 2.0), (2.5, 2.5, 0.5)]
    m0 = [(0.9, 1.0, 2.0), (2.0, 2.0, 2.0)]
    gain = frontier.arch_compare(m1, m0)
    # (1,1,2) weakly dominated by m0's (0.9,1.0,2.0) → excluded; the
    # trade-off point (2.5,2.5,0.5) is dominated by nothing → newly attainable
    assert gain == [(2.5, 2.5, 0.5)]


def test_dimensions_are_pinned_agnotstic():
    """pareto_frontier/arch_compare stay correct on 4+ dimensions (pin)."""
    pts4 = [(1.0, 2.0, 1.0, 0.5), (1.0, 2.0, 1.2, 0.4), (2.0, 1.0, 2.0, 0.6)]
    front = frontier.pareto_frontier(pts4)
    assert (1.0, 2.0, 1.0, 0.5) in front and (1.0, 2.0, 1.2, 0.4) in front
    assert (2.0, 1.0, 2.0, 0.6) in front
    assert frontier.arch_compare(pts4, [(9.9, 9.9, 9.9, 9.9)]) == sorted(pts4)


def test_axes_registry_roundtrip():
    frontier.register_axis("sleep_error", THREE_AXES["sleep_error"])
    try:
        names = frontier.axes_from_names(["paid_work_error", "leisure_social_error", "sleep_error"])
        v = frontier.assemble_error_vector(
            score_result(0.2, 0.1, 0.1, 0.3), config="c", seed=7, fixture=True, axes=names
        )
        assert v["sleep_error"] == pytest.approx(0.3)
    finally:
        frontier.AXES.pop("sleep_error", None)
