"""Regression: A(M1|M0) must exclude WEAKLY dominated points (docs/MCF_GAWORLD.md §4, paper Eq. 2).

Tie on one axis: (1,3) weakly dominates (2,3) (better-or-equal everywhere,
strict on x) — so the (2,3) point must NOT appear in the gain set.
"""

from importlib import import_module


def _frontier():
    return import_module("benchmark.mcf.frontier")


def test_arch_compare_excludes_weakly_dominated_tie():
    f = _frontier()
    assert f.arch_compare([(2, 3)], [(1, 3)]) == []


def test_arch_compare_weak_dominance_multi_axis():
    f = _frontier()
    # equal-or-better on both axes excludes; equal points already excluded
    assert f.arch_compare([(2, 3), (1, 4)], [(1, 3)]) == []
    # worse on one axis, better on the other survives (not comparable)
    assert f.arch_compare([(3, 2)], [(1, 3)]) == [(3, 2)]


def test_pareto_frontier_tie_semantics_unchanged():
    # Pareto extraction keeps its documented strict-domination semantics:
    # points tied on one axis both survive on the front.
    f = _frontier()
    assert sorted(f.pareto_frontier([(2, 3), (1, 3)])) == [(1, 3), (2, 3)]
