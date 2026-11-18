"""RED tests — frontier None/NaN robustness (task 3753 sub-task B).

A point carrying a None or NaN component must never enter the dominance
computation: it is excluded (with a machine-readable reason), not crashed on
and not silently compared. Documented choice: exclusion-with-reason, because
one malformed leg should not void the whole frontier computation.
"""

from __future__ import annotations

from benchmark.mcf.frontier import pareto_frontier


def test_clean_points_unchanged():
    points = [(1.0, 5.0), (2.0, 3.0), (3.0, 3.0)]
    assert pareto_frontier(points) == [(1.0, 5.0), (2.0, 3.0), (3.0, 3.0)]


def test_none_component_excluded_with_reason():
    report: list[dict] = []
    front = pareto_frontier([(1.0, 5.0), (None, 1.0)], report=report)
    assert front == [(1.0, 5.0)]
    assert len(report) == 1
    assert report[0]["point"] == [None, 1.0]
    assert "None" in report[0]["reason"] or "none" in report[0]["reason"].lower()


def test_nan_component_excluded():
    report: list[dict] = []
    front = pareto_frontier([(1.0, 5.0), (float("nan"), 1.0), (2.0, 2.0)], report=report)
    assert (1.0, 5.0) in front and (2.0, 2.0) in front
    assert len(report) == 1


def test_all_points_invalid_yields_empty_front_not_crash():
    report: list[dict] = []
    assert pareto_frontier([(None, None), (float("nan"), 0.0)], report=report) == []
    assert len(report) == 2


def test_invalid_point_does_not_win_by_default_nan_compare():
    # NaN comparisons are always False, so an unsanitized NaN point would
    # silently survive dominance; sanitized it must not appear on the front.
    front = pareto_frontier([(float("nan"), float("nan")), (1.0, 1.0)])
    assert front == [(1.0, 1.0)]
