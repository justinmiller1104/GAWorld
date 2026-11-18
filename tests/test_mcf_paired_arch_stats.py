"""Step 10 — matched-config paired architecture statistics.

Per axis, over seeds matched between two architectures at the same config:
per-seed paired differences, improvement counts, standardized mean difference
dz, and an exact two-sided sign test — axes are never collapsed into one
score. Ties count separately; missing seeds are reported, not silently
dropped from the summary.
"""

from __future__ import annotations

import pytest

from benchmark.mcf import frontier

AXES = ["paid_work_error", "leisure_social_error"]


def vec(seed, paid, leisure):
    return {
        "config": "cfg",
        "seed": seed,
        "paid_work_error": paid,
        "leisure_social_error": leisure,
        "excluded": False,
    }


def test_paired_stats_improvement_on_both_axes():
    m0 = {7: vec(7, 0.20, 0.10), 11: vec(11, 0.22, 0.12), 13: vec(13, 0.30, 0.16)}
    m1 = {7: vec(7, 0.10, 0.06), 11: vec(11, 0.14, 0.08), 13: vec(13, 0.20, 0.10)}
    out = frontier.paired_arch_stats("m1", m1, "m0", m0, axes=AXES)
    for axis in AXES:
        stat = out["axes"][axis]
        assert stat["n"] == 3
        assert stat["improvements"] == 3
        assert stat["worsenings"] == 0
        assert stat["ties"] == 0
        assert stat["dz"] > 0
        assert stat["sign_test_p"] <= 0.25  # exact binomial: 3/3 → p = 0.25 two-sided
        assert stat["verdict"] == "improved"
    # axes are reported separately, never collapsed
    assert set(out["axes"]) == set(AXES)
    assert "combined" not in out


def test_paired_stats_per_seed_differences_exposed():
    m0 = {7: vec(7, 0.20, 0.10), 11: vec(11, 0.22, 0.12)}
    m1 = {7: vec(7, 0.10, 0.06), 11: vec(11, 0.14, 0.08)}
    out = frontier.paired_arch_stats("m1", m1, "m0", m0, axes=AXES)
    stat = out["axes"]["paid_work_error"]
    assert stat["diffs"] == pytest.approx([-0.10, -0.08])
    assert stat["matched_seeds"] == [7, 11]


def test_paired_stats_ties_and_mixed_direction():
    m0 = {7: vec(7, 0.20, 0.10), 11: vec(11, 0.22, 0.12), 13: vec(13, 0.30, 0.16)}
    m1 = {7: vec(7, 0.20, 0.12), 11: vec(11, 0.10, 0.12), 13: vec(13, 0.40, 0.14)}
    out = frontier.paired_arch_stats("m1", m1, "m0", m0, axes=AXES)
    paid = out["axes"]["paid_work_error"]
    assert paid["ties"] == 1
    assert paid["improvements"] == 1
    assert paid["worsenings"] == 1
    assert paid["verdict"] == "no-difference"  # tie excluded from the sign test
    leisure = out["axes"]["leisure_social_error"]
    assert leisure["improvements"] == 1 and leisure["worsenings"] == 1 and leisure["ties"] == 1
    assert leisure["verdict"] == "no-difference"
    paid = out["axes"]["paid_work_error"]
    assert paid["mean_diff"] == pytest.approx(-1 / 150)  # (0 − 0.12 + 0.10)/3 over all three pairs


def test_paired_stats_missing_seed_reported():
    m0 = {7: vec(7, 0.20, 0.10), 11: vec(11, 0.22, 0.12), 13: vec(13, 0.30, 0.16)}
    m1 = {7: vec(7, 0.10, 0.06), 11: vec(11, 0.14, 0.08)}  # seed 13 leg failed
    out = frontier.paired_arch_stats("m1", m1, "m0", m0, axes=AXES)
    assert out["unmatched"] == {"m0": [13], "m1": []}
    assert out["axes"]["paid_work_error"]["matched_seeds"] == [7, 11]
    assert out["complete"] is False


def test_paired_stats_sign_test_counts_ties_as_neither():
    m0 = {7: vec(7, 0.20, 0.10), 11: vec(11, 0.20, 0.10)}
    m1 = {7: vec(7, 0.10, 0.10), 11: vec(11, 0.10, 0.10)}
    out = frontier.paired_arch_stats("m1", m1, "m0", m0, axes=AXES)
    paid = out["axes"]["paid_work_error"]
    assert paid["improvements"] == 2 and paid["sign_test_p"] == pytest.approx(0.5)
    leisure = out["axes"]["leisure_social_error"]
    assert leisure["ties"] == 2 and leisure["sign_test_p"] == pytest.approx(1.0)


def test_paired_stats_default_axes():
    m0 = {7: vec(7, 0.20, 0.10)}
    m1 = {7: vec(7, 0.10, 0.06)}
    out = frontier.paired_arch_stats("m1", m1, "m0", m0)
    assert set(out["axes"]) == {"paid_work_error", "leisure_social_error"}
