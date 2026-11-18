"""Phase 2.8 RED tests — multi-arm aggregation with honest intervals (spec §5).

``benchmark/mcf/phase28_aggregate.py`` does not exist yet (GREEN deliverable):
every test fails with ModuleNotFoundError — missing feature, not a syntax error.

Contract under test, on labeled synthetic fixtures (6 runs across 3 arms):
- per-arm admission/denial/redirect rates with 95% Bayesian credible
  intervals (beta-binomial, Jeffreys Beta(0.5, 0.5) prior);
- incomplete runs / runs with missing outcomes must raise LOUDLY or be
  excluded with an explicit marker — they must NEVER silently contribute to
  a valid-looking effect estimate.
"""

from __future__ import annotations

import pytest


def _mod():
    import benchmark.mcf.phase28_aggregate as m

    return m


def _run(run_id, arm, admitted, denied, redirected=0):
    """A complete run fixture: labeled arm + full outcome counts."""
    attempts = admitted + denied + redirected
    rows = (
        [{"outcome": "admitted"}] * admitted
        + [{"outcome": "denied"}] * denied
        + [{"outcome": "redirected"}] * redirected
    )
    return {
        "run_id": run_id,
        "arm": arm,
        "venue.admission": [{"outcome": r["outcome"]} for r in rows],
        "attempts": attempts,
    }


def _six_runs():
    return [
        _run("r1", "baseline", admitted=8, denied=1, redirected=1),
        _run("r2", "baseline", admitted=9, denied=1, redirected=0),
        _run("r3", "neutral_control", admitted=8, denied=2, redirected=0),
        _run("r4", "neutral_control", admitted=7, denied=2, redirected=1),
        _run("r5", "capacity_shock", admitted=4, denied=4, redirected=2),
        _run("r6", "capacity_shock", admitted=3, denied=5, redirected=2),
    ]


class TestPerArmRates:
    def test_six_runs_three_arms(self):
        agg = _mod().aggregate(_six_runs())
        assert set(agg["arms"]) == {"baseline", "neutral_control", "capacity_shock"}

    def test_per_arm_rates_pool_their_runs(self):
        agg = _mod().aggregate(_six_runs())
        base = agg["arms"]["baseline"]
        assert base["attempts"] == 20
        assert base["admission_rate"] == pytest.approx(17 / 20)
        assert base["denial_rate"] == pytest.approx(2 / 20)
        assert base["redirect_rate"] == pytest.approx(1 / 20)

    def test_jeffreys_credible_interval_brackets_the_point(self):
        agg = _mod().aggregate(_six_runs())
        base = agg["arms"]["baseline"]
        lo, hi = base["admission_ci95"]
        assert 0.0 <= lo <= base["admission_rate"] <= hi <= 1.0

    def test_interval_width_shrinks_with_more_attempts(self):
        few = _mod().aggregate([_run("r1", "baseline", admitted=1, denied=0)])
        many = _mod().aggregate([_run("r1", "baseline", admitted=90, denied=10)])
        w_few = few["arms"]["baseline"]["admission_ci95"][1] - few["arms"]["baseline"]["admission_ci95"][0]
        w_many = many["arms"]["baseline"]["admission_ci95"][1] - many["arms"]["baseline"]["admission_ci95"][0]
        assert w_many < w_few, "Jeffreys beta-binomial CI must narrow with n"

    def test_zero_attempts_arm_reports_no_fabricated_rate(self):
        agg = _mod().aggregate([_run("r1", "baseline", admitted=0, denied=0)])
        base = agg["arms"]["baseline"]
        assert base["attempts"] == 0
        assert base["admission_rate"] is None
        assert base["admission_ci95"] is None


class TestIncompleteRunsAreLoud:
    def test_run_with_missing_outcome_rows_raises(self):
        broken = {"run_id": "rX", "arm": "baseline", "venue.admission": None, "attempts": None}
        with pytest.raises(_mod().IncompleteRunError):
            _mod().aggregate(_six_runs() + [broken])

    def test_run_without_arm_label_raises(self):
        broken = _run("rX", None, admitted=1, denied=0)
        with pytest.raises(_mod().IncompleteRunError):
            _mod().aggregate(_six_runs() + [broken])

    def test_excluded_runs_are_reported_not_silently_dropped(self):
        broken = {"run_id": "rX", "arm": "baseline", "venue.admission": None, "attempts": None}
        agg = _mod().aggregate(_six_runs() + [broken], on_incomplete="exclude")
        assert "rX" in agg["excluded"], "an excluded run must be named in the output"
        assert agg["arms"]["baseline"]["attempts"] == 20  # unchanged by the broken run


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
