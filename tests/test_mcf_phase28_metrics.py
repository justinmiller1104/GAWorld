"""Phase 2.8 RED tests — explicit-denominator rate metrics (TASK-3675, spec §2).

``benchmark/mcf/phase28_metrics.py`` does not exist yet: the whole module is
the GREEN-phase deliverable, so every test below fails today with
ModuleNotFoundError (a missing feature, not a syntax error).

Contract under test:
- denominators are EXPLICIT: ``attempts == len(venue.admission rows)``;
- ``admission_rate`` / ``redirect_rate`` / ``denial_rate`` are outcome counts
  divided by attempts;
- zero attempts → rates are ``None`` (undefined), never fabricated as 0.0;
- the legacy outcome tables ``venue.redirect`` / ``action.denied`` are passed
  through untouched (their row counts are returned as-is for compatibility).
"""

from __future__ import annotations

import pytest


def _mod():
    import benchmark.mcf.phase28_metrics as m

    return m


def _row(outcome, agent_id=1, to="ShopA"):
    return {"agent_id": agent_id, "to": to, "outcome": outcome}


class TestExplicitDenominators:
    """attempts = number of venue.admission rows, whatever their outcomes."""

    def test_attempts_is_row_count(self):
        rows = [_row("admitted"), _row("denied", 2), _row("redirected", 3)]
        assert _mod().attempts_of(rows) == 3

    def test_attempts_of_empty_is_zero(self):
        assert _mod().attempts_of([]) == 0


class TestRatesFromAdmissionRows:
    def test_all_admitted(self):
        rows = [_row("admitted"), _row("admitted", 2)]
        rates = _mod().admission_rates(rows)
        assert rates["attempts"] == 2
        assert rates["admission_rate"] == 1.0
        assert rates["redirect_rate"] == 0.0
        assert rates["denial_rate"] == 0.0

    def test_mixed_outcomes(self):
        rows = [
            _row("admitted"),
            _row("admitted", 2),
            _row("redirected", 3),
            _row("denied", 4),
            _row("denied", 5),
        ]
        rates = _mod().admission_rates(rows)
        assert rates["attempts"] == 5
        assert rates["admission_rate"] == pytest.approx(2 / 5)
        assert rates["redirect_rate"] == pytest.approx(1 / 5)
        assert rates["denial_rate"] == pytest.approx(2 / 5)

    def test_zero_attempts_rates_are_none_never_fabricated(self):
        rates = _mod().admission_rates([])
        assert rates["attempts"] == 0
        assert rates["admission_rate"] is None, "undefined rate must be None, not 0.0"
        assert rates["redirect_rate"] is None
        assert rates["denial_rate"] is None

    def test_rates_are_between_zero_and_one_when_defined(self):
        rows = [_row("admitted"), _row("denied", 2), _row("redirected", 3)]
        rates = _mod().admission_rates(rows)
        for key in ("admission_rate", "redirect_rate", "denial_rate"):
            assert 0.0 <= rates[key] <= 1.0, key


class TestLegacyTablesUntouched:
    """The metrics layer must not rewrite or merge legacy outcome tables."""

    def test_redirect_and_denied_rows_pass_through(self):
        redirects = [{"agent_id": 1, "from": "ShopA", "to": "ShopB"}]
        denied = [{"agent_id": 2, "action": "move", "reason": "full"}]
        rates = _mod().admission_rates([], redirects=redirects, denied=denied)
        assert rates["venue_redirect_rows"] == 1
        assert rates["action_denied_rows"] == 1


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
