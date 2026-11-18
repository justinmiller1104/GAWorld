"""Phase 2.8 RED tests — per-run outcome extraction (spec §4).

``benchmark/mcf/phase28_outcomes.py`` does not exist yet (GREEN deliverable):
every test fails with ModuleNotFoundError — missing feature, not a syntax error.

Contract under test: from EXISTING recorded tables (location series,
``venue.admission``, ``venue.redirect``, ``action.denied``) code computes
per-run outcome measures:
- admission / denial / redirect rates (denominators = venue.admission rows);
- redirect frequency, destination changes count;
- travel time / distance, time travelling vs participating, activity
  completion, venue occupancy / congestion — each only when the source data
  actually supports it;
- any metric whose source data is missing must return the explicit
  ``NOT_MEASURED`` marker — never a fabricated number.
"""

from __future__ import annotations

import pytest

NOT_MEASURED = "not_measured"


def _mod():
    import benchmark.mcf.phase28_outcomes as m

    return m


def _admission_rows():
    return [
        {"agent_id": 1, "to": "ShopA", "outcome": "admitted", "capacity": 2,
         "occupancy": 1, "agents_represent": 1.0, "_day": 1, "_time": "09:00"},
        {"agent_id": 2, "to": "ShopA", "outcome": "redirected", "capacity": 1,
         "occupancy": 1, "agents_represent": 1.0, "_day": 1, "_time": "09:10"},
        {"agent_id": 3, "to": "ShopA", "outcome": "denied", "capacity": 1,
         "occupancy": 1, "agents_represent": 1.0, "_day": 1, "_time": "09:20"},
    ]


class TestRatesFromTables:
    def test_rates_use_admission_rows_as_denominator(self):
        m = _mod()
        s = m.summarize({"venue.admission": _admission_rows()})
        assert s["admission"]["attempts"] == 3
        assert s["admission"]["admission_rate"] == pytest.approx(1 / 3)
        assert s["admission"]["redirect_rate"] == pytest.approx(1 / 3)
        assert s["admission"]["denial_rate"] == pytest.approx(1 / 3)

    def test_rates_without_attempts_are_not_measured(self):
        s = _mod().summarize({})
        assert s["admission"]["admission_rate"] == NOT_MEASURED
        assert s["admission"]["attempts"] == 0


class TestRedirectAndDestinationMetrics:
    def test_redirect_frequency_from_rows(self):
        s = _mod().summarize({"venue.admission": _admission_rows()})
        assert s["redirect_frequency"] == pytest.approx(1 / 3)

    def test_destination_changes_from_location_series(self):
        locations = [
            {"agent_id": 1, "_day": 1, "_time": "09:00", "current": "Home"},
            {"agent_id": 1, "_day": 1, "_time": "10:00", "current": "ShopA"},
            {"agent_id": 1, "_day": 1, "_time": "11:00", "current": "ShopA"},
            {"agent_id": 1, "_day": 1, "_time": "12:00", "current": "Home"},
        ]
        s = _mod().summarize({"locations": locations})
        assert s["destination_changes"] == 2


class TestTravelMetrics:
    def test_travel_time_from_movement_records(self):
        # travel.status episodes with departed/in_transit/arrived markers
        movement = [
            {"agent_id": 1, "_day": 1, "_time": "09:00", "status": "departed", "to": "ShopA"},
            {"agent_id": 1, "_day": 1, "_time": "09:30", "status": "arrived", "at": "ShopA"},
            {"agent_id": 1, "_day": 1, "_time": "10:00", "status": "departed", "to": "Home"},
            {"agent_id": 1, "_day": 1, "_time": "10:20", "status": "arrived", "at": "Home"},
        ]
        s = _mod().summarize({"movement": movement})
        assert s["travel_minutes"] == pytest.approx(50.0)

    def test_time_travelling_vs_participating(self):
        movement = [
            {"agent_id": 1, "_day": 1, "_time": "09:00", "status": "departed", "to": "ShopA"},
            {"agent_id": 1, "_day": 1, "_time": "09:30", "status": "arrived", "at": "ShopA"},
            {"agent_id": 1, "_day": 1, "_time": "10:30", "status": "departed", "to": "Home"},
            {"agent_id": 1, "_day": 1, "_time": "11:00", "status": "arrived", "at": "Home"},
        ]
        s = _mod().summarize({"movement": movement})
        assert s["minutes_travelling"] == pytest.approx(60.0)
        assert s["minutes_participating"] == pytest.approx(60.0)

    def test_travel_distance_without_route_data_is_not_measured(self):
        # No route/segment lengths anywhere in the tables → must NOT be a number.
        s = _mod().summarize({"venue.admission": _admission_rows()})
        assert s["travel_distance"] == NOT_MEASURED

    def test_activity_completion_without_schedule_is_not_measured(self):
        s = _mod().summarize({"venue.admission": _admission_rows()})
        assert s["activity_completion"] == NOT_MEASURED


class TestOccupancyCongestion:
    def test_peak_occupancy_from_admission_rows(self):
        rows = [
            {"agent_id": 1, "to": "ShopA", "outcome": "admitted", "occupancy": 1,
             "capacity": 3, "_day": 1, "_time": "09:00"},
            {"agent_id": 2, "to": "ShopA", "outcome": "admitted", "occupancy": 2,
             "capacity": 3, "_day": 1, "_time": "10:00"},
            {"agent_id": 3, "to": "ShopB", "outcome": "denied", "occupancy": 3,
             "capacity": 3, "_day": 1, "_time": "11:00"},
        ]
        s = _mod().summarize({"venue.admission": rows})
        assert s["congestion"]["peak_occupancy"]["ShopA"] == 2
        assert s["congestion"]["peak_load_ratio"]["ShopB"] == pytest.approx(1.0)


class TestHonestyInvariants:
    """Every metric key must be either a number or the explicit marker —
    never None-by-accident, never a fabricated default."""

    def test_all_reported_keys_are_number_or_marker(self):
        s = _mod().summarize({})
        for key, value in s.items():
            if key == "admission":
                continue  # checked elsewhere
            if isinstance(value, dict):
                values = value.values()
            else:
                values = [value]
            for v in values:
                assert v == NOT_MEASURED or isinstance(v, (int, float)), (key, v)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
