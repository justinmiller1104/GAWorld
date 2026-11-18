"""Phase 2.8 — per-run outcome extraction from existing recorded tables.

Every metric marks in code whether its source data exists. A metric whose
source data is missing returns the explicit ``NOT_MEASURED`` marker —
never ``None``-by-accident, never a fabricated number.

Sources (all pre-existing tables):
- ``venue.admission`` (Phase 2.8): per-attempt outcomes → rates,
  redirect frequency, congestion;
- ``locations``: destination changes;
- ``movement`` (travel.status episodes): travel / participating minutes.
Route segment lengths and schedule completion are not recorded anywhere
today, so travel distance and activity completion are always
``NOT_MEASURED`` rather than derived guesses.
"""

from __future__ import annotations

NOT_MEASURED = "not_measured"


def _minutes_between(day0, time0, day1, time1):
    def _clock(day, time_str):
        h, _, m = str(time_str).partition(":")
        return int(day) * 1440 + int(h) * 60 + int(m)

    try:
        return _clock(day1, time1) - _clock(day0, time0)
    except (TypeError, ValueError):
        return None


def _admission_metrics(tables):
    from benchmark.mcf.phase28_metrics import admission_rates

    rates = admission_rates(tables.get("venue.admission") or [])
    if rates["attempts"] == 0:
        for key in ("admission_rate", "redirect_rate", "denial_rate"):
            rates[key] = NOT_MEASURED
        rates["redirect_frequency"] = NOT_MEASURED
    else:
        rates["redirect_frequency"] = rates["redirected"] / rates["attempts"]
    return rates


def _destination_changes(tables):
    locations = tables.get("locations")
    if not locations:
        return NOT_MEASURED
    changes = 0
    by_agent = {}
    for row in sorted(locations, key=lambda r: (r.get("_day", 0), str(r.get("_time", "")))):
        prev = by_agent.setdefault(row.get("agent_id"), row.get("current"))
        if row.get("current") != prev:
            changes += 1
        by_agent[row.get("agent_id")] = row.get("current")
    return changes


def _travel_metrics(tables):
    movement = tables.get("movement")
    if not movement:
        return {"travel_minutes": NOT_MEASURED,
                "minutes_travelling": NOT_MEASURED,
                "minutes_participating": NOT_MEASURED}
    travel = 0.0
    participating = 0.0
    open_trip = None  # (day, time) of a departed without its arrived yet
    arrival = None  # (day, time) of the last arrived, until next departure
    for row in sorted(movement, key=lambda r: (r.get("_day", 0), str(r.get("_time", "")))):
        status = row.get("status")
        if status == "departed":
            if arrival is not None and open_trip is None:
                span = _minutes_between(*arrival, row.get("_day"), row.get("_time"))
                if span is not None:
                    participating += span
                arrival = None
            open_trip = (row.get("_day"), row.get("_time"))
        elif status == "arrived" and open_trip is not None:
            span = _minutes_between(*open_trip, row.get("_day"), row.get("_time"))
            if span is not None:
                travel += span
            open_trip = None
            arrival = (row.get("_day"), row.get("_time"))
    return {
        "travel_minutes": travel,
        "minutes_travelling": travel,
        "minutes_participating": participating,
    }


def _congestion(tables):
    peak_occupancy = {}
    peak_load_ratio = {}
    for row in tables.get("venue.admission") or []:
        venue = row.get("to")
        occupancy, capacity = row.get("occupancy"), row.get("capacity")
        if venue is None or occupancy is None or capacity in (None, 0):
            continue
        peak_occupancy[venue] = max(peak_occupancy.get(venue, occupancy), occupancy)
        ratio = occupancy / capacity
        peak_load_ratio[venue] = max(peak_load_ratio.get(venue, ratio), ratio)
    return {"peak_occupancy": peak_occupancy, "peak_load_ratio": peak_load_ratio}


def summarize(tables):
    """Per-run outcome measures from recorded tables (see module docstring).

    Travel distance and activity completion have no recorded source data
    today — they are always ``NOT_MEASURED``.
    """
    admission = _admission_metrics(tables)
    result = {
        "redirect_frequency": admission.pop("redirect_frequency"),
        "admission": admission,
        "destination_changes": _destination_changes(tables),
        "travel_distance": NOT_MEASURED,  # no route/segment lengths recorded
        "activity_completion": NOT_MEASURED,  # no schedule-completion table
    }
    result.update(_travel_metrics(tables))
    congestion = _congestion(tables)
    result["congestion"] = congestion if congestion["peak_occupancy"] else NOT_MEASURED
    return result
