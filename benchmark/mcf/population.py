"""Population resolution for MCF experiments (city + residents + duration).

Bridges the experiment API to the real simulation inputs:

- ``gaworld.city.bundle`` city bundles: each city is a directory whose
  ``agents.csv`` roster carries the resident IDs and the explicit
  ``employment`` classification column the economy/population pipeline uses;
- the simulator's CONFIG keys (``city``, ``csv_path``, ``map_path``,
  ``agent_ids``, ``sim_days``) that ``generative_city_sim.run_simulation``
  consumes at module scope.

Nothing here invents residents: every ID comes from the selected city's
roster, and employment comes from that roster's ``employment`` column —
never guessed from names.
"""

from __future__ import annotations

import csv
import random
from typing import Any

from gaworld.city.bundle import CityNotFoundError, resolve_city

__all__ = [
    "PopulationError",
    "city_roster",
    "resolve_population",
    "random_sample",
    "config_overrides_for",
]

EMPLOYED_VALUE = "employed"


class PopulationError(ValueError):
    """Raised when a population selection cannot be honoured as asked."""


def _bundle(city: str) -> Any:
    try:
        return resolve_city(city)
    except CityNotFoundError as exc:
        raise PopulationError(f"unknown city: {city!r}") from None


def city_roster(city: str) -> list[dict[str, Any]]:
    """The selected city's residents: exact int IDs + employment column."""
    bundle = _bundle(city)
    csv_path = bundle.state_csv_path
    if not csv_path.exists():
        raise PopulationError(f"city {city!r} has no resident roster ({csv_path})")
    roster: list[dict[str, Any]] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                resident_id = int(row["id"])
            except (KeyError, TypeError, ValueError):
                continue
            roster.append(
                {
                    "id": resident_id,
                    "name": row.get("name", ""),
                    "employment": (row.get("employment") or "").strip(),
                }
            )
    if not roster:
        raise PopulationError(f"city {city!r} has an empty resident roster")
    return roster


def random_sample(ids: list[int], size: int, seed: int) -> list[int]:
    """Reproducible sample WITHOUT replacement from the complete id list."""
    if size > len(ids):
        raise PopulationError(
            f"sample size {size} exceeds the city population ({len(ids)} residents)"
        )
    rng = random.Random(seed)
    return sorted(rng.sample(list(ids), size))


def resolve_population(population: dict[str, Any]) -> dict[str, Any]:
    """Validate and fully resolve a wizard population selection.

    Returns the persisted, provenance-carrying form: explicit agent IDs
    (expanded even for random/all so the manifest never depends on re-sampling),
    derived employed IDs from the roster's employment column, and the city
    bundle identity.
    """
    city = str(population.get("city") or "").strip()
    if not city:
        raise PopulationError("population.city is required")
    method = str(population.get("method") or "").strip()
    if method not in ("manual", "random", "all"):
        raise PopulationError("population.method must be one of: manual, random, all")

    roster = city_roster(city)
    roster_ids = [r["id"] for r in roster]
    by_id = {r["id"]: r for r in roster}

    if method == "manual":
        requested = population.get("agent_ids") or []
        if isinstance(requested, str):
            requested = [part for part in requested.replace(";", ",").split(",") if part.strip()]
        try:
            ids = sorted({int(i) for i in requested})
        except (TypeError, ValueError):
            raise PopulationError("population.agent_ids must be a list of integer resident IDs") from None
        if not ids:
            raise PopulationError("manual selection needs at least one resident")
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            raise PopulationError(
                f"resident IDs not in city {city!r}: {unknown[:10]}{'…' if len(unknown) > 10 else ''}"
            )
    elif method == "random":
        try:
            size = int(population.get("sample_size"))
            seed = int(population.get("sample_seed"))
        except (TypeError, ValueError):
            raise PopulationError("random selection needs integer sample_size and sample_seed") from None
        if size < 1:
            raise PopulationError("sample_size must be >= 1")
        if seed < 0:
            raise PopulationError("sample_seed must be a non-negative integer")
        ids = random_sample(roster_ids, size, seed)
    else:  # all
        ids = sorted(roster_ids)

    id_set = set(ids)
    employed = sorted(r["id"] for r in roster if r["id"] in id_set and r["employment"] == EMPLOYED_VALUE)
    unknown_employment = sum(1 for r in roster if r["id"] in id_set and not r["employment"])

    bundle = _bundle(city)
    return {
        "city": bundle.slug,
        "city_name": bundle.name,
        "method": method,
        "sample_seed": int(population["sample_seed"]) if method == "random" else None,
        "agent_ids": ids,
        "n_agents": len(ids),
        "employed_agent_ids": employed,
        "n_employed": len(employed),
        "unknown_employment": unknown_employment,
        "population_source": str(bundle.state_csv_path),
    }


def config_overrides_for(city: str) -> dict[str, Any]:
    """The CONFIG overrides that point the simulator at this city's data."""
    bundle = _bundle(city)
    overrides = bundle.paths_for_config()
    overrides["city"] = bundle.slug
    return overrides
