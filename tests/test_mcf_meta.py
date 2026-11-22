"""Offline contract test for the /api/mcf/meta metadata endpoint."""

from gaworld.apps import mcf_api


def test_meta_gates_match_scorer_constants():
    meta = mcf_api.meta()
    from benchmark.mcf import time_use

    assert meta["gates"]["min_comparison_days"] == time_use.MIN_COMPARISON_DAYS
    assert meta["gates"]["min_comparison_population"] == time_use.MIN_COMPARISON_POPULATION


def test_meta_targets_are_real_2024_survey_values():
    meta = mcf_api.meta()
    t = meta["targets"]
    assert t["year"] == 2024 and t["units"] == "minutes/day"
    assert t["categories"]["paid_work"]["minutes"] == 383
    assert t["categories"]["leisure"]["minutes"] == 153


def test_meta_objectives_match_frontier_axes():
    meta = mcf_api.meta()
    keys = [o["key"] for o in meta["objectives"]]
    assert keys == ["paid_work_error", "leisure_social_error"]
    assert all(o["lower_is_better"] for o in meta["objectives"])


def test_bands_expose_sources():
    bands = mcf_api.bands()["bands"]
    style = next(b for b in bands if b["name"] == "style_fit_amplitude")
    assert style["low"] == 0.1 and style["high"] == 0.4
    assert isinstance(style["sources"], list) and style["sources"]
