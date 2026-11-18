"""Step 8 — Sobol sampler over the admissible config space (RED first).

The sampler must give deterministic, low-discrepancy points strictly inside
every admissible band, respect the chosen (non-excluded) parameter set, and
work for power-of-2 and arbitrary n. ``config_space.sample`` is the
one-call surface the MCF API layer is written against.
"""

from __future__ import annotations

import pytest

from benchmark.mcf import config_space


def _bands(names):
    return [config_space.BANDS[name] for name in names]


def _assert_interior(points, bands):
    assert len(points) > 0
    for point in points:
        assert len(point) == len(bands)
        for value, band in zip(point, bands):
            assert band["low"] < value < band["high"], (value, band)


# ---------------------------------------------------------------- sampler


def test_sobol_points_strictly_inside_every_band():
    from benchmark.mcf import sampler

    bands = _bands(["style_fit_amplitude", "contagion_weight", "spending_daily_variance"])
    points = sampler.sample(16, bands, seed=0)
    assert len(points) == 16
    _assert_interior(points, bands)


def test_sobol_is_reproducible_for_seed_and_changes_with_seed():
    from benchmark.mcf import sampler

    bands = _bands(["style_fit_amplitude", "temperature"])
    a = sampler.sample(8, bands, seed=0)
    b = sampler.sample(8, bands, seed=0)
    c = sampler.sample(8, bands, seed=7)
    assert a == b
    assert a != c


def test_sobol_arbitrary_n_and_power_of_two():
    from benchmark.mcf import sampler

    bands = _bands(["contagion_weight"])
    assert len(sampler.sample(5, bands, seed=0)) == 5
    assert len(sampler.sample(8, bands, seed=0)) == 8
    assert len(sampler.sample(1, bands, seed=0)) == 1


def test_sobol_low_discrepancy_coverage():
    from benchmark.mcf import sampler

    bands = _bands(["style_fit_amplitude", "spending_daily_variance"])
    points = sampler.sample(64, bands, seed=0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    # points reach into both tails of every band instead of clumping
    assert min(xs) < config_space.BANDS["style_fit_amplitude"]["low"] + 0.02
    assert max(xs) > config_space.BANDS["style_fit_amplitude"]["high"] - 0.02
    assert min(ys) < config_space.BANDS["spending_daily_variance"]["low"] + 0.015
    assert max(ys) > config_space.BANDS["spending_daily_variance"]["high"] - 0.015
    # no two points coincide
    assert len({tuple(p) for p in points}) == len(points)


def test_sobol_respects_excluded_params():
    """Excluded parameters can never be sampled: they are not manipulable bands."""
    from benchmark.mcf import sampler

    for excluded in config_space.EXCLUDED_PARAMS:
        assert excluded not in {b["path"] for b in config_space.BANDS.values()}
    bands = _bands(["style_fit_amplitude", "contagion_weight", "spending_daily_variance", "temperature"])
    points = sampler.sample(8, bands, seed=0)
    assert all(len(p) == 4 for p in points)


# ------------------------------------------------------------ config_space


def test_config_space_sample_sobol_returns_valid_patches():
    patches = config_space.sample(12, method="sobol")
    assert len(patches) == 12
    flat_paths = {b["path"] for b in config_space.BANDS.values()}
    for patch in patches:
        assert set(patch) <= flat_paths
        nested = config_space.validate_config(patch)  # must not raise
        assert nested


def test_config_space_sample_endpoints_matches_endpoint_configs():
    patches = config_space.sample(99, method="endpoints")
    # endpoint_configs patches are nested; sample() returns the flat form
    expected = []
    for ep in config_space.endpoint_configs():
        nested = ep["patch"]
        flat = {}
        stack = [(nested, "")]
        while stack:
            obj, prefix = stack.pop()
            for key, value in obj.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    stack.append((value, path))
                else:
                    flat[path] = value
        expected.append(flat)
    assert patches == expected


def test_config_space_sample_is_deterministic():
    assert config_space.sample(8, method="sobol") == config_space.sample(8, method="sobol")


def test_config_space_sample_rejects_unknown_method():
    with pytest.raises(ValueError):
        config_space.sample(4, method="latin-hypercube")
