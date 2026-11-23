"""M1 RED tests — admissible Θ config space + endpoint sampler.

Spec (docs/MCF_GAWORLD.md §3, MCF_COMPLETION_PLAN.md M1):
four admissible bands, excluded params rejected, stress 0.9 stays outside,
2-endpoint pairs, reproducible config ids, patch exactness.

The dotted paths below were verified against the REAL config implementation
(``gaworld.settings.build_config()``):
- ``personality.style_fit_amplitude``      -> 0.3  (band [0.10, 0.40], ADMISSIBLE_STYLE_FIT)
- ``personality.emotion_baseline.contagion_weight`` -> 0.06 (there is NO
  top-level ``behavior.contagion_weight``; the admissible knob is the
  emotion-baseline one)
- ``economy.spending.daily_variance``      -> 0.25
- temperature: no global ``llm.temperature`` exists; the real numeric knobs
  are per-provider ``llm.providers.<name>.temperature`` (0.2 where set). The
  band must therefore be attached to a per-provider path that round-trips.
"""

from __future__ import annotations

import copy

import pytest

from benchmark.mcf.config_space import (  # noqa: F401
    BANDS,
    EXCLUDED_PARAMS,
    STRESS_VALUES,
    apply_patch,
    config_id,
    endpoint_configs,
    validate_config,
)
from gaworld.settings import CONFIG, build_config

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _read_path(config: dict, path: str):
    node = config
    for part in path.split("."):
        assert isinstance(node, dict) and part in node, f"path {path} does not exist in the real config"
        node = node[part]
    return node


def _flat(config: dict, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flat(value, path))
        else:
            out[path] = value
    return out


# ---------------------------------------------------------------------------
# band encoding — real dotted paths
# ---------------------------------------------------------------------------


class TestBands:
    def test_four_bands_defined(self):
        assert set(BANDS) == {
            "style_fit_amplitude",
            "contagion_weight",
            "spending_daily_variance",
            "temperature",
        }

    def test_style_fit_band_reuses_time_use_admissible_constant(self):
        from benchmark.mcf.time_use import ADMISSIBLE_STYLE_FIT

        band = BANDS["style_fit_amplitude"]
        assert (band["low"], band["high"]) == ADMISSIBLE_STYLE_FIT == (0.10, 0.40)

    def test_style_fit_path_is_the_real_config_path(self):
        assert BANDS["style_fit_amplitude"]["path"] == "personality.style_fit_amplitude"
        assert _read_path(build_config(), BANDS["style_fit_amplitude"]["path"]) == 0.3

    def test_contagion_band(self):
        band = BANDS["contagion_weight"]
        assert (band["low"], band["high"]) == (0.02, 0.12)

    def test_contagion_path_is_the_real_config_path(self):
        # there is no top-level behavior.contagion_weight; the knob that the
        # pilot actually turns is personality.emotion_baseline.contagion_weight
        assert BANDS["contagion_weight"]["path"] == "personality.emotion_baseline.contagion_weight"
        assert isinstance(_read_path(build_config(), BANDS["contagion_weight"]["path"]), (int, float))

    def test_spending_variance_band(self):
        band = BANDS["spending_daily_variance"]
        assert (band["low"], band["high"]) == (0.10, 0.40)
        assert BANDS["spending_daily_variance"]["path"] == "economy.spending.daily_variance"
        assert isinstance(_read_path(build_config(), BANDS["spending_daily_variance"]["path"]), (int, float))

    def test_temperature_band(self):
        band = BANDS["temperature"]
        assert (band["low"], band["high"]) == (0.0, 0.7)

    def test_temperature_path_round_trips_into_the_real_config(self):
        # whatever path the module picks, it must exist in the real config,
        # resolve to a numeric value, and survive a patch_for round-trip
        path = BANDS["temperature"]["path"]
        assert isinstance(path, str) and path
        current = _read_path(build_config(), path)
        assert isinstance(current, (int, float)), (
            f"temperature path {path} resolves to {type(current).__name__}, not a number"
        )


# ---------------------------------------------------------------------------
# excluded params
# ---------------------------------------------------------------------------


class TestExcludedParams:
    def test_excluded_set_lists_the_doc_params(self):
        for name in (
            "economy.initial_savings_months_min",
            "economy.initial_savings_months_max",
            "economy.spending.engel_curve",
        ):
            assert name in EXCLUDED_PARAMS, f"{name} missing from EXCLUDED_PARAMS"

    @pytest.mark.parametrize(
        "param",
        [
            "economy.initial_savings_months_min",
            "economy.initial_savings_months_max",
            "economy.spending.engel_curve",
        ],
    )
    def test_validate_config_rejects_excluded_params(self, param):
        with pytest.raises(ValueError, match=param.split(".")[-1]):
            validate_config({param: 6})


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_valid_mid_band_value_accepted(self):
        normalized = validate_config({"personality.style_fit_amplitude": 0.25})
        assert normalized == {"personality": {"style_fit_amplitude": 0.25}}

    @pytest.mark.parametrize(
        "param,value",
        [
            ("style_fit_amplitude", 0.10),
            ("style_fit_amplitude", 0.40),
            ("contagion_weight", 0.02),
            ("contagion_weight", 0.12),
            ("spending_daily_variance", 0.10),
            ("spending_daily_variance", 0.40),
            ("temperature", 0.0),
            ("temperature", 0.7),
        ],
    )
    def test_exact_band_edges_accepted(self, param, value):
        normalized = validate_config({BANDS[param]["path"]: value})
        assert _flat(normalized)[BANDS[param]["path"]] == value

    def test_stress_style_fit_0_9_rejected(self):
        with pytest.raises(ValueError, match=r"0\.9|stress"):
            validate_config({"personality.style_fit_amplitude": 0.9})

    def test_out_of_band_below_rejected(self):
        with pytest.raises(ValueError):
            validate_config({"personality.emotion_baseline.contagion_weight": 0.0})

    def test_out_of_band_above_rejected(self):
        with pytest.raises(ValueError):
            validate_config({"economy.spending.daily_variance": 0.5})

    def test_temperature_above_band_rejected(self):
        with pytest.raises(ValueError):
            validate_config({BANDS["temperature"]["path"]: 0.9})

    def test_unknown_param_rejected(self):
        with pytest.raises(ValueError, match=r"personality\.strength"):
            validate_config({"personality.strength": 0.8})

    def test_unknown_top_level_rejected(self):
        with pytest.raises(ValueError):
            validate_config({"no.such.path": 1})

    def test_validator_is_reused_not_duplicated(self, monkeypatch: pytest.MonkeyPatch):
        # the style-fit check must be delegated to the existing
        # time_use.validate_style_fit (M1 spec: "do NOT duplicate the validator")
        from benchmark.mcf import time_use

        calls: list[float] = []

        def spy(amplitude: float) -> None:
            calls.append(amplitude)
            time_use.validate_style_fit.__wrapped__(amplitude) if hasattr(
                time_use.validate_style_fit, "__wrapped__"
            ) else None

        monkeypatch.setattr(time_use, "validate_style_fit", spy)
        validate_config({"personality.style_fit_amplitude": 0.3})
        assert calls == [0.3], "validate_config must delegate to time_use.validate_style_fit"


# ---------------------------------------------------------------------------
# endpoint sampler
# ---------------------------------------------------------------------------


class TestEndpoints:
    def test_two_endpoints_per_parameter(self):
        endpoints = endpoint_configs()
        per_param: dict[str, list[float]] = {}
        for item in endpoints:
            per_param.setdefault(item["param"], []).append(item["value"])
        assert set(per_param) == set(BANDS)
        for param, values in per_param.items():
            assert sorted(values) == [BANDS[param]["low"], BANDS[param]["high"]]

    def test_eight_endpoints_total(self):
        assert len(endpoint_configs()) == 8

    def test_deterministic_across_calls(self):
        assert endpoint_configs() == endpoint_configs()

    def test_patches_use_only_the_band_path(self):
        for item in endpoint_configs():
            patch = item["patch"]
            flat = _flat(patch)
            assert set(flat) == {BANDS[item["param"]]["path"]}, (
                f"endpoint patch for {item['param']} touches unrelated keys: {set(flat)}"
            )

    def test_endpoints_apply_cleanly_to_the_real_config(self):
        for item in endpoint_configs():
            config = copy.deepcopy(CONFIG)
            apply_patch(config, item["patch"])
            assert _read_path(config, BANDS[item["param"]]["path"]) == item["value"]


# ---------------------------------------------------------------------------
# config ids
# ---------------------------------------------------------------------------


class TestConfigIds:
    def test_same_patch_same_id(self):
        patch = {"personality.style_fit_amplitude": 0.1}
        assert config_id(patch) == config_id({"personality.style_fit_amplitude": 0.1})

    def test_key_order_irrelevant(self):
        a = config_id({"personality.style_fit_amplitude": 0.1, "temperature": 0.2})
        b = config_id({"temperature": 0.2, "personality.style_fit_amplitude": 0.1})
        assert a == b, "config id must be derived from sorted patch keys"

    def test_different_values_different_ids(self):
        a = config_id({"personality.style_fit_amplitude": 0.1})
        b = config_id({"personality.style_fit_amplitude": 0.4})
        assert a != b

    def test_id_is_a_stable_string(self):
        cid = config_id({"personality.style_fit_amplitude": 0.1})
        assert isinstance(cid, str) and cid
        assert int(cid, 16) >= 0  # stable-hex identifier


# ---------------------------------------------------------------------------
# patch application / exactness against the real config
# ---------------------------------------------------------------------------


class TestPatchExactness:
    @pytest.mark.parametrize(
        "param,value",
        [
            ("style_fit_amplitude", 0.25),
            ("contagion_weight", 0.05),
            ("spending_daily_variance", 0.3),
            ("temperature", 0.5),
        ],
    )
    def test_only_the_named_key_changes(self, param, value):
        band = BANDS[param]
        before = _flat(copy.deepcopy(CONFIG))
        config = copy.deepcopy(CONFIG)
        apply_patch(config, {band["path"]: value})
        after = _flat(config)

        changed = {k for k in before if before[k] != after.get(k)}
        assert changed == {band["path"]}, (
            f"applying {band['path']}={value} also changed {changed - {band['path']}}"
        )
        assert after[band["path"]] == value
