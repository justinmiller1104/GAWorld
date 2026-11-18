"""M1 — admissible Θ config space + endpoint sampler.

Four admissible bands (docs/MCF_GAWORLD.md §3). Dotted paths verified against
``gaworld.settings.build_config()``:

- ``personality.style_fit_amplitude``                 (band reused from time_use)
- ``personality.emotion_baseline.contagion_weight``   (no top-level behavior knob)
- ``economy.spending.daily_variance``
- ``llm.providers.omlx_qwen.temperature``             (temperature is per-provider;
                                                       no global llm.temperature exists)

The style-fit band/validator are NOT duplicated here: validation delegates to
``benchmark.mcf.time_use.validate_style_fit``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from benchmark.mcf import time_use
from gaworld.parallel.sweep import patch_for

BANDS: dict[str, dict[str, Any]] = {
    "style_fit_amplitude": {
        "path": "personality.style_fit_amplitude",
        "low": time_use.ADMISSIBLE_STYLE_FIT[0],
        "high": time_use.ADMISSIBLE_STYLE_FIT[1],
    },
    "contagion_weight": {
        "path": "personality.emotion_baseline.contagion_weight",
        "low": 0.02,
        "high": 0.12,
    },
    "spending_daily_variance": {
        "path": "economy.spending.daily_variance",
        "low": 0.10,
        "high": 0.40,
    },
    "temperature": {
        "path": "llm.providers.omlx_qwen.temperature",
        "low": 0.0,
        "high": 0.7,
    },
}

# Parameters the docs exclude from manipulation (structural / calibration
# knobs, not treatment variables).
EXCLUDED_PARAMS: frozenset[str] = frozenset(
    {
        "economy.initial_savings_months_min",
        "economy.initial_savings_months_max",
        "economy.spending.engel_curve",
    }
)

# Stress probes: never admissible manipulation points.
STRESS_VALUES: tuple[float, ...] = time_use.STRESS_STYLE_FIT

_BAND_BY_PATH = {band["path"]: (name, band) for name, band in BANDS.items()}


def validate_config(patch: dict[str, Any]) -> dict[str, Any]:
    """Validate a flat ``{dotted.path: value}`` patch and return the nested patch.

    Raises ``ValueError`` for excluded params, unknown paths, or values outside
    the admissible band (including stress values).
    """
    nested: dict[str, Any] = {}
    for path, value in patch.items():
        if path in EXCLUDED_PARAMS:
            raise ValueError(f"parameter {path!r} is excluded from manipulation")
        if path in _BAND_BY_PATH:
            name, band = _BAND_BY_PATH[path]
            if name == "style_fit_amplitude":
                # reuse the existing validator (module-attribute lookup so the
                # delegation stays monkeypatchable and the band has one source
                # of truth)
                time_use.validate_style_fit(value)
            elif not band["low"] <= value <= band["high"]:
                raise ValueError(
                    f"{path} = {value} outside the admissible band [{band['low']}, {band['high']}]"
                )
        else:
            raise ValueError(f"unknown configuration parameter {path!r}")
        nested.update(patch_for(path, value))
    return nested


def endpoint_configs() -> list[dict[str, Any]]:
    """Deterministic 2-endpoint pairs (low/high) per band parameter."""
    endpoints: list[dict[str, Any]] = []
    for name, band in BANDS.items():
        for value in (band["low"], band["high"]):
            endpoints.append({"param": name, "value": value, "patch": patch_for(band["path"], value)})
    return endpoints


def config_id(patch: dict[str, Any]) -> str:
    """Reproducible identifier: stable hex hash over sorted (key, value) pairs.

    ``patch`` is flat (dotted keys); key order is irrelevant.
    """
    items = sorted(patch.items(), key=lambda kv: kv[0])
    blob = json.dumps(items, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def apply_patch(config: dict[str, Any], patch: dict[str, Any]) -> None:
    """Deep-merge ``patch`` into ``config`` in place — only named keys change."""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            apply_patch(config[key], value)
        else:
            config[key] = value
