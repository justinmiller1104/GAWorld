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

# ``sources`` states each range's origin independently of the evaluation
# metrics (MCF §3). PILOT-calibrated ranges cite the calibration record;
# heuristic ranges say so honestly rather than borrowing authority.
BANDS: dict[str, dict[str, Any]] = {
    "style_fit_amplitude": {
        "path": "personality.style_fit_amplitude",
        "low": time_use.ADMISSIBLE_STYLE_FIT[0],
        "high": time_use.ADMISSIBLE_STYLE_FIT[1],
        "sources": [
            "docs/MCF_GAWORLD_INTEGRATION_FEASIBILITY.md §4.4: acceptance band pre-registered in "
            "gaworld/settings/personality.py from the literature argument that personality must "
            "not explain >~15% of behavioural variance; 0.40/0.60 fail the "
            "scripts/big5_effect_ceiling.py windows (literature-anchored calibration)",
            "PILOT calibration: validated by benchmark.mcf.time_use.validate_style_fit; "
            "0.9 is a stress value outside the band by construction (STRESS_STYLE_FIT)",
        ],
    },
    "contagion_weight": {
        "path": "personality.emotion_baseline.contagion_weight",
        "low": 0.02,
        "high": 0.12,
        "sources": [
            "docs/MCF_GAWORLD_INTEGRATION_FEASIBILITY.md §4.4: social-contagion effect sizes "
            "in the literature are small; the parameter exists to fix a documented defect "
            "(_cognition.py hard-coded 0.1); default 0.06 is the band midpoint (literature "
            "anchor on effect-size magnitude, exact endpoints heuristic)",
            "docs/MCF_GAWORLD_PHASE_0_5_VALIDATION.md §5: effect size never measured — "
            "range is plausible-but-unverified, kept away from the scorer by construction",
        ],
    },
    "spending_daily_variance": {
        "path": "economy.spending.daily_variance",
        "low": 0.10,
        "high": 0.40,
        "sources": [
            "docs/MCF_GAWORLD_INTEGRATION_FEASIBILITY.md §4.4: plausible consumption-noise "
            "range from household-budget survey dispersion (default 0.25); independent of "
            "D2/D3 by construction and not the engel_curve itself (pilot heuristic, not "
            "literature-justified endpoints)",
        ],
    },
    "temperature": {
        "path": "llm.providers.omlx_qwen.temperature",
        "low": 0.0,
        "high": 0.7,
        "sources": [
            "docs/MCF_GAWORLD_INTEGRATION_FEASIBILITY.md §4.4: standard LLM-provider creative-"
            "task diversity range (provider docs convention); 0.0 = deterministic floor, "
            "0.7 stays below values where open-ended generation degrades into noise",
        ],
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


def sample(n: int, method: str = "endpoints", params: list[str] | None = None) -> list[dict[str, Any]]:
    """Sample ``n`` flat config patches ({dotted.path: value}) over the bands.

    ``method="sobol"`` uses :mod:`benchmark.mcf.sampler` (deterministic,
    strictly inside every band); ``method="endpoints"`` returns the
    one-at-a-time low/high endpoints (``endpoint_configs``, flat form, ``n``
    is a no-op cap kept for interface symmetry). Unknown methods raise.
    """
    if method not in ("endpoints", "sobol"):
        raise ValueError(f"unknown sampling method {method!r} (use 'sobol' or 'endpoints')")
    names = list(params) if params is not None else list(BANDS)
    chosen = [BANDS[name] for name in names]
    if method == "endpoints":
        patches = []
        for band in chosen:
            for value in (band["low"], band["high"]):
                patches.append({band["path"]: value})
        return patches if n >= len(patches) else patches[:n]
    from benchmark.mcf import sampler

    points = sampler.sample(n, chosen)
    return [{band["path"]: value for band, value in zip(chosen, point, strict=True)} for point in points]
