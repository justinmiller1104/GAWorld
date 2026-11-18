"""Sobol sampler over admissible parameter bands (plan step 8).

Deterministic scrambled Sobol sequences via ``scipy.stats.qmc`` (scipy is
already a repo dependency through ``gaworld.parallel.causal``), remapped from
the unit cube into each band. Points are strictly inside every band: edge
hits are filtered and re-drawn rather than clamped, so no sampled config sits
on a band endpoint by construction.

``n`` may be any positive integer. Power-of-2 n keeps the full balance
properties of the digital net; other values take a deterministic prefix of
the sequence (scipy emits a balance warning, which is why n=power-of-2 is
recommended in the docs). Sampling only ever touches the bands handed in —
``config_space.EXCLUDED_PARAMS`` are not bands, so they are unreachable.
"""

from __future__ import annotations

from typing import Any, Sequence

__all__ = ["sample"]


def _draw(n: int, d: int, seed: int) -> list[tuple[float, ...]]:
    from scipy.stats import qmc

    sobol = qmc.Sobol(d, scramble=True, seed=seed)
    points: list[tuple[float, ...]] = []
    # draw in power-of-2 chunks: the sequence's balance properties hold per chunk
    chunk = 1 << max(1, (n - 1).bit_length())
    while len(points) < n:
        for row in sobol.random(chunk):
            points.append(tuple(float(u) for u in row))
    return points[:n]


def sample(n: int, bands: Sequence[dict[str, Any]], seed: int = 0) -> list[list[float]]:
    """``n`` Sobol points, one value per band, strictly inside [low, high].

    ``bands`` is a list of ``config_space.BANDS`` entries (``low``/``high``);
    excluded parameters never appear among them.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if not bands:
        raise ValueError("at least one band is required")

    candidates = _draw(n * 2, len(bands), seed)  # headroom for filtered edge hits
    points = []
    for row in candidates:
        values = [b["low"] + u * (b["high"] - b["low"]) for u, b in zip(row, bands)]
        if all(b["low"] < v < b["high"] for v, b in zip(values, bands)):
            points.append(values)
            if len(points) == n:
                break
    if len(points) < n:  # keep drawing deterministically until n interior points
        offset = len(candidates)
        while len(points) < n:
            for row in _draw(n * 2, len(bands), seed + offset):
                values = [b["low"] + u * (b["high"] - b["low"]) for u, b in zip(row, bands)]
                if all(b["low"] < v < b["high"] for v, b in zip(values, bands)):
                    points.append(values)
                    if len(points) == n:
                        break
            offset += n * 2
    return points[:n]
