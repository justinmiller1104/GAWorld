"""Experiment manifest (plan step 12): one JSON per experiment binding
configs, seeds, model info, per-leg outcomes, results and verdicts.

The manifest is the reopenable record the GUI results endpoint serves:
``load_manifest(path)`` returns the full structure. Building it is a pure
function of (experiment spec, leg table, computed results) — identical
inputs yield byte-identical JSON. Failures and incomplete legs are carried
loudly (``complete: false`` + an explicit ``failures`` list), never dropped.
"""

from __future__ import annotations

import json
import os
from typing import Any

from benchmark.mcf.config_space import config_id

__all__ = ["build_manifest", "save_manifest", "load_manifest"]

_SCHEMA_VERSION = 1


def build_manifest(
    exp: dict[str, Any],
    legs: list[dict[str, Any]],
    results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the experiment spec, leg table and results into one manifest.

    ``exp`` is the persisted experiment dict (mcf_api shape); ``legs`` the
    per-leg rows written by the orchestrator; ``results`` the frontier
    computation (``mcf_api._compute_results`` shape) or ``None`` while no
    results exist yet.
    """
    leg_entries = []
    failures = []
    for row in legs:
        entry = {
            "arch": row.get("arch"),
            "config_id": row.get("config_id"),
            "config_patch_flat": row.get("config_patch_flat") or row.get("config_patch") or {},
            "seed": row.get("seed"),
            "status": row.get("status"),
            "stop_reason": row.get("stop_reason", ""),
            "usage": row.get("usage") or {},
            "usage_source": row.get("usage_source", ""),
            "run_path": row.get("run_path", ""),
        }
        leg_entries.append(entry)
        if entry["status"] != "completed":
            failures.append(
                {
                    "arch": entry["arch"],
                    "config_id": entry["config_id"],
                    "seed": entry["seed"],
                    "status": entry["status"],
                    "stop_reason": entry["stop_reason"],
                }
            )

    # distinct swept configs with their reproducible ids
    seen: dict[str, dict] = {}
    for entry in leg_entries:
        patch = entry["config_patch_flat"]
        cid = entry["config_id"] or config_id(patch)
        if cid not in seen:
            seen[cid] = {"config_id": cid, "patch": patch}
    configs = [seen[cid] for cid in sorted(seen)]
    manifest: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": {
            "id": exp["id"],
            "name": exp.get("name", ""),
            "note": exp.get("note", ""),
            "created_at": exp.get("created_at"),
            "status": exp.get("status", ""),
        },
        "architectures": [
            {"id": a["id"], "label": a.get("label", a["id"]), "patch": dict(a.get("patch") or {})}
            for a in exp.get("architectures", [])
        ],
        "reference_architecture": exp.get("reference_architecture"),
        "objectives": list(exp.get("objectives") or []),
        "sampler": exp.get("sampler", "endpoints"),
        "parameters": list(exp.get("parameters") or []),
        "n_configs": exp.get("n_configs"),
        "seeds": list(exp.get("seeds") or []),
        "model": dict(exp.get("model") or {}),
        "configs": configs,
        "legs": leg_entries,
        "failures": failures,
        "complete": not failures,
        "results": results,
    }

    archs = manifest["architectures"]
    reference = manifest["reference_architecture"]
    verdicts: dict[str, Any] = {}
    nondominated: dict[str, Any] = {}
    arch_compare: dict[str, Any] = {}
    if results is None:
        for arch in archs:
            verdicts[arch["id"]] = {"verdict": "no scored results yet"}
    else:
        per_arch = {entry["arch"]: entry for entry in results.get("architectures", [])}
        for arch in archs:
            aid = arch["id"]
            entry = per_arch.get(aid, {})
            nondominated[aid] = entry.get("frontier", [])
            arch_compare[aid] = {"gain": entry.get("gain", []), "lost": entry.get("lost", [])}
            if aid == reference:
                verdicts[aid] = {"verdict": "reference"}
            else:
                gain = entry.get("gain", [])
                lost = entry.get("lost", [])
                verdict = (
                    "newly-attainable-points" if gain else ("no-gain" if not lost else "no-gain-lost-points")
                )
                verdicts[aid] = {"verdict": verdict, "gain": gain, "lost": lost}
    manifest["nondominated"] = nondominated
    manifest["arch_compare"] = arch_compare
    manifest["verdicts"] = verdicts
    if results is not None:
        manifest["diagnostics"] = results.get("diagnostics")
    return manifest


def save_manifest(path: str, manifest: dict[str, Any]) -> str:
    """Write the manifest atomically; returns the path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path


def load_manifest(path: str) -> dict[str, Any] | None:
    """Full manifest structure for the GUI results endpoint; None if absent."""
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
