"""Baseline-vs-intervention outcome comparison (Phase 2.2).

Pure, deterministic functions over run directories in the score_directory
shape (``agent_<id>.jsonl`` episode files + ``config.json``). The report
separates:

- effect differences: pre/post intervention-window slicing, per-day series
  over simulation time, repeated-run variation (spread alongside mean);
- honest data status: ``ok`` / ``MISSING`` / ``UNRELIABLE`` markers — absent
  or sub-threshold data is never reported as a zero effect;
- MCF validation attach (per-arm ``score_directory`` results), clearly kept
  out of the effect section: baseline realism is NOT evidence of
  intervention-response accuracy, and MCF validates but never selects a
  winning intervention.

No LLM calls, no network, no mutation of the run directories.
"""

import json
from pathlib import Path

from benchmark.mcf.time_use import MIN_COMPARISON_DAYS, MIN_COMPARISON_POPULATION, score_directory

__all__ = ["compare_arms", "compare_repeated_runs", "time_to_minute", "NEUTRAL_VERDICTS"]


def time_to_minute(time_str):
    """"14:00" (HH:MM) -> minute of day (840); the trigger-tick arithmetic."""
    hours, minutes = time_str.split(":")
    return int(hours) * 60 + int(minutes)

# Verdicts meaning "no measurable effect". A MISSING/UNRELIABLE arm must
# never map to one of these (insufficient data is not evidence of no effect).
NEUTRAL_VERDICTS = ("no_effect", "neutral", "zero_effect")

MCF_LIMITATIONS = (
    "MCF validation is attached for context only and is kept separate from the "
    "effect-difference section: baseline realism is not evidence of "
    "intervention-response accuracy; MCF validates, never selects the winning "
    "intervention. The per-arm overall_error describes fit against survey "
    "targets, not the size or direction of the intervention effect."
)


def compare_arms(baseline_dir, intervention_dir, intervention_day, intervention_time=None, tick_minutes=10):
    """Compare one baseline run dir against one intervention run dir.

    Returns a report dict with ``windows`` (pre/post window statistics:
    ``max_abs_diff`` as a raw diagnostic, ``episode_disagreement_rate`` —
    mismatches / |common episode keys in the window| — the headline
    comparability statistic, and ``variable_diffs`` — max |Δ| per state
    variable, the variable named), ``per_day`` (differences over simulation
    time), ``first_divergence_tick`` (earliest differing (agent, day, time),
    ``None`` when the arms are identical), ``effect`` (``verdict`` +
    ``beneficial``, judged on the disagreement rate — never on
    ``max_abs_diff``), ``data_status`` per arm (``ok`` / ``MISSING`` /
    ``UNRELIABLE``), and ``mcf_validation`` (per-arm scores +
    ``limitations``).

    With ``intervention_time`` ("HH:MM"), the pre/post window is sliced at
    episode-time (tick) resolution instead of day granularity: a record is
    post when its day is after ``intervention_day`` or it is on the trigger
    day at/after ``intervention_time``. The report then also carries
    ``per_time``, per-tick differences tagged with their window; the trigger
    tick itself belongs to the post window.
    """
    baseline_records = _load_run(baseline_dir)
    intervention_records = _load_run(intervention_dir)
    status = {
        "baseline": _data_status(baseline_records),
        "intervention": _data_status(intervention_records),
    }

    per_day = {}
    for day in sorted(set(baseline_records["days"]) | set(intervention_records["days"])):
        per_day[day] = {"max_abs_diff": _day_diff(baseline_records, intervention_records, day)}

    first_divergence_tick = _first_divergence_tick(baseline_records, intervention_records)
    if intervention_time is not None:
        trigger_minute = time_to_minute(intervention_time)
        times = sorted(
            {k[2] for k in baseline_records["records"]} | {k[2] for k in intervention_records["records"]}
        )
        per_time = {}
        # The series is a running max over the window up to each tick, so a
        # tick that itself never diverges (e.g. an unchanged late-night slot)
        # still carries the divergence reached before it; ticks before the
        # first divergence stay clean.
        running = 0.0
        for time in times:
            running = max(running, _time_diff(baseline_records, intervention_records, time))
            per_time[time] = {
                "max_abs_diff": running,
                "window": "post" if time_to_minute(time) >= trigger_minute else "pre",
            }
        pre = [v["max_abs_diff"] for v in per_time.values() if v["window"] == "pre"]
        post = [v["max_abs_diff"] for v in per_time.values() if v["window"] == "post"]
    else:
        per_time = None
        pre = [v["max_abs_diff"] for d, v in per_day.items() if d < intervention_day]
        post = [v["max_abs_diff"] for d, v in per_day.items() if d >= intervention_day]
    windows = {
        "pre": {"max_abs_diff": max(pre) if pre else 0.0},
        "post": {"max_abs_diff": max(post) if post else 0.0},
    }
    # Corrected window statistics (Phase 2.5): each window's denominator is
    # the number of COMMON episode keys inside that window — never a
    # whole-run count (the Phase 2.4 discrepancy: 2/288 per-slot mismatches
    # divided by 288 whole-run records shrank a 100% slot disagreement ~120x).
    for window_name, window_keys in _window_key_sets(
        baseline_records, intervention_records, intervention_day, trigger_minute if intervention_time is not None else None
    ).items():
        windows[window_name].update(
            _window_stats(baseline_records, intervention_records, window_keys)
        )

    effect = _effect_verdict(status, windows)
    report = {
        "intervention_dir": str(intervention_dir),
        "intervention_day": intervention_day,
        "windows": windows,
        "per_day": per_day,
        "first_divergence_tick": first_divergence_tick,
        "effect": effect,
        "data_status": status,
        "mcf_validation": _mcf_attach(baseline_dir, intervention_dir),
    }
    if per_time is not None:
        report["intervention_time"] = intervention_time
        report["per_time"] = per_time
    return report


def compare_repeated_runs(baseline_dirs, intervention_dirs, intervention_day):
    """Compare repeated (baseline, intervention) run pairs.

    Same report shape as :func:`compare_arms` (from the first pair) plus a
    ``variation`` section: per-run post-window differences, their mean, and
    the spread across runs — variation is reported alongside the mean
    differences, never collapsed into a single number.
    """
    reports = [
        compare_arms(b, i, intervention_day)
        for b, i in zip(baseline_dirs, intervention_dirs)
    ]
    report = dict(reports[0])
    per_run = [r["windows"]["post"]["max_abs_diff"] for r in reports]
    report["variation"] = {
        "per_run": per_run,
        "mean": sum(per_run) / len(per_run) if per_run else 0.0,
        "spread": (max(per_run) - min(per_run)) if per_run else 0.0,
    }
    return report


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _load_run(run_dir):
    """Load ``agent_*.jsonl`` episode records: {(agent_id, day, time): activity}
    plus the per-episode ``state`` dicts the named variable diffs read."""
    records = {}
    states: dict[tuple, dict] = {}
    agents: set[int] = set()
    days: set[int] = set()
    for path in sorted(Path(run_dir).glob("agent_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            key = (ep["agent_id"], ep["day"], ep["time"])
            records[key] = ep.get("final_activity", "")
            state = ep.get("state")
            states[key] = state if isinstance(state, dict) else {}
            agents.add(ep["agent_id"])
            days.add(ep["day"])
    return {"records": records, "states": states, "agents": agents, "days": days}


def _data_status(run_data):
    if not run_data["records"]:
        return "MISSING"
    n_agents = len(run_data["agents"])
    n_days = len(run_data["days"])
    if n_days < MIN_COMPARISON_DAYS or n_agents < MIN_COMPARISON_POPULATION:
        return "UNRELIABLE"
    return "ok"


def _day_diff(baseline, intervention, day):
    """Mismatched episode records on ``day``, normalized by the baseline arm's
    full episode count (comparability across days of different coverage)."""
    if not baseline["records"]:
        return 0.0
    b_keys = {k for k in baseline["records"] if k[1] == day}
    i_keys = {k for k in intervention["records"] if k[1] == day}
    common = b_keys & i_keys
    if not common:
        return 0.0
    mismatches = sum(
        1 for k in common if baseline["records"][k] != intervention["records"][k]
    )
    return mismatches / len(baseline["records"])


def _time_diff(baseline, intervention, time):
    """Mismatched episode records at time-of-day ``time`` across all days,
    normalized like :func:`_day_diff` by the baseline arm's full episode count."""
    if not baseline["records"]:
        return 0.0
    b_keys = {k for k in baseline["records"] if k[2] == time}
    i_keys = {k for k in intervention["records"] if k[2] == time}
    common = b_keys & i_keys
    if not common:
        return 0.0
    mismatches = sum(
        1 for k in common if baseline["records"][k] != intervention["records"][k]
    )
    return mismatches / len(baseline["records"])


def _effect_verdict(status, windows):
    if "MISSING" in status.values():
        return {"verdict": "insufficient_data", "beneficial": False}
    # Judged on the corrected per-window disagreement rate (explicit window
    # denominator). max_abs_diff stays a raw window diagnostic and never
    # appears as — or inside — the headline verdict.
    post_rate = windows["post"]["episode_disagreement_rate"]
    if post_rate == 0.0:
        return {"verdict": "no_effect", "beneficial": False}
    # Differences are reported as divergence; no hardcoded success criterion,
    # so an effect is never auto-labelled beneficial by this module.
    return {"verdict": "divergence", "beneficial": False, "post_episode_disagreement_rate": post_rate}


def _window_key_sets(baseline, intervention, intervention_day, trigger_minute=None):
    """Pre/post episode-key windows for both arms.

    Pre = strictly before the trigger (day < intervention_day, or on the
    trigger day before ``trigger_minute`` when a within-day trigger is
    given); post = the rest, the trigger tick included.
    """

    def _in_pre(key):
        if key[1] < intervention_day:
            return True
        if trigger_minute is None or key[1] != intervention_day:
            return False
        return time_to_minute(key[2]) < trigger_minute

    pre = {k for k in baseline["records"] if _in_pre(k)} & {k for k in intervention["records"] if _in_pre(k)}
    post = (set(baseline["records"]) & set(intervention["records"])) - pre
    return {"pre": pre, "post": post}


def _window_stats(baseline, intervention, common_keys):
    """Corrected per-window statistics over ``common_keys``.

    - ``episode_disagreement_rate``: mismatches / |common keys| (explicit
      window denominator);
    - ``variable_diffs``: max |Δ| per named state variable across the
      window's common records (missing/non-numeric read as 0 contribution);
    (``max_abs_diff`` is computed by the caller from the activity diffs and
    stays a raw diagnostic only.)
    """
    if not common_keys:
        return {"episode_disagreement_rate": 0.0, "variable_diffs": {}}
    mismatches = 0
    variable_diffs: dict[str, float] = {}
    for key in common_keys:
        if baseline["records"][key] != intervention["records"][key]:
            mismatches += 1
        b_state = baseline["states"].get(key, {})
        i_state = intervention["states"].get(key, {})
        for var in set(b_state) | set(i_state):
            b_val = b_state.get(var)
            i_val = i_state.get(var)
            if isinstance(b_val, (int, float)) and isinstance(i_val, (int, float)):
                diff = abs(b_val - i_val)
            elif isinstance(b_val, (int, float)) or isinstance(i_val, (int, float)):
                diff = abs(b_val if isinstance(b_val, (int, float)) else i_val)
            else:
                continue
            variable_diffs[var] = max(variable_diffs.get(var, 0.0), diff)
    return {
        "episode_disagreement_rate": mismatches / len(common_keys),
        "variable_diffs": variable_diffs,
    }


def _first_divergence_tick(baseline, intervention):
    """Earliest differing (agent, day, time) across the common records;
    ``None`` when the arms' activities are identical everywhere."""
    common = set(baseline["records"]) & set(intervention["records"])
    differing = [k for k in common if baseline["records"][k] != intervention["records"][k]]
    if not differing:
        return None
    agent_id, day, time = sorted(differing, key=lambda k: (k[1], time_to_minute(k[2]), k[0]))[0]
    return {"agent_id": agent_id, "day": day, "time": time}


def _mcf_attach(baseline_dir, intervention_dir):
    """Per-arm MCF scores (validation layer, kept out of the effect section)."""
    scores = {}
    for arm, run_dir in (("baseline", baseline_dir), ("intervention", intervention_dir)):
        result = score_directory(run_dir, employed_agent_ids=(), tick_minutes=10)
        scores[arm] = {
            "overall_error": result.overall_error,
            "coverage": result.coverage,
            "frontier_valid": result.frontier_valid,
            "uncertainty_warnings": list(result.uncertainty_warnings),
        }
    scores["limitations"] = MCF_LIMITATIONS
    return scores
