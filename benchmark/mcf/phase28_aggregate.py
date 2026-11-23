"""Phase 2.8 — multi-arm aggregation with honest Bayesian intervals.

Per-arm admission / denial / redirect rates are pooled over the arm's
runs and carry a 95% credible interval from a Jeffreys Beta(0.5, 0.5)
beta-binomial prior.

Honesty contract: a run with a missing arm label or missing outcome rows
raises :class:`IncompleteRunError` by default; with ``on_incomplete=
"exclude"`` it is named in the output's ``excluded`` list instead. It is
never silently dropped and never contributes to a valid-looking estimate.
"""

from __future__ import annotations

from scipy.stats import beta


class IncompleteRunError(ValueError):
    """A run lacks the arm label or outcome rows needed for aggregation."""


def _validate_run(run):
    problems = []
    if not run.get("arm"):
        problems.append("missing arm label")
    if run.get("venue.admission") is None:
        problems.append("missing venue.admission rows")
    if problems:
        raise IncompleteRunError(f"run {run.get('run_id')!r} is incomplete: {'; '.join(problems)}")
    return run


def _arm_summary(runs, arm):
    attempts = 0
    counts = {"admitted": 0, "redirected": 0, "denied": 0}
    for run in runs:
        if run["arm"] != arm:
            continue
        for row in run["venue.admission"]:
            outcome = row.get("outcome")
            if outcome in counts:
                counts[outcome] += 1
            attempts += 1

    def _rate(n):
        return n / attempts if attempts else None

    def _ci95(n):
        if not attempts:
            return None
        # Jeffreys Beta(0.5, 0.5) posterior: Beta(n + 0.5, attempts - n + 0.5)
        dist = beta(0.5 + n, 0.5 + attempts - n)
        return [float(dist.ppf(0.025)), float(dist.ppf(0.975))]

    return {
        "attempts": attempts,
        "admission_rate": _rate(counts["admitted"]),
        "redirect_rate": _rate(counts["redirected"]),
        "denial_rate": _rate(counts["denied"]),
        "admission_ci95": _ci95(counts["admitted"]),
        "redirect_ci95": _ci95(counts["redirected"]),
        "denial_ci95": _ci95(counts["denied"]),
    }


def aggregate(runs, on_incomplete="raise"):
    """Aggregate labeled runs into per-arm rates with 95% credible intervals.

    ``on_incomplete``: ``"raise"`` (default) fails loudly on an incomplete
    run; ``"exclude"`` names it in the output's ``excluded`` list instead.
    """
    if on_incomplete not in ("raise", "exclude"):
        raise ValueError(f"unknown on_incomplete policy: {on_incomplete!r}")
    valid = []
    excluded = []
    for run in runs:
        try:
            valid.append(_validate_run(run))
        except IncompleteRunError:
            if on_incomplete == "exclude":
                excluded.append(run.get("run_id"))
            else:
                raise
    arms = sorted({run["arm"] for run in valid})
    return {
        "arms": {arm: _arm_summary(valid, arm) for arm in arms},
        "excluded": excluded,
    }
