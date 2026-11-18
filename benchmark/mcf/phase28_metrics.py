"""Phase 2.8 — explicit-denominator admission metrics + manifest honesty.

Rates are always outcome counts divided by ``attempts``, where
``attempts == len(venue.admission rows)`` — never a silent denominator
swap. A run with zero attempts has undefined rates: they are ``None``,
never fabricated as 0.0.

Manifest honesty (RESULT-phase28-checkpoint §b): a run may only be called
checkpoint restoration when its manifest's ``launch_mode`` says so. A
fresh seeded start plus strict sequence replay is labeled
``fresh-start-replay`` and carries a ``replay`` block
{window, events_total, events_consumed, mismatches}.
"""

from __future__ import annotations

VALID_LAUNCH_MODES = ("fresh-start-replay", "checkpoint-restore")


def attempts_of(admission_rows):
    """The explicit denominator: every venue.admission row is one attempt."""
    return len(admission_rows)


def admission_rates(admission_rows, redirects=None, denied=None):
    """Per-outcome rates over the explicit attempts denominator.

    Legacy outcome tables are passed through untouched — their row counts
    are returned as-is for compatibility, never merged into the rates.
    """
    attempts = attempts_of(admission_rows)
    counts = {"admitted": 0, "redirected": 0, "denied": 0}
    for row in admission_rows:
        outcome = row.get("outcome")
        if outcome in counts:
            counts[outcome] += 1

    def _rate(n):
        return n / attempts if attempts else None

    return {
        "attempts": attempts,
        "admitted": counts["admitted"],
        "redirected": counts["redirected"],
        "denied": counts["denied"],
        "admission_rate": _rate(counts["admitted"]),
        "redirect_rate": _rate(counts["redirected"]),
        "denial_rate": _rate(counts["denied"]),
        "venue_redirect_rows": len(redirects or []),
        "action_denied_rows": len(denied or []),
    }


def launch_mode_of(manifest):
    """Return the manifest's validated launch_mode, or raise ValueError.

    A legacy manifest with no ``launch_mode`` key is rejected: without the
    label the run's launch provenance is simply unknown.
    """
    mode = manifest.get("launch_mode")
    if mode not in VALID_LAUNCH_MODES:
        raise ValueError(
            f"manifest launch_mode must be one of {VALID_LAUNCH_MODES}, got {mode!r}"
        )
    return mode


def replay_block(manifest):
    """Return the manifest's complete replay block, or raise ValueError.

    All four keys — window, events_total, events_consumed, mismatches —
    are required; a partial block would invite fabricated completeness.
    """
    block = manifest.get("replay")
    required = ("window", "events_total", "events_consumed", "mismatches")
    if not isinstance(block, dict) or any(key not in block for key in required):
        raise ValueError(
            f"replay block requires {required}, got {block!r}"
        )
    return block


def may_claim_checkpoint_restore(manifest):
    """True only when the manifest itself says launch_mode == checkpoint-restore."""
    try:
        return launch_mode_of(manifest) == "checkpoint-restore"
    except ValueError:
        return False


def describe_launch(manifest):
    """Human label for a run's launch. Unlabeled manifests may never be
    described as checkpoint restoration."""
    if may_claim_checkpoint_restore(manifest):
        return "checkpoint-restoration"
    if manifest.get("launch_mode") == "fresh-start-replay":
        return "fresh-start-replay"
    return "unlabeled"
