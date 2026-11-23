"""Phase 2.8 RED tests — manifest honesty: launch_mode + replay block (spec §3).

``benchmark/mcf/phase28_metrics.py`` (launch-label half) does not exist yet,
so these tests fail with ModuleNotFoundError — missing feature, not a syntax
error.

Contract under test (RESULT-phase28-checkpoint §(b)):
- a run manifest must carry ``launch_mode`` in
  {"fresh-start-replay", "checkpoint-restore"};
- a run that is a fresh seeded start plus strict sequence replay must be
  labeled "fresh-start-replay" and must carry a ``replay`` block
  {window, events_total, events_consumed, mismatches};
- analysis code must REFUSE to label a run "checkpoint-restoration" unless
  the manifest's ``launch_mode`` says so — an unlabeled / legacy manifest
  (today's reality: no ``launch_mode`` key exists anywhere, see
  gaworld/worldrun/manifest.py V2_FIELDS and benchmark/mcf/experiment.py)
  may never be described as checkpoint restoration.
"""

from __future__ import annotations

import pytest

VALID_MODES = {"fresh-start-replay", "checkpoint-restore"}


def _mod():
    import benchmark.mcf.phase28_metrics as m

    return m


class TestLaunchModeField:
    def test_fresh_start_replay_manifest_has_valid_launch_mode(self):
        manifest = {
            "launch_mode": "fresh-start-replay",
            "replay": {
                "window": {"start": 1, "stop": 3},
                "events_total": 594,
                "events_consumed": 594,
                "mismatches": 0,
            },
        }
        mode = _mod().launch_mode_of(manifest)
        assert mode in VALID_MODES
        assert mode == "fresh-start-replay"

    def test_checkpoint_restore_manifest_is_accepted(self):
        manifest = {
            "launch_mode": "checkpoint-restore",
            "replay": {
                "window": {"start": 3, "stop": 3},
                "events_total": 0,
                "events_consumed": 0,
                "mismatches": 0,
            },
        }
        assert _mod().launch_mode_of(manifest) == "checkpoint-restore"

    def test_unknown_launch_mode_is_rejected(self):
        manifest = {"launch_mode": "vibes"}
        with pytest.raises(ValueError):
            _mod().launch_mode_of(manifest)

    def test_legacy_manifest_without_launch_mode_is_rejected(self):
        # Today's manifests: benchmark/mcf/experiment.py writes fork.checkpoint
        # + replay_window; gaworld/worldrun/manifest.py has no launch_mode.
        with pytest.raises(ValueError):
            _mod().launch_mode_of({"fork.checkpoint": "output/x", "replay_window": "1-3"})


class TestReplayBlock:
    def test_replay_block_fields_are_required(self):
        manifest = {"launch_mode": "fresh-start-replay"}
        with pytest.raises(ValueError):
            _mod().replay_block(manifest)
        manifest["replay"] = {"window": {"start": 1, "stop": 3}}
        with pytest.raises(ValueError):
            # events_total / events_consumed / mismatches missing
            _mod().replay_block(manifest)

    def test_complete_replay_block_round_trips(self):
        block = {
            "window": {"start": 1, "stop": 3},
            "events_total": 594,
            "events_consumed": 594,
            "mismatches": 0,
        }
        assert _mod().replay_block({"launch_mode": "fresh-start-replay", "replay": block}) == block


class TestNoCheckpointRestorationClaims:
    """The honesty gate: nothing is "checkpoint-restoration" unless the
    manifest says launch_mode == "checkpoint-restore"."""

    def test_unlabeled_run_is_never_checkpoint_restoration(self):
        assert _mod().may_claim_checkpoint_restore({}) is False
        assert _mod().describe_launch({}) != "checkpoint-restoration"

    def test_fresh_start_replay_is_never_checkpoint_restoration(self):
        manifest = {
            "launch_mode": "fresh-start-replay",
            "replay": {
                "window": {"start": 1, "stop": 3},
                "events_total": 594,
                "events_consumed": 594,
                "mismatches": 0,
            },
        }
        assert _mod().may_claim_checkpoint_restore(manifest) is False
        assert _mod().describe_launch(manifest) == "fresh-start-replay"

    def test_checkpoint_restore_label_requires_launch_mode_saying_so(self):
        manifest = {
            "launch_mode": "checkpoint-restore",
            "replay": {
                "window": {"start": 3, "stop": 3},
                "events_total": 0,
                "events_consumed": 0,
                "mismatches": 0,
            },
        }
        assert _mod().may_claim_checkpoint_restore(manifest) is True
        assert _mod().describe_launch(manifest) == "checkpoint-restoration"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
