"""TDD RED tests (Phase 1.0 MCF RED audit, TASK-3217).

Four audit areas, adversarial-first:

1. Sleep accounting — the 1,380-minute question. The locked Phase 0.9 test
   ``test_mcf_time_use.py::test_score_directory_scores_every_agent_file``
   expects ``sleep sim == 1380`` for an input whose ONLY evidence of sleep is
   a 00:00 record, a 08:00 work anchor, and a single 09:00 ``睡觉`` record.
   480 min are justified (00:00–08:00, pinned by the anchored work start);
   the remaining 900 min (09:00 → 24:00) are credited to sleep purely by the
   tail rule — and the sleep branch of that rule (``benchmark/mcf/time_use.py``,
   ``compute_daily_minutes`` tail handling) has NO evening guard, unlike the
   non-sleep branch which caps at ``last_start >= 18:00`` + 2*tick. A sleep
   record at 09:00 may be a nap or a data glitch; 15 unobserved hours must
   become ``unknown`` (+ a warning), never silently credited sleep.

2. style_fit_amplitude admissible range — the independently justified band
   is [0.10, 0.40] (docstring of ``gaworld/settings/personality.py``,
   enforced operationally by ``scripts/big5_effect_ceiling.py``); the
   Phase 0.9 spec's 0.1 vs 0.9 uses 0.9 as a STRESS value, not an admissible
   manipulation point. The scorer/spec module must carry the admissible range
   as a constant, must validate against it, and must separate stress values.

3. Survey-comparison honesty — a 2–3-day / small-cohort comparison against
   weekly-average survey targets is not a like-for-like check. The scorer
   must refuse or explicitly mark frontier-invalid comparisons and declare
   its minimum duration/population thresholds.

4. (No tests: cost model is spec-only — see RESULT-3217 for the call-site
   concurrency table and the canary spec. ``resolve_max_workers`` defaults
   and the serial tick loop are production facts, documented not pinned here
   to keep this file free of production-code imports.)

Imports are done inside each test via :func:`_mod` so a missing feature
fails that test (with a clear message), not the whole module at collection.

No LLM calls, no network, no simulation. Serial pytest only.
"""

from __future__ import annotations

import pytest


def _mod():
    import benchmark.mcf.time_use as m

    return m


def _ep(agent_id: int, day: int, time: str, activity: str) -> dict[str, object]:
    return {
        "agent_id": agent_id,
        "day": day,
        "time": time,
        "final_activity": activity,
        "scheduled_activity": activity,
    }


# ===========================================================================
# 1. Sleep accounting — unobserved time must become unknown, never sleep
# ===========================================================================


class TestSleepAccounting:
    def test_single_midday_sleep_record_does_not_credit_unobserved_tail_to_sleep(self):
        """Adversarial sparse day: ONE 09:00 睡觉 episode, nothing else.

        The 00:00–09:00 head is unobserved (genuinely unknown). The
        09:00–24:00 tail is also unobserved: a 09:00 sleep record does not
        evidence 15 h of sleep (could be a nap). Expected: sleep ≤ 60 min
        (at most a bounded vouch), ≥ 1320 min unknown, incomplete-day
        warning fires, coverage < 0.85.
        """
        m = _mod()
        minutes = m.compute_daily_minutes([_ep(1, 1, "09:00", "睡觉")], tick_minutes=10)[1][1]
        assert minutes["sleep"] <= 60, (
            f"a single 09:00 sleep record was credited {minutes['sleep']} min of sleep; "
            "unobserved tail time must become unknown, not sleep"
        )
        assert minutes["unknown"] >= 1320, (
            f"only {minutes['unknown']} min of the unobserved day landed in unknown"
        )
        assert sum(minutes.values()) == 1440
        result = m.score_against_targets({1: {1: minutes}}, employed_agents=[])
        assert any("incomplete" in w for w in (result.uncertainty_warnings or [])), (
            "a day that is almost entirely unobserved must raise an uncertainty warning"
        )
        assert result.coverage < 0.85

    def test_locked_1380_scenario_over_attribution_is_corrected(self):
        """The exact input behind the locked sleep==1380 expectation.

        Evidence: 00:00 sleep (480 min justified via the 08:00 work anchor),
        08:00 work (60 min, anchored by the next record), 09:00 sleep — and
        NOTHING until midnight. The 09:00–24:00 span (900 min) is unobserved;
        crediting it as sleep inflates the mean by ~62% of the 586-min
        target from one pre-noon record. Expected: sleep ≤ 540, unknown ≥ 840,
        warning fires, coverage < 0.85.
        """
        m = _mod()
        episodes = [
            _ep(31, 1, "00:00", "睡觉"),
            _ep(31, 1, "08:00", "上班工作"),
            _ep(31, 1, "09:00", "睡觉"),
        ]
        minutes = m.compute_daily_minutes(episodes, tick_minutes=10)[31][1]
        assert 480 <= minutes["sleep"] <= 540, (
            f"sleep={minutes['sleep']}: the observed 00:00–08:00 span (480 min) is "
            "justified, but the 09:00–24:00 tail must NOT be silently credited to sleep"
        )
        assert minutes["unknown"] >= 840, (
            f"the unobserved 09:00–24:00 span must land in unknown, got "
            f"{minutes['unknown']} min"
        )
        assert sum(minutes.values()) == 1440
        result = m.score_against_targets({31: {1: minutes}}, employed_agents=[31])
        assert any("incomplete" in w for w in (result.uncertainty_warnings or [])), (
            "a day with 900 unobserved minutes must raise an uncertainty warning"
        )
        assert result.coverage < 0.85

    def test_sleep_tail_evening_guard_matches_non_sleep_rule(self):
        """A sleep tail is only 'evidence of sleeping on' when the record is
        an EVENING one (>= 18:00), mirroring the non-sleep branch's guard.
        A 15:00 sleep record vouches for at most a bounded tail; the rest is
        unknown."""
        m = _mod()
        episodes = [
            _ep(2, 1, "08:00", "上班工作"),
            _ep(2, 1, "15:00", "睡觉"),  # afternoon nap, not evening sleep
        ]
        minutes = m.compute_daily_minutes(episodes, tick_minutes=10)[2][1]
        # 08:00–15:00 paid_work is justified (pinned by the sleep record as a
        # boundary anchor). The 15:00–24:00 tail (540 min) is unobserved.
        assert minutes["paid_work"] == 420
        assert minutes["sleep"] <= 60, (
            f"an afternoon (15:00) sleep record credited {minutes['sleep']} min of "
            "sleep through midnight; only an evening sleep record may extend"
        )
        assert minutes["unknown"] >= 480

    def test_empty_input_reports_no_data_rather_than_zero_error(self):
        """Scoring zero agent-days currently returns overall_error == 0.0 and
        NO warnings — which reads as 'perfect fit'. It must warn that nothing
        was scored."""
        m = _mod()
        result = m.score_against_targets({}, employed_agents=[])
        assert result.uncertainty_warnings, (
            "an empty comparison must warn that no agent-days were scored, "
            "not return a silent zero error"
        )


# ===========================================================================
# 2. style_fit_amplitude admissible range [0.10, 0.40]; stress values separate
# ===========================================================================


class TestStyleFitAmplitudeRange:
    def test_admissible_range_constant_declared(self):
        m = _mod()
        rng = getattr(m, "ADMISSIBLE_STYLE_FIT", None)
        assert rng is not None, (
            "benchmark.mcf.time_use must declare ADMISSIBLE_STYLE_FIT = (0.10, 0.40); "
            "the band currently lives only in prose (personality.py docstring) and "
            "scripts/big5_effect_ceiling.py output, not as a pinned constant"
        )
        assert rng == pytest.approx((0.10, 0.40))

    def test_validation_rejects_out_of_range_amplitude(self):
        m = _mod()
        validate = getattr(m, "validate_style_fit", None)
        assert validate is not None, (
            "benchmark.mcf.time_use must expose validate_style_fit(amplitude) that "
            "raises ValueError outside [0.10, 0.40]"
        )
        for bad in (0.0, 0.05, 0.09, 0.41, 0.9, 1.5, -0.2):
            with pytest.raises(ValueError, match="style_fit"):
                validate(bad)
        # Admissible values pass, including the configured default 0.30.
        for good in (0.10, 0.20, 0.30, 0.40):
            validate(good)

    def test_stress_values_are_marked_not_admissible(self):
        """Phase 0.9's 'high = 0.9' is a stress-test point: it must be declared
        as a stress value and must sit OUTSIDE the admissible range."""
        m = _mod()
        stress = getattr(m, "STRESS_STYLE_FIT", None)
        assert stress is not None, (
            "benchmark.mcf.time_use must declare STRESS_STYLE_FIT (e.g. (0.9,)) so "
            "stress points are separated from admissible manipulation points"
        )
        lo, hi = getattr(m, "ADMISSIBLE_STYLE_FIT", (0.10, 0.40))
        for value in stress:
            assert not (lo <= value <= hi), f"stress value {value} must be outside the admissible band"

    def test_phase09_high_point_outside_admissible_band(self):
        """Guard the Phase 1.0 correction directly: 0.9 (the Phase 0.9 'high')
        is not an admissible amplitude. If the range ever widens to include
        it, this audit finding is void and must be re-justified."""
        m = _mod()
        lo, hi = getattr(m, "ADMISSIBLE_STYLE_FIT", (0.10, 0.40))
        assert not (lo <= 0.9 <= hi), (
            "0.9 must not be inside ADMISSIBLE_STYLE_FIT — personality explaining "
            ">15% of behavioural variance exceeds literature support"
        )


# ===========================================================================
# 3. Survey-comparison honesty — duration and population gates
# ===========================================================================


class TestSurveyComparisonHonesty:
    def test_minimum_thresholds_are_declared(self):
        m = _mod()
        min_days = getattr(m, "MIN_COMPARISON_DAYS", None)
        min_pop = getattr(m, "MIN_COMPARISON_POPULATION", None)
        assert min_days is not None, (
            "declare MIN_COMPARISON_DAYS: a weekly-average survey target needs >= 7 "
            "simulated days (or an explicit workday/restday split with basis declared)"
        )
        assert min_days >= 7
        assert min_pop is not None, (
            "declare MIN_COMPARISON_POPULATION: category means need a cohort whose "
            "agent-days give a stable mean (see RESULT-3217 CI argument)"
        )
        assert min_pop >= 20, "a population below ~20 agents cannot ground a weekly-average comparison"

    def test_short_comparison_is_frontier_invalid(self):
        """2 simulated days vs weekly-average survey targets: must be refused
        or explicitly marked frontier-invalid, with a warning — not reported
        as a plain overall_error."""
        m = _mod()
        minutes = {
            "sleep": 500,
            "personal_care": 56,
            "meals": 103,
            "paid_work": 400,
            "housework": 77,
            "leisure": 153,
            "socializing": 18,
            "commuting": 50,
            "unknown": 0,
        }
        daily = {1: {day: dict(minutes) for day in (1, 2)}}  # 2 days only
        result = m.score_against_targets(daily, employed_agents=[1])
        validity = getattr(result, "frontier_valid", None)
        assert validity is False, (
            f"a 2-day comparison reported frontier_valid={validity!r}; short comparisons "
            "must be marked invalid (or the scorer must refuse) vs weekly-average targets"
        )
        assert any("day" in w.lower() for w in (result.uncertainty_warnings or [])), (
            "a sub-threshold duration must raise an explanatory warning"
        )

    def test_small_population_is_marked_exploratory(self):
        """1 agent × 10 days: duration is enough for a weekly average, but a
        single resident cannot ground a population-level category mean. Must
        be marked exploratory/invalid with a warning."""
        m = _mod()
        minutes = {
            "sleep": 550,
            "personal_care": 56,
            "meals": 103,
            "paid_work": 383,
            "housework": 77,
            "leisure": 153,
            "socializing": 18,
            "commuting": 50,
            "unknown": 0,
        }
        daily = {1: {day: dict(minutes) for day in range(1, 11)}}  # 10 days, 1 agent
        result = m.score_against_targets(daily, employed_agents=[1])
        validity = getattr(result, "frontier_valid", None)
        assert validity is False, (
            f"a 1-resident comparison reported frontier_valid={validity!r}; a cohort of "
            "one cannot support a population-level comparison"
        )
        assert any(
            ("population" in w.lower()) or ("cohort" in w.lower()) or ("agent" in w.lower())
            for w in (result.uncertainty_warnings or [])
        ), "a sub-threshold population must raise an explanatory warning"

    def test_adequate_comparison_is_valid(self):
        """The documented adequate design — >= MIN_COMPARISON_DAYS days ×
        >= MIN_COMPARISON_POPULATION agents — must NOT be flagged invalid."""
        m = _mod()
        min_days = getattr(m, "MIN_COMPARISON_DAYS", 7)
        min_pop = getattr(m, "MIN_COMPARISON_POPULATION", 20)
        minutes = {
            "sleep": 560,
            "personal_care": 56,
            "meals": 103,
            "paid_work": 383,
            "housework": 77,
            "leisure": 153,
            "socializing": 18,
            "commuting": 50,
            "unknown": 0,
        }
        daily = {a: {d: dict(minutes) for d in range(1, min_days + 1)} for a in range(1, min_pop + 1)}
        result = m.score_against_targets(daily, employed_agents=list(range(1, min_pop + 1)))
        assert getattr(result, "frontier_valid", True) is True, (
            "a comparison meeting the declared duration+population thresholds must "
            "not be marked frontier-invalid"
        )
