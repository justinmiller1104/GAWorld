# MCF × GAWorld — Phase 0.8 Offline Audit

**Status:** offline, zero LLM inference, no simulations.
**Labels used:** `Verified repo` (checked directly in this repository), `Computed` (deterministic script output), `Assumption`, `Proposed`.
**Scope:** scorer corrections (`benchmark/mcf/time_use.py`), evaluation-dimension independence analysis, survey comparability, and the final micro-run design (Stage 1/2/3 with stop criteria).

---

## 1. Executive Summary

Phase 0.8 closes the offline phase of the MCF benchmark integration. All 52 scorer
tests are GREEN (43 pre-existing contract tests + 9 realism tests added by the
Phase 0.8 audit, RESULT-3201). The scorer now (a) accounts for the full 24-hour
day, recovering ~7–8 h of overnight sleep per agent-day that the previous
implementation dropped, (b) surfaces uncertainty instead of hiding it
(`uncertainty_warnings`, incomplete-day tails no longer pad leisure),
(c) declares survey bases (`survey_basis`, `day_types`) so weekday-only sims are
never silently compared to weekly averages, and (d) exposes `per_agent_day` and
`n_agent_days` so the 24h-budget coupling question is answerable from real runs.

The dimension-independence analysis (`Computed`, script
`~/work/crew/implement/OUTBOX/independence_analysis_3203.py`) shows the paid-work
and leisure dimensions are **not redundant**: under independently generated
behaviour the two error dimensions are essentially uncorrelated across a config
sweep (r = −0.186), while under perfectly coupled behaviour they co-move
strongly (r = +0.833). Co-movement therefore measures behavioural coupling, not
a normalization artifact — but the 1440-min budget does produce a small
spurious negative correlation, so configs must never be ranked on one dimension
alone.

Go/No-Go (§9): **conditional Go** — Stage 1 micro-run may proceed; Stage 2/3
gates are specified below.

---

## 2. Scorer Audit and Corrections

Findings are from RESULT-3201 (audit against the real episode writer,
`gaworld/sim/_stages.py:901–951` and `gaworld/memory/experience.py:79–86`).
All fixes are in `benchmark/mcf/time_use.py` (`Verified repo`, stdlib-only,
pure functions, no production-execution-path changes).

| # | Finding (RESULT-3201) | Correction | Label |
|---|---|---|---|
| 1 | Cross-midnight head-of-day loss (~7–8 h sleep/day vanished) | Day = full calendar day. The head (00:00 → first episode) is carried into the current day as the previous recorded day's terminal category when that is sleep; otherwise it defaults to sleep unless the log itself opens on sleep | `Verified repo` |
| 2 | Sleep-after-paid_work tail rule suspected artifact | Kept: the locked contract tests (`tests/test_mcf_time_use.py::test_score_directory_scores_every_agent_file`) pin that a sleep episode directly following paid_work ends the recorded day. It is now *documented* as the nap/ambiguous-shift case, and combined with fix #1 the overnight sleep is recovered on the following day's head instead of vanishing | `Verified repo` |
| 3 | Scheduled-anchor gap pinning realistic, undocumented | Kept and documented (module docstring): sleep/paid_work/meals/personal_care start timestamps pin the previous activity's end | `Verified repo` |
| 4 | No uncertainty signal; incomplete days padded the last category | `uncertainty_warnings` on `ScoreResult` (low coverage < 0.85, incomplete days); a non-sleep last episode observed late in the day (≥ 18:00) vouches for at most 2×tick, the unobserved evening goes to `unknown` and is flagged — never padded into leisure | `Verified repo` |
| 5 | Out-of-order timestamps silently clamped | Deterministic: inversions > 1 h mean the record is unordered → the day is sorted by time and treated as contiguous; smaller inversions are write jitter → record order kept, negatives clamp to zero | `Verified repo` |
| 6 | No survey-basis metadata / day-type flag | `survey_basis` on every target (2024 = `weekly_average`, 2018 = `workday_restday_split`); `score_against_targets(..., day_types=...)` scores paid_work over workdays only | `Verified repo` |
| 7 | Participants-basis opacity | Per-category `n_agent_days` reports how many agent-days each mean covers (employed + workday for paid_work) | `Verified repo` |
| 8 | 24h coupling not analysable | `per_agent_day` exposes per-agent per-category minutes on `ScoreResult` | `Verified repo` |

**Accounting basis (documented in the module docstring):** a day spans its full
1440 minutes. Sleep tails extend to 24:00 (evening sleep implies sleeping on
through midnight, except after a work shift); heads are carried backward from
the previous night; non-sleep evening tails and long unanchored gaps go to
`unknown`, which is reported and warned on, never dropped.

---

## 3. Evaluation-Dimension Independence

Script: `~/work/crew/implement/OUTBOX/independence_analysis_3203.py`
(raw output alongside: `independence_analysis_3203.out`). Deterministic, seed
20261105, 24 configs × 500 synthetic agents, sweep over mean work 400–840 min;
errors measured against the 2024 targets (paid_work 383, leisure 153).
All numbers `Computed`.

| Generator | paid_work error range | leisure error range | r(work_err, leisure_err) | monotone co-movement |
|---|---|---|---|---|
| Independent (work, leisure drawn separately; residual absorbs budget) | 50.7 – 453.4 min | 94.1 – 109.9 min | **−0.186** | No |
| Perfectly coupled (each extra work minute taken from leisure) | 50.2 – 459.8 min | 50.1 – 153.0 min | **+0.833** | No |

**Compositional-artifact check:** recomputing errors as relative (share-of-target)
distances instead of raw minutes leaves the correlations unchanged (−0.186 /
+0.833) — the normalization choice is not what creates or destroys the
trade-off. The pure budget constraint shows up only as the small negative
co-movement (−0.186) in the independent generator, from the residual bucket
absorbing the budget.

**Disproof tests** (implemented in the script; (ii) also a Stage-2 gate):

1. *Redundancy:* the dimensions degenerate to one iff their errors co-move
   monotonically across **any** Θ variation. Both generators fail that
   condition (`Computed`), so the two dimensions carry different information.
2. *Noise domination:* if a dimension's within-config SD exceeds its
   across-config range, it measures mapper/coverage noise, not behaviour —
   re-checked on real episodes at Stage 2 (§7).
3. *Compositional artifact:* with a hard 1440 budget, raw-minute errors
   co-move even for independent behaviours (here mildly, r = −0.186).
   Correction applied: report both dimensions; never rank configs on one
   dimension alone.

**Verdict:** the paid-work and leisure dimensions provide genuinely different
information. They degenerate to a single dimension only when behaviour itself
is strongly coupled (r → +0.83 and above) — which is a finding about residents,
not about the metric — or when mapper coverage is so low that both dimensions
are dominated by the same unknown-time noise (gate at §7 Stage 2).

---

## 4. Synthetic Validation Results

- Test files: `tests/test_mcf_time_use.py` (43 tests, pre-existing contract,
  unchanged and passing) and `tests/test_mcf_time_use_realism.py` (9 realism
  tests from RESULT-3201). **52 passed, 0 failed** (`Verified repo`,
  serial scoped run, < 1 s).
- The 9 realism RED tests now pass for the pinned reasons: overnight sleep
  reaches the ~586-min target scale (sim 495–540 min/day on the synthetic
  schedules vs 30 min before the fix); incomplete days warn instead of padding
  leisure; low coverage (< 0.85) warns; `survey_basis` metadata present;
  `day_types` accepted; out-of-order timestamps sorted deterministically;
  `per_agent_day` and `n_agent_days` exposed.
- Hand-calculated exact-minute cases (full 24h day at tick 30) remain exact.

---

## 5. Survey Comparability

`Verified repo` unless noted:

- **Participants vs all-persons bases never mixed.** paid_work is scored only
  over employed agents' workdays; `n_agent_days` makes the exclusion
  (NILF vs present-but-zero) checkable.
- **Basis mismatch surfaced.** `TARGETS_2024[*].survey_basis =
  "weekly_average"`; a weekday-only sim compared against it without
  `day_types` is now detectable. `Assumption`: the scorer warns via metadata
  only (no automatic weekday/weekend inference) — callers must pass
  `day_types` for workday/restday comparisons.
- **Categories consistent.** 2018's combined personal-care+meals bucket stays
  under `meals` with a documented caveat; `study` excluded by default
  (child-inflated), both surveys.
- **Residual mismatches (documented, unresolved by design):**
  - 2024 survey covers ages 6+; the GAWorld cohort is adults — study and
    housework means still carry slight cohort skew. `Assumption`.
  - 2024 weekly averages include rest days; workday-only runs must use
    `day_types` and, for strict comparability, the 2018 workday column.
  - Classification errors cannot reward unrealistic behaviour: unknown time
    counts as full-target error in its true category and is additionally
    reported (`leftover_minutes`) and warned on.

---

## 6. Remaining Scientific Risks

1. **Keyword-mapper ceiling.** `classify_activity` is a closed keyword list;
   episodes outside it inflate `unknown`, which depresses every category's
   accuracy simultaneously. Mitigation: coverage gate ≥ 0.85 before any
   comparison is read (Stage 2 gate).
2. **Head/tail conventions are conventions.** The sleep-default head and the
   18:00 non-sleep-tail threshold are defensible but not ground-truthed; a
   sensitivity check (±2 h threshold) is `Proposed` for Stage 2.
3. **Compositional entanglement.** Improving paid_work realism can move
   leisure error up purely through the budget; interpret dimensions jointly.
4. **Synthetic-vs-real gap.** The 9 realism tests use hand-written schedules;
   real episode streams (one episode per tick) may exercise the repair paths
   (sort/contiguous) differently. Stage 1 checks schema sanity on real output.

---

## 7. Final Micro-Run Design (seed-count resolution)

Resolution of the 2-seed vs 3-seed inconsistency: the micro-run measures **call
cost and mapper coverage, not sensitivity** — 2 seeds suffice. Sensitivity
(config effect vs noise) requires ≥ 3 seeds per config and is Stage 2.
`Proposed` (design), with `Verified repo` cost hooks.

### Stage 1 — Micro-run (cost + coverage)

- **Configs:** 2 matched-seed configs (`style_fit_amplitude` low vs high) ×
  **2 seeds** × **≥ 10 employed agents** (minimum for participants-basis
  paid_work scoring), **2–3 sim days**, normal mode.
- **Purpose:** per-tick call cost; mapper coverage on real episodes; episode
  schema sanity.
- **Estimated calls:** tick grid × agents × days (e.g. 10-min ticks → 144
  ticks/day → 144 × 10 × 2.5 ≈ 3,600 agent-ticks per run pair).
- **Cost procedure:** `run_manifest` LLM totals
  (`gaworld/core/run_manifest.py:373–420`, `build()` folds the bound
  `llm_stats.snapshot()`) ÷ wall-clock.
- **Stop/kill criteria:** server errors or DNS failures on > 5% of calls →
  kill and reschedule; coverage < 0.85 on real episodes → fix mapper before
  Stage 2; any crash corrupting run state → kill.

### Stage 2 — Sensitivity probe

- **Design:** 2 configs × **3 seeds** × **≥ 5 days** × reduced cohort.
- **Gates (stop criteria):**
  - mapper coverage < 0.85 → stop, fix mapper first;
  - seed SD > config range on **both** dimensions → stop (no signal);
  - cost/day > threshold set from Stage 1 → renegotiate scope before
    continuing.
- **Noise-domination check** (§3 disproof test ii) runs here on real data.

### Stage 3 — Full pilot

- **Design:** 2-D frontier over the paired architecture
  (`channels.rules` on/off), matched seeds, per RESULT-3192.
- **Proceed criteria:** Stage 2 shows ≥ 1 dimension with config range >
  2× seed SD **and** coverage ≥ 0.85. Otherwise: no Stage 3.

---

## 8. Estimated Costs

- **Stage 1 (measured target):** ≈ 3,600 agent-ticks per run pair (2 configs ×
  2 seeds × ≥10 agents × 2–3 days, 10-min tick grid). The authoritative number
  comes from the run's own `run_manifest` LLM totals ÷ wall-clock
  (`Verified repo` hook location; counts `Assumption` until Stage 1 runs).
- **Stage 2:** scales to 2 configs × 3 seeds × ≥5 days — ≥ 3.5× Stage 1
  compute; only started if Stage 1 cost/day is within the negotiated
  threshold.
- **Offline phase (this document):** zero model calls, zero spend
  (`Verified repo`: scorer tests run in < 1 s, CPU only).

---

## 9. Go / No-Go

**Verdict: conditional GO.**

- Go: Stage 1 micro-run may start immediately — scorer GREEN (52/52),
  full-suite regression at documented baseline (see RESULT), accounting basis
  and warnings in place, cost-measurement hook identified.
- Conditions carried into Stage 1: mapper coverage on real episodes must be
  reported alongside cost; DNS/server instability kills the run (do not
  retry-loop).
- No-Go for Stage 2 until: coverage ≥ 0.85 on real episodes AND Stage 1
  cost/day within threshold. No-Go for Stage 3 until Stage 2 shows a dimension
  with config range > 2× seed SD.
- Residual blockers: none offline. The cohort-vs-survey age mismatch (§5) is a
  documented interpretation limit, not a blocker.

---

## Change Record (Phase 0.8 GREEN)

- `benchmark/mcf/time_use.py` — accounting basis, warnings, metadata, per-agent
  outputs (§2). No commits made (task constraint); changes left in worktree.
- `tests/test_mcf_time_use.py`, `tests/test_mcf_time_use_realism.py` — locked
  test files, **not modified** by this task.
- `docs/MCF_GAWORLD_PHASE_0_8_OFFLINE_AUDIT.md` — this document (new).
- Analysis script + raw output — `~/work/crew/implement/OUTBOX/`
  (`independence_analysis_3203.py`, `.out`); repo untouched by these.
