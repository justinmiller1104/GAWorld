# MCF × GAWorld — Phase 0.9 Preflight (GREEN + Stage 1 Spec)

**Status:** offline, zero LLM inference, no simulations, serial scoped pytest only.
**Labels used:** `Verified repo` (checked directly in this repository), `Computed` (deterministic script/grep output), `Assumption`, `Proposed`.
**Inputs:** BRIEF-MCF-phase09-green; the Phase 0.8 offline audit (`docs/MCF_GAWORLD_PHASE_0_8_OFFLINE_AUDIT.md`); RESULT-3206 (cancelled by the orchestrator — its call-site inventory was rebuilt in this task directly from the repo, §3).

---

## 1. Executive Summary

Phase 0.9 applies the six preflight corrections from the Task A audit to the
MCF time-use scorer and finalizes the exact Stage 1 execution specification.
All 52 MCF tests are GREEN (`Verified repo`: `pytest tests/test_mcf_time_use.py
tests/test_mcf_time_use_realism.py` → 52 passed, < 1 s, serial). The full test
suite shows no new failures vs the recorded baseline (§2, §8).

The one deliberate contract change: the sleep-after-paid_work tail exception
was **removed** — it contradicted the overnight head-carry introduced in Phase
0.8 (the same evening span was "unrecorded" at day N's tail and then counted
as sleep at day N+1's head, systematically deflating sleep for employed
agents). The locked legacy test
`tests/test_mcf_time_use.py::test_score_directory_scores_every_agent_file` was
updated accordingly and the change is documented in §2 and the module
docstring's change record.

**Stage 1 readiness: GO** (§9) — all tests green, spec complete, cost hooks
verified, kill thresholds explicit.

---

## 2. Scorer Preflight Corrections

All fixes in `benchmark/mcf/time_use.py`, stdlib-only, pure functions, no
production-execution-path changes. Labels `Verified repo` unless noted.

| # | Correction | Behavior now | Locked-test impact |
|---|---|---|---|
| 1 | **Sleep-after-work artifact removed** | A sleep episode directly following paid_work extends to 24:00 as sleep like any other evening sleep — consistent with the overnight head-carry (which had been attributing the same span as sleep on the next day). The artifact deflated sleep only for employed agents, biasing the participants-basis comparison. | `test_score_directory_scores_every_agent_file` updated deliberately (allowed: MCF evaluation test, not production): sleep sim 480 → **1380** (480 overnight + 900 evening tail). **Documented contract change.** |
| 2 | **Keyword precision** | The bare `吃` keyword over-matched non-meal compounds (吃亏 / 吃惊 / 吃苦 / 吃力 / …). Specific meal keywords match first; bare `吃` is a last-resort fallback rejected when the next character starts a non-meal compound. `吃亏了` → unknown; `吃饭` / `吃午饭` unchanged. | none |
| 3 | **Zero-employed participants** | With zero employed agent-days, paid_work is reported **unscored**: `sim=None`, `error=None`, `n_agent_days=0`, plus an uncertainty warning — never a silent 383-minute error. `overall_error` skips unscored categories. | none (empty-input contract still yields `overall_error == 0.0`) |
| 4 | **Exploratory basis marking** | Scoring paid_work with an explicit `day_types` map (workday-only subset) marks the entry `basis="exploratory"` with a `basis_note`: a workday subset vs a weekly-average survey target is not a like-for-like check. | none |
| 5 | **`ages_6_plus` caveat** | `TARGETS_2024["study"]["ages_6_plus"] = True` — the survey's inflated scope covers ages 6+, while the GAWorld cohort is adults (cohort skew documented in the Phase 0.8 audit §5). | none |
| 6 | **Tick-mismatch guard** | A splittable gap (> 2×`tick_minutes`) that is **not a multiple of** `tick_minutes` raises `ValueError` naming agent/day/gap/tick — the file was emitted at a different tick. Previously this silently inflated `unknown`. Anchored gaps (≤ 2×tick or boundary-anchored categories) are unaffected. | none |

`Assumption` (unchanged from Phase 0.8): head/tail conventions (sleep-default
head, 18:00 late-tail threshold) are conventions, not ground truth; the Stage 2
sensitivity check (±2 h threshold) remains `Proposed`.

---

## 3. Stage 1 Size Recalculation (agent-ticks vs LLM calls)

**RESULT-3206 was cancelled** — its 12-row call-site inventory is unavailable.
The inventory below was rebuilt directly from the repo (`Computed`, grep over
`gaworld/sim/`, `Verified repo` module responsibilities):

| # | Call site | Module | Task label | Fires per |
|---|---|---|---|---|
| 1 | perception | `sim/_cognition.py` | `perception` | cognition tick (throttled) |
| 2 | action decision | `sim/_action.py` | `actions` | action tick |
| 3 | action retry | `sim/_action.py` | `actions` | only on invalid/failed parse |
| 4 | location action | `sim/_action.py` | `location_actions` | location-constrained ticks |
| 5 | routine change | `sim/_step.py` | `routine_change` | probability-gated |
| 6 | transient thought | `sim/_step.py` | (thought) | probability-gated |
| 7 | curiosity probe | `sim/_curiosity.py` | (curiosity) | probability-gated |
| 8 | memory review | `sim/_memory_recall.py` | `memory_review` | recall ticks |
| 9 | news reaction | `sim/_news.py` | `news_reaction` | on unread news |
| 10 | info-seek / web-search reaction | `sim/_news.py` | `info_seek_reaction` / `web_search_reaction` | event-driven |
| 11 | RAG bootstrap | `sim/_rag.py` | `external_rag_bootstrap` | once per agent at start |
| 12 | daily summary | `sim/_diary.py` | `summary` | 1× per agent-day |

**The critical distinction:** agent-ticks ≠ LLM calls. Rows 1 and 3–12 are
throttled or probability-gated; only row 2 approaches once-per-action-tick.
Costs are therefore reported as **conservative bounds**, and the authoritative
number is always the run's own `run_manifest` LLM totals
(`gaworld/core/run_manifest.py`, `build()` folds `llm_stats.snapshot()` —
`Verified repo`) ÷ wall-clock.

---

## 4. Survey Basis Resolution

- **No mixed bases.** paid_work is scored on a participants basis (employed
  agent-days only, workdays when `day_types` is passed); every other category
  on an all-persons basis; study excluded by default (`child_inflated`,
  `ages_6_plus` caveat). `Verified repo`.
- **Exploratory marking.** A workday-only paid_work mean is stamped
  `basis="exploratory"` — 2024 targets are weekly averages, so a weekday-only
  sim must not be read as a strict comparison. `Verified repo`.
- **Population caveats.** The 2024 survey covers ages 6+; the cohort is adults.
  2018 robustness comparisons require the explicit `day_types` flag (2018 is a
  workday/restday split). Zero-employed cohorts yield unscored, not an
  error. `Verified repo` (code) / `Assumption` (cohort-skew magnitude).

---

## 5. Bias Audit Results

Fixed in this phase: sleep-after-work artifact (#1), keyword precision (#2),
zero-employed handling (#3), exploratory marking (#4), `ages_6_plus` (#5),
tick-mismatch guard (#6) — §2.

Kept (no-fix, with verdicts): scheduled-anchor gap pinning (realistic,
documented), sorted-repair for >1 h inversions (deterministic, lossless),
unknown-time reporting (never dropped, never rewarded). All `Verified repo`.

**Stage 2 residual:** keyword-mapper ceiling (coverage gate ≥ 0.85 on real
episodes); head/tail convention sensitivity (±2 h check); noise-domination
check on real data; compositional entanglement (never rank configs on one
dimension alone — Phase 0.8 §3).

---

## 6. Exact Stage 1 Execution Specification

All numbers `Computed` unless labeled otherwise.

| Parameter | Value | Basis |
|---|---|---|
| City bundle | `data/cities/绍兴柯桥/` (`agents.csv`: **80 residents** = **33 employed** + **47 NILF**; ages 6–92) | `Computed` from the CSV |
| Stage 1 cohort | **12 employed agents** (minimum 10 for participants-basis paid_work; identical IDs across all configs/seeds) | `Proposed` (audit §7 minimum + margin) |
| Configurations | 2, `personality.style_fit_amplitude` **low = 0.1** vs **high = 0.9** (default 0.3) | `Assumption` (manipulation points; 0.9 documented as the strong end in `config_docs.py`) |
| Seeds | 2 matched seeds per config (`random_seed`) | `Verified repo` hook, `Proposed` count (audit resolution: micro-run measures cost/coverage, not sensitivity) |
| Runs total | 2 configs × 2 seeds = **4 runs** | `Computed` |
| Duration | **3 sim days**, normal tick mode (day 1 treated as warm-up; scorer reads days 2–3) | `Proposed` |
| Tick grid | `time_grid_snap: true`, `time_step_minutes: 10` → **144 ticks/day**, fixed timeline | `Verified repo` (`generative_city_sim.py` TIME_STEP_MINUTES / TIME_GRID_SNAP) |
| Agent-ticks per run | 144 × 12 agents × 3 days = **5,184** | `Computed` |
| Agent-ticks, all runs | 5,184 × 4 = **20,736** | `Computed` |
| Expected LLM calls | **lower ≈ 10k, upper ≈ 60k** (lower: ~0.5 calls/agent-tick — action calls only on active ticks with gated rows firing rarely; upper: 3 calls/agent-tick = perception + action + retry). Authoritative: each run's `run_manifest.llm` totals. | `Computed` bounds, `Assumption` until measured |
| Wall-clock | serial ≈ **14–20 h** total (20–60k calls at ~2.5–3.5 s/call); with **8 workers ≈ 2–3 h**. Re-baselined from the first run's first 30 minutes (`Assumption` — zero inference allowed in this phase). | `Assumption` |
| Outputs | `output/memory/agent_<id>_episodes.jsonl` (episode stream, `gaworld/memory/experience.py`); `run_manifest` with `llm.call_count` / `by_task` per run | `Verified repo` |
| Scoring | `benchmark/mcf/time_use.py::score_directory` over the episode files, `tick_minutes=10`, employed list = cohort IDs, `day_types` passed; coverage + `uncertainty_warnings` reported alongside cost | `Verified repo` |

---

## 7. Cost Limits and Stop Conditions

- **Hard kill thresholds** (any one triggers): server/DNS errors on > **5%** of
  LLM calls (do not retry-loop — Phase 0.8 lesson); total calls > **60,000**
  (2× upper bound); a crash corrupting run state.
- **Soft thresholds:** cost/day exceeding the first-run baseline × 2 → pause
  and renegotiate scope before continuing.
- **Quality gate:** mapper coverage < **0.85** on real episodes → fix the
  mapper before any Stage 2 work; coverage is reported with cost in every
  Stage 1 summary.
- **Zero-inference guarantee of this phase:** no model calls were made; the
  spec's cost numbers are bounds and hooks only. `Verified repo`.

---

## 8. Change Record

- `benchmark/mcf/time_use.py` — six preflight corrections (§2) + module
  docstring change record. No production-execution-path changes.
- `tests/test_mcf_time_use.py` — **one deliberate contract change**:
  `test_score_directory_scores_every_agent_file` sleep expectation 480 → 1380
  with an updated comment explaining the artifact removal. No other test
  modified; no test written or weakened.
- `docs/MCF_GAWORLD_PHASE_0_9_PREFLIGHT.md` — this document (new).
- No commits made (task constraint); changes left in the worktree
  (`git status`: modified MCF scorer + one MCF test + new doc only).

**Test results** (`Verified repo`):
- `pytest tests/test_mcf_time_use.py tests/test_mcf_time_use_realism.py` → **52 passed**.
- Full suite `pytest tests` → **5 failed / 4709 passed / 22 skipped** vs the
  recorded baseline **4 failed / 4710 passed** (RESULT-3203). The 4 baseline
  failures reproduce unchanged (`test_auth_frontend.py` ×2,
  `test_model_routing.py`, `test_population_permissions.py` — dashboard
  modules, no MCF linkage). The 5th (`test_serious_game.py::…http`) **passes
  in isolation** — a full-suite flake, not a new failure. **No new failures.**

---

## 9. Go / No-Go Decision

**Verdict: GO for Stage 1.**

- All MCF tests GREEN (52/52); full-suite regression at the documented
  baseline with no new failures.
- The exact Stage 1 spec (§6) is complete: population, cohort, configs, seeds,
  duration, tick grid, agent-ticks, LLM-call bounds, wall-clock, outputs, and
  scoring procedure.
- Kill thresholds and stop conditions are explicit (§7) and the cost
  measurement hook (`run_manifest.llm`) is verified in-repo.
- Conditions carried into Stage 1: report mapper coverage alongside cost;
  DNS/server instability kills the run; every run's authoritative cost comes
  from its own manifest, not the §6 bounds.
- Stage 2 remains gated (no-go until coverage ≥ 0.85 on real episodes AND
  Stage 1 cost/day within threshold), unchanged from Phase 0.8 §9.
