# MCF × GAWorld — Phase 1.0 Engineering Audit

**Status:** Phase 1.0 GREEN complete. All MCF tests pass. No simulation run, no LLM call.
**Predecessors:** `MCF_GAWORLD_PHASE_0_9_PREFLIGHT.md` (authoritative end of the Phase 0.5→0.9 chain, GO verdict §9), `MCF_GAWORLD_INTEGRATION_FEASIBILITY.md` (living root doc).
**Evidence in:** `tests/test_mcf_phase10_red.py` (RED audit, TASK-3217), this doc (GREEN + corrections, TASK-3220).

---

## 1. MCF issues found & corrected

### 1.1 Sleep accounting — the 1,380-minute question (corrected)

**Verdict (from RESULT-3217): OVER-ATTRIBUTION.** The Phase 0.9 locked expectation
`sleep sim == 1380` (`tests/test_mcf_time_use.py::test_score_directory_scores_every_agent_file`)
decomposed, for its input (00:00 睡觉 / 08:00 上班工作 / 09:00 睡觉, tick=10), as:

| Span | Minutes | Rule | Verdict |
|---|---|---|---|
| 00:00–08:00 | 480 | head→first-gap, next = paid_work ∈ boundary-anchored | **justified** — sleep observed at 00:00, end pinned by the anchored work start |
| 08:00–09:00 | 60 | boundary-anchored gap → paid_work | **justified** |
| 09:00–24:00 | 900 | sleep tail: `last_category == "sleep"` extends to midnight | **invention** — a single 09:00 record credited 15 unobserved hours; at 09:00 this may be a nap or a glitch |

The defect was precise: the tail rule's **non-sleep** branch guarded with
`last_start >= 18:00` and capped at `2*tick`, but the **sleep** branch had **no time guard
at all** — any sleep record, even at 09:00, extended to midnight. 900 min ≈ 1.54× the
entire 586-min target attributed from one pre-noon observation with zero evidence.

**Correction applied (`benchmark/mcf/time_use.py`, Phase 1.0 change record):**

- *Sleep-tail corroboration:* a last sleep record evidences "sleeping on through midnight"
  only when it is an **evening** one (≥ 18:00), or a **≥ 10:00 record outside a work shift**
  (the resident has demonstrably left the workday and turned in). An **early-morning
  record (≤ 09:00)** — indistinguishable from a nap or a glitch — or a **mid-day sleep
  directly after paid_work** (a work-break nap) vouches for at most `2*tick`; the
  unobserved remainder goes to `unknown`, never silently to sleep. The corrected
  expectation for the 1,380 scenario is **500** (480 anchored + 20 capped); the locked
  test was updated deliberately, as the brief authorized (RESULT-3217 verdict: inflated).
- *First-day head honesty:* a **first recorded day** defaults its pre-midnight head to
  sleep only when the day itself ends in a corroborated sleep tail (evidence the log spans
  the night); otherwise — including a log that opens on sleep, or one whose last record is
  an uncorroborated nap — the head is genuinely `unknown`. A day with a recorded previous
  day ending in sleep still carries the overnight head as sleep (the Phase 0.8 head-carry
  is preserved; verified by `test_overnight_head_of_day_is_attributed_not_dropped` and
  `test_cross_midnight_sleep_reaches_sleep_target_scale`).
- *Empty comparison:* `score_against_targets({})` previously returned
  `overall_error == 0.0` with no warning — reading as "perfect fit". It now warns
  "no agent-days scored". The mean error also skips categories reported unscored
  (error `None`, e.g. paid_work with zero employed agent-days).

### 1.2 style_fit_amplitude range correction 0.9 → stress-only (corrected)

The admissible band **[0.10, 0.40]** existed only in prose
(`gaworld/settings/personality.py` docstring; `scripts/big5_effect_ceiling.py` operational
check). The Phase 0.9 spec (§6 table) used low=0.1 vs high=**0.9**, which read as an
admissible manipulation point. 0.9 is a **stress value**: personality explaining >~15% of
behavioural variance is louder than anything the literature supports.

**Correction applied:** `benchmark/mcf/time_use.py` now pins
`ADMISSIBLE_STYLE_FIT = (0.10, 0.40)`, exposes `validate_style_fit(amplitude)`
(`ValueError` outside the band, stress values rejected), and declares
`STRESS_STYLE_FIT = (0.9,)` separately — outside the band by construction, with a guard
that the band can never silently grow to include it.

### 1.3 Cost-model correction: the 8-worker assumption is unsupported (corrected)

The Phase 0.9 cost estimate assumed `day_routine_workers=8` could bring Stage 1 to
~2–3 h. That assumption is **unsupported**:

- `concurrency.enabled` defaults to **False** and `concurrency.day_routine_workers`
  defaults to **1** (`gaworld/settings/runtime.py:392,394`) — the default path is fully serial.
- Only **two** call sites use `parallel_map` (daily routine `generative_city_sim.py:3232`,
  fast-forward digest `:2825`). Everything else — perception, action (+retry), location
  actions, routine change/thought, curiosity, memory review, news/info-seek reactions,
  RAG bootstrap, daily summary — fires **inside the serial per-agent tick loop**; no
  `parallel_map` call site exists outside those two (grep over `gaworld/`: only
  `core/runner.py` + the two entry-point sites).
- With 8 workers, only routine+digest calls parallelize: 12 agents × 3 days × 4 runs =
  **144 calls of 20–60k (< 1%)**. The other ~99% are serial regardless of the knob.
  Realistic best case with 8 workers ≈ 13–20 h — the Phase 0.9 estimate treated the wrong
  stage as the bottleneck.

**Conservative Stage 1 wall-clock (default config, serial): 14–20 h total** for
20,736 agent-ticks at the Phase 0.9 bound of 10–60k calls at 2.5–3.5 s/call.
Authoritative per-run numbers remain each run's `run_manifest.llm` totals
(`gaworld/core/run_manifest.py`).

### 1.4 Survey-honesty gates (added)

The scorer computed the same `overall_error` for 2 agent-days as for 700 — no duration
or population gate existed. Added:

- `MIN_COMPARISON_DAYS = 7` — the 2024/2018 targets are weekly averages; a shorter
  comparison over-weights whatever weekday mix was sampled and cannot capture weekend
  behaviour (or an explicit workday/restday split with `basis` declared).
- `MIN_COMPARISON_POPULATION = 20` — with per-agent-day category SD ~120 min,
  ±10-min precision at 95% needs N ≥ (1.96·120/10)² ≈ **553 agent-days** (~80 agents at
  7 days/agent); the 20×7 floor (~140 agent-days) is the minimum, not survey-grade.
- `ScoreResult.frontier_valid` — `False` (plus explanatory warnings) when either gate is
  unmet: the comparison stays scored but is marked exploratory, never reported as a
  like-for-like frontier check. The Phase 0.9 12-agent cohort remains usable for
  cost/coverage measurement, not for a frontier survey comparison.

## 2. Canary specification (NOT executed)

Purpose: measure real per-call latency + tokens before any Stage 1 commit.

- **Scale:** 1 config × 1 seed × 2 agents × half-day (72 ticks), normal tick mode, default config (serial).
- **Hard limits (kill without finishing):**
  - max **400 LLM calls**;
  - max **250k tokens**;
  - max **$1.00** (assumes the repo's default provider pricing at ~$0.3–1/M tokens blended; recheck against the configured model before launch);
  - max **30 min** wall-clock;
  - kill condition: server/DNS errors on **> 5% of calls** (no retry loops — Phase 0.6/0.8 lesson).
- **Outputs:** per-task call counts + latency (from `llm_stats`); measured calls/agent-tick
  replaces the 0.5–3 calls/tick assumption; Stage 1 serial wall-clock is then
  (measured rate) × 20,736 × measured latency.
- **Status: specified only. NOT executed in Phase 1.0** (zero-inference constraint).

## 3. Revised Stage 1 specification (zero-inference)

Unchanged from Phase 0.9 §6 except where noted:

- Cohort 12 employed agents, configs/seeds as locked; **no worker-acceleration
  assumption** — plan against **serial 14–20 h** (default config), not 2–3 h.
- Survey-honesty gates active: a frontier comparison needs ≥ 7 days × ≥ 20 agents
  (~553 agent-days for ±10-min precision); the 12-agent Stage 1 output is
  `frontier_valid=False` / exploratory by construction.
- Canary (§2) must run first and pass its hard limits before any full Stage 1 commit.

## 4. Exact test results (serial, no LLM, no network)

| Suite | Result |
|---|---|
| `tests/test_mcf_phase10_red.py` (RED audit file, untouched) | **12 passed** (was 10 failed, 2 passed) |
| `tests/test_mcf_time_use.py` | **43 passed** (1 expectation corrected: 1380 → 500, per §1.1) |
| `tests/test_mcf_time_use_realism.py` | **9 passed** |
| **All MCF** | **64 passed** |
| Full suite | 4721 passed, 5 failed, 22 skipped — the 5 failures (`test_auth_frontend[True/False]`, `test_model_routing`, `test_population_permissions`, `test_serious_game`) are pre-existing on the base commit and touch no MCF code; no NEW failures vs the 4-real-failures + 1-known-flake baseline |

## 5. Unresolved problems

1. **Canary not executed** — the latency/token measurements that would replace the
   0.5–3 calls/tick assumption are still spec-only (§2).
2. **The banded sleep-tail rule is calibrated on audited scenarios, not on real logs** —
   the ≤ 09:00 / 10:00–18:00 / ≥ 18:00 bands and the post-work-nap rule are principled
   but fitted to the adversarial cases; the canary/Stage 1 logs should be spot-checked
   for mass `unknown` inflation.
3. **Survey-grade precision needs ~80 agents × 7 days (~553 agent-days)** — no current
   run meets it; any frontier claim before then stays exploratory by design.
4. **The 12-agent cohort cannot be widened without cost** — at serial 14–20 h per
   Stage-1-scale run, cohort expansion multiplies wall-clock linearly; a concurrency
   design that parallelizes *inside the tick loop* (not just routine/digest) is the only
   real lever and does not exist yet.

## Phase 1.1 correction — Stage 1 serial-cost range recomputed

The §1.3 headline "serial 14–20 h" does not follow from the Phase 0.9 inputs
(10–60k calls at 2.5–3.5 s/call). Recomputation:

- 10,000 × 2.5 s = **7 h** — lower bound assumes the low call inventory
  (~0.5 calls/agent-tick: action calls only on active ticks, gated rows firing
  rarely) *and* the fastest latency.
- 60,000 × 3.5 s = **58 h** — upper bound assumes the high inventory
  (3 calls/agent-tick: perception + action + retry) *and* the slowest latency.

The old 14–20 h is exactly 20,736 agent-ticks × 2.5–3.5 s — i.e. it implicitly
assumed **1 call per agent-tick (20,736 calls)**, a rate that appears nowhere in
the Phase 0.9 call-site inventory (0.5–3 calls/tick).

**Corrected planning range (default config, serial): 7–58 h**, spanning the
inventory bounds. Because the true calls/agent-tick is unmeasured, the range is
wide; the canary's *measured* per-call latency and measured calls/agent-tick
(§2) supersede every assumed latency above, and Stage 1 wall-clock must be
re-planned as (measured rate) × 20,736 × (measured latency) once it runs. The
wider range strengthens the §2 canary gate: no Stage 1 commitment before it.
