# MCF × GAWorld — Authoritative Reference

**Status:** consolidation of the superseded MCF phase documents (Phase 0.5–1.0).
This is the single authoritative doc for the Model Capacity Frontier (MCF)
integration into GAWorld's evaluation infrastructure. The full history of the
phase documents — including the complete evidence trail and change records —
lives **outside the repository** in the external agent workspace:

```
/home/justin/gaworld-agent-work/mcf-archive/     # external workspace (Phase 1.1)
```

The in-repo copies of the superseded phase docs remain on disk (untracked)
under `docs/`; `docs/MCF_GAWORLD_INTEGRATION_FEASIBILITY.md` (the living root
doc, contains the uncommitted 2026-11-05 data-audit revision) and
`docs/MCF_GAWORLD_PHASE_1_0_ENGINEERING_AUDIT.md` remain tracked.

Labels: **Verified (repo/paper/computed)** = checked against the repo, the MCF
paper, or deterministic computation; **Proposed** / **Assumption** as in the
phase docs.

---

## 1. Objective & scientific framing

The **Model Capacity Frontier (MCF)** (**Verified (paper)**, Zhao, Li & Shirado,
arXiv:2610.24168v2) is the nondominated set of expected empirical-error vectors
attainable by a model *architecture* across its admissible configuration space:

> f(θ) = E_ξ[e(θ,ξ)] over admissible θ ∈ Θ_M; F_M = {f(θ) | θ ∈ Θ_M, ∄ θ′ : f(θ′) ≺ f(θ)}

The error vector e ∈ R^k scores one run against k real-world empirical targets;
dominance is componentwise. Architecture development becomes frontier
estimation (sampling Θ_M, extracting the Pareto set); architecture comparison
becomes newly-attainable-set analysis, A(M1|M0).

**Verdicts (Feasibility doc §13):**
- **Conditional GO** for the full social-simulation layer, via a dedicated
  additive `benchmark/mcf/` module (Option B); design the sampler abstraction so
  Option C (workbench integration) can build on it later, but do not build C now.
- **NO-GO for ARENA** (`gaworld/apps/arena_api.py`): 10 single-agent LLM-judged
  tasks — no error vector over real observations, no meaningful Θ.
- The architecture/config boundary (are prompts inside M or part of θ?) must be
  declared and frozen per experiment (**Verified (paper)**: every MCF conclusion
  holds relative to this boundary).

## 2. What is evaluable in GAWorld today

### 2.1 Time-use scorer (`benchmark/mcf/time_use.py`)

The one fully validated dimension family. Architecture (**Verified (repo)**):

- `TARGETS_2024`: cat → minutes + basis. sleep 586, personal_care 56, meals 103,
  **paid_work 383 (`basis="participants"`)**, housework 77, leisure 153,
  socializing 18, commuting 50, study 107 (`basis="child_inflated"`,
  `ages_6_plus`, excluded from default scoring).
- `TARGETS_2018`: urban robustness vector (2018 is a workday/restday split —
  comparisons require the explicit `day_types` flag).
- `compute_daily_minutes(episodes, tick_minutes)`: timestamp differencing with
  24-hour accounting (Phase 0.8 overnight head-carry recovered ~7–8 h of sleep
  per agent-day); gaps ≤ 2×tick → previous category; larger gaps capped at
  2×tick + remainder `unknown`; tick-mismatch guard raises `ValueError`.
- `score_against_targets(...)` → `ScoreResult` with `per_category`,
  `overall_error` (skips unscored categories), `coverage`,
  `uncertainty_warnings`, `per_agent_day`, `n_agent_days`, `frontier_valid`.
- Zero-employed cohorts report paid_work **unscored** (`sim=None`,
  `error=None`), never a silent 383-minute error.

Bias-audit corrections baked in (Phase 0.9 + 1.0): sleep-tail corroboration
bands (evening ≥ 18:00 or ≥ 10:00 outside a work shift corroborates a through-
midnight sleep tail; early-morning or post-work nap records vouch for at most
2×tick, the rest goes to `unknown`); meal-keyword precision (bare `吃` is a
last-resort fallback); first-day head honesty; empty-comparison warning. The
Phase 1.0 corrected expectation for the audited 1,380-minute scenario is
**500** (480 anchored + 20 capped).

**Phase 1.7 Track 2 correction (TASK-3439, offline, pilot16 logs):** the
scorer previously read only `final_activity`, whose tick snapshots are
「前往X」/「处理待办」 strings matching no keyword (coverage 0.11–0.14 on the
pilot agents). The scorer now also reads the structured episode fields:
an in-transit `travel.status` (departed/in_transit) or the writer's
乘坐…移动 `action` phrase maps to commuting; stale 「前往X」 labels after
arrival (status arrived/stationary) and 「处理待办」 stay unknown — no POI
guessing, no sleep invention. `split_run_segments()` detects run
contamination (episode files were appended across runs, duplicating 75
ticks) and segments must be scored separately, never merged. On the
pilot's live segment this lifts coverage to ~0.94, but ~90% of classified
time is commuting — partly real (status-verified in-transit ticks), partly
the locked early-stop rule persisting the last commuting label across the
unobserved tail, and the agents travel through 00:00–07:00, so **sleep
remains 0 everywhere and time use is still not survey-comparable from tick
snapshots**. Smallest logging fix: write one machine-readable span record
`{"agent_id", "day", "category", "start_min", "end_min", "mode"}` per
activity transition (computable from `travel.status` transitions + the
schedule slot), or add `activity_category` + `activity_started_at` to each
tick snapshot; either way the episode file must be reset between runs.
Benchmark-population limitation: an all-employed cohort (e.g. Stage 1's
12) cannot ground the all-persons categories (leisure 153 / sleep 586 /
housework 77 min are whole-population weekly averages); below
`MIN_COMPARISON_POPULATION=20` results stay exploratory
(`frontier_valid=False`).

#### 2.1.1 Observed-only time accounting (Phase 1.8, Contract 1)

`compute_observed_minutes(episodes, tick_minutes, window_start=None,
window_end=None)` is the honest twin of `compute_daily_minutes`: it never
credits activity minutes past the last observed record. A sim ending at
12:20 accounts exactly 740 observed minutes; the 12:20→24:00 tail is
**unobserved**, never leisure/work/unknown-observed. The only extension
past the last record is the Phase 1.0 corroborated evening-sleep tail
(≥ 18:00 sleep records evidence sleeping on through midnight — this is
also what carries a 23:30→00:30 sleep span across the day boundary and
credits the next day's 00:00→00:30 head as sleep). Each agent-day yields
an `ObservedBreakdown` with four separate buckets: `observed_minutes`
(classified + unknown observed), `classified_minutes`, `unknown_minutes`,
`unobserved_minutes` (1440 − observed); the mapping protocol exposes the
per-category partition, which sums to the full 1440 with unobserved
minutes folded into the `unknown` bucket (a scoring bucket, never a
category). `coverage_breakdown(episodes, tick_minutes)` reports
`observed_coverage` (classified / observed) and `full_day_coverage`
(classified / 1440) as separate numbers — a half-observed day is never
diluted into what looks like poor classification of a full day. An
explicit `window_start`/`window_end` pins the observed span exactly
(08:00→12:20 = 260 observed minutes).

#### 2.1.2 Structured activity records (Phase 1.8, Contract 2)

`ActivityRecord(run_id, agent_id, category, start_min, end_min,
evidence_source)` is the machine-readable span record recommended above;
it is captured from authoritative tick/schedule state with **zero
additional LLM inference**. `category` may be the explicit `unknown` —
nothing is fabricated (no invented sleep/work/leisure from an empty or
unclassifiable record). Overlapping spans are merged deterministically
by `dedupe_activity_records` (sort by (start, end, category, agent,
evidence), union same-category overlaps) so no minute is ever
double-counted. When a tick snapshot carries the structured
`activity_category` field it takes precedence over keyword inference;
keyword hits are the weakest evidence and are marked as such in
`evidence_source`.

#### 2.1.3 Travel vs commuting classification policy (Phase 1.8, Contract 3)

`classify_activity_record` precedence: (1) structured `activity_category`
→ evidence `structured`; (2) `travel.status == in_transit` → **travel**,
never commuting, unless a structured `destination_purpose` names a
verified purpose (e.g. `paid_work`) → `commuting` with evidence
`destination_purpose`; (3) keyword inference; (4) `unknown`.
Purpose-unknown journeys stay `travel`/unknown and are **excluded from
commuting** in survey comparisons (`survey_comparison_categories()`:
`travel` is a first-class bucket and `commuting_aliases` contains only
`commuting`), so unknown-purpose travel cannot inflate the commuting
estimate against the survey targets.

#### 2.1.4 Run isolation (Phase 1.8, `benchmark/mcf/launch.py`)

`create_run(base_dir, config)` gives every run a unique `run_id`
(timestamp + uuid suffix) and its own `run_dir` under `base_dir`; episode
files live inside it, so two consecutive runs can never append to each
other's files and an interrupted run's partial outputs stay untouched by
the next run. A partial manifest (`outcome="in_progress"`, empty
`finished_at` — the exact format of the real interrupted pilot16 run
`20261109_133955`) is written at creation; `finalize_run(handle,
outcome=...)` rewrites it with the real outcome and a finish timestamp,
and the manifest binds the run's own `output_dir` so results are always
attributable to their run. `validate_observation_window(records,
window_start_min, window_end_min, days=None)` flags episode records whose
time falls outside the run's observation window or whose `day` is not one
of the run's days (schedule-consistency guard).

#### 2.1.5 Nighttime behaviour — defect D1 fixed, D2/D3 retained (Phase 1.8 Part D)

Of the three nighttime gating defects identified in the genuine-behaviour
investigation (RESULT-nighttime.md), defect D1 — the head-of-day sleep
coverage hole — is fixed: `ensure_sleep_in_schedule` now stretches the
inserted rest window back over midnight (a `00:00` sleep slot derived from
the schedule's own slots), so `get_activity_for_time` never falls back to
个人时间 before the first morning slot. D2 (travel-labelled activities
bypass the time-fit keyword gate) and D3 (no hour gate on movement /
venue-open) are retained as documented limitations.

Phase 1.9 extension (post-PILOT19, RESULT-3475): the D1 guard was widened —
a schedule whose existing `00:00` head slot is NON-sleep (e.g. 个人时间) is
now converted into sleep coverage (the head slot becomes sleep and the
unscheduled night slots before the first morning slot are dropped; times
still derive only from the schedule's own slots), fixing the pilot-proven
no-op. The move path (`gaworld/sim/_location.py` `move_agent`) now logs a
structured `destination_purpose` in its travel dicts, derived
deterministically from the movement's scheduled activity (None when
undeterminable — never guessed) and persisted on `locations` so in-transit
snapshots carry the departure purpose. `classify_activity_record` promotes
in-transit travel to commuting ONLY for work-related purposes (`paid_work`);
unknown or non-work purposes stay `travel`.

### 2.2 Valid dimensions

- **Paid-work vs leisure/social time-use error** — the two-dimensional frontier
  (**Verified (computed)**, Phase 0.8 independence analysis: under independent
  behaviour r = −0.186, under perfectly coupled behaviour r = +0.833, so
  co-movement measures behavioural coupling; the 1440-min budget adds a small
  spurious negative correlation — never rank configs on one dimension alone).
- **Wealth Gini** — **static day-0 level check only**. ΔGini as a frontier
  coordinate is **REJECTED** (RESULT-3192): no published short-horizon ΔGini
  statistic exists; deterministic worst-case |ΔGini| ≈ 0.032/30 days is below
  1 SD of the seed-noise floor; matched-seed pinning alters the input
  distribution. Report once per run against the labelled savings-distribution
  range.

### 2.3 Rejected measures (with reasons)

- **Engel coefficient + savings rate** — input echoes of
  `spending.engel_curve` (`METRIC_PROVENANCE` grade "c", 回显): one dimension
  counted twice; measures the configured lookup, not modelled consumption.
- **Stateless D2/D3 as frontier coordinates** (classic paradigms, replication
  fidelity) — demoted to add-on interviews / Phase-0 unlockables; LLM residents
  likely know the famous classic-paradigm results.
- Commute minutes — reserve manipulation-check report-out only (inferred from
  episode `travel`, likely Θ-insensitive).
- Track B stylized facts, Track C causal checks, DEMAND phenomena, Track D —
  no observed magnitudes / no real-world causal target / not quantitative.

## 3. Validity requirements

- **Admissible Θ** (`benchmark/mcf/time_use.py::ADMISSIBLE_STYLE_FIT`):
  `personality.style_fit_amplitude` ∈ **[0.10, 0.40]** (validated by
  `validate_style_fit`; **0.9 is a stress value**, `STRESS_STYLE_FIT = (0.9,)`,
  outside the band by construction — it must never silently re-enter the band).
  Other pilot-eligible parameters (Feasibility §4.4): contagion_weight
  [0.02, 0.12], spending.daily_variance [0.10, 0.40], temperature [0.0, 0.7].
  Excluded: `initial_savings_months_min/max` and `engel_curve` rows
  (circularity/tautological bound-pressing). Ranges must be justified
  independently of the evaluation metrics.
- **Matched seeds:** per-agent trait streams (`Random(seed·1000003 + agent_id)`)
  are config-independent; the U(1,6) initial-draw Gini component cancels at
  t=0 under matched seeds.
- **Survey-basis rules:** paid_work on a participants basis (employed
  agent-days only; workday-only means stamped `basis="exploratory"` — 2024
  targets are weekly averages); all other categories on an all-persons basis;
  study excluded by default; 2018 robustness needs the explicit `day_types`
  flag; population caveat (survey covers ages 6+, cohort is adults).
- **Frontier-valid gates:** `MIN_COMPARISON_DAYS = 7` AND
  `MIN_COMPARISON_POPULATION = 20` agents; otherwise `frontier_valid=False`
  (exploratory). Survey-grade precision needs ~80 agents × 7 days
  (~553 agent-days for ±10-min precision at per-agent-day SD ~120 min).

## 4. Architecture comparison: `personality.channels.rules` on/off

The recommended architecture pair (**Verified (repo)**, Phase 0.6 §6):

- M1 = `personality.channels.rules = true` (default), M0 = `false`.
- **Clean matched-seed ablation:** 12 mechanism sites switch off; trait values
  are identical in both arms (per-agent seeding independent of the toggle);
  the toggle is consumed solely via `traits_of(agent, "rules")`; no config key
  becomes required; RNG divergence ruled out (isolated per-call `residual()`
  stream).
- The five action-selection sites cut across both time-use dimensions — this
  is why one pair serves as the contrast for both coordinates simultaneously.
- Comparison statistic: A(M1|M0) (newly-attainable set) plus matched-config
  paired stats (medians, improvement counts, dz), reusing
  `bootstrap_ci` / `sign_flip_p` from `gaworld/parallel/causal.py`.
- Interpretation rule (**Verified (paper)**): asymmetric — extension beyond a
  densely sampled frontier is strong evidence of capacity gain; an apparent
  shortfall may reflect sparse exploration. Always report configs × seeds.

## 5. Experiment ladder: canary → Stage 1 → 2 → 3

### Canary (Phase 1.0 §2 — specified, NOT yet executed; must run first)

- Scale: 1 config × 1 seed × 2 agents × half-day (72 ticks), normal mode,
  default serial config, **enforcement pre-call**.
- Hard kill limits: **400 LLM calls · 250k tokens · $1.00 · 30 min wall-clock ·
  server/DNS errors on >5% of calls** (no retry loops — Phase 0.6/0.8 lesson).
- Purpose: replace the assumed 0.5–3 calls/agent-tick and 2.5–3.5 s/call with
  measured values. Corrected serial-cost context for Stage 1:
  **7–58 h** (10k×2.5s – 60k×3.5s); the old "14–20 h with 8 workers" figure was
  wrong (concurrency disabled by default; only routine+digest parallelize,
  < 1% of calls). Stage 1 wall-clock = (measured rate) × 20,736 × (measured
  latency), planned only after the canary passes.
- Launch (opt-in, implemented in `benchmark/mcf/launch.py`; default runs are
  unaffected — nothing in production imports the canary):

  ```sh
  # zero-inference plumbing check (no provider dispatch):
  GAWORLD_CANARY=1 python -m benchmark.mcf.launch --dry-run
  # live canary (requires verified pricing in CONFIG):
  GAWORLD_CANARY=1 python -m benchmark.mcf.launch --authorize
  ```

  with `CONFIG["canary"] = {"enabled": true, "dry_run": false,
  "est_tokens_in": …, "est_tokens_out": …, "pricing": {"<model>":
  {"input": $/1M, "output": $/1M}}}` in `dashboard_config.json`. The wrapper
  rebinds `gaworld.llm.providers.call_llm` (the one dispatch point every
  kernel inference path uses) before the entry module is imported, and every
  call is enforced pre-call against the 400-call / 250k-token / $1 / 30-min /
  >5%-error caps; termination writes `output/canary/canary_manifest_*.json`
  with the stop reason and partial usage. NOT yet executed.

#### Token accounting: measured vs estimated (Phase 1.5 Track 2)

LM Studio's OpenAI-compatible `/chat/completions` responses DO carry a
`usage` object (`prompt_tokens` / `completion_tokens` / `total_tokens`) in
blocking mode; streaming responses do not unless `stream_options:
{"include_usage": true}` is requested (NOT enabled — it would change
production behaviour). GAWorld's provider wrapper returns only the text, so
the raw usage was previously DISCARDED: every token figure in the canary14
manifests (`output/canary/`, `/home/justin/Documents/gaworld-agent-work/
canary14/`) is the conservative 400-token/call ESTIMATE, never observed —
40 calls × 500? or 280 × 400 = 112,000 etc. are all estimate-derived.

As of Phase 1.5 Track 2 the provider stashes the blocking response's
`usage` (`gaworld.llm.providers.pop_last_token_usage()`, thread-local) and
the canary wrapper records OBSERVED counts for the ledger when present,
falling back to the estimate (marked estimated) when the report is missing
or malformed — never presenting an estimate as observed. Pre-call
reservations stay estimate-based. The post-run manifest gains
`token_accounting: {measured_calls, estimated_calls, tokens_measured,
tokens_estimated}`. Tests: `tests/test_mcf_token_usage.py` (12 cases,
mocked).

#### Stage 1 execution readiness (Phase 1.5 Track 2)

- **Budget enforcement** — solid: opt-in wrapper, pre-call check +
  in-flight reservation, post-kill freeze + manifest, `restore()` always
  rebinds the original provider.
- **Resume** — **NOT reliable for mid-day kills; recommend
  restart-per-run for Stage 1.** Checkpoints are day-boundary only
  (`docs/CHECKPOINTS.md`): a kill between day boundaries leaves the
  partial day's records already appended to the append-only recorders
  (`gaworld/kernel/recorder.py`, `records/agent.step.jsonl`, timeline,
  memory episodes — flushed per record, never rolled back; the checkpoint
  explicitly excludes outputs). Restoring from the previous day's
  checkpoint then RERUNS that partial day → duplicate activity records and
  doubled timeline/episode entries for that day (sim-clock timestamps stay
  consistent; duplication is the failure mode, not corruption). A kill
  exactly at a day boundary is safe (no duplicates). Stage 1 canaries are
  3 sim days ≈ 840 calls ≈ 45 min measured — cheaper to restart than to
  audit de-duplication.
- **Recommended Stage 1 execution caps** (from measured canary14 local
  performance: avg 3.17 s/call, 0.97 calls/agent-tick, 280 calls/day-sim
  at 2 agents × 144 ticks): Stage 1's 20,736 agent-ticks × 0.97 ≈ 20.1k
  calls ≈ 17.7 h serial wall-clock (the doc's earlier 10k–60k range
  brackets this); keep the existing 400-call/30-min canary caps for smoke
  legs, and for full Stage 1 legs set `max_calls` ≈ 21,000, `max_tokens`
  ≈ 11M (21k × 500 est), `max_elapsed_seconds` ≈ 80,000 (~22 h, 1.25× the
  measured projection — no retry loops, so latency is the only variance),
  `max_error_rate` 5% unchanged.

### Stage 1 — micro-run (cost + coverage) — spec LOCKED (Phase 0.9 §6, GO)

12 employed agents (of the 80-agent 绍兴柯桥 cohort: 33 employed + 47 NILF),
2 configs (`style_fit_amplitude` — within the admissible band, e.g. 0.1 vs 0.4),
2 matched seeds, 3 sim days (day 1 warm-up; scorer reads days 2–3),
10-min tick grid (`time_grid_snap: true` → 144 ticks/day), 20,736 agent-ticks,
expected 10k–60k calls. Outputs: `output/memory/agent_<id>_episodes.jsonl`
+ per-run `run_manifest.llm` (the authoritative cost). Scoring:
`score_directory(..., tick_minutes=10, day_types=...)`, coverage + uncertainty
warnings reported alongside cost. Stop conditions: kill thresholds above;
coverage < 0.85 → fix mapper before Stage 2. Output is `frontier_valid=False`
by construction (12 < 20 agents) — cost/coverage measurement, not a survey check.

### Stage 2 — sensitivity probe

2 configs × 3 seeds × ≥5 days × reduced cohort. Stop conditions: coverage
< 0.85; seed SD > config range on **both** dimensions (no signal); cost/day
> 2× the Stage 1 baseline → pause and renegotiate scope. Noise-domination
check runs here on real data.

### Stage 3 — full pilot

2-D frontier over the `channels.rules` on/off pair, matched seeds, Θ with
`style_fit_amplitude` primary; static day-0 Gini reported once as a labelled
sanity check. Proceed criteria: Stage 2 shows ≥ 1 dimension with config range
> 2× seed SD AND coverage ≥ 0.85 — otherwise no Stage 3. Falsifier fallback:
a plainly-labelled single-dimension dose sweep + architecture contrast, never a
manufactured two-axis frontier.

### Standing success / abandon criteria

- Success: frontier survives seed-doubling on nondominated points; ≥ 1
  interpretable bound-pressing or flat-dimension diagnostic.
- Abandon: seed SD / observed config range > 1 on every dimension even at
  n = 5 seeds — no configuration contrast resolvable anywhere.

## 6. Key evidence & provenance

External archive (authoritative preserved copies):
**`/home/justin/gaworld-agent-work/mcf-archive/`** (workspace created by the
Phase 1.1 cleanup; the originally specified path
`/home/justin/Documents/gaworld-agent-work/` was unavailable — read-only mount —
see the workspace's `cleanup-inventory/change-log.md`).

| Evidence | Value | Source |
|---|---|---|
| D1 seed-noise floor | SD = **0.0385** (mean Gini 0.4421, range 0.349–0.545; 200-draw Monte Carlo, seed 20261105) | PHASE_0_6 §2.1 (archived) |
| ΔGini adversarial bound | |ΔGini| ≤ ≈ 0.032 / 30 days < 1 SD noise floor; 2-SD threshold 0.077 | PHASE_0_7 (archived) |
| Dimension independence | paid-work vs leisure/social: r = −0.186 (independent) / +0.833 (coupled) | PHASE_0_8 §1 (archived) |
| Sleep-tail 1,380 audit | verdict OVER-ATTRIBUTION; corrected expectation 500 | PHASE_1_0 §1.1 (tracked) |
| Stage 1 GO verdict | Phase 0.9 §9 (52/52 tests GREEN; no new full-suite failures) | PHASE_0_9 (archived) |
| 2018 time-use survey | 全国时间利用调查公报, NBS 2019-01-25 — https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900224.html | RESULT-3188 |
| 2024 time-use survey | 公报第二号 https://www.stats.gov.cn/sj/zxfb/202410/t20241031_1957216.html · 第三号 …/t20241031_1957215.html | RESULT-3188 |

Supersession chain: FEASIBILITY (living) → PHASE_0_5 → 0_6 → 0_7 → 0_8 →
**0_9** (end of offline chain, Stage 1 GO) → **1_0** (correction record + canary
spec + revised Stage 1) → **this doc** (consolidation). Each phase doc names its
predecessor; 0.5–0.8 surviving content is restated in 0.9/1.0 above.

## 7. Outstanding questions / decisions

1. **Canary execution** — still spec-only; no Stage 1 commitment before it
   passes (its measured latency + calls/tick supersede every assumed number).
2. **Sleep-tail bands on real logs** — the ≤09:00 / 10:00–18:00 / ≥18:00 rule is
   calibrated on audited adversarial scenarios; spot-check canary/Stage 1 logs
   for mass `unknown` inflation.
3. **Survey-grade cohort** — ~80 agents × 7 days (~553 agent-days) for any
   frontier claim; currently no run meets it. Widening the cohort multiplies
   serial wall-clock linearly; in-tick-loop concurrency is the only real lever
   and does not exist yet.
4. **Track E cost metric** — define per-config cost accounting (calls × price
   vs wall-clock vs tokens) before adding a cost axis; fold
   `gaworld/llm/stats.py::GLOBAL_STATS` into run manifests.
5. **D1 target re-basing** — income-Gini ≈ 0.45–0.5 (国家统计局 工资性收入)
   recommended over the CHFS wealth-Gini 0.6–0.75 (population mismatch).
6. **Held-out instance** — plan a hold-out event/scenario before any
   frontier-guided iteration (overfitting control, paper §7).

---

*Consolidated by the Phase 1.1 cleanup task (TASK-3232) from the six phase docs,
REPO_HYGIENE_AUDIT.md, and research RESULTs 3170/3171/3176/3187/3188/3192/3201/
3206/3217. No code, tests, or default-run behaviour were changed by this doc.*
