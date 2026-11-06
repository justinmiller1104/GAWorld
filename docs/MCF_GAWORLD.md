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
