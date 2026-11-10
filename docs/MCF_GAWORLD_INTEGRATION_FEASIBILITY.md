# MCF × GAWorld Integration Feasibility

**Subject:** Whether and how the *Model Capacity Frontier* framework (Zhao, Li & Shirado, arXiv:2610.24168v2) can be integrated into GAWorld's experiment and evaluation infrastructure.

**Scope:** documentation only. All GAWorld claims below are labeled **Verified (repo)** (original analysis, HEAD `9697471`) or **Verified (repo, HEAD 80f8c1f)** (2026-11-05 data audit, RESULT-3176); all paper claims are labeled **Verified (paper)**; design proposals are labeled **Proposed**; unstated but necessary premises are labeled **Assumption**.

---

## 1. Executive Summary

The **Model Capacity Frontier (MCF)** is — **Verified (paper)** — the nondominated set of expected empirical-error vectors attainable by a model *architecture* across its admissible configuration space:

> f(θ) = E_ξ[e(θ,ξ)] over admissible θ ∈ Θ_M; F_M = {f(θ) | θ ∈ Θ_M, ∄ θ′ ∈ Θ_M : f(θ′) ≺ f(θ)} (paper Eq. 1)

where the error vector e ∈ R^k≥0 scores one run against k real-world empirical targets and dominance is componentwise. MCF turns architecture development into frontier estimation (sampling Θ_M, extracting the Pareto set) and architecture comparison into newly-attainable-set analysis, A(M1|M0) (paper Eq. 2).

**Bottom line (Proposed):**

- **Conditional GO** on a dedicated MCF module (Option B, §9) for the **full social-simulation layer only**.
- **NO-GO for ARENA** (`gaworld/apps/arena_api.py`): it is a 10-task single-agent benchmark with no multi-dimensional error vector over real observations and no meaningful configuration space — **Verified (repo)** — so MCF has nothing to push a frontier against.
- **Partial GO without dense microdata** — **Verified (repo, HEAD 80f8c1f)** (RESULT-3176 data audit): a minimal pilot can run on **2–3 genuinely independent error dimensions** that already exist in the repo — labour-force wealth Gini (re-based to an income-Gini target is recommended), classic-paradigm human effects (six published references already hard-coded in `gaworld/experiments/classics/paradigms.py`), and replication-tab fidelity against **one** uploaded published survey. The original Phase-0 hard gate (dense individual-level ground truth) is **relaxed**: dense microdata would improve the frontier's resolution but is no longer a precondition. Two honest caveats remain: Track A's headline anchors (engel, savings) are **input echoes** of `spending.engel_curve` (`METRIC_PROVENANCE` grade "c", 回显) and must not appear in the error vector, and D3 still needs one uploaded survey (hours of work, zero simulation — §5, §11 Phase 0).

---

## 2. Existing GAWorld Architecture

All paths and symbols below are **Verified (repo)** against the working tree on 2026-11-05 (HEAD `9697471`, main).

- **Orchestrator & kernel.** `generative_city_sim.py::run_simulation` (line 2332, line-budget-capped by `tests/test_line_budget.json` / `tests/test_line_budget.py`) drives the day loop over a microkernel (`gaworld/kernel/`: clock/bus/registry/controller/recorder, `build_kernel()`), with ~9 built-in plugins assembled by `gaworld/plugins/builtin_plugins.py`. LLM calls route through `gaworld/llm/providers.py::call_llm` (line 1316) with an `LLMRouter` across OpenAI/Anthropic/Ollama providers and cross-provider fallback.
- **Parallel / counterfactual machinery.** `gaworld/parallel/runner.py` (seeded parallel worlds), `gaworld/parallel/spec.py::normalize_experiment` (spec validation, `RESERVED_CONFIG_KEYS`, max 8 worlds), `gaworld/parallel/sweep.py` (**1-D only**, `MAX_WORLDS = 8`), `gaworld/parallel/causal.py::estimate_effects` (deterministic: `bootstrap_ci`, `sign_flip_p`, `bh_qvalues`, `dose_response`, `heterogeneity`, `pool_replicates`).
- **Research workbench.** `gaworld/research/protocol.py::Protocol` (compiled study definition), `gaworld/research/study.py` / `workbench.py` / `batch_runner.py` (orchestration), `gaworld/research/evaluate.py::evaluate` (paired estimates gated on per-seed paired tests), `gaworld/research/replication.py` (paper-extracted human results → fidelity scoring), `gaworld/research/report.py` (study reports).
- **GAWorld-Bench.** `benchmark/gaworld_bench.py`, Tracks A–E: Track A `track_a_macro_fit()` (macro anchors: Engel coefficient, savings rate, wealth Gini; `gini()` at line 187), Track B stylized facts from playground games, Track C causal checks (known-sign/placebo/determinism, multiseed with checkpointing), Track D who's-human judge consistency, **Track E (cost/reproducibility) stubbed**. Auto-reports to `benchmark/results/report.md`.
- **ARENA.** `gaworld/apps/arena_api.py::TASK_BANK` — 10 hand-written single-agent tasks, LLM-judged; separate from the main simulation eval.
- **Provenance & comparability.** `gaworld/core/run_manifest.py::start_manifest` (per-run JSON/HTML manifest with `gaworld/llm/stats.py::GLOBAL_STATS`) and `gaworld/core/comparability.py` epochs (cross-epoch pooling refused).
- **Dashboard.** Route table `gaworld/apps/routes.py`, OpenAPI document `gaworld/apps/openapi.py` (enforced by `tests/test_openapi.py`), Python client `gaworld/client.py`, ~60 route modules under `gaworld/apps/`.
- **Config layering.** `gaworld/settings/` (CONFIG defaults + overrides), `dashboard_config.json`, CLI args, and `GAWORLD_CONFIG_OVERRIDES` env injection for subprocess runs. The numeric sweep surface comes from `gaworld/parallel/sweep.py::tunables` / `current_value` (numeric config paths only; integer settings only get integers). There is **no multi-parameter grid/random/Bayesian search** — the sweep is strictly 1-D with ≤ 8 worlds.

---

## 3. Proposed MCF Design

**Proposed:** a new `benchmark/mcf/` module, additive to the existing tree (so `tests/test_line_budget.py` stays untouched and no new `Epoch` is needed — MCF would be opt-in and must not change default-run output). Components:

1. **(a) Admissible config-space spec** — a declarative Θ_M with literature-justified bounds per parameter (**Verified (paper):** admissibility "ranges justified by the literature, theory, or measurements independent of the evaluation metrics" is what makes bound-pressing diagnostics meaningful). Reuse the numeric-path enumeration ideas of `sweep.py::tunables` but with explicit per-parameter ranges and multi-dimensional combinations. **Assumption:** prompts sit inside the architecture M (fixed within a version), consistent with the paper's default boundary; this must be declared per experiment.
2. **(b) Multi-config seeded runner** — sample Θ_M (Sobol low-discrepancy or uniform random; **Verified (paper):** MCF is agnostic to the sampler) and dispatch each configuration × seed through the existing parallel-world runner (`gaworld/parallel/runner.py`, `gaworld/research/batch_runner.py`) with `GAWORLD_CONFIG_OVERRIDES` env injection for subprocess runs. Either extend `sweep.py` beyond 1-D or add a new sampler; a new sampler is cleaner since `MAX_WORLDS = 8` and the reserved-key guards are tuned for dose sweeps.
3. **(c) Error-vector scorer** — assemble one error vector per (config, seed) by reusing: the wealth-Gini scorer (`benchmark/gaworld_bench.py::track_a_macro_fit::gini` — **Gini only**; the engel/savings anchors are input echoes and excluded per the data audit, §4.3), the replication-tab fidelity machinery (`gaworld/research/replication.py` + `gaworld/research/evaluate.py::evaluate`), the classic-paradigm scoring (`gaworld/experiments/classics/`), and the deterministic causal estimators (`gaworld/parallel/causal.py::estimate_effects`) where a target is a contrast rather than a level. Aggregate across seeds (mean) and, where multiple empirical instances exist, adopt the paper's worse-of-two-instances rule (**Verified (paper):** f_i = max over instances, to force reproduction of the recurring pattern rather than one realization's noise).
4. **(d) Pareto frontier extraction** — new but small: a pure componentwise-dominance filter over seed-averaged error vectors. No surrogate models, no smoothing (**Verified (paper):** the estimated frontier is just the nondominated subset of the sampled attainable set).
5. **(e) Architecture comparison** — compute A(M1|M0), the newly attainable set (points in Ŷ_M1 not weakly dominated by any Ŷ_M0 point), plus matched-configuration paired statistics for the same θ across architectures (medians, improvement counts, dz effect sizes — **Verified (paper)** protocol). Reuse `bootstrap_ci` and `sign_flip_p` from `gaworld/parallel/causal.py`.
6. **(f) Report generation** — frontier plots/tables and comparison verdicts via `gaworld/research/report.py`, following the paper's asymmetric interpretation rule for sparse frontiers (§7).

**Where MCF applies — and where it does not (Verified (repo) facts, Proposed verdicts):**

- **ARENA (single-agent): MCF not appropriate.** `TASK_BANK` is 10 hand-written tasks judged by an LLM; there is no error vector over real observations, no admissible configuration space worth sampling, and no meaningful frontier.
- **Full social simulation: applicable, and a minimal pilot is feasible now.** The architecture (kernel + plugins + config layering) supports multi-config runs, and the data audit (**Verified (repo, HEAD 80f8c1f)**) identified 2–3 genuinely independent error dimensions already in the repo (§4) — dense trajectory-level ground truth is no longer a blocker for a pilot, though it would still improve frontier resolution (§7, §11 Phase 0).

---

## 4. Minimal MCF Pilot: Proposed Experimental Design

**Proposed**, with every factual input **Verified (repo, HEAD 80f8c1f)** (RESULT-3176 data audit).

### 4.1 Scenario and cost

- **Cohort:** the 80-agent 绍兴柯桥 cohort (`data/cities/绍兴柯桥/agents.csv`), run in fast-forward mode (**Verified (repo):** 1 LLM call / agent / day per `docs/TUTORIAL.md`).
- **Smoke leg:** 16 configs × 1 seed × 10 days ≈ **12,800 calls ≈ 3.5 h**. Run this first; it must also verify that the economy plugin still ticks daily under `--fast-forward` (**unverified** — no FF run outputs exist on disk; if it does not, D1 needs its own full-fidelity leg).
- **Full leg:** 16 configs × 2 seeds × 30 FF days × 80 agents ≈ **76,800 calls ≈ 20 h wall-clock** at 8 workers on a cheap domestic model; budget accordingly on Minimax-M2.7-class reasoning models. D2/D3 are stateless resident interviews and survive the FF proxy intact; only D1 needs full fidelity (or the FF smoke check above).
- **Cost-axis caveat:** archived run manifests show `llm: {}` — per-run cost is currently **unmeasurable** until `gaworld/llm/stats.py::GLOBAL_STATS` totals are folded into run manifests. Until then the frontier's cost axis is an estimate, not a measurement.

### 4.2 Error vector — three genuinely independent dimensions

Each dimension scores against real published human data from sources independent of the repo; aggregation for D2/D3 uses the paper's worse-of-instances rule.

- **D1 — labour-force wealth Gini level.** Scored by the existing `benchmark/gaworld_bench.py::gini` + `track_a_macro_fit`. Target: the CHFS/CS household-wealth Gini range 0.6–0.75 measured at range midpoint — with an honest population-mismatch caveat — or, **recommended**, re-based to an income-Gini ≈ **0.45–0.5** (国家统计局 工资性收入), which a labour-force cohort can actually represent. Emergent distribution statistic, sensitive to `investment.*`, `inheritance_*`, income bands. Measured as the worse of 2 snapshots.
- **D2 — classic-paradigm effects.** Six published human references already hard-coded in `gaworld/experiments/classics/paradigms.py` as `human={...}`: framing 72% (N=152) vs 22% (N=155); anchoring index ≈ 0.5; dictator ≈ 28% / ultimatum ≈ 40%; responder rejections; trust ≈ 50% (d≈0.3); public-goods first round 40–60%. Error = normalized |agent effect − human effect| / human range, aggregated as the **worse of 2–3 paradigms** (six *different* psychological mechanisms — prospect theory, anchoring, fairness, reciprocity, parochialism, cooperation). Caveats: LLM residents likely know these famous results (the code itself warns anchoring will bias small); the interview is a cheap add-on to any config since it does not read run output.
- **D3 — replication fidelity vs one uploaded published survey.** The existing `gaworld/research/replication.py` machinery (per-cell error, CI containment, effect agreement, fidelity score) against **one** uploaded survey table with reported human results — the smallest missing dataset; the machinery exists but `output/research/` does not: no survey has ever been uploaded. Upload happens through the existing replication tab: hours of work, zero simulation.

### 4.3 Explicitly rejected measures (with reasons)

**Verified (repo, HEAD 80f8c1f):**

- **Engel coefficient + savings rate** — both are **input echoes** of the same `spending.engel_curve` table (`METRIC_PROVENANCE` grade "c", 回显): one dimension counted twice, and it measures the configured lookup, not modelled consumption. MCF-independent fails and bound-pressing is trivial.
- **Commute minutes / transit share** — `city_wide` scope, excluded from scoring by `SCORED_SCOPES`; `state/commute_summary.csv` is not in default output.
- **Track B stylized facts** — direction-only conventions (`PANIC_EXTREME_MAX = 0.2`, `HELP_RATE_MIN = 0.5`), no observed magnitudes, and no archived games exist.
- **Track C causal checks** — no real-world causal target ("not an independent test against real-world causality").
- **DEMAND phenomena** — targets are simulation-internal properties (no confounding, transition-width ordering), not quantitative human effect sizes.
- **Track D who's-human** — no archived rooms (`output/games/` does not exist).

### 4.4 Θ — four admissible parameters

Ranges justified **independently of the evaluation metrics** (**Verified (paper):** this independence is what makes bound-pressing diagnostics meaningful):

| Parameter | Range | Independent justification |
|---|---|---|
| `personality.style_fit_amplitude` | [0.10, 0.40] | Acceptance band pre-registered in `gaworld/settings/personality.py` from the literature argument that personality must not explain >~15% of behavioural variance; 0.40/0.60 fail `scripts/big5_effect_ceiling.py` windows. Independent of Gini and of survey answers. |
| `personality.emotion_baseline.contagion_weight` | [0.02, 0.12] (default 0.06) | Social-contagion effect sizes in the literature are small; the parameter exists to fix a documented defect (`_cognition.py` hard-coded 0.1). Affects social dynamics, not consumption tables. |
| `economy.spending.daily_variance` | [0.10, 0.40] (default 0.25) | Plausible consumption-noise range from household-budget survey dispersion; independent of D2/D3 by construction, and it is *not* the `engel_curve` itself. |
| LLM sampling temperature (provider config) | [0.0, 0.7] | Standard LLM diversity range; affects every dimension, making it the right "architecture-noise" probe. |

**Explicitly excluded (declared in the Θ spec):** `economy.initial_savings_months_min/max` — varying them moves wealth Gini almost mechanically at snapshot time, which would let MCF "attain fit through implausible settings" (the paper's own admissibility rationale) and turn bound-pressing into a tautology. Same for `spending.engel_curve` rows (circularity with D1 and admissibility).

### 4.5 Stopping rules (practical)

Frame as power, not a rigid threshold. After a 2-config × 3-seed pilot, estimate the per-dimension seed SD σ; the minimal detectable config effect at n seeds is d ≈ (z_{α/2}+z_β)·σ·√(2/n):

- **Per dimension:** if the full Θ-range effect observed in the smoke is < 2×σ, the dimension is noise-bound — raise seeds to n = 2(1.96+0.84)²σ²/d² (usually ≤ 5 for d≈σ) **or drop the dimension** from the vector. Do not abandon the pilot for one flat dimension — read a flat dimension as an *architecture diagnostic* (the paper's rule), not a failure.
- **Frontier stability:** seed-doubling test — re-run all nondominated points at 2× seeds; if points reshuffle, report configs × seeds and interpret asymmetrically; do not conclude.
- **Uncertainty on frontier points:** bootstrap CIs via the existing `gaworld/parallel/causal.py::bootstrap_ci` per coordinate; claim a dominance edge only when the CIs separate in the deciding coordinate.
- **Narrowing Θ:** if a parameter's dose–response (`dose_response` already fits sweeps) is monotone-into-a-bound with no interior optimum, freeze it at the bound-adjacent value and spend the budget on the remaining dimensions.
- **Abandon:** only when **max over dimensions of (seed SD / observed config range) > 1 even at n = 5 seeds** — i.e., no configuration contrast is resolvable anywhere. The old rigid rule ("abandon when seed variance > config variance") survives only in this refined, per-vector form.

### 4.6 Success / failure criteria

- **Success:** a 3-dimension frontier is produced end-to-end and survives seed-doubling on its nondominated points; at least one bound-pressing or flat-dimension diagnostic fires with an interpretable reading; the full-leg cost lands within ~20 h wall-clock as estimated.
- **Failure:** the abandon rule above fires; or the FF smoke shows the economy does not tick and a full-fidelity D1 leg pushes the budget beyond feasibility; or the uploaded survey (D3) cannot be obtained, collapsing the vector to 2 dimensions without an agreed fallback.

---

## 5. Reusable Components

**Verified (repo)** — drawn from the repository analysis (RESULT-3170):

| Need | Reusable | Path / symbol |
|---|---|---|
| Multi-config experiment running | 1-D sweep expansion with guards | `gaworld/parallel/sweep.py::tunables/current_value`, `MAX_WORLDS = 8` |
| | World-spec validation & config patching | `gaworld/parallel/spec.py::normalize_experiment`, `world_overrides` |
| | Seeded parallel-world runner | `gaworld/parallel/runner.py`; `gaworld/research/batch_runner.py` |
| | Env-var config injection for subprocess runs | `GAWORLD_CONFIG_OVERRIDES` |
| | Run provenance | `gaworld/core/run_manifest.py::start_manifest`, `gaworld/core/comparability.py` epochs |
| Evaluation against real data | Wealth-Gini scoring (the only non-echo Track A metric) | `benchmark/gaworld_bench.py::track_a_macro_fit`, `gini` — engel/savings anchors are 回显 echoes, excluded (§4.3) |
| | Paper-replication fidelity scoring | `gaworld/research/replication.py` + `gaworld/research/evaluate.py::evaluate` |
| Classic-paradigm human references | `gaworld/experiments/classics/paradigms.py` (`human={...}`), `classics/analysis.py` |
| Deterministic causal estimates | `gaworld/parallel/causal.py::estimate_effects`, `bootstrap_ci`, `sign_flip_p`, `bh_qvalues` |
| | Stylized-fact / judge tracks | `track_b_games`, `track_d_whois`, `benchmark/rubric/` |
| Capacity-frontier analysis | Partial — dose–response fit reads a sweep directly | `gaworld/parallel/causal.py::dose_response` |
| Reporting | Auto bench report + scorecard renderer | `benchmark/gaworld_bench.py` report writer, `benchmark/results/*.md` |
| | Dashboard API + OpenAPI client | `gaworld/apps/routes.py`, `gaworld/apps/openapi.py`, `gaworld/client.py` |
| | Study reports & interpret (1 LLM call) | `gaworld/research/report.py`, `gaworld/parallel/interpret.py` |

---

## 6. Required Engineering Changes

**Proposed**, with effort ratings:

| Change | Effort | Notes |
|---|---|---|
| Multi-parameter sampler (Sobol/random over Θ_M) | new, moderate | `sweep.py` is strictly 1-D with `MAX_WORLDS = 8`; a new sampler module under `benchmark/mcf/` avoids disturbing sweep guards. |
| Pareto/frontier extraction | new, easy | Pure dominance filter; no Pareto code exists anywhere in `gaworld/` today (**Verified (repo)**). |
| Error-vector assembly with epoch stamps | moderate | Combine Track A / replication / causal scorers into one vector per run; stamp every run via `gaworld/core/comparability.py` and record provenance via `start_manifest`. |
| Architecture comparison + matched pairing | moderate | A(M1\|M0) plus matched-config paired stats; reuse `bootstrap_ci`, `sign_flip_p`. |
| Per-config cost telemetry | easy | Read `gaworld/llm/stats.py::GLOBAL_STATS` per run (surfaced by the run manifest); Track E of the bench is stubbed, so this doubles as its cost feed. |
| Dashboard/OpenAPI surface | moderate | New routes must be documented in `gaworld/apps/openapi.py` — `tests/test_openapi.py` fails on an undocumented route; keep it green. One delegate line in `gaworld/apps/routes.py`, not a handler branch. |

**Gates to respect (Verified (repo)):** `tests/test_line_budget.py` (orchestrator line caps — untouched if MCF is additive), the ruff ratchet (`scripts/lint_ratchet.py` + `scripts/lint_baseline.json`), black (110), mypy strict on `gaworld/`. A new `Epoch` in `gaworld/core/comparability.py` is required **only if** default-run output changes — design MCF as opt-in so this is avoided.

---

## 7. Scientific Feasibility (critical)

This is the decisive section. **Verified (repo):** GAWorld's real-human targets are thin. Track A anchors are **1–3 scalars** (Engel coefficient, savings rate, wealth Gini); the replication flow scores against *paper-reported summaries*, not microdata. **Verified (paper):** the MCF demonstration's power came from dense video-derived multi-dimensional errors — three metrics (temporal occupancy, spatial Wasserstein, flow-direction cosine), each computed per year and aggregated as the worse of two years.

Consequences:

1. **Too few error dimensions today.** With 1–3 scalar targets, a "frontier" degenerates: dominance ordering over 2–3 scalars sampled by a 1-D sweep is not an informative capacity picture. A meaningful frontier needs **≥ 2–3 independent error dimensions** plus **bound-pressing diagnostics** (best configs pressing an empirically justified parameter bound; a failing dimension flat across Θ_M; required behavior absent from the action space — **Verified (paper):** "diagnostics rather than proofs").
2. **Dense microdata would still help — but the audit shows it is not a precondition.** The original read of this section was "without new data, MCF yields a frontier over weak targets." The data audit (**Verified (repo, HEAD 80f8c1f)**) revised that: 2–3 genuinely independent dimensions already exist (§4), enough for an informative minimal pilot. What dense microdata would still add is resolution (trajectory-level errors, more recurring instances for worse-of scoring) — it is a quality upgrade, not a blocker.
3. **Overfitting controls are available but must be designed in.** **Verified (paper):** iterative development on the same empirical data risks fitting one event's idiosyncrasies; mitigations are grounding refinements in observed behavior, holding Θ fixed across architectures, worse-of-two-instances scoring, and — decisively — a **held-out event**. GAWorld must plan hold-out events/scenarios before any frontier-guided iteration.
4. **Sparse-seed LLM frontiers are optimistic.** **Verified (paper):** sparse sampling makes the estimated frontier conservative, but stochastic evaluation makes it optimistic (nondominated points partly drew favorable noise; the bias grows as seed counts shrink), and "stochastic variation in the language model is not separated from configuration effects" at 1 seed. Interpret sparse frontiers **asymmetrically**: extension beyond a densely sampled frontier is strong evidence of capacity gain; an apparent shortfall may reflect sparse exploration. Always report configurations × seeds per frontier.
5. **Don't confuse data limits with architecture limits.** A frontier gap under sparse/dense-but-weak targets must not be read as an architectural shortcoming.
6. **When MCF adds little — and the honest challenge to this proposal.** Does the existing bench already capture what MCF would report? Partly: dose–response sweeps already exist for 1-D parameter probes, and with only scalar targets the current scorecard (`benchmark/gaworld_bench.py`) suffices. **MCF adds value only for trade-offs across ≥ 2 dimensions.** But the strongest counter-argument dissolves on inspection of the record (**Verified (repo, HEAD 80f8c1f)**): the only two real Track A runs are 2026-07-04 Track A 0.87 **PASS** on **n=5 snapshots** and 2026-10-02 Track A 0.0 **FAIL** on **n=1** (a 伦敦 bundle) — the existing evaluation has never been run at a sample size that supports any conclusion, so "the bench already tells us this" is not true today. MCF's seed/config discipline would be new information. Related honesty requirements:
   - **Sparse-data misleading frontiers:** always report configs × seeds and interpret asymmetrically (item 4); a sparse frontier can look better than reality because nondominated points partly drew favourable noise.
   - **Config meaningfulness:** the pilot's 4 parameters vary real mechanisms with literature-justified ranges (§4.4). The sweep surface exposes **512 numeric paths** (`sweep.py::tunables`), but most combinations are untested — a frontier over untested corners is a hypothesis, not a verdict.
   - **Two-architecture comparison:** defer past the pilot. Possible axes when it comes: an LLM provider swap, or a mechanism on/off toggle (e.g. contagion weight at 0 vs its range) — both are cheap and honest "architecture" boundaries.
   - **Simpler alternatives:** for a single question about one parameter, a 1-D dose–response sweep answers it; adopt MCF machinery only when the question is a trade-off across ≥ 2 error dimensions.

---

## 8. Risks and Rewards

**Proposed**, anchored on **Verified (paper)** cost/bias facts and **Verified (repo)** constraints:

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Compute cost of multi-config runs | High | High | Paper: M0 ran 512×3 but LLM architectures only 16×1 — "evaluation costs can differ by orders of magnitude across architectures." Adopt **matched budgets** per architecture; start with the smoke leg (§4.1); use `gaworld/llm/stats.py` telemetry to watch spend — noting the cost axis is unmeasurable until run manifests carry `llm` totals. |
| Sparse-sampling optimism (frontier looks better than reality) | High | Medium | ≥ 3 seeds where affordable; report configs × seeds; apply the paper's asymmetric interpretation rule; seed-doubling stability check before trusting a frontier. |
| Added complexity / maintenance | Medium | Medium | Keep MCF as an additive `benchmark/mcf/` module; reuse existing runners/scorers; no changes to the orchestrator or default pipeline. |
| Bench / test breakage | Low | Medium | Additive module; keep `tests/test_openapi.py`, `tests/test_line_budget.py`, ruff ratchet, mypy strict green; no new `Epoch` (opt-in only). |
| Data-availability blocker (no dense human ground truth) | ~~High~~ **Low (downgraded by audit)** | ~~High (go/no-go)~~ Medium | **Revised (Verified (repo, HEAD 80f8c1f)):** no longer a hard gate — the pilot runs on the 3 existing dimensions (§4). Phase 0 is now cheap and quick (§11): upload one survey table, optionally re-base the D1 target, verify economy under FF. Dense microdata remains a quality upgrade, not a precondition. |
| Data-vs-model confound (improvements reflect data quirks, not capacity) | Medium | High | Worse-of-instances scoring, held-out instance as the decisive test, refinements grounded in observation/theory. |

**Rewards:** a principled answer to "does this architecture change add capacity?" (nonempty A(M1|M0)); early detection of architectural limits via bound-pressing/flat-dimension diagnostics; exposure of compensating errors (a frontier that recedes after a well-motivated change is informative — **Verified (paper)**); a cost/quality frontier that finally fills the stubbed Track E role — once run manifests record LLM totals.

---

## 9. Options A / B / C

**Proposed**, effort estimates are rough engineering estimates (Assumption):

| Option | Description | Effort | Verdict |
|---|---|---|---|
| **A — post-hoc Pareto over existing results** | Add a dominance filter over already-archived run/sweep outputs. | ~3–5 person-days | **Low value:** only 1-D sweeps (≤ 8 worlds) and scalar-anchor runs exist, so there is no multi-dimensional attainable set to filter. Not recommended beyond a throwaway prototype. |
| **B — dedicated `benchmark/mcf/` module** | Sampler + error-vector scorer + frontier extraction + comparison + reports, wired to the existing parallel-world runners and scorers (§3). | ~16–22 person-days engineering (§10); pilot compute ~77k LLM calls (§4.1) | **Recommended** — conditional on the Phase 1 pilot gate, not on dense microdata (§11, §13). |
| **C — full Bench+Workbench integration with scheduler** | MCF as a first-class study mode in the workbench, with cost-aware run scheduling, dashboard views, and Track E rewrite. | ~2–3 months; high maintenance | **Premature** before B proves value. However, **design B's sampler abstraction so C can build on it** (sampler as a pluggable component from day one). |

---

## 10. Estimated Development Effort

**Proposed / Assumption** (person-days, Option B scope):

| Component | Effort |
|---|---|
| Admissible config-space spec format + validation | 2–3 d |
| Multi-parameter sampler (Sobol/random) + runner dispatch | 4–5 d |
| Error-vector scorer (Track A + replication + causal assembly, epoch-stamped) | 3–4 d |
| Pareto extraction + A(M1\|M0) + matched-config paired stats | 2–3 d |
| Report generation (frontier plots, comparison verdicts) | 2–3 d |
| Cost telemetry plumbing (`gaworld/llm/stats.py` → per-config records) | 1 d |
| Tests + lint/mypy/OpenAPI conformance | 2–3 d |
| Dashboard/OpenAPI surface (optional, Phase 3) | 3–5 d |
| **Engineering total (core, no dashboard)** | **~16–22 person-days** |

**LLM-run cost dominates over MCF code.** Each frontier point is a full simulation run; at paper-like matched budgets (16 configs × 1 seed per LLM architecture) the compute bill will exceed the engineering cost. Budget runs before budgeting code.

---

## 11. Recommended Roadmap

**Proposed**, revised per the data audit (**Verified (repo, HEAD 80f8c1f)**) — four phases, no dense-microdata hard gate:

- **Phase 0 — data feasibility (cheap, hours, zero simulation).** Upload **one** published survey/experiment table with reported human results through the existing replication tab (unlocks D3); optionally re-base the D1 target to an income-Gini ≈ 0.45–0.5 (国家统计局) to fix the population mismatch; run the FF smoke leg (§4.1) to verify the economy ticks under `--fast-forward`. **Success:** the survey spec normalizes and scores (`replication.py::normalize_spec`); the smoke produces state output. **Failure condition:** no suitable survey can be obtained → fall back to a 2-dimension vector (D1+D2) or NO-GO if D1 is also judged unusable.
- **Phase 1 — minimal research demonstration.** The pilot of §4: 16 configs × 2 seeds × 30 FF days × 80 agents (~77k calls, ~20 h at 8 workers) plus a full-fidelity leg for D1 if needed; 3-dimension error vector; pure dominance filter for the frontier; stopping rules of §4.5. **Success:** a frontier is produced end-to-end and survives seed-doubling on nondominated points; at least one bound-pressing or flat-dimension diagnostic is interpretable. **Failure condition:** the abandon rule of §4.5 fires (no resolvable config contrast at n = 5 seeds).
- **Phase 2 — dedicated `benchmark/mcf/` module.** Build the permanent module (sampler + scorer + frontier extraction + comparison, §3) **only if** the pilot shows added value over the existing bench. **Success:** tests/lint/mypy/OpenAPI green, opt-in, no new `Epoch`. **Failure condition:** pilot insights duplicate what a 1-D dose–response sweep already reports → stay with the existing bench.
- **Phase 3 — Bench/dashboard integration.** Routes via `gaworld/apps/openapi.py` + one delegate line in `gaworld/apps/routes.py`; frontier reports through `gaworld/research/report.py`; per-config cost telemetry (fold `GLOBAL_STATS` totals into run manifests) feeding Track E. **Success:** a cost/quality frontier with measured (not estimated) cost axis.

**Standing success criteria:** frontier stable under seed doubling; bound-pressing or plateau diagnostics fire on at least one known-tight dimension; a matched-config comparison detects a known improvement in the expected dimension.

**Standing abandon conditions:** the §4.5 abandon rule (seed SD / config range > 1 on every dimension even at n = 5); or insights duplicate what the existing bench scorecard already reports.

---

## 12. Open Questions and Blockers

1. **Which survey unlocks D3?** One published survey or experiment table with reported human results and per-item statistics, suitable for `gaworld/research/replication.py`'s spec format. *(Smallest missing dataset; Phase 0, hours of work.)* Dense trajectory/microdata would further improve resolution but is no longer a blocker.
2. **GAWorld's architecture/configuration boundary.** Are prompts inside architecture M (fixed per version) or configurations θ? **Verified (paper):** the boundary is a modeling choice every MCF conclusion holds relative to; treating prompt variants as configurations would enlarge Θ_M and could widen frontiers. GAWorld must declare and freeze this per experiment.
3. **Track E cost metric definition.** The bench's cost/reproducibility track is stubbed (**Verified (repo)**); per-config cost accounting (LLM calls × price vs wall-clock vs tokens) needs a definition before the frontier can include a cost axis.
4. **Per-config run budget.** Sized for the pilot (§4.1): 16 configs × 2 seeds as the full leg, smoke-tested at 16 × 1 × 10 days first. For multi-architecture comparisons, keep budgets matched across architectures (paper matched 16 configs × 1 seed for LLM architectures against a 512×3 conventional frontier).

---

## 13. Final Go / No-Go

- **Conditional GO — revised per the data audit (Verified (repo, HEAD 80f8c1f)).** The path is now: **Phase 0** (cheap, hours — upload one survey, optionally re-base D1, FF smoke check) → **Phase 1 minimal pilot** (~1 week of work + ~20 h compute, §4) → **gate** → Phase 2 module if the pilot earns it. **Dense individual-level data is no longer a hard prerequisite** — the pilot runs on the 3 genuinely independent dimensions of §4.2. Honest notes retained: D3 still needs **one uploaded survey** (without it the vector is 2-dimensional), and Track A's headline anchors (engel, savings) are unusable as targets — they are input echoes of `spending.engel_curve`.
- **NO-GO for ARENA** (`gaworld/apps/arena_api.py`): single-agent, 10 hand-written tasks, no multi-dimensional error vector over real observations, no meaningful admissible configuration space — MCF is structurally inapplicable there.
- Build B so its sampler abstraction can later graduate into Option C, but do not build C now.

---

*Sources: RESULT-3170 (verified GAWorld repo analysis, HEAD `9697471`, 2026-11-05); RESULT-3171 / `paper-zhao-2026-mcf.md` (verified summary of Zhao, Li & Shirado, "Model Capacity Frontier: A Development Framework for LLM-Supported Social Simulation," arXiv:2610.24168v2); RESULT-3176 (verified data audit + minimal pilot design, HEAD `80f8c1f`, 2026-11-05). No code was changed in producing this document.*
