# MCF × GAWorld Integration Feasibility

**Subject:** Whether and how the *Model Capacity Frontier* framework (Zhao, Li & Shirado, arXiv:2610.24168v2) can be integrated into GAWorld's experiment and evaluation infrastructure.

**Scope:** documentation only. All GAWorld claims below are labeled **Verified (repo)**; all paper claims are labeled **Verified (paper)**; design proposals are labeled **Proposed**; unstated but necessary premises are labeled **Assumption**.

---

## 1. Executive Summary

The **Model Capacity Frontier (MCF)** is — **Verified (paper)** — the nondominated set of expected empirical-error vectors attainable by a model *architecture* across its admissible configuration space:

> f(θ) = E_ξ[e(θ,ξ)] over admissible θ ∈ Θ_M; F_M = {f(θ) | θ ∈ Θ_M, ∄ θ′ ∈ Θ_M : f(θ′) ≺ f(θ)} (paper Eq. 1)

where the error vector e ∈ R^k≥0 scores one run against k real-world empirical targets and dominance is componentwise. MCF turns architecture development into frontier estimation (sampling Θ_M, extracting the Pareto set) and architecture comparison into newly-attainable-set analysis, A(M1|M0) (paper Eq. 2).

**Bottom line (Proposed):**

- **Conditional GO** on a dedicated MCF module (Option B, §8) for the **full social-simulation layer only**.
- **NO-GO for ARENA** (`gaworld/apps/arena_api.py`): it is a 10-task single-agent benchmark with no multi-dimensional error vector over real observations and no meaningful configuration space — **Verified (repo)** — so MCF has nothing to push a frontier against.
- The GO is **gated on Phase 0: acquiring real human observation data** (dense micro-level ground truth for at least one scenario). **Verified (repo):** GAWorld currently holds only statistical anchors (Track A of `benchmark/gaworld_bench.py`), paper-reported human results (replication flow), and distillation portraits — no trajectory or microdata. Without dense targets, MCF yields a frontier over weak targets and adds little beyond the existing bench (§6).

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
3. **(c) Error-vector scorer** — assemble one error vector per (config, seed) by reusing: Track A anchor scoring (`benchmark/gaworld_bench.py::track_a_macro_fit`, `gini`), the replication-tab fidelity machinery (`gaworld/research/replication.py` + `gaworld/research/evaluate.py::evaluate`), and the deterministic causal estimators (`gaworld/parallel/causal.py::estimate_effects`) where a target is a contrast rather than a level. Aggregate across seeds (mean) and, where multiple empirical instances exist, adopt the paper's worse-of-two-instances rule (**Verified (paper):** f_i = max over instances, to force reproduction of the recurring pattern rather than one realization's noise).
4. **(d) Pareto frontier extraction** — new but small: a pure componentwise-dominance filter over seed-averaged error vectors. No surrogate models, no smoothing (**Verified (paper):** the estimated frontier is just the nondominated subset of the sampled attainable set).
5. **(e) Architecture comparison** — compute A(M1|M0), the newly attainable set (points in Ŷ_M1 not weakly dominated by any Ŷ_M0 point), plus matched-configuration paired statistics for the same θ across architectures (medians, improvement counts, dz effect sizes — **Verified (paper)** protocol). Reuse `bootstrap_ci` and `sign_flip_p` from `gaworld/parallel/causal.py`.
6. **(f) Report generation** — frontier plots/tables and comparison verdicts via `gaworld/research/report.py`, following the paper's asymmetric interpretation rule for sparse frontiers (§6).

**Where MCF applies — and where it does not (Verified (repo) facts, Proposed verdicts):**

- **ARENA (single-agent): MCF not appropriate.** `TASK_BANK` is 10 hand-written tasks judged by an LLM; there is no error vector over real observations, no admissible configuration space worth sampling, and no meaningful frontier.
- **Full social simulation: applicable but blocked on data.** The architecture (kernel + plugins + config layering) supports multi-config runs and the eval layer can produce error vectors — but the only real-human targets today are 1–3 macro scalars (Track A) and paper-reported summary results (replication tab). Dense trajectory-level ground truth must be acquired first (§6, §10 Phase 0).

---

## 4. Reusable Components

**Verified (repo)** — drawn from the repository analysis (RESULT-3170):

| Need | Reusable | Path / symbol |
|---|---|---|
| Multi-config experiment running | 1-D sweep expansion with guards | `gaworld/parallel/sweep.py::tunables/current_value`, `MAX_WORLDS = 8` |
| | World-spec validation & config patching | `gaworld/parallel/spec.py::normalize_experiment`, `world_overrides` |
| | Seeded parallel-world runner | `gaworld/parallel/runner.py`; `gaworld/research/batch_runner.py` |
| | Env-var config injection for subprocess runs | `GAWORLD_CONFIG_OVERRIDES` |
| | Run provenance | `gaworld/core/run_manifest.py::start_manifest`, `gaworld/core/comparability.py` epochs |
| Evaluation against real data | Macro-anchor scoring | `benchmark/gaworld_bench.py::track_a_macro_fit`, `gini` |
| | Paper-replication fidelity scoring | `gaworld/research/replication.py` + `gaworld/research/evaluate.py::evaluate` |
| | Deterministic causal estimates | `gaworld/parallel/causal.py::estimate_effects`, `bootstrap_ci`, `sign_flip_p`, `bh_qvalues` |
| | Stylized-fact / judge tracks | `track_b_games`, `track_d_whois`, `benchmark/rubric/` |
| Capacity-frontier analysis | Partial — dose–response fit reads a sweep directly | `gaworld/parallel/causal.py::dose_response` |
| Reporting | Auto bench report + scorecard renderer | `benchmark/gaworld_bench.py` report writer, `benchmark/results/*.md` |
| | Dashboard API + OpenAPI client | `gaworld/apps/routes.py`, `gaworld/apps/openapi.py`, `gaworld/client.py` |
| | Study reports & interpret (1 LLM call) | `gaworld/research/report.py`, `gaworld/parallel/interpret.py` |

---

## 5. Required Engineering Changes

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

## 6. Scientific Feasibility (critical)

This is the decisive section. **Verified (repo):** GAWorld's real-human targets are thin. Track A anchors are **1–3 scalars** (Engel coefficient, savings rate, wealth Gini); the replication flow scores against *paper-reported summaries*, not microdata. **Verified (paper):** the MCF demonstration's power came from dense video-derived multi-dimensional errors — three metrics (temporal occupancy, spatial Wasserstein, flow-direction cosine), each computed per year and aggregated as the worse of two years.

Consequences:

1. **Too few error dimensions today.** With 1–3 scalar targets, a "frontier" degenerates: dominance ordering over 2–3 scalars sampled by a 1-D sweep is not an informative capacity picture. A meaningful frontier needs **≥ 2–3 independent error dimensions** plus **bound-pressing diagnostics** (best configs pressing an empirically justified parameter bound; a failing dimension flat across Θ_M; required behavior absent from the action space — **Verified (paper):** "diagnostics rather than proofs").
2. **Without new data, MCF yields a frontier over weak targets = limited insight.** The simulation-side machinery (config variation, seeded runs) exists — the 1-D sweep already proves parameter sensitivity — but the empirical side does not. **This is a data blocker, not an architecture blocker.**
3. **Overfitting controls are available but must be designed in.** **Verified (paper):** iterative development on the same empirical data risks fitting one event's idiosyncrasies; mitigations are grounding refinements in observed behavior, holding Θ fixed across architectures, worse-of-two-instances scoring, and — decisively — a **held-out event**. GAWorld must plan hold-out events/scenarios before any frontier-guided iteration.
4. **Sparse-seed LLM frontiers are optimistic.** **Verified (paper):** sparse sampling makes the estimated frontier conservative, but stochastic evaluation makes it optimistic (nondominated points partly drew favorable noise; the bias grows as seed counts shrink), and "stochastic variation in the language model is not separated from configuration effects" at 1 seed. Interpret sparse frontiers **asymmetrically**: extension beyond a densely sampled frontier is strong evidence of capacity gain; an apparent shortfall may reflect sparse exploration. Always report configurations × seeds per frontier.
5. **Don't confuse data limits with architecture limits.** A frontier gap under sparse/dense-but-weak targets must not be read as an architectural shortcoming.
6. **When MCF adds little:** with only scalar targets, the current bench's best-fit reporting (`benchmark/gaworld_bench.py` scorecard + auto report) already suffices. Building MCF before Phase 0 data exists would produce machinery with nothing meaningful to compute.

---

## 7. Risks and Rewards

**Proposed**, anchored on **Verified (paper)** cost/bias facts and **Verified (repo)** constraints:

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Compute cost of multi-config runs | High | High | Paper: M0 ran 512×3 but LLM architectures only 16×1 — "evaluation costs can differ by orders of magnitude across architectures." Adopt **matched budgets** per architecture; start with cheap configs and small worlds; use `gaworld/llm/stats.py` telemetry to watch spend. |
| Sparse-sampling optimism (frontier looks better than reality) | High | Medium | ≥ 3 seeds where affordable; report configs × seeds; apply the paper's asymmetric interpretation rule; seed-doubling stability check before trusting a frontier. |
| Added complexity / maintenance | Medium | Medium | Keep MCF as an additive `benchmark/mcf/` module; reuse existing runners/scorers; no changes to the orchestrator or default pipeline. |
| Bench / test breakage | Low | Medium | Additive module; keep `tests/test_openapi.py`, `tests/test_line_budget.py`, ruff ratchet, mypy strict green; no new `Epoch` (opt-in only). |
| Data-availability blocker (no dense human ground truth) | **High** | **High (go/no-go)** | Phase 0 gate (§10): if no suitable dataset is acquired by the deadline, NO-GO; fall back to existing bench reporting. |
| Data-vs-model confound (improvements reflect data quirks, not capacity) | Medium | High | Worse-of-two-instances scoring, held-out event as the decisive test, refinements grounded in observation/theory. |

**Rewards:** a principled answer to "does this architecture change add capacity?" (nonempty A(M1|M0)); early detection of architectural limits via bound-pressing/flat-dimension diagnostics; exposure of compensating errors (a frontier that recedes after a well-motivated change is informative — **Verified (paper)**); a cost/quality frontier that finally fills the stubbed Track E role.

---

## 8. Options A / B / C

**Proposed**, effort estimates are rough engineering estimates (Assumption):

| Option | Description | Effort | Verdict |
|---|---|---|---|
| **A — post-hoc Pareto over existing results** | Add a dominance filter over already-archived run/sweep outputs. | ~3–5 person-days | **Low value:** only 1-D sweeps (≤ 8 worlds) and scalar-anchor runs exist, so there is no multi-dimensional attainable set to filter. Not recommended beyond a throwaway prototype. |
| **B — dedicated `benchmark/mcf/` module** | Sampler + error-vector scorer + frontier extraction + comparison + reports, wired to the existing parallel-world runners and scorers (§3). | ~2–4 person-weeks engineering **plus** separate human-data acquisition; moderate compute | **Recommended** — conditional on the Phase 0 data gate. |
| **C — full Bench+Workbench integration with scheduler** | MCF as a first-class study mode in the workbench, with cost-aware run scheduling, dashboard views, and Track E rewrite. | ~2–3 months; high maintenance | **Premature** before B proves value. However, **design B's sampler abstraction so C can build on it** (sampler as a pluggable component from day one). |

---

## 9. Estimated Development Effort

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

## 10. Recommended Roadmap

**Proposed.**

- **Phase 0 — data acquisition (BLOCKING gate).** Acquire a dense human-observation dataset for **one** scenario (trajectory- or microdata-level, ≥ 2–3 independent error dimensions, ideally ≥ 2 recurring instances for worse-of-two scoring). **If unavailable by the deadline → NO-GO**; MCF stays unimplemented and the existing bench remains the evaluation of record.
- **Phase 1 — MVP.** Sampler + error-vector scorer with ≥ 2 error dimensions; ~64 configurations × 3 seeds on the cheapest admissible configuration; frontier extraction. Success: the pipeline produces a stable frontier end-to-end.
- **Phase 2 — rigor.** Frontier reports via `gaworld/research/report.py`; architecture-comparison protocol (matched configurations, A(M1|M0), asymmetric sparse-frontier interpretation); tests; lint/mypy/OpenAPI conformance.
- **Phase 3 — surface.** Dashboard/OpenAPI routes; multi-architecture comparison; optional Track E (cost) feed-in.

**Success criteria:**

- Frontier is **stable under seed doubling** (points don't reshuffle).
- **Bound-pressing or plateau diagnostics fire** on at least one known-tight dimension.
- A **matched-config comparison detects a known improvement** (a deliberately added mechanism moves the frontier outward in the expected dimension).

**Abandon conditions:**

- No real data acquired by the Phase 0 deadline.
- **Seed variance > config variance** (the frontier measures noise, not architecture).
- Insights duplicate what the existing bench scorecard already reports.

---

## 11. Open Questions and Blockers

1. **Which scenario gets dense ground truth?** One city, one behavior domain (e.g., daily mobility, market participation, event response) must be chosen so that ≥ 2–3 error dimensions and a held-out instance can be defined. *(Blocker for Phase 0.)*
2. **GAWorld's architecture/configuration boundary.** Are prompts inside architecture M (fixed per version) or configurations θ? **Verified (paper):** the boundary is a modeling choice every MCF conclusion holds relative to; treating prompt variants as configurations would enlarge Θ_M and could widen frontiers. GAWorld must declare and freeze this per experiment.
3. **Track E cost metric definition.** The bench's cost/reproducibility track is stubbed (**Verified (repo)**); per-config cost accounting (LLM calls × price vs wall-clock vs tokens) needs a definition before the frontier can include a cost axis.
4. **Per-config run budget.** Configs × seeds per architecture, matched across architectures (paper matched 16 configs × 1 seed for LLM architectures against a 512×3 conventional frontier). A pilot-based sizing like the paper's Appendix F has not been run for GAWorld scenarios.

---

## 12. Final Go / No-Go

- **Conditional GO — Option B** (dedicated `benchmark/mcf/` module) for the full social-simulation layer, **conditional on passing the Phase 0 data gate** (dense real human observation data with ≥ 2–3 error dimensions and a hold-out instance). No data → **NO-GO**.
- **NO-GO for ARENA** (`gaworld/apps/arena_api.py`): single-agent, 10 hand-written tasks, no multi-dimensional error vector over real observations, no meaningful admissible configuration space — MCF is structurally inapplicable there.
- Build B so its sampler abstraction can later graduate into Option C, but do not build C now.

---

*Sources: RESULT-3170 (verified GAWorld repo analysis, HEAD `9697471`, 2026-11-05) and RESULT-3171 / `paper-zhao-2026-mcf.md` (verified summary of Zhao, Li & Shirado, "Model Capacity Frontier: A Development Framework for LLM-Supported Social Simulation," arXiv:2610.24168v2). No code was changed in producing this document.*
