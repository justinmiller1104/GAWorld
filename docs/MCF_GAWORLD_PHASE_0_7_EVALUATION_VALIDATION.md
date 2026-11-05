# MCF × GAWorld — Phase 0.7: Evaluation & Validation

**Date:** 2026-11-05
**Scope:** documentation only — no code changes, no simulations, no LLM calls in this phase.
**Sources:** RESULT-3187 / RESULT-3188 (context), RESULT-3191 (TDD RED), RESULT-3194 (TDD GREEN), RESULT-3192 (ΔGini reassessment + options analysis).
**Label conventions:** **Verified (repo)** = direct code read with file:line · **Verified (test)** = executed test result quoted verbatim · **Computed** = deterministic calculation executed and archived in the source RESULT · **Assumption** = unmeasured inference · **Proposed** = recommended, not implemented.

---

## 1. Executive Summary

**The time-use scorer is built and tested.** `benchmark/mcf/time_use.py` was implemented against the locked test contract `tests/test_mcf_time_use.py` via strict TDD: RED state first confirmed (43 tests/params failing at collection with `ModuleNotFoundError: No module named 'benchmark.mcf.time_use'`, RESULT-3191), then GREEN (**43 passed**, RESULT-3194). The full suite after implementation reads **17 failed / 4657 passed / 10 skipped, 1243 subtests passed — identical to the documented baseline (baseline-3191.txt), i.e. no regressions**. `ruff check` / `ruff format --check` clean; `python scripts/lint_ratchet.py` reports "nothing got worse".

**D1 (wealth inequality) is settled: ΔGini is REJECTED as any frontier coordinate.** Three independent grounds (RESULT-3192): (i) no published monthly/short-horizon ΔGini statistic exists — NBS and CHFS publish annual aggregates only; (ii) the deterministic worst-case adversarial projection bounds |ΔGini| ≈ **0.032 over 30 days**, below 1 SD of the day-0 seed-noise floor (**0.0385**) and under half the 2-SD detection threshold (**0.077**); (iii) matched-seed pinning, while mechanically achievable with zero production changes (config-only pin, probe-verified), alters the input distribution (Gini(0) collapses to ≈ **0.4385** at k=3.5) and cannot rescue the coordinate. **Gini is retained conditionally as a static day-0 level check only** — reported once per run against the labelled savings-distribution range, never a Pareto/error axis.

**The recommended design is two-dimensional and both dimensions are time-use** (Option B of RESULT-3192): **paid-work pattern accuracy vs leisure+social pattern accuracy**, under the `personality.channels.rules` on/off matched-seed architecture pair (clean ablation, RESULT-3187 §4), with `style_fit_amplitude` as primary Θ. Commute minutes (episode `travel` field) is held as the single Option-C reserve checkpoint, not a third axis.

**Honest limitation, stated up front:** both dimensions share the same underlying activity records and the same keyword mapper, so partial correlation risk is real; falsifiers and a single-dimension downgrade path are pre-registered (§5, §7).

**Verdict: conditional GO** (§9), gated on (i) real-episode mapper coverage measured in an authorized micro-run and (ii) a config-sensitivity probe. NO-GO on any frontier work before both gates.

---

## 2. Time-Use Scorer Implementation and Tests

### 2.1 Files and API surface

**Verified (repo):** `benchmark/mcf/time_use.py` (321 lines, stdlib-only, pure functions, `from __future__ import annotations`); package marker `benchmark/mcf/__init__.py`; locked contract `tests/test_mcf_time_use.py` (29 test cases in RESULT-3191's count; **43 passed** as executed in RESULT-3194, parametrization expands the case count).

Public API (module `__all__`, Verified repo):

| symbol | contract |
|---|---|
| `CATEGORIES` | the 9 survey categories: sleep, paid_work, housework, personal_care, meals, leisure, socializing, commuting, study |
| `TARGETS_2024` | cat → `{"minutes", "basis"}` — sleep 586, personal_care 56, meals 103, **paid_work 383 (`basis="participants"`)**, housework 77, leisure 153, socializing 18, commuting 50, study 107 (`basis="child_inflated"`, `comparable_by_default: False`, excluded from default scoring) |
| `TARGETS_2018` | 2018 urban robustness vector (RESULT-3188 §5) |
| `classify_activity(text)` | Chinese keyword map → `(category \| None, "keyword" \| "unknown")`; unknown/empty → `(None, "unknown")` |
| `load_episodes(path)` | JSONL → `[Episode]` with `.agent_id` (int, parsed from `agent_<id>_episodes.jsonl`, with filename fallback), `.day`, `.time`, `.final_activity`; empty file → `[]` |
| `compute_daily_minutes(episodes, tick_minutes)` | timestamp differencing → `{agent: {day: {category: minutes}}}`; gap ≤ 2×tick → previous category; gap > 2×tick → min(gap, 2×tick) previous + remainder `unknown`; overlaps clamped ≥ 0; last episode extends to 24:00; missing days absent |
| `coverage(result)` | unknown time counts against coverage (coverage < 1.0 with unknowns) |
| `score_against_targets(daily, employed_agents)` | `ScoreResult` with `.per_category` (sim/target/error), `.overall_error`, `.coverage`, `.leftover_minutes`; **paid_work over employed agents only**, others all-agents; zero-division-safe |
| `score_directory(dir, employed_agent_ids, tick_minutes)` | `ScoreResult` over many agent files |

**Duration-model rules documented in the module docstring** (Verified repo): scheduled-anchor pinning — when the next episode is a scheduled anchor (sleep / paid_work / meals / personal_care), its timestamp pins the previous activity's end so the whole gap is attributed to the previous category (leisure timestamps do not pin, since drifting into free time is exactly where the switch moment is unobservable); and sleep-ends-day — the day's last episode extends to 24:00 in its own category, *except* a sleep episode directly after paid_work, where the sleep timestamp ends the recorded day and the remainder is `unknown`. Both exceptions are the minimal reverse-fitting of the locked hand-calculated tests (RESULT-3194 §Why).

### 2.2 Test coverage

**Verified (test, RESULT-3191/3194):** the suite pins, among others:

- the **hand-calculated full-day case** with exact minutes — sleep 510 / personal_care 30 / commuting 60 / paid_work 600 / meals 30 / housework 60 / leisure 150, **sum 1440, coverage 1.0**, tick_minutes=30;
- unknown-gap attribution (≤ 2×tick vs truncation + remainder), the anchor-pinning exception, the sleep-after-work 24:00-boundary exception;
- classifier keyword mapping (24 parametrized cases) and unknown/empty handling;
- determinism (same input → same output) and population basis (paid_work employed-only; zero-division-safe empty input);
- multi-agent `score_directory` aggregation.

### 2.3 Test results (verbatim)

RESULT-3191 (RED):

> `pytest tests/test_mcf_time_use.py` → 1 collection error:
> `E ModuleNotFoundError: No module named 'benchmark.mcf.time_use'` — fails solely because the
> scorer module is missing, as required.

RESULT-3194 (GREEN):

> - `pytest tests/test_mcf_time_use.py` → **43 passed** (0.4 s)
> - Full suite → 17 failed, 4657 passed, 10 skipped, 1243 subtests passed (all 17 failures pre-existing per baseline-3191.txt)
> - `ruff check` + `ruff format --check` clean; `python scripts/lint_ratchet.py` → "nothing got worse"

---

## 3. Wealth Inequality Reassessment (D1)

### 3.1 Matched-seed formalization

**Verified (probe, RESULT-3192 §1.2):** the U(1,6) initial-savings draw (`gaworld/economy/finance.py:2154–2157`) sits on a module-level shared stream `_rng` (`finance.py:61`), seeded per run as `f"economy:{int(seed)}"`. `random.Random.uniform` computes `a + (b−a)·random()` — the `random()` call happens regardless of `[a,b]` width — so setting `initial_savings_months_min = initial_savings_months_max = k` pins the *value* while consuming exactly one draw: with `Random(42)`, pinned `uniform(6.0, 6.0)` vs unpinned `uniform(1.0, 6.0)` leave **identical downstream 3-draw sequences**. All later shared-stream draws (housing fund, spending noise at `finance.py:2287–2295`, events, wage jitter) therefore stay bit-aligned across arms, provided no config difference changes a draw count (none found within the `rules` toggle, RESULT-3187 §4.2).

- **What cancels under matched seeds:** the U(1,6) initial-draw component of Gini variance (the SD **0.0385** of RESULT-3187 §2.1 cancels exactly at t=0); per-agent trait streams (per-agent `Random(seed·1000003 + agent_id)`, config-independent).
- **What remains in ΔGini:** spending-noise effects (config-dependent via `daily_variance`), `wealth_drive` wage/savings tilts (`finance.py:1366`, coarse wage factor `0.90 + 0.35·drive`), LLM sampling temperature.

**Admissibility of the config pin (Proposed, RESULT-3192 §1.3):** admissible for a paired contrast or as a sanity device, but it **alters the input distribution** — at k=3.5 the pinned Gini(0) is **0.4385** (Computed) vs variant-B mean ≈0.48, i.e. the wealth dispersion the Gini should measure is manufactured by the very draw being pinned. Must be declared in the run manifest; never silent.

### 3.2 Mechanism bound vs noise floor (Computed)

Cohort `data/cities/绍兴柯桥/agents.csv`, labour force n=33, variant-B formula, no inheritance term. Cross-seed spread of Gini(0) over 3 fixed seeds: **0.114** (seeds 11 / 20261105 / 777 → Gini(0) 0.4232 / 0.5374 / 0.5335). Worst-case 30-day adversarial projection (daily_variance ±40% and maximum `wealth_drive` wage-factor swing, both bent Gini-maximising / minimising per agent):

| direction | Gini(30) | ΔGini |
|---|---|---|
| Gini-maximising | 0.5408 | +0.0034 |
| Gini-minimising | 0.5051 | −0.0323 |

**Bound |ΔGini| ≤ ≈0.032**, below 1 SD of the noise floor (0.0385) and under half the 2-SD threshold (0.077). Assumption noted in source: the projection's expense share (0.62 of income); even at 100% the bound stays below 0.05.

### 3.3 External-target check

**No published statistic exists for the change in wealth or income Gini over ~1 month or any short horizon** (domain knowledge of NBS/CHFS publications, with the caveat that only a reachability probe of stats.gov.cn was executable; no counter-example known). Official series are annual: NBS disposable-income Gini (≈0.462 for 2015, ≈0.465–0.47 territory recently), CHFS wealth Gini (≈0.72+, survey-wave basis, biennial). A simulated ΔGini therefore has **no external target to be scored against** — it fails as an MCF error coordinate quite apart from being noise-bound.

### 3.4 Verdict table

| quantity | verdict | condition |
|---|---|---|
| ΔGini (Gini(T) − Gini(0)) | **Reject** | in every form: frontier coordinate, error dimension, config-contrast outcome (bound 0.032 < threshold 0.077; no external target) |
| Gini day-0 level | **Retain conditionally** | static plausibility check vs the CHFS ≈0.7 wealth-Gini range **with the population caveat**: the cohort's ≈0.44–0.54 initial Gini is a *savings* distribution and must be labelled as such; reported once per run, never a Pareto/error axis |
| matched-seed pin (min=max) | Admissible, declared | only as a declared distribution change for paired contrasts in *other* dimensions |

---

## 4. Candidate Evaluation Dimensions

| option | content | verdict | grounds |
|---|---|---|---|
| **A** | time-use + ΔGini | **Reject** (the ΔGini half) | time-use itself is sound (RESULT-3187 §3.5, feasible in normal mode); ΔGini is triple-dead (§3.2–3.4); any pairing inherits the dead half |
| **B** | paid-work pattern accuracy vs leisure+social pattern accuracy | **Retain — the recommended pair** | §5 |
| **C** | one other recorded observable | **one reserve candidate: commute minutes**; all else rejected | see below |

**Option C reserve — commute minutes.** Episodes record `travel` and `location` per tick (`_stages.py:589, 910`); commute is keyword-mappable (通勤/乘坐…移动, `_stages.py:603–612`). Targets: 2024 公报 交通出行 50 min/day all-persons (rate 80.5%); 2018 附表3 城镇 participants 75 min, 附表1 城镇交通 44 min. Scoring: employed agents only, per-day inferred duration from timestamp differencing. Honest weaknesses (Assumption): inferred not recorded (boundary hazard, RESULT-3187 §3.2), likely small and possibly Θ-insensitive (commute follows the LLM/archetype-generated schedule skeleton, only weakly style-fit-coupled). **Recommendation: pre-registered manipulation-check / report-out, not a third frontier axis.**

**Rejected candidates** (RESULT-3192 §2): `valence` / `salience` (no external distribution to score against — survey affect series are not time-budget-compatible); `social_partners` (no published per-day unique-contact benchmark; Social Survey modules irregular); `need_snapshot` (no external analogue); **internet use 5h37m / 92.9%** (2024 公报 — superficially attractive but *not recorded as an observable*: episodes carry no internet-use field and the keyword mapper cannot reliably separate 手机伴随 use); `location` distribution (no published destination-type share table at matching granularity).

---

## 5. Recommended Two-Dimensional Design

| element | recommendation |
|---|---|
| **Dimension 1** | Paid-work pattern error: per-employed-agent daily paid-work minutes vs **2024 participants 383 min** (fallback 2018 城镇 participants 457); population = 33 employed agents; normal-mode episodes; timestamp-differenced durations |
| **Dimension 2** | Leisure+social pattern error: daily 文化休闲娱乐 + 社会交往 (± 运动健身) minutes vs **2024 all-persons 206** (= 153 + 18 + 35; or 171 excluding sport); population = all 80 agents |
| **Architecture pair** | `personality.channels.rules` on/off, matched seeds — clean ablation: identical shared-RNG sequences, identical trait values, no required-key changes (RESULT-3187 §4) |
| **Θ** | `style_fit_amplitude` [0.10, 0.40] — **Verified to activity choice** (`traits.py:208–234` → `_action.py:679–683`), cuts across both dimensions, plausible tension source; `contagion_weight` [0.02, 0.12] — Verified to emotion (`_cognition.py:183–210`), Plausible to leisure via `low_mood_avoidance` (`_action.py:685`), secondary axis. `daily_variance` **excluded from Θ** for these dimensions (Verified: zero path to activity, RESULT-3187 §3) — it belongs only to the dead wealth coordinate. LLM temperature optional, Plausible. |
| **D1's role** | static day-0 Gini reported once per run against the labelled savings-distribution range; never a coordinate |
| **Reserve checkpoint** | commute minutes vs 50 min (employed agents) — manipulation check only |

**Why these two dimensions (Verified repo + independent targets).** The mechanism sets are largely **disjoint in code**: paid-work composition is driven by schedule archetypes keyed on job/work_style keywords (`_heuristic_schedule`, `gaworld/sim/_schedule.py:221–281`; the income-seek pre-step hook can force work activity, `_stages.py:425` region), wealth-drive-modulated effort and wage booking (`finance.py:1366`, `2512` region), and `maybe_adjust_activity` derailing with trait-modified interrupt resistance (`_stages.py:376–419`, `dynamic.py:192`); leisure+social composition is driven by `spontaneity_chance` (`dynamic.py:333`), dynamic social/leisure insertions (`_stages.py:404–411`), the social-encounter trait branch (`dynamic.py:588–604`), and emotion-avoidance fed by contagion with its set-point mechanism. The 2024 公报 targets are different survey categories with different bases and population definitions (participants vs all-persons) — two coordinates, not one split twice. **Plausible trade-off:** a higher `style_fit_amplitude` pulls each agent toward its trait-styled mix (conscientious → work, extraverted → leisure/social), so the two fits can move in opposite directions across Θ — wiring Verified, effect sizes unmeasured (**Assumption**).

**Scoring populations declared separately:** paid work conditional on employment (participants basis; the cohort is 33 employed / 59% NILF, RESULT-3187 §4.3); leisure all-persons.

**Falsifiers (pre-registered, RESULT-3192 §3):**

1. The deterministic scorer shows keyword-mapper coverage < ~85% of episode minutes, or unresolvable timestamp boundaries > ~10% of intervals, on real episodes → the dimensions are unmeasurable as designed.
2. The live probe shows config-range Δ in either dimension < 2× seed-to-seed SD of that dimension (across ≥3 seeds) → noise-bound like D1 → honest single-dimension downgrade.
3. Both dimensions move *together* monotonically with every Θ parameter (no sign divergence anywhere in Θ) → no trade-off exists → report as a dose sweep, not a frontier.

**Downgrade path:** if any falsifier fires, the honest fallback is the RESULT-3179 §7 alternative — single-dimension dose sweep plus the `channels.rules` architecture contrast, plainly labelled — not a manufactured two-axis frontier.

**Shared-mapper correlation risk (honest limitation):** both dimensions are computed from the same episode records through the same keyword mapper, so the two "independent" coordinates can partially co-move; falsifier (3) is the guard, and §7 carries the limitation.

**Cost warning carried forward:** both dimensions require **normal-mode** runs (no episodes under fast-forward, RESULT-3187 §2.3); per-day call cost is unmeasured and must be measured (§8) before any pilot.

---

## 6. Parameter-to-Outcome Dependency Matrix

Updated matrix. Labels: **Verified** = code path read end-to-end; **Plausible** = wiring exists, effect size unmeasured; **None** = no path found.

| parameter ↓ / outcome → | paid-work time-use | leisure/social time-use | commute | Gini (static) |
|---|---|---|---|---|
| `style_fit_amplitude` | **Verified** (`_action.py:679–683` via `traits.py:208–234`) | **Verified** (same site) | Plausible (weak, archetype-skeleton-mediated) | None |
| `contagion_weight` | Plausible (via mood → action choice) | **Plausible** (`low_mood_avoidance`, `_action.py:685`) | Plausible (weak) | None |
| LLM temperature | Plausible | Plausible | Plausible | Plausible (weak) |
| `daily_variance` | **None** (Verified zero path, RESULT-3187 §3) | **None** | None | Plausible (spending noise only) |
| `channels.rules` (architecture pair) | **Verified** (action-selection sites #1–5) | **Verified** (same sites) | Plausible | None |
| initial-savings pin (min=max) | None | None | None | **Verified** (collapses Gini(0) variance; distribution change) |
| schedule archetype (data, not Θ) | **Verified** (`_schedule.py:221–281`) | Plausible | **Verified** (通勤 anchors) | None |

The `channels.rules` action-selection sites (RESULT-3187 audit, carried into this matrix): the five sites where the rules toggle changes action selection, cutting across both time-use dimensions — this is why it serves as the architecture contrast for both coordinates simultaneously.

---

## 7. Remaining Scientific Limitations

1. **Shared-mapper correlation risk.** Both dimensions derive from the same activity records and keyword mapper; partial correlation of the two "independent" coordinates is possible (guarded by falsifier 3, not eliminated).
2. **Scorer unvalidated on real episodes.** Mapper coverage and timestamp-boundary resolution are unknown until the first authorized run measures them (gate i, §9); the 43-test suite validates the contract, not real-data coverage.
3. **Θ effect sizes unmeasured.** `style_fit_amplitude` / `contagion_weight` magnitudes on both dimensions require the authorized micro-run.
4. **Weekly-avg vs workday basis mismatch.** The 公报 targets are week-averages (including weekends); the scorer computes per-day minutes over whatever days exist in the episode file — a workday-only episode set biases paid-work comparisons upward unless the basis is declared (**Assumption** until measured).
5. **Population mismatch on paid-work scoring.** The participants-basis target (383 min) is conditional on scoring only employed agents (33 of 80; 59% NILF); the population definition must be honoured exactly, or the comparison is invalid by construction.
6. **Normal-mode cost unmeasured.** Both dimensions need normal-mode runs; per-day LLM call cost is unknown (§8).

---

## 8. Estimated Cost of the Eventual Live Pilot

| item | cost |
|---|---|
| Time-use scorer + test suite | **done — ¥0** (deterministic, stdlib-only, no LLM calls; 43 tests passing, no regressions) |
| Mapper-coverage & sensitivity micro-run | authorized run required: 2 configs (`channels.rules` on/off) × 2 seeds × short normal-mode run on a reduced cohort; cost **not yet estimable** — the per-day normal-mode call count is unmeasured; a tick-grid extrapolation is indicative only and must be **flagged as unmeasured** until gate (ii) executes |
| Full 2-D frontier pilot | budget framework carried from Phase 0.5/0.6 (grid over Θ × matched-seed architecture pair), **with the normal-mode multiplier caveat**: normal mode costs strictly more per simulated day than fast-forward, and the Phase 0.5/0.6 figures were fast-forward-basis — apply the measured multiplier from the micro-run, never reuse the FF numbers directly |

---

## 9. Go / No-Go

**Conditional GO.** Gates, in order:

1. **Real-episode mapper coverage (gate i).** Run the scorer on a real episode file produced by an authorized micro-run; require keyword-mapper coverage ≥ ~85% of episode minutes and unresolvable timestamp boundaries ≤ ~10% of intervals. If it fails → **fix the mapper** (deterministic keyword work, cheap, no pilot cost) and re-measure.
2. **Config-sensitivity probe (gate ii).** On the micro-run, verify config-range Δ in at least one dimension exceeds 2× the seed-to-seed SD of that dimension (≥3 seeds). If it fails → honest **single-dimension dose sweep** or stop.
3. **If both pass → 2-D frontier pilot** (paid-work accuracy vs leisure/social accuracy, `channels.rules` on/off matched-seed pair, Θ = `style_fit_amplitude` primary), with the static day-0 Gini reported once per run as a labelled sanity check and commute minutes as the manipulation-check report-out.

**NO-GO on any frontier work before gates (i) and (ii).** A frontier built on unmeasured mapper coverage or unmeasured Θ sensitivity would repeat exactly the failure mode that killed ΔGini: a coordinate that is either unmeasurable or noise-bound — discoverable only after the compute is spent.
