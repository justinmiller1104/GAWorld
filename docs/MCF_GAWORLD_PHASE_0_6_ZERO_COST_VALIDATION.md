# MCF × GAWorld Phase 0.6 Zero-Cost Validation Report

**Status:** FINAL · 2026-11-05 · Based on repo `task/3189` worktree (read-only validation; no code changed, no simulations launched, no LLM calls spent)
**Sources:** All facts come from RESULT-3187 (Monte Carlo + logging audit + dependency traces + rules-toggle ablation) and RESULT-3188 (stats.gov.cn time-use survey data, primary sources fetched verbatim). Prior context: `docs/MCF_GAWORLD_INTEGRATION_FEASIBILITY.md`, `docs/MCF_GAWORLD_PHASE_0_5_VALIDATION.md`.
**Label conventions:** Every key claim is labelled —
**Verified (repo)** = direct code read with file:line citation · **Verified (source)** = fetched from the official stats.gov.cn communiqué ·
**Computed** = zero-cost calculation actually executed in this validation · **Assumption** = unmeasured inference ·
**Proposed** = recommended but not implemented.

---

## 1. Executive Summary

**Zero-cost validation complete. The Phase 0.5 pilot as specified is abandoned; a redesigned pilot on two dimensions is now recommended.**

- **(a) D1 is noise-bound at n=33 — the abandon condition fires; do not spend calls on D1.** A 200-draw Monte Carlo (seed 20261105, **Computed**) pins D1's seed-noise floor at **SD = 0.0385** (mean Gini 0.4421, range 0.349–0.545). The dynamic headroom to the re-based anchor (0.45 target sits 0.008 above the input mean; 0.50 sits 0.058 above) is less than 2×SD (0.077) at every anchor setting — a config effect could not be detected even if simulated dynamics moved the Gini by the *entire* anchor gap. The real init formula variant B (with the `econ_security` factor, finance.py:2156–2157) has mean 0.4786 **inside** the 0.45–0.5 anchor band with zero simulated dynamics — the circularity is worse than Phase 0.5 assumed.
- **(b) The time-use scorer is feasible in normal mode only.** Per-tick **episodes** (`output/memory/agent_<id>_episodes.jsonl`, schema composed at `gaworld/sim/_stages.py:901–951`, **Verified (repo)**) carry `final_activity` + timestamps but no durations (inferable by timestamp differencing on the tick grid). **Fast-forward writes no episodes** — only a deterministic fallback diary, memory lines and numeric state metrics — so under FF the only activity information is the static `base_schedule_map` skeleton, which is Θ-invariant (**Verified (repo)**).
- **(c) Real-world targets are confirmed from stats.gov.cn primary sources.** The 2024 第三次全国时间利用调查 communiqués give a ≥6-category target vector (sleep 586, care+meals 159, paid work 383 participants-only, housework 77, leisure+social+sport 206, commute 50 min/day; **Verified (source)**). No employment-status cross-tab is published, so paid work must be scored conditional-on-employment against participants' averages.
- **(d) Dependency matrix: only `style_fit_amplitude` is Verified to activity.** `daily_variance` is Verified to spending and **cannot touch activity choice** (activity text is an input to expense allocation, never an output of it); `contagion_weight` is Verified to emotion and Plausible to activity; LLM temperature is Plausible (**Verified (repo)** traces, §5).
- **(e) `personality.channels.rules` on/off is a clean matched-seed ablation.** 12 mechanism sites switch off, RNG divergence is ruled out (isolated per-call `residual()` stream, per-agent trait seeding independent of the toggle), and no config key becomes required (**Verified (repo)**, §6).

**Revised pilot (Proposed):** a 2-dimension MCF — **time-use error (primary, normal-mode runs)** + **wealth Gini retained only as a static sanity check, not a frontier coordinate** — with the architecture pair `channels.rules` on/off, matched seeds. Cost is revised: normal-mode runs are ~1 LLM call/agent/tick (vs FF's ~1 call/agent/day), so per-day cost is far higher — a cost table is required before any pilot (§7, §10).

---

## 2. Monte Carlo Results (Computed)

Script: `~/work/crew/research/OUTBOX/monte_carlo_d1.py`; raw output: `~/work/crew/research/OUTBOX/monte_carlo_d1_out.txt`. 200 draws, seed `20261105`, cohort `data/cities/绍兴柯桥/agents.csv` (80 rows; labour force per `LABOUR_FORCE = {"employed","unemployed"}` = **33 agents, 0 unemployed**). Gini reimplemented identically to `benchmark/gaworld_bench.py::gini` (line 187). Multiplier U(`initial_savings_months_min`=1.0, `initial_savings_months_max`=6.0) per `gaworld/settings/economy.py:20–21`; the draw applied at `gaworld/economy/finance.py:2153–2156`.

### 2.1 Variant A — income × U(1,6) (the brief / Phase 0.5 spec)

| statistic | value (Computed) |
|---|---|
| draws × agents | 200 × 33 |
| mean | **0.4421** |
| median | 0.4430 |
| SD | **0.0385** |
| P5 | 0.3768 |
| P25 | 0.4172 |
| P75 | 0.4687 |
| P95 | 0.5015 |
| min | 0.3487 |
| max | 0.5445 |
| range | 0.1958 |

Labour-force income Gini = 0.3953 (mean ¥8,524, range ¥1,347–¥36,951) — matches Phase 0.5's single-draw figure (0.441 ≈ this run's mean 0.442).

### 2.2 Variant B — sensitivity: finance.py's actual shape

The real initialisation at `finance.py:2156–2157` is `net_monthly × U(1,6) × (0.7 + 0.6·econ_security)` plus `_infer_inheritance_amount` (not modelled — deterministic per-agent but requires the inheritance inference chain; flagged as the unmodelled component). With the `econ_security` factor applied (read from the cohort CSV):

| statistic | value (Computed) |
|---|---|
| mean | **0.4786** |
| median | 0.4803 |
| SD | 0.0397 |
| P5 / P95 | 0.4136 / 0.5374 |
| min / max | 0.3801 / 0.5779 |

Variant B's mean (0.479) lands **inside** the re-based 0.45–0.5 anchor band with zero simulated dynamics — under the real formula the "anchor fit" is even more an input artifact than under the brief's simplified spec (circularity strengthened).

### 2.3 Paired / common-random-numbers case

With one U(1,6) vector per draw reused across hypothetical configs, Gini at t=0 is **deterministic given the draw** — config-to-config Gini variance at day 0 is exactly zero (Computed). Decomposition:

- **Initial-draw component:** SD 0.0385 (Computed, §2.1).
- **Later-dynamics component** (spending noise, wage drift, wealth_drive effects over the run): **unobservable in this zero-cost validation** — running the paired sims was out of scope. Stated explicitly: what is measured here is start-population uncertainty; simulation-behaviour variation across draws is not, and can only be estimated by a live probe.

### 2.4 Decision-rule reading (Computed → verdict)

- Dynamic headroom to the 0.45 anchor: 0.45 − 0.4421 = **0.008**; to 0.50: **0.058**.
- Noise floor SD = 0.0385; a 2-SD detection threshold = **0.0770**.
- **Headroom (0.008–0.058) < 2·SD (0.077)** at every anchor setting. Under the standard "config range must exceed 2× seed noise" rule, D1 cannot certify any config effect at n=33 even if simulated dynamics moved the Gini by the entire anchor gap.

**Verdict: D1 is seed-noise-bound; the abandon condition fires before any frontier work (Computed).** The Phase 0.5 Assumption (SD ≈ 0.03–0.06) is now a measured 0.0385 — near the middle of the guessed band.

---

## 3. Activity Logging Audit (Verified repo)

There is no `gaworld/core/recorder*`; the writers live in the modules below.

### 3.1 What a run records

1. **Episodes — the only structured per-tick activity record.** Composed in `gaworld/sim/_stages.py:901–951`, persisted by `append_agent_episode` (`gaworld/memory/experience.py:79–86`) to **`output/memory/agent_<id>_episodes.jsonl`** (dir from `CONFIG["memory_dir"]`, default `output/memory`). Schema keys: `episode_id, day, time, scheduled_activity, final_activity, action, location, target_location, travel, env_events, life_events, policy_event, social_partners, perception, plan, outcome, reflection, transient_thought, state_before, state_after, need_snapshot, delta, tags, salience, valence, decision_driver, change_reason, commitment_level, created_at_day`. One JSONL line per executed tick per agent — this *is* an activity diary.
2. **Schedules** — `output/memory/agent_<id>_schedule.json` via `save_agent_schedule` (`gaworld/memory/store.py:200–206`): `[{time:"HH:MM", activity}, …]`, ~9 anchors/day. Generated by LLM routine or the heuristic `_heuristic_schedule` (`gaworld/sim/_schedule.py:221–281`) with fixed archetype templates (retired/student/late/standard).
3. **State history** — `save_state_history` (`gaworld/sim/_plots.py:117–134`) → `output/state/agent_state_history.csv`: long format `agent_id, step, metric, value`, **numeric metrics only**. No activity field.
4. **Agent log** — `output/logs/agent_<id>.log` via `append_agent_log` (`gaworld/memory/store.py:276–281`): free text with `[TodayRoutine Day N …] @ HH:MM` headers. Narrative, not structured.
5. **Diary markdown** — `save_daily_diary` (`gaworld/sim/_diary.py`): prose only.
6. **Environment timeline** — `output/environment/timeline.jsonl`: environment events, not agent activities.

### 3.2 Durations

Episodes carry a timestamp (`time`) but **no duration field**; durations must be inferred by differencing consecutive episode timestamps within a day, using the tick grid (`_timeline_step_minutes`, generative_city_sim.py:1202–1218; default step from `build_master_timeline`, line 2280). Feasible in normal mode; **boundary steps and skipped ticks are the main inference hazard**. Schedules give anchor-to-anchor spans directly but are the *planned* day, not the realised one (`final_activity` vs `scheduled_activity` diverge via `maybe_adjust_activity` / `dynamic_insert_activity`, `_stages.py:376–419`).

### 3.3 Fast-forward behaviour (Verified repo)

`_run_fast_forward_step` (generative_city_sim.py:2721+): **no episodes are written.** Per agent per period it writes: applied state deltas + jitter (~lines 2860–2875), relationship nudges, a deterministic fallback diary `_fallback_daily_diary` (lines 2957–2964; prose from top memory lines — no structured activity), one memory line (line 2944), and numeric `state_history` appends (2988–2991). Economy hooks run with `coarse=True` booking approximate wages (2823–2845, 2743–2745). The only activity information under FF is the **static** `base_schedule_map` skeleton (`schedule_map = base_schedule_map`, line 2753) — the plan-of-record, never realised per day.

**FF verdict: a time-use scorer under FF can only score the intended schedule skeleton (constant per agent, so Θ-invariant) or the brief's prose — it cannot measure realised activity composition. The scorer requires normal-mode runs (Verified repo).**

### 3.4 Category labels

Activities are Chinese free text (`上午工作`, `通勤`, `睡前`, `个人时间`, `买菜`, `晨练`, …). The only existing classifier, `ACTIVITY_CATEGORY_MAP` (`gaworld/world/city_map.py:1610–1637` + `activity_to_categories`:1689), maps to **location** categories (commerce/education/leisure/medical), not time-use survey categories. `is_sleep_activity` (`_schedule.py:305`) covers sleep. **No time-use-survey mapping exists** (sleep / paid work / housework / leisure / commute / study); a deterministic keyword mapper must be written (Proposed, ~half a day) — e.g. 工作/上班/加班→paid work, 通勤→commute, 吃饭/买菜/家务→housework+eating, 个人时间/散步/娱乐/健身→leisure, 学习/上课→study, 睡前/睡觉/午休→sleep. The episode schema has everything the mapper needs.

### 3.5 Scorer feasibility verdict

**Feasible in normal mode** (Verified structure + Proposed implementation): episodes × 80 agents × 30 days ≈ tens of thousands of records; deterministic timestamp differencing + keyword mapping; scored against the published 时间利用调查 shares (§4). **Missing for FF:** realised-activity records. **Missing fields everywhere:** duration (inferable), category (mappable), and a per-tick step-minutes record alongside episodes (inferable from the timeline builder).

---

## 4. Published Time-Use Data Assessment (Verified source)

### 4.1 2018年全国时间利用调查

**Primary source:** 2018年全国时间利用调查公报, 国家统计局, released 2019-01-25, https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900224.html

Stratified multi-stage random sampling in 11 provinces — 北京、河北、黑龙江、上海、浙江、安徽、河南、广东、四川、云南、甘肃 (**Zhejiang included**, so Hangzhou-area relevance is plausible). Subjects: 15+ year-old permanent household members; 20,226 households / 48,580 persons (城镇 29,557 / 农村 19,023). Fielded May 2018: one workday + one weekend day, 15-minute-interval 24h diary. Shanghai's APP pilot is excluded from this bulletin.

Published categories & daily averages (附表1, minutes/day, 合计|男|女|城镇|农村):
- 个人生理必需活动 713|708|718|713|713 (睡觉休息 559; 个人卫生护理 50; 用餐 104)
- 有酬劳动 264|315|215|239|301 (就业工作 177|217|139|**197 城镇 all-persons**|145; 家庭生产经营 87)
- 无酬劳动 162|92|228|165|159 (家务劳动 86|45|126|**79 城镇**|97)
- 个人自由支配活动 236|253|220|**250 城镇**|213 (健身锻炼 31; 看电视 100; 休闲娱乐 65; 社会交往 24; 阅读 9)
- 学习培训 27|28|27|29|24
- 交通活动 38|44|33|**44 城镇**|30
- 使用互联网 (伴随活动): 162|174|150|203|98

Participation rates (附表2): 就业工作 38.4% (城镇 43.2%); 家务劳动 58.5%; 看电视 66.5%; 交通 50.8%. Participants' averages (附表3): 就业工作 461 min (**城镇 457**); 家务劳动 147; 交通 75; 学习培训 372.

### 4.2 2024 第三次全国时间利用调查

**Primary sources (released 2024-10-31, fielded May 2024):**
- 公报第二号: https://www.stats.gov.cn/sj/zxfb/202410/t20241031_1957216.html
- 公报第三号: https://www.stats.gov.cn/sj/zxfb/202410/t20241031_1957215.html

Methodology changes vs 2018 (公报第二号): scope extended to all 31 provinces + 新疆生产建设兵团; subjects extended down to 6–14 year-olds; activity categories expanded to 34; first self-administered 手机小程序填报; results reported on a 按一周7天计算 (weekly-average) basis rather than the 2018 workday/restday split.

Published values (all persons, weekly-avg daily): 个人生理必需 12h27m; 有酬劳动 3h27m (participants 6h23m, rate 54.1%); 无酬劳动 2h5m (participants 2h45m, rate 75.6%); 个人自由支配 3h24m; 学习培训 1h47m (rate 25.1%); 交通 50m (rate 80.5%). Categories: 睡觉休息 9h46m; 个人卫生护理 56m; 用餐 1h43m; 交通出行 50m; 劳动就业 3h28m (**participants 6h23m = 383 min, rate 54.4%**); 学习培训 1h47m; 家务劳动 1h17m (participants 1h59m, rate 64.9%); 陪伴照料家人 30m; 购买商品或服务 15m; 看病就医 2m; 运动健身 35m; 文化休闲娱乐 2h33m; 社会交往 18m; 互联网使用 5h37m (rate 92.9%).

Breakdowns (公报第三号): gender, 城镇/农村, and three age bands (6–17, 18–59, 60+). Examples: 有酬劳动 rate 男性 60.2% / 女性 47.7% / 城镇 52.1% / 18–59岁 75.6%; 城镇互联网使用 6h42m vs 农村 4h43m; 18–59岁无酬劳动参与者 2h42m, rate 82.7%.

**Employment-status breakdown: NOT published** in either communiqué — only urban/rural, gender, and the 3 coarse age bands.

### 4.3 Population match (honest assessment)

GAWorld cohort: ~80 synthetic urban-district adults, Hangzhou area, 33 employed / 0 unemployed / 47 not in labor force.

Mismatches to flag: (a) national averages fold in rural residents and children (2024 includes 6–14 year-olds, which depresses 学习培训-vs-2018 comparability); (b) no published "urban employed 15–64" subpopulation table exists in either communiqué — the closest 2024 bracket is 18–59岁 all-persons; (c) the cohort's 47/80 (59%) not-in-labor-force share means headcount-averaged paid-work minutes will grossly undershoot any national target — **paid-work time must be scored conditional on employment** against participants' averages (2018 城镇 就业工作 457 min/day; 2024 劳动就业 participants 383 min/day, noting 2024 participants include part-time workers, lowering the average); (d) 2018 fielded in Zhejiang (good), but the 2024 Hangzhou subsample is not separately published.

### 4.4 Proposed target vector (from RESULT-3188 §5)

2024 national, weekly-average daily minutes, source 公报第二号; paid work scored conditionally on employment; all values confirmed from fetched primary text:

| Category | Target (min/day) | Basis | Source |
|---|---|---|---|
| Sleep (睡觉休息) | 586 | all persons | 2024 公报第二号 (9h46m) |
| Personal care + meals (个人卫生护理+用餐) | 159 | all persons | 2024 公报第二号 (56m + 1h43m) |
| Paid work (劳动就业) | 383 | participants only (employed agents) | 2024 公报第二号 (6h23m); alternative 2018 城镇 participant 457 |
| Housework (家务劳动) | 77 | all persons | 2024 公报第二号 (1h17m) |
| Leisure/culture/social/sport (文化休闲娱乐+社会交往+运动健身) | 206 | all persons | 2024 公报第二号 (2h33m + 18m + 35m) |
| Commute (交通出行) | 50 | all persons | 2024 公报第二号 |
| Study (学习培训) | 107 | all persons — **FLAG:** inflated by 6–17 inclusion; for an adult cohort prefer participants' 429 min × low adult rate, or drop | 2024 公报第二号 |

Robustness alternative: the 2018 城镇 column of 附表1 (sleep 556, meals+care 154, paid work 197 all-persons / 457 participants, housework 79, free time 250, traffic 44, study 29). Normalize against 1440 min/day.

**Unavailable items (honestly listed):** employment-status cross-tabs (2024), 中国统计年鉴 appendix tables (not fetched), Beijing/Shanghai city-level primary tables (existence partially confirmed, values not retrieved), CFPS/CGSS diary variables (unverified — do not use as targets), 2024 公报第一号 URL (anti-bot blocked; No.2/No.3 contain all needed numbers).

---

## 5. Parameter-to-Activity Dependency Matrix

From RESULT-3187 §3, every trace directly read in code this session:

| parameter | → recorded activity/time allocation | label |
|---|---|---|
| `personality.style_fit_amplitude` | **Verified.** Read at `gaworld/personality/plugin.py:203` into per-agent `tuning.amplitude` → consumed by `style_fit` (`gaworld/personality/traits.py:208–234`: `amplitude * strength * tanh(...)`) → the `trait_style_fit` component of `choose_action` (`gaworld/sim/_action.py:679–683`) → weighted action draw → realised `final_activity` recorded in episodes. Direct, single-hop to the activity record. Note `trait_modifier` uses `band` (default 0.25), not the amplitude, so the amplitude's economic reach is only via activity mix. | **Verified** |
| `personality.emotion_baseline.contagion_weight` | **Verified to emotion**: read per call in `social_influence` (`gaworld/sim/_cognition.py:202`, `cfg.get("contagion_weight", 0.06)`), gated on config present *and* non-empty rules traits (183–194); fallback 0.1 averaging when gated off (194–196). **Plausible to activity**: emotion feeds `low_mood_avoidance` (`state["emotion"] < 0.4` → +1.0 avoidance bonus, `_action.py:685`); ≥3 hops to `final_activity`, effect size unmeasured. | **Verified (emotion) / Plausible (activity)** |
| `economy.spending.daily_variance` | **Verified to spending, None to activity choice.** Sole read at `gaworld/economy/finance.py:2287–2295`: `hourly_cost *= _rng.uniform(1−v, 1+v)` inside expense estimation. The activity/action/location text enters that function as an *input* (expense keyword allocation, line 2297+); the parameter never feeds back into `choose_action`, schedules, or any recorded activity. It moves D1 (wealth) but **cannot** move a time-use dimension. | **Verified (spending) / None (activity)** |
| LLM temperature (provider default) | **Plausible.** Default 0.2 in `gaworld/settings/llm.py:52,122`; applied to sim calls that pass no per-call temperature (`gaworld/llm/providers.py:397–402`). Reaches activity only through LLM-generated text (routines, actions, digests) — the heuristic paths (`_heuristic_schedule`, fallback actions) are temperature-free. Magnitude unknown; measurable only with a live probe. | **Plausible** |

**Implication:** Θ reaches the time-use dimension through `style_fit_amplitude` (Verified) and contagion/temperature (Plausible). `daily_variance` reaches only wealth ⇒ it can stay in Θ as the wealth-side parameter but is **NOT a time-use lever**.

---

## 6. Architecture Comparison Validation (Verified repo)

### 6.1 Mechanism inventory — every behavioural effect of the `channels.rules` toggle

Gating is per-agent: `traits_of(agent, "rules")` returns neutral z-scores when the seeded record's `channels` list lacks `"rules"` (`gaworld/personality/traits.py:102–123`); the channel list is written at seed time from `plugin.py:197` (`_channels()`), consumed nowhere except through `traits_of`. Complete consumer inventory:

| # | site | effect when rules off |
|---|---|---|
| 1 | `gaworld/sim/_action.py:679–683` | `style_fit` → 0.0 exactly; `trait_style_fit` component key vanishes |
| 2 | `gaworld/sim/_action.py:595–598` | `decision_noise` jitter width multiplier `trait_modifier(agent,"decision_noise")` → 1.0 (base 0.18 unchanged) |
| 3 | `gaworld/sim/_action.py:830` | `impulse_gate` random-action bypass chance multiplier → 1.0 |
| 4 | `gaworld/behavior/dynamic.py:192` | `interrupt_threshold` multiplier → 1.0 (activity-derail resistance) |
| 5 | `gaworld/behavior/dynamic.py:333` | `spontaneity_chance` multiplier → 1.0 |
| 6 | `gaworld/behavior/dynamic.py:588, 602` | `has_traits` false → social-encounter branch flips from trait modifier to the keyword fallback (`is_extrovert` +0.10); **a branch change, not just a neutral multiplier** |
| 7 | `gaworld/behavior/dynamic.py:735–741` | `_classify_personality` archetype scoring skipped → profile-blob keyword classification |
| 8 | `gaworld/sim/_cognition.py:192–196` | emotion update reverts to raw contagion averaging with hard-coded 0.1 — the trait-anchored **set-point** and **N-dependent recovery** (`_emotion_setpoint`, `n_recovery_slope`, lines 200–204) are removed entirely |
| 9 | `gaworld/economy/finance.py:1366` | `wealth_drive` multiplier → 1.0 (`MODIFIERS` table `traits.py:244`: c 0.20, n 0.10) — savings/consumption/effort paths (finance.py:2377, 2512, 2794, 2901) lose trait tilt |
| 10 | `gaworld/infosources/diet.py:233` | `openness_z` → 0.0 (information-diet openness term) |
| 11 | `gaworld/travel/trigger.py:163` | travel-wanderlust openness term → 0.0 (travel module default-off, inert unless enabled) |
| 12 | `gaworld/personality/anchors.py:147` | prompt/voice channel anchoring — unaffected by rules (separate channels; prompt defaults off, `settings/personality.py:43`) |

Three mechanism *families* switch off at once: trait-conditioned action selection (#1–3, 7), trait-modified behaviour/economy dynamics (#4–6, 9–11), trait-anchored emotion (#8).

### 6.2 Hidden-change check

- **RNG consumption: no divergence (Verified repo).** `residual()` (`traits.py:172–181`) uses an isolated per-call `random.Random(blake2b(agent_id|name))` — a private stream that never touches the shared `random` module. With rules off, `style_fit` returns 0.0 *before* calling residual (traits.py:221–223), so only the private stream's call count changes; the shared stream sequence is untouched. The gated branches (#2, 3, 4, 5, 9) change multiplier *values*, not draw counts; #6's `random.random()` encounter draw executes in both branches (dynamic.py:604). The emotion path draws nothing. ⇒ two arms with the same seed see identical shared-RNG sequences.
- **Seeding/population: matched.** Trait values come from `data/agents_big5.csv` (frozen on disk) or, for unlisted agents, `prior_traits` with a **per-agent** stream `Random(seed*1000003 + agent_id)` — draw independent of roster position and of the toggle. Same seed CSV ⇒ identical trait *values* in both arms; the toggle changes only the record's `channels` list (`plugin.py:266–270`), which is read solely by `traits_of`.
- **Config keys:** nothing becomes required. `style_fit_amplitude`/`modifier_band`/`residual_ratio` are still seeded into `tuning` in M0 but become inert dials (documented intent, `settings/personality.py:9–11`). `voice` stays on and `prompt` stays off in both arms by default — the diary/voice channel is a controlled constant across arms.
- **Inert writes:** `_emotion_setpoint` caches into `agent.ext.big_five` only when the traits path runs (M1-only side cache, `_cognition.py:196–198`) — observation metadata, not behaviour.

### 6.3 Caveats (declare in any run manifest)

(a) The social-encounter branch (#6) changes *which fallback* supplies personality, so M0 is not literally "no personality influence" there — it is the pre-personality keyword behaviour, which is the documented contract. (b) `contagion_weight` remains live in *both* arms (config read at `_cognition.py:202` in M1, hard-coded 0.1 in M0's fallback) — moving it is within-architecture variation.

### 6.4 Verdict

**Clean, interpretable, matched-seed mechanism ablation: YES (Verified repo).** Both arms share the same seed CSV/initial population and identical shared-RNG sequences; M0 removes a documented mechanism bundle rather than a dial position; attribution is unconfounded within the rules channel. Expected effect locus is wealth via `wealth_drive` and, if the time-use scorer is built on normal-mode runs, activity composition via #1–5.

---

## 7. Revised Pilot Recommendation

1. **D1 demoted to a static sanity check.** The Monte Carlo verdict (§2.4) removes it as a frontier coordinate; report day-0 Gini against the anchor once per run as a plausibility check, never as a Pareto axis.
2. **Time-use error as the primary frontier coordinate** (multi-category, target vector §4.4). It **needs normal-mode runs** — cost implication: vs FF's ~1 call/agent/day, normal mode is ~1 call/agent/tick ⇒ the Phase 0.5 budget is void; re-estimate with a smaller cohort, fewer days, or a reduced tick grid (§10).
3. **A 2-dim frontier remains possible if wealth dynamics accrue on normal runs** — economy ticks in both modes (day-boundary hooks preserved in `_run_fast_forward_step`, generative_city_sim.py:2721–2745). Candidate dimensions: time-use error vs Gini error, both measured on the same normal-mode run. Θ = `style_fit_amplitude`, `contagion_weight`, temperature (+ `daily_variance` optional, wealth-side only per §5).
4. **Architecture pair: `channels.rules` on/off, matched seeds** (§6 — clean ablation, Verified).
5. **Smallest live probe redefined:** 2 configs × 2 seeds × a short normal-mode run on a reduced cohort. Cost estimate: tick grid × 1 call/agent/tick at ~7.4 s/call (**estimate, unmeasured** — must be measured by a micro-run before committing, §9 gate ii).

---

## 8. Remaining Scientific and Engineering Risks

- **Normal-mode cost possibly 5–20× FF (unmeasured)** — the single biggest budget unknown; measure before designing the pilot.
- **Time-use mapper coverage/keyword risk** — free-text activities may fall outside the keyword list; unmapped minutes must be reported, not silently dropped.
- **Population mismatch of survey targets** — conditional-scoring assumptions for paid work (59% NILF cohort vs participants-only averages; 2024 participants include part-time workers).
- **Tick-grid boundary inference errors** — durations inferred by timestamp differencing; boundary steps and skipped ticks are the documented hazard (§3.2).
- **Θ effect sizes still unknown** — only the wiring is Verified; magnitudes require the live probe.
- **Category target validity** — weekly-average vs workday bases differ between 2018 and 2024; the study category is child-inflated (§4.4 FLAG).
- **Scorer unvalidated against any run** — it exists only as a design; validate deterministically first (§9 gate i).

---

## 9. Go / No-Go Verdict

**Conditional GO on a redesigned pilot (time-use primary), with gates:**

1. **(i)** Build the keyword mapper + scorer and validate deterministically on synthetic episodes — **zero LLM cost**.
2. **(ii)** Measure normal-mode per-day call cost with a tiny authorized run before any pilot commitment.
3. **(iii)** D1 excluded as a frontier coordinate (noise-bound, Computed SD 0.0385 > headroom 0.058).

**NO-GO on the Phase 0.5 pilot as specified** (D1-primary 3-dim frontier).

---

## 10. Costs and Next Smallest Step

**Next step (zero LLM cost):** build the deterministic time-use keyword mapper + scorer from existing code (episode schema §3.1 already carries everything needed), then validate it on synthetic episodes. Engineering estimate **~0.5–1 day, 0 LLM calls**. Only after that: one authorized micro-run to measure normal-mode per-day cost (gate ii of §9), from which the pilot's cost table is computed.

---

*Report generated from RESULT-3187 and RESULT-3188; no repository code was modified in producing this document.*
