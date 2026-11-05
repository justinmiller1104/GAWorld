# MCF × GAWorld Phase 0.5 科学验证报告

**Status:** FINAL · 2026-11-05 · 基于 repo `80f8c1f`（只读验证，未改任何代码，未跑任何仿真、未花任何模型调用）
**来源:** 全部事实来自 RESULT-3179（Phase 0.5 验证研究，file:line 引用）；纸面协议引自 RESULT-3171（paper-zhao-2026-mcf）。
**标签约定:** 每条关键论断标注 —
**Verified (repo)** = 直接代码阅读 file:line 引用 · **Verified (paper)** = 论文协议原文 ·
**Verified (computed)** = 本验证实际执行的零成本计算 · **Assumption** = 未测量的合理推断 ·
**Proposed** = 建议但未实现。

---

## 1. Executive Summary

**拟议的 pilot 按现有设计是根本性缺陷的（fundamentally flawed as designed）。**

- **D2 与 D3 是无状态 persona 问答，对 Θ 恒定。** D2: `paradigms.py` 模块文档写明 "each in its **own stateless call**, so no answer can see another"（paradigms.py:12–13, **Verified (repo)**），且 `ClassicSpec.temperature = 1.0`（classics/runner.py:36, **Verified (repo)**）；D3: `replication.py` 从**初始** roster + profile MD 构建 prompt（replication.py:956, 1000–1005, **Verified (repo)**），`temperature=0.9` 硬编码于 `replication_api.py:121`（**Verified (repo)**）。连温度轴都够不到它们。
- **只有 D1 能动，而其唯一 Verified 的参数链路是 `daily_variance`**（finance.py:2287–2295 → 余额 → snapshot, **Verified (repo)**）。
- **零成本 Monte-Carlo 探针显示 day-0 Gini 被输入主导**：33 人劳动力的初始财富 Gini ≈ 0.441（n=33, income × U(1,6) 一次抽样, **Verified (computed)**），相对重定锚目标 0.45–0.5 的可测动态余量 ≤ 0.06；种子噪声（估计 SD ≈ 0.03–0.06, **Assumption**）很可能与 Θ 效应同量级。
- **因此：当前 3 个维度下不可能存在 Pareto 前沿。** 依赖矩阵只有一个活跃列 ⇒ 无权衡 ⇒ 无前沿（§3）。

**裁定（Proposed）：**
1. **不要按现有设计运行 pilot。** 改用重排序的 Phase 1 便宜验证门：免费的 Monte-Carlo 噪声地板 → 4,800 调用的小探针 → 通过后才做前沿工作（§5）。
2. 架构对比对采用 `personality.channels.rules` on/off（§6）。
3. 候选第二维：确定性时间利用（time-use）评分器替代 D2/D3（§7）。

**对现有可行性文档的两处修正（均 Verified (repo)，属文档过时警告，非代码缺陷）：**
- **经济在 fast-forward 下确实会 tick。** §4.1 的 "economy-under-FF unverified" 是错的：`_run_fast_forward_step`（generative_city_sim.py:2721–2745）明确保留日界钩子（growth/interests/economy），`coarse=True` 按块记近似工资（2743–2745）。
- **run manifest 现在会记录 LLM 总量。** `run_manifest.py:373–420`（`bind_llm_stats`, `_run_scoped_llm`）已存在；文档的 `llm: {}` 警告反映的是归档的 partial manifest，不是当前代码（磁盘上尚无完成的 manifest 可比对，见 §9）。

---

## 2. D1/D2/D3 科学效度评估

### D1 — 劳动力财富 Gini：Suitable with modifications（弱；短期内主要是输入伪象）

**Verified (repo) 计算链路：**
- 评分函数 `benchmark/gaworld_bench.py::gini`（line 187），在 `track_a_macro_fit`（line 203）内，作用于 `LABOUR_FORCE = {"employed","unemployed"}`（line 87），净财富 = `balance + housing_fund − debt`，读 `output/<run>/economy/wealth_snapshot.csv`（~237–247 行）。
- 初始财富在 `gaworld/economy/finance.py:2155–2156` 设为 月净收入 × U(`initial_savings_months_min`=1.0, `initial_savings_months_max`=6.0)（默认值, lines 89–90）。`METRIC_PROVENANCE["wealth_gini"]`（gaworld_bench.py:101–104）自评为 "c"：“开局存款 = 月净收入 × U(1, 6) 个月（initial_savings_months_*，定的）”。
- Θ 可达的初始后动态：`wealth_drive` × `trait_modifier(agent, "wealth_drive")`（finance.py:1350–1371, 1366；traits 定义于 `gaworld/personality/traits.py:259`）驱动储蓄/投资/努力路径（finance.py:2377, 2512, 2794, 2901）。日消费噪声由 `economy.spending.daily_variance` 缩放（finance.py:2287–2295, 小时成本上 `_rng.uniform(1−v, 1+v)`）。情绪仅通过行动选择到达消费（`state["emotion"] < 0.4` 时的 `low_mood_avoidance`, `gaworld/sim/_action.py:685`）。

**输入伪象量化（Verified (computed)，本验证在 pilot cohort `data/cities/绍兴柯桥/agents.csv`, n=80 上实际执行）：**
- 就业列：33 employed，**0 unemployed**，47 `not_in_labor_force` ⇒ Gini 实际作用的人口是 **33 人**（LABOUR_FORCE 的 unemployed 半边在这份数据上是死代码）。
- 劳动力月收入 Gini = **0.395**（均值 ¥8,524，范围 ¥1,347–36,951）。全队列收入 Gini = 0.699。
- 一次初始财富抽样（income × U(1,6), seed 1）：**Gini ≈ 0.441**。

**后果：**
1. 原来的 0.6–0.75 家庭财富锚**不可达也无意义**——80 名居民的原始收入离散度已经是 0.70，无需任何仿真动态。
2. 即使采用建议的收入 Gini 重定锚（0.45–0.5），也只比 day-0 输入值 0.44 高 ~0.06–0.10——30 天仿真动态必须在锚差距内移动 Gini，D1 的"拟合"大部分就是输入抽样本身（RESULT-3176 指出的循环性，现在被量化了）。
3. n=33 时 Gini 估计量的种子间 SD 估计 ≈0.03–0.06（**Assumption**，33 个相关的 U(1,6) 乘数抽样叠在偏斜收入向量上）。`daily_variance` [0.10, 0.40] 或 `style_fit_amplitude` [0.10, 0.40] 扫 30 个 FF 天产生的财富位移同级或更小——**config range ≲ seed noise**，这正是文档自己的 §4.5 放弃条件，很可能在任何前沿存在之前就触发。

**保留 D1 所需的修正（Proposed）：** 把目标重定为在**同一人口定义**上测得的收入 Gini 锚（城镇就业劳动力、工资性收入）；诚实申报锚差距预期（输入基线 0.44 vs 目标 0.45–0.5 ⇒ 可测动态余量 ≤ 0.06）；扩大评分人口（或声明保留地纳入 `not_in_labor_force`，或构造含失业者的队列——当前队列一个失业者都没有）。

**FF 修正（Verified (repo)）：** D1 在 FF 下确实近似累积（§1 的 generative_city_sim.py:2721–2745）——必须写明"近似"，而不是当作未知。

### D2 — 经典范式：Unsuitable as an MCF dimension（无状态；对 Θ 恒定）

**Verified (repo)：**
- `paradigms.py:12–13`（模块文档）："Each resident answers **every** condition, each in its **own stateless call**, so no answer can see another."
- Prompt = `SYSTEM`（paradigms.py:44–47, "你正在扮演一位真实的中国城市居民…"）+ `subject_block(subject)`（paradigms.py:355–370）：姓名/年龄/性别/职业/居住地/月收入、`work_rhythm`、`personality`、`daily_life` —— 全是静态属性。
- 属性来自 `load_subjects`（`gaworld/experiments/subjects.py:120–130`），读 `CONFIG["csv_path"]`/`CONFIG["md_path"]`——**初始**种子 CSV 和 profile Markdown，不读 `output/memory/`、不读 run state、不读关系、不读经济。不存在从 run 中晋升/选拔被试；被试就是"按 id 顺序的居民"。
- 温度被 runner 钉死：`ClassicSpec.temperature: float = 1.0`（`classics/runner.py:36`），逐调用传入（`call_llm(..., temperature=spec.temperature)`）；逐调用温度覆盖 provider 默认（`gaworld/llm/providers.py:397–402`："A per-call temperature wins over the provider default"）。

因此：改变 `contagion_weight`、`daily_variance`、`style_fit_amplitude` 或 LLM provider 默认温度，D2 的**每一个输入字节都不变**。D2 在 Θ 上是常数（温度 1.0 的抽样噪声除外——那是纯噪声测量，不是信号）。**裁定：Unsuitable**（对 Θ 敏感性而言；作为 persona 渲染的静态基准仍然合适）。

### D3 — 复制保真度：Unsuitable（同一无状态机制用第二遍；与 D2 冗余）

**Verified (repo)：**
- `gaworld/research/replication.py::run_replication`（line 956）从 `load_population(city)`（roster CSV）+ `profile_block` → `persona_block(card)`（lines 1000–1005）构建每个受访者。`answer_prompt`（line 522）= persona 文本 + `你正在参加{kind}` + 【说明】+【你看到的材料】+【问题】。无记忆、无仿真状态、无 run 输出进入 prompt。
- 应答调用在 `gaworld/apps/replication_api.py:121` **硬编码 `temperature=0.9`**——同样是 Θ 温度轴够不到的逐调用覆盖。
- 种子与样本在流程内抽取（`random.Random(seed).sample(...)`, ~line 1010），来自同一静态 roster。

D3 与 D2 是同一无状态 QA 机制配不同评分壳（`compare`/`contrast`, replication.py:648/745）。它检测不到任何配置差异，且不独立于 D2——它是 D2 的孪生。文档说 D2/D3 "survive the FF proxy intact" 是真的但无关紧要：它们存活恰恰因为与仿真断连，这正是它们不能当前沿坐标的原因。**裁定：Unsuitable。**

---

## 3. 参数-结果依赖矩阵

**Verified (repo) 追踪注记：** `style_fit_amplitude` 只在 `gaworld/personality/plugin.py:203` 被读入 tuning dict（`amplitude`），由 `choose_action` 中的 `style_fit` 消费（`gaworld/sim/_action.py:679–683`——"Personality enters the decision loop here and nowhere else in this function"）；`trait_modifier` 用 `modifier_band` 而非 amplitude，故 amplitude 的经济触达只经活动组合。`contagion_weight` 在 `social_influence` 中逐调用读取（`gaworld/sim/_cognition.py:202`, `cfg.get("contagion_weight", 0.06)`），门控于 `personality.enabled` + `emotion_baseline.enabled` + 非空 rules traits（lines 183–194）；硬编码 0.1（line 195）只是 personality 关闭时的 fallback——**该设置是接通的，不是死的**（见 §6 注意）。LLM 温度默认在 `gaworld/settings/llm.py:52,122`（0.2），作用于不传逐调用温度的 sim 调用（管线的多数；`providers.py:1335–1339`）。

| 参数 ↓ / 维度 → | D1 wealth Gini | D2 classics | D3 replication |
|---|---|---|---|
| `personality.style_fit_amplitude` [0.10, 0.40] | **Plausible but unverified to Gini**（Verified 到行动选择：`_action.py:679–683`；活动→收入/支出→财富的剩余跳数是多步且未量化） | **None**（输入是静态 CSV/profile；参数不被读取） | **None**（同） |
| `personality.emotion_baseline.contagion_weight` [0.02, 0.12] | **Plausible but unverified**（Verified：情绪更新 `_cognition.py:202`；Plausible：情绪→`low_mood_avoidance` `_action.py:685`→活动→消费；≥3 跳，无测得的效应量） | **None** | **None** |
| `economy.spending.daily_variance` [0.10, 0.40] | **Verified**（直接代码路径 finance.py:2287–2295 → 余额 → snapshot；效应量未测但一阶） | **None** | **None** |
| LLM temperature（provider 配置）[0.0, 0.7] | **Plausible but unverified**（接线已验证：provider 默认作用于 sim 调用, `providers.py:397–402,1339`；对 Gini 的量级未知） | **None**（钉死 1.0, runner.py:36） | **None**（钉死 0.9, replication_api.py:121） |

**Pareto 裁定：** 只有一列是活的，且其唯一 **Verified** 行是 `daily_variance`。不存在响应**不同**参数的维度对，故**不可能有真正的权衡，不存在 Pareto 前沿**——每个点都将在单一标量上比较。文档自己的 §7 写明 MCF "adds value only for trade-offs across ≥ 2 dimensions"；拟议的 pilot 失败于该前提。Θ 之内没有任何东西能在单维内制造张力：`daily_variance` 增大噪声（Gini 保真更差，微观行为现实性 arguably 更好），但没有第二个坐标来登记这一增益。

---

## 4. 现有数据与复用机会

**盘点结果（Verified (repo)，本验证实际检查磁盘）：基本无可复用。**
- 经济输出只存在于一个城市：`output/cities/wuzhen/economy/wealth_snapshot.csv` —— **2 个 agent 行**（文件共 3 行），来自 partial run；`output/cities/wuzhen/run_manifests/` 的全部 12 个 manifest 都是 `.partial.json`，`"llm": []`，`"outcome": "in_progress"`。`output/` 下不存在 伦敦/绍兴柯桥 的经济输出。
- `output/experiments/classics/` 不存在（磁盘上无 classics 结果）；`output/research/` 不存在（从未跑过 replication）。
- `output/comparisons/` 有 164 个目录，全部 `simtree-*` 带 `divergence_metrics.csv`——compare-event 状态增量，无 wealth snapshot，且无差异化 Θ manifest 可对比。
- **结论：不存在零 LLM 调用的重评分可能；任何验证都需要新 run。**

**成本遥测修正（Verified (repo)）：** run 范围 LLM 总量的 manifest 机制**已存在**——`gaworld/core/run_manifest.py:373–420`（`bind_llm_stats`, `_llm_stats_provider.snapshot()`, `_run_scoped_llm`）。文档 §4.1 的 `llm: {}` 警告反映归档 partial manifest，不是当前代码；完成的 manifest 应携带总量，成本轴大概率已可测（未在完成的 run 上验证——磁盘上不存在）。

**静态无 LLM 探针（已完成，零成本，已有信息量）：** 解析地重推 D1 的输入——队列 CSV 的收入 Gini 与 income×U(1,6) 抽样的 Gini（§2）。这是最便宜的敏感性测试，且已显示 day-0 Gini 被输入主导。稍大的版本——对 33 人劳动力做 200 次 U(1,6) 初始化的 Monte-Carlo 抽样、算 Gini 分布——只需几秒，即给出 D1 的**种子噪声地板**，不花一次调用。若该地板的 SD（估计 0.03–0.06）超过合理的 Θ 效应，D1 在花一次调用之前就是噪声受限的。

---

## 5. 最便宜的有效验证实验

**三步重排序验证门（Proposed）：**

1. **Monte-Carlo D1 噪声地板** —— 200 次 income × U(1,6) 抽样（33 人劳动力），秒级、零成本、无 LLM。输出 Gini 分布的 SD = D1 的种子噪声地板。
2. **最小活体探针** —— 2 个极端配置（`daily_variance` 0.10 vs 0.40，其余默认）× 3 seeds × 10 FF 天 × 80 agents = **4,800 调用**；按 1 调用/agent/天（FF）与 RESULT-3176 的锚（7.4 s/调用，8 workers）≈ **1.5 h 墙钟**；廉价国产模型上 token 成本 trivial（≪ pilot 单腿的 12,800 调用 smoke）。用 `track_a_macro_fit` 离线确定性评分。
3. **判定规则：** 若配置间 Gini range < 2× 种子 SD（可从 3 个 seeds 加 Monte-Carlo 地板估计），则 D1 在任何可负担的 n 下都是噪声受限 ⇒ pilot 零活维 ⇒ **pilot 按现有设计 NO-GO**（不是"把 seeds 提到 5"：唯一维度的 range/σ < 1 时，文档 §4.5 的放弃规则本来就会触发）。按 §2 的数字，预期结果是规则触发。**此探针应先于任何 16-config 投资。**

**同时明确：不要在 Θ 变化下花调用跑 D2/D3 —— 证明性地无信息**（prompt 逐字节相同；只有温度 1.0/0.9 的抽样噪声不同）。

---

## 6. 拟议的双架构对比（MCF 的 matched-config A(M1|M0) 协议）

**先解决 contagion 注意事项：** `contagion_weight` **不是**架构开关。它在 CONFIG 中逐调用读取（`_cognition.py:202`），默认 0.06，文档标注为缺陷修复（"原来写死是 0.1", `gaworld/settings/config_docs.py:2535`）。`_cognition.py:195` 的硬编码 0.1 只是 personality config/traits 缺失时才走的 *fallback 分支*。在 [0.02, 0.12] 内移动 `contagion_weight` 是**同一架构内的参数变化**。RESULT-3176 的注记关乎血统，不是死接线——该设置是活的。

**推荐架构对：personality `channels.rules` on/off（Proposed，定义 Verified (repo)）。**
- M1 = `personality.channels.rules = true`（默认, `gaworld/settings/personality.py:24`）；M0 = `rules: false`。
- 为什么是架构级而非参数级：settings 模块自己的契约——"Turning a channel off yields exactly the pre-personality behaviour on that path"（`settings/personality.py:9–11`）——即 M0 移除的是**机制**（特质条件化行动选择、特质修正的财富驱动、特质锚定的情绪定点），不是某个拨盘位置。rules 关闭时：`style_fit` 恰好贡献 0.0 且组件键消失（`_action.py:673–683`，"the draw is bit-identical to the pre-personality build"）；`trait_modifier` 返回 1.0（`traits.py:259–280`，空 traits）；`traits_of(agent,"rules")` 为空，`social_influence` 完全丢弃定点/恢复机制，退回带硬编码 0.1 的原始平均（`_cognition.py:194–196`）。三个机制一次关断，且播种通道（`plugin.py:194–199`）保证 M0 的 traits 是缺失而非置零。
- **Matched-config 协议（Verified (paper)）：** 对共享 Θ 样本中的每个 θ（4 个参数、同样范围），跑两个架构；按论文比较种子平均误差向量——A(M1|M0) = 新可达且不被任何 M0 点弱占优的点——外加 matched-θ 配对统计，复用 `bootstrap_ci`/`sign_flip_p`（`gaworld/parallel/causal.py`）。
- 预期效应落点：D1 经 `wealth_drive`（finance.py:1366）与活动组合；D2/D3 上的诊断预期为**平**（它们对两个架构都是盲的——§7 说明这意味着什么）。

**已评估并否决的候选（均 Verified (repo)）：**
- (a) `contagion_weight` —— config，不是架构（上文）；
- (b) memory consolidation —— 暴露的旋钮是 `behavior.daily_consolidation_top_k`（`settings/behavior.py:193`），容量拨盘；consolidation 的 `enabled` 门（`memory/consolidation.py:159`，默认关）切换的子系统几乎没有已发表目标触及；
- (c) `HUMAN_REALISM_ENABLED`（`_cognition.py:33`，import 时读取一次）是真实的机制开关，但同时改变社会抽样**和**传染加权，归因混淆；且 import-time 读取；
- (d) needs-driven intentions（`cognition/realism.py::build_daily_intentions`）vs uniform 是可辩护的对，但其效应落在 diary/routine 文本上，没有对应的外部误差维度。

`channels.rules` 是唯一干净、被文档标注为 exact-ablation、且接通唯一活维（D1）的候选。次优的诚实对比对（静态、无仿真）：LLM provider 交换——这是 D2/D3 唯一响应的轴，也是 §7 建议把它们降级到这个对比而非前沿的原因。

---

## 7. 修订后的实验设计

**替代维度提案（Proposed）：** 确定性**时间利用 / 活动构成评分器**，对仿真状态日志评分，对照《中国居民时间利用调查》公开发布的总量（2018/2024，就业成人的全国时间预算份额：睡眠、有酬劳动、家务、休闲）。理由：状态路径存在（`state/` 输出，活动流驱动 `_estimate_behavior_expenses`）；评分器是纯代码（无 LLM）；目标是真实、独立、按活动类别多维——且它经由行动选择响应全部 4 个 Θ 参数，为前沿补上缺失的第二个坐标和可能的真张力（例如 `style_fit_amplitude` 可能改善时间利用拟合而 `daily_variance` 恶化 Gini 保真）。

**若评分器不建的诚实替代（Proposed）：**
- 保留 D1 为唯一前沿坐标，跑 §5 探针，若 D1 存活，做 MCF 但以 1-D 向量**诚实标注为 dose sweep**；
- 或放弃 MCF，改用现有 `dose_response` + §6 的架构对、仅以 D1 评分（这正是论文协议减去前沿表演）。

**旗标（Assumption / 待验证）：** 时间利用评分器提案需要先确认默认 run 输出中**确实会产出按天的活动日志**（state CSV schema 在 RESULT-3179 中未重新核验）。§9 列为残余 blocker。

---

## 8. 估算成本与工程量

| 项 | 成本 | 性质 |
|---|---|---|
| Monte-Carlo D1 噪声地板 | **免费**（秒级、纯代码、零 LLM） | Verified (computed)（小版已做；200 抽样版 Proposed） |
| 最小活体探针 | **4,800 调用 / ≈1.5 h 墙钟 / token 成本 trivial**（≪ pilot 单腿的 12,800 调用 smoke） | Proposed（设计见 §5；未经授权不花调用） |
| 时间利用评分器 | **~2–4 person-days** | Proposed |
| 门通过后的完整修订 pilot | 复用此前 16–22 person-day 估计，**减去现已不需要的 D2/D3 glue** | Proposed |

（4,800 调用/1.5 h 的换算基于 1 调用/agent/天的 FF 速率与 RESULT-3176 的 7.4 s/调用、8 workers 锚。）

---

## 9. 风险与残余 blocker

1. **误导性 3-D 前沿风险** —— 按现有设计跑 pilot 会产出一个"三维前沿"，其中两个额外坐标是常数噪声；论文的非对称解释规则救不了动态范围为零的维度。**由 §5 验证门消解**（前沿工作被排在两个便宜检查之后）。
2. **D1 噪声受限风险** —— 基于 §2 的数字，**预期结果**是 §5 的放弃规则触发。
3. **活动日志 schema 未验证** —— 时间利用评分器依赖按天活动日志存在于默认 run 输出；未核验。
4. **完成 manifest 的 LLM 总量** —— 代码级已验证（`run_manifest.py:373–420`）但未对磁盘上的完成 manifest 验证（不存在）。在第一个完成的 run 上核对。
5. **Gini 噪声数字是解析估计** —— 待 §5 的 Monte-Carlo 探针替换为实测分布。

---

## 10. Go / No-Go 建议

**Proposed 裁定：**

- **对 pilot 按现有设计：Conditional NO-GO。** D2/D3 对 Θ 无状态、恒定；D1 输入主导、余量 ≤ 0.06、种子噪声 ≈ 效应量；3 个维度下不存在 Pareto 前沿。
- **对重排序的便宜验证门：GO。** 先免费 Monte-Carlo 噪声地板，再 4,800 调用/1.5 h 探针，二者之后才允许任何前沿投资。
- **明确指令：不要在 Θ 变化下花任何调用跑 D2/D3** —— 证明性地无信息（prompt 恒定，仅温度 1.0/0.9 的抽样噪声不同）。若需要 D2/D3，把它们降级为静态 matched-config 检查（例如 §6 的 provider 交换对比）。
- 架构对比对采用 `personality.channels.rules` on/off；第二维候选采用确定性时间利用评分器，前提是 §9-3 的活动日志旗标先被核验。
