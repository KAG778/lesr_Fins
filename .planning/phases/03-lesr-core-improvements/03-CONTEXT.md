# Phase 3: LESR Core Improvements - Context

**Gathered:** 2026-04-15 (updated)
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix prompts, quality gates, COT feedback, and feature selection to stabilize LESR performance. LLM-generated features must be economically meaningful, syntactically valid, dimensionally correct, and consistently outperform raw features across runs. **Working directory is exp4.15** (clean copy from exp4.9_c). Architecture follows Phase 2 decisions: LLM outputs structured JSON selecting from a predefined feature library (no longer generates Python code); intrinsic_reward is decoupled as fixed rules.

</domain>

<decisions>
## Implementation Decisions

### Prompt 设计策略 (LESR-01)
- **D-01:** Prompt 为市场感知型 — 注入训练集基础统计摘要（波动率、趋势强度、平均成交量、日均收益率，~100 tokens）引导 LLM 根据市场状态选择适合的特征
- **D-02:** 迭代上下文为精选模式 — 仅提供上一轮的选择+COT反馈+历史最优组合信息（~2k tokens/迭代），不传完整历史
- **D-03:** 特征选择使用主题包引导 — 4个固定包，LLM 可跨包组合：
  - 趋势跟踪：RSI, MACD, EMA_Cross, Momentum
  - 波动率：Bollinger, ATR, Volatility
  - 均值回归：Stochastic, Williams_%R, CCI
  - 成交量：OBV, Volume_Ratio, ADX
- **D-04:** Prompt 重写为 JSON 输出模式 — 不再要求生成 Python 代码，而是要求选择主题包中的指标组合并附 rationale

### 质量控制与特征筛选 (LESR-02, LESR-04)
- **D-05:** 质量守门采用静态验证+因子预评估 — JSON 解析成功 + 指标在库中存在 + 参数范围合理 + 用训练数据计算 IC/方差，拒绝 IC≈0 或方差≈0 的特征
- **D-06:** 最终传给 DQN 的特征数量固定为 5-10 个
- **D-07:** 筛选标准为 IC + 方差，使用宽松阈值 — IC > 0.02 且方差 > 1e-6，给 LLM 更多尝试空间
- **D-08:** 同类型指标冲突时保留强者 — 如 RSI(14) 和 RSI(21) 同时被选中，保留 IC 更高的那个
- **D-09:** 需新增 NaN/Inf 检查 — 特征中的 NaN 会传播到 DQN 训练

### COT 反馈机制 (LESR-03)
- **D-10:** COT 反馈包含策略绩效 + 因子级别评估 — 每个候选组合的 Sharpe/MaxDD/TotalReturn + 每个特征的 IC/IR
- **D-11:** 负面指导为具体拒绝原因 — 明确列出被拒绝特征及原因（如"RSI(14) IC=0.001 被拒绝"），加总结性"不要再选低 IC 的动量指标"指导
- **D-12:** 批量反馈 — 一次迭代所有候选评估完后统一生成反馈，能对比不同组合的优劣
- **D-13:** 确保调用 `check_prompt_for_leakage()` — 已定义但从未调用，必须激活

### 特征稳定性评估 (LESR-05)
- **D-14:** 稳定性评估使用固定子期间划分 — 将训练集分成 3-4 个等长时间段，计算每个特征在每个子期间的 IC
- **D-15:** 稳定性标准为 IC 均值 + 波动比 — 特征 IC 均值 > 阈值 且 IC 标准差 < 2 * IC 均值
- **D-16:** 不稳定特征直接移除，在 COT 反馈中告知 LLM 该特征不稳定

### 特征库设计 (Phase 3 新增决策)
- **D-17:** 特征归一化使用 Z-score 标准化 — 对每段训练数据计算均值/标准差，输出均值 0、方差 1
- **D-18:** 指标参数化为连续参数 — LLM 可自由指定 window/range 值，系统验证在合理范围内（如 window ∈ [5, 60]）
- **D-19:** 特征库实现为纯 Python + NumPy — 无外部 ta-lib 或 pandas_ta 依赖，完全可控易调试
- **D-20:** 特征库范围扩展到 20+ 个指标 — 在 Phase 2 定义的 14 个基础上增加均线类（SMA_Cross, DEMA）、动量类（Williams_Alligator）、统计类（Skewness, Kurtosis）等
- **D-21:** JSON→函数组装使用闭包方式 — 根据 LLM 返回的 JSON 选择方案，运行时通过闭包动态生成 `revise_state` 函数，无需 exec/eval

### 固定奖励规则 (Phase 3 新增决策)
- **D-22:** intrinsic_reward 解耦为 5-6 条固定规则 — 风险规避、趋势跟随、波动率抑制、动量保护、均值回归。替代当前 LLM 生成的 intrinsic_reward 函数
- **D-23:** 奖励权重保持 — intrinsic_weight=0.02, regime_bonus_weight=0.01，合并为统一的固定奖励函数

### 迭代超参数 (Phase 3 新增决策)
- **D-24:** 迭代配置为 3 候选 × 5 轮，固定轮数停止 — 每轮 LLM 采样 3 个候选方案，共 5 轮迭代，总评估 15 次

### Claude's Discretion
- 具体的 prompt 模板措辞和格式
- 主题包中每个指标的默认参数范围
- 报告格式（markdown tables vs LaTeX）
- IC 阈值的具体数值微调
- 扩展指标的精确列表和参数范围
- 5-6 条固定奖励规则的具体阈值和奖励值

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Architecture Decisions (Phase 2 — locked)
- `.planning/phases/02-evaluation-framework-redesign/02-CONTEXT.md` — Feature library design (D-09, D-10), intrinsic_reward decoupling (D-01, D-02), evaluation metrics
- `.planning/phases/02-evaluation-framework-redesign/02-ARCHITECTURE-REFLECTION.md` — Full architecture reflection with rationale for decoupling and structured feature library

### Core Code to Modify (exp4.15 working directory)
- `exp4.15/core/prompts.py` — Current prompt templates — **must rewrite** from Python code generation to JSON feature selection
- `exp4.15/core/lesr_controller.py` — LLM optimization loop, code validation, COT feedback generation — adapt for JSON mode
- `exp4.15/core/dqn_trainer.py` — State assembly, evaluation, reward computation — integrate fixed reward + feature library
- `exp4.15/core/feature_analyzer.py` — Feature importance analysis (Spearman + SHAP) — reuse as-is
- `exp4.15/core/metrics.py` — Performance + factor metrics (IC/IR/quantile_spread) — reuse as-is
- `exp4.15/core/regime_detector.py` — 3-dim regime vector — reuse as-is
- `exp4.15/core/baseline.py` — DQN baseline — minor import path updates
- **New `exp4.15/core/feature_library.py`** — Feature library module: indicator implementations + JSON→function assembler

### Phase 1 Infrastructure (reusable)
- `exp4.7/diagnosis/analyze_existing.py` — Post-hoc analysis tool
- `exp4.7/diagnosis/stats_reporter.py` — Statistical comparison (t-test, bootstrap)
- `exp4.7/diagnosis/feature_quality.py` — Feature quality diagnostics

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `regime_detector.py` (exp4.15/core): 3-dim regime vector — directly reusable for market state summary in prompt
- `metrics.py` (exp4.15/core): IC/IR/quantile_spread already implemented — directly usable for feature pre-evaluation and stability assessment
- `feature_analyzer.py` (exp4.15/core): Spearman + SHAP importance — reusable for feature ranking
- `filter_cot_metrics()` + `check_prompt_for_leakage()` (exp4.15/core/lesr_controller.py): Leakage prevention already defined — needs activation
- `compute_regime_bonus()` (exp4.15/core/dqn_trainer.py:184-197): Current 2-rule bonus — extend to 5-6 rules

### Established Patterns
- JSON config-driven experiments via YAML — extend for feature library config
- `_validate_code()` pattern — adapt from Python code validation to JSON feature validation
- COT feedback rendering via string templates (exp4.15/core/prompts.py) — keep template approach for new JSON-mode prompts
- Subprocess-isolated training workers — keep this pattern
- `sys.path.insert` pattern for core/ sibling imports — already set up in exp4.15

### Integration Points
- `prompts.py` — complete rewrite: INITIAL_PROMPT, get_financial_cot_prompt, get_iteration_prompt → JSON mode
- `lesr_controller.py` — `_extract_code()` → `_extract_json()`, `_validate_code()` → `_validate_selection()`
- `lesr_controller.py` — `_call_llm()` — response parsing from Python code to JSON
- `lesr_controller.py` — `_train_ticker_worker()` — no more tempfile/importlib, use feature_library closure
- `dqn_trainer.py` — `compute_regime_bonus()` → unified fixed reward with 5-6 rules
- `dqn_trainer.py` — `_build_enhanced_state()` — revise_state_func source changes but interface stays same
- **New `feature_library.py`** — indicator functions + `build_revise_state(selection_json)` closure assembler
- `lesr_controller.py` — must invoke `check_prompt_for_leakage()` before each LLM call

### Key Gaps (confirmed by code analysis)
- No NaN/Inf validation on features at generation time
- `check_prompt_for_leakage()` defined but never called (line 83-98 in lesr_controller.py)
- No feature library or catalog — everything generated from scratch each iteration
- `lesr_strategy.py` `on_data()` calls `revise_state()` directly without building full enhanced_state (raw + regime + features) — must fix to use `_build_enhanced_state` equivalent
- `intrinsic_reward` still expects LLM-generated function — must replace with fixed rules

### Data Flow (current → target)
```
CURRENT:
LLM → Python code string → _extract_code() → _validate_code() → module.revise_state → DQN
LLM → Python code string → intrinsic_reward function → DQN reward

TARGET:
LLM → JSON selection → _extract_json() → _validate_selection() → feature_library.build_revise_state(json) → DQN
Fixed rules → compute_fixed_reward(regime, action) → DQN reward (no LLM)
```

</code_context>

<specifics>
## Specific Ideas

- Fixed reward extends `compute_regime_bonus()` from 2 rules to ~5-6 rules (risk management, trend following, volatility dampening, momentum protection, mean reversion)
- Feature library uses Z-score normalization to prevent scale mismatch across indicators
- LLM rationale field is important for paper writing — captures the "why" behind each feature selection
- Theme packs map naturally to financial theory categories, making the LLM's reasoning more interpretable for the paper
- Same-type indicator conflict resolution (keep higher IC) is a simple but effective deduplication strategy
- Closure-based function assembly avoids exec/eval — safer and debuggable
- 3 候选 × 5 轮 gives enough exploration while keeping compute cost manageable (15 total evaluations)
- exp4.15 uses core/ subdirectory — all internal imports use sys.path.insert to core/

</specifics>

<deferred>
## Deferred Ideas

- Custom feature proposal by LLM (outside the library) — future phase after v1 proves stable
- Adaptive feature library (growing over iterations) — v2 territory
- Rolling IC-based stability assessment — more granular but higher compute cost
- Market regime-stratified stability evaluation — could be added later
- Early stopping mechanism for optimization loop — could replace fixed rounds if needed

</deferred>

---

*Phase: 03-lesr-core-improvements*
*Context gathered: 2026-04-15 (updated with feature library, reward rules, iteration config decisions)*
