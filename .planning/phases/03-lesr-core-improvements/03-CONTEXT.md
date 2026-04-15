# Phase 3: LESR Core Improvements - Context

**Gathered:** 2026-04-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix prompts, quality gates, COT feedback, and feature selection to stabilize LESR performance. LLM-generated features must be economically meaningful, syntactically valid, dimensionally correct, and consistently outperform raw features across runs. Baseline code is exp4.9_c. Architecture follows Phase 2 decisions: LLM outputs structured JSON selecting from a predefined feature library (no longer generates Python code); intrinsic_reward is decoupled as fixed rules.

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
- **D-09:** 需新增 NaN/Inf 检查 — 当前 exp4.9_c 的 `_validate_code` 缺少此检查，特征中的 NaN 会传播到 DQN 训练

### COT 反馈机制 (LESR-03)
- **D-10:** COT 反馈包含策略绩效 + 因子级别评估 — 每个候选组合的 Sharpe/MaxDD/TotalReturn + 每个特征的 IC/IR
- **D-11:** 负面指导为具体拒绝原因 — 明确列出被拒绝特征及原因（如"RSI(14) IC=0.001 被拒绝"），加总结性"不要再选低 IC 的动量指标"指导
- **D-12:** 批量反馈 — 一次迭代所有候选评估完后统一生成反馈，能对比不同组合的优劣
- **D-13:** 确保调用 `check_prompt_for_leakage()` — 当前 exp4.9_c 已定义但从未调用

### 特征稳定性评估 (LESR-05)
- **D-14:** 稳定性评估使用固定子期间划分 — 将训练集分成 3-4 个等长时间段，计算每个特征在每个子期间的 IC
- **D-15:** 稳定性标准为 IC 均值 + 波动比 — 特征 IC 均值 > 阈值 且 IC 标准差 < 2 * IC 均值
- **D-16:** 不稳定特征直接移除，在 COT 反馈中告知 LLM 该特征不稳定

### Claude's Discretion
- 具体的 prompt 模板措辞和格式
- 主题包中每个指标的默认参数范围
- 特征库的具体实现（函数签名、输出归一化方式）
- 报告格式（markdown tables vs LaTeX）
- IC 阈值的具体数值微调

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Architecture Decisions (Phase 2 — locked)
- `.planning/phases/02-evaluation-framework-redesign/02-CONTEXT.md` — Feature library design (D-09, D-10), intrinsic_reward decoupling (D-01, D-02), evaluation metrics
- `.planning/phases/02-evaluation-framework-redesign/02-ARCHITECTURE-REFLECTION.md` — Full architecture reflection with rationale for decoupling and structured feature library

### Core Code to Modify (exp4.9_c baseline)
- `exp4.9_c/prompts.py` — Current prompt templates (lines 13-277) — **must rewrite** from Python code generation to JSON feature selection
- `exp4.9_c/lesr_controller.py` — LLM optimization loop (lines 218-255), code validation (lines 317-393), COT feedback generation (lines 490-524)
- `exp4.9_c/dqn_trainer.py` — State assembly (lines 167-182), evaluation (lines 332-459)
- `exp4.9_c/feature_analyzer.py` — Feature importance analysis (Spearman + SHAP)
- `exp4.9_c/metrics.py` — Performance + factor metrics including IC/IR/quantile_spread
- `exp4.9_c/regime_detector.py` — 3-dim regime vector (reusable as-is)

### Phase 1 Infrastructure (reusable)
- `exp4.7/diagnosis/analyze_existing.py` — Post-hoc analysis tool
- `exp4.7/diagnosis/stats_reporter.py` — Statistical comparison (t-test, bootstrap)
- `exp4.7/diagnosis/feature_quality.py` — Feature quality diagnostics

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `regime_detector.py` (exp4.9_c): 3-dim regime vector — directly reusable for market state summary in prompt
- `metrics.py` (exp4.9_c): IC/IR/quantile_spread already implemented — directly usable for feature pre-evaluation and stability assessment
- `feature_analyzer.py` (exp4.9_c): Spearman + SHAP importance — reusable for feature ranking
- `filter_cot_metrics()` + `check_prompt_for_leakage()` (exp4.9_c/lesr_controller.py): Leakage prevention already defined — needs activation

### Established Patterns
- JSON config-driven experiments via YAML — extend for feature library config
- `_validate_code()` pattern (exp4.9_c/lesr_controller.py:317-393) — adapt from Python code validation to JSON feature validation
- COT feedback rendering via Jinja2-style string templates (exp4.9_c/prompts.py) — keep template approach for new JSON-mode prompts
- Subprocess-isolated training workers — keep this pattern

### Integration Points
- `prompts.py` — complete rewrite needed: INITIAL_PROMPT, get_financial_cot_prompt, get_iteration_prompt
- `lesr_controller.py` — `_validate_code()` → `_validate_feature_selection()`, `_generate_cot_feedback()` → factor-level feedback
- `lesr_controller.py` — `_call_llm()` (line 199-216) — response parsing changes from Python code extraction to JSON parsing
- `dqn_trainer.py` — `_build_enhanced_state()` — feature assembly changes from calling `revise_state()` to looking up feature library
- New `feature_library.py` module needed — implements indicator calculations, called by dqn_trainer
- `lesr_controller.py` — must actually invoke `check_prompt_for_leakage()` before each LLM call

### Key Gaps
- No NaN/Inf validation on features at generation time
- `check_prompt_for_leakage()` defined but never called
- No feature library or catalog — everything generated from scratch each iteration
- `lesr_strategy.py` line 67 calls `revise_state()` directly without regime prepend — inconsistency with DQN training state format

</code_context>

<specifics>
## Specific Ideas

- Fixed reward extends `compute_regime_bonus()` from 2 rules to ~5-6 rules (risk management, trend following, volatility dampening) — already decided in Phase 2
- Feature library should normalize all outputs to [0,1] or z-scored to prevent scale mismatch
- LLM rationale field is important for paper writing — captures the "why" behind each feature selection
- Theme packs map naturally to financial theory categories, making the LLM's reasoning more interpretable for the paper
- Same-type indicator conflict resolution (keep higher IC) is a simple but effective deduplication strategy

</specifics>

<deferred>
## Deferred Ideas

- Custom feature proposal by LLM (outside the library) — future phase after v1 proves stable
- Adaptive feature library (growing over iterations) — v2 territory
- Rolling IC-based stability assessment — more granular but higher compute cost
- Market regime-stratified stability evaluation — could be added later

</deferred>

---

*Phase: 03-lesr-core-improvements*
*Context gathered: 2026-04-15*
