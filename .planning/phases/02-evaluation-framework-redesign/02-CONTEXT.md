# Phase 2: Evaluation Framework Redesign - Context

**Gathered:** 2026-04-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Redesign training/validation/testing to prevent leakage and enable robust multi-dimensional assessment. Build evaluation infrastructure for the **new LESR architecture** (fixed reward + structured feature library). Baseline code is exp4.9_c.

</domain>

<decisions>
## Implementation Decisions

### Architecture Direction (from reflection session)
- **D-01:** intrinsic_reward is decoupled — fixed as human-designed regime-based rules. LLM no longer generates reward functions.
- **D-02:** LLM output changes from free-form Python code to structured JSON selecting features from a predefined library (RSI, MACD, Bollinger, Momentum, etc.)
- **D-03:** This phase builds evaluation for the NEW architecture, not exp4.9_c as-is

### Walk-Forward Validation (EVAL-01)
- **D-04:** Use sliding-window walk-forward. exp4.9_c already has config_221_SW*.yaml and run_sliding_parallel.py — reuse and adapt this infrastructure

### Multi-Metric Assessment (EVAL-02)
- **D-05:** Extend DQNTrainer.evaluate() to compute Sharpe, Sortino, Max Drawdown, Calmar ratio, Win Rate. Current evaluate() only returns Sharpe/MaxDD/TotalReturn

### Factor Evaluation Metrics (新增 — 金融因子评估维度)
- **D-11:** metrics.py 除策略绩效指标外，还应包含**因子评估指标**：IC（Information Coefficient，因子值与未来收益的秩相关）、IR（Information Ratio，IC均值/IC标准差）、Quantile Spread（因子 Top 组 vs Bottom 组的收益差）。这些指标评估的是单个特征/因子的预测能力，而非策略整体绩效
- **D-12:** evaluate() 在返回策略绩效的同时，也返回每个特征维度的因子评估结果（IC, IR, quantile_spread per feature），使得 Phase 3 的特征筛选（LESR-04）有直接的量化依据
- **D-13:** 论文叙事中，"LLM 生成的特征具有显著 IC" 比 "LLM 策略 Sharpe 更高" 更有说服力——因子评估是核心学术贡献之一

### Data Leakage Prevention (EVAL-03)
- **D-06:** COT feedback must only use training-set analysis. Verify that _generate_cot_feedback() and get_iteration_prompt() do not pass validation/test metrics to LLM

### Regime-Stratified Evaluation (EVAL-04)
- **D-07:** Use existing regime_detector.py (3-dim: trend/volatility/risk) to label test periods, then report per-regime Sharpe and MaxDD. regime_detector is already in exp4.9_c

### Cross-Stock/Cross-Window Reports (EVAL-05)
- **D-08:** Build on Phase 1's analyze_existing.py to aggregate results across result_221_SW* directories into publication-ready comparison tables

### Feature Library Design
- **D-09:** v1 feature library: RSI, MACD, Bollinger_Band, Momentum, Volatility, Volume_Ratio, ROC, EMA_Cross, Stochastic_Osc, OBV, ATR, Williams_%R, CCI, ADX — 10-15 indicators with parameterized windows
- **D-10:** LLM outputs JSON: `{"features": [{"indicator": "RSI", "params": {"window": 14}}, ...], "rationale": "..."}`

### Claude's Discretion
- Exact feature library implementation (which indicators, default params)
- Fixed reward rule details and thresholds
- Report format (markdown tables vs LaTeX)
- Sliding window sizes and overlap

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Architecture Decisions
- `.planning/phases/02-evaluation-framework-redesign/02-ARCHITECTURE-REFLECTION.md` — Full architecture reflection with rationale for decoupling intrinsic_reward and structured feature library

### Existing Code (exp4.9_c baseline)
- `exp4.9_c/lesr_controller.py` — Current LESR optimization loop (to be redesigned)
- `exp4.9_c/dqn_trainer.py` — DQN trainer with regime conditioning (evaluate() needs extension)
- `exp4.9_c/regime_detector.py` — 3-dim regime vector (reusable as-is)
- `exp4.9_c/prompts.py` — Current prompt templates (to be redesigned for JSON output)
- `exp4.9_c/run_sliding_parallel.py` — Sliding window runner (reusable infrastructure)
- `exp4.9_c/config_221_SW01.yaml` — Example sliding window config

### Phase 1 Infrastructure (reusable)
- `exp4.7/diagnosis/analyze_existing.py` — Post-hoc analysis tool (already adapted for exp4.9_c formats)
- `exp4.7/diagnosis/stats_reporter.py` — Statistical comparison (t-test, bootstrap)
- `exp4.7/diagnosis/feature_quality.py` — Feature quality diagnostics

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `regime_detector.py`: 3-dim regime vector — directly reusable for EVAL-04
- `run_sliding_parallel.py` + `build_221_configs.py`: Sliding window infrastructure — 80% of EVAL-01 already exists
- `analyze_existing.py`: Handles exp4.9_c result_221_SW* directories — foundation for EVAL-05
- `stats_reporter.py`: Welch's t-test, BCa bootstrap — reusable for statistical reports

### Established Patterns
- State layout: [raw(120) + regime(3) + features(N)] — this remains, but features come from library not LLM code
- Config-driven experiments via YAML — extend this pattern for new architecture
- Subprocess-isolated training workers in `_train_ticker_worker()` — keep this pattern

### Integration Points
- New `feature_library.py` module needed — implements indicator calculations, called by dqn_trainer
- New `llm_feature_selector.py` needed — parses LLM JSON output, calls feature library
- `prompts.py` needs rewrite — from "generate Python code" to "select from feature library"
- `dqn_trainer.py` evaluate() needs extension — add Sortino, Calmar, Win Rate
- `metrics.py` 需要包含因子评估函数 — IC (spearman rank corr), IR (IC mean / IC std), quantile_spread — 这些是 numpy/scipy 操作，不需要外部依赖
- evaluate() 返回 dict 需要新增 `factor_metrics` 键 — 包含每个特征的 IC/IR/quantile_spread

</code_context>

<specifics>
## Specific Ideas

- Fixed reward should extend current `compute_regime_bonus()` (only 2 rules now) to ~5-6 rules covering risk management, trend following, and volatility dampening
- Feature library should normalize all outputs to [0,1] or z-scored to prevent scale mismatch
- LLM rationale field is important for paper writing — captures the "why" behind each feature selection
- IC 计算用 Spearman 秩相关（`scipy.stats.spearmanr`），不用 Pearson，因为金融因子和收益的关系经常是非线性的
- IR = mean(IC) / std(IC)，需要 rolling IC 先算出 IC 序列（默认 window=20 天），再取均值和标准差
- Quantile Spread: 将特征值排序分成 5 组，计算 Top 组 vs Bottom 组的下期收益差
- 因子评估函数签名建议：`ic(feature_values, forward_returns, method='spearman') -> float`、`rolling_ic(feature_values, forward_returns, window=20) -> np.ndarray`、`information_ratio(rolling_ic_series) -> float`、`quantile_spread(feature_values, forward_returns, n_quantiles=5) -> float`

</specifics>

<deferred>
## Deferred Ideas

- Custom feature proposal by LLM (outside the library) — future phase after v1 proves stable
- Ensemble DQN / multi-agent — independent improvement, not in scope
- Adaptive feature library (growing over iterations) — Phase 3 territory
- CPCV (Combinatorial Purged Cross-Validation) — v2 requirement

</deferred>

---

*Phase: 02-evaluation-framework-redesign*
*Context gathered: 2026-04-14*
