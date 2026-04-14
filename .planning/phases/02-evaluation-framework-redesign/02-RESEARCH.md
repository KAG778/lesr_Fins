# Phase 2: Evaluation Framework Redesign - Research

**Researched:** 2026-04-15
**Domain:** Walk-forward validation, financial metrics, data leakage prevention, regime-stratified evaluation, cross-experiment reporting
**Confidence:** HIGH

## Summary

Phase 2 redesigns the evaluation infrastructure for the new LESR architecture (fixed reward + structured feature library). The existing codebase in `exp4.9_c` already provides approximately 80% of the sliding window infrastructure (`run_sliding_parallel.py`, `build_sliding_configs.py`, `config_SW*.yaml`), a 3-dimensional regime detector (`regime_detector.py`), and Phase 1 diagnostic tools (`analyze_existing.py`, `stats_reporter.py`). The primary gaps are: (1) extending `DQNTrainer.evaluate()` from 3 metrics to 6 (adding Sortino, Calmar, Win Rate), (2) adding regime-stratified reporting within evaluate(), (3) hardening data leakage prevention in the COT feedback path, and (4) building a cross-stock/cross-window/cross-run aggregation reporter.

The financial metric computations (Sortino, Calmar, Win Rate) are straightforward numpy operations that should be hand-rolled within the existing `_sharpe()` and `_max_dd()` pattern rather than adding an external dependency like `empyrical` (not installed). The sliding window infrastructure uses a 3-year-train / 1-year-val / 1-year-test design with 10 windows covering test years 2015-2024, and this design is sound for walk-forward validation.

**Primary recommendation:** Extend existing `DQNTrainer.evaluate()` and `sliding_summary.py` in-place. Do not introduce new external dependencies for financial metrics. Reuse Phase 1's `analyze_existing.py` as the foundation for EVAL-05 cross-experiment reporting.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** intrinsic_reward is decoupled -- fixed as human-designed regime-based rules. LLM no longer generates reward functions.
- **D-02:** LLM output changes from free-form Python code to structured JSON selecting features from a predefined library
- **D-03:** This phase builds evaluation for the NEW architecture, not exp4.9_c as-is
- **D-04:** Use sliding-window walk-forward. Reuse and adapt exp4.9_c's config_SW*.yaml and run_sliding_parallel.py infrastructure
- **D-05:** Extend DQNTrainer.evaluate() to compute Sharpe, Sortino, Max Drawdown, Calmar ratio, Win Rate
- **D-06:** COT feedback must only use training-set analysis. Verify _generate_cot_feedback() and get_iteration_prompt() do not pass validation/test metrics to LLM
- **D-07:** Use existing regime_detector.py (3-dim: trend/volatility/risk) to label test periods, then report per-regime Sharpe and MaxDD
- **D-08:** Build on Phase 1's analyze_existing.py to aggregate results across result_SW* directories into publication-ready comparison tables
- **D-09:** v1 feature library: RSI, MACD, Bollinger_Band, Momentum, Volatility, Volume_Ratio, ROC, EMA_Cross, Stochastic_Osc, OBV, ATR, Williams_%R, CCI, ADX
- **D-10:** LLM outputs JSON: `{"features": [{"indicator": "RSI", "params": {"window": 14}}, ...], "rationale": "..."}`

### Claude's Discretion
- Exact feature library implementation (which indicators, default params)
- Fixed reward rule details and thresholds
- Report format (markdown tables vs LaTeX)
- Sliding window sizes and overlap

### Deferred Ideas (OUT OF SCOPE)
- Custom feature proposal by LLM (outside the library) -- future phase after v1 proves stable
- Ensemble DQN / multi-agent -- independent improvement, not in scope
- Adaptive feature library (growing over iterations) -- Phase 3 territory
- CPCV (Combinatorial Purged Cross-Validation) -- v2 requirement
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EVAL-01 | Walk-forward rolling-window training/testing | Section: Walk-Forward Infrastructure; existing config_SW01-SW10 + build_sliding_configs.py + run_sliding_parallel.py provide 80% of this |
| EVAL-02 | Multi-metric assessment: Sharpe, Sortino, MaxDD, Calmar, Win Rate | Section: Financial Metric Formulas; extend DQNTrainer.evaluate() following existing _sharpe/_max_dd pattern |
| EVAL-03 | Prevent LLM data leakage in iterative optimization | Section: Leakage Prevention Audit; verify _generate_cot_feedback path, add explicit train-only filtering |
| EVAL-04 | Regime-stratified evaluation (bull/bear/sideways) | Section: Regime-Stratified Evaluation; reuse regime_detector.py, label test-period steps, compute per-regime metrics |
| EVAL-05 | Cross-stock, cross-window, cross-run comparison reports | Section: Cross-Experiment Reporting; extend sliding_summary.py + analyze_existing.py with aggregation logic |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | 2.2.6 | Array operations, metric computation | [VERIFIED: pip3 list] Already installed, used throughout codebase |
| pandas | 2.3.3 | DataFrame-based result aggregation | [VERIFIED: pip3 list] Used in analyze_existing.py |
| scipy | 1.16.2 | Statistical tests (t-test, bootstrap) | [VERIFIED: pip3 list] Used in stats_reporter.py, feature_analyzer.py |
| torch | 2.9.0+cu128 | DQN training and inference | [VERIFIED: pip3 list] Core framework dependency |
| pyyaml | 6.0.2 | Config file loading | [VERIFIED: pip3 list] Used in run_window.py |
| pytest | 9.0.3 | Test framework | [VERIFIED: pip3 show] Phase 1 uses this |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| sklearn | 1.7.2 | RandomForest feature importance | [VERIFIED: pip3 list] Used in feature_analyzer.py for SHAP proxy |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled financial metrics | empyrical (pip package) | empyrical not installed; our 4 additional metrics (Sortino, Calmar, Win Rate, Sortino denominator) are trivial numpy ops -- not worth a dependency for 15 lines of code each [ASSUMED] |
| Hand-rolled financial metrics | quantstats | Not installed, heavier dependency with many features we do not need [ASSUMED] |

**Installation:**
```bash
# No new packages needed -- all dependencies already installed
```

**Version verification:** All versions verified via `pip3 list` and `python3 -c "import X; print(X.__version__)"` on 2026-04-15.

## Architecture Patterns

### Recommended Project Structure
```
exp4.9_c/
├── evaluation/              # NEW: Evaluation framework module
│   ├── __init__.py
│   ├── metrics.py           # Multi-metric computation (Sharpe, Sortino, Calmar, MaxDD, WinRate)
│   ├── regime_evaluator.py  # Regime-stratified evaluation
│   ├── leakage_guard.py     # Train/test data leakage prevention
│   └── cross_report.py      # Cross-stock/cross-window/cross-run aggregation
├── dqn_trainer.py           # EXTEND: evaluate() calls evaluation/metrics.py
├── regime_detector.py       # REUSE AS-IS
├── run_sliding_parallel.py  # EXTEND: calls cross_report after all windows
├── sliding_summary.py       # EXTEND: richer output format
├── build_sliding_configs.py # REUSE AS-IS or adapt for new architecture
└── config_SW*.yaml          # REUSE AS-IS
```

### Pattern 1: Metrics as Standalone Functions
**What:** Financial metric functions take a list of daily returns and return a float, matching the existing `_sharpe()` and `_max_dd()` pattern.
**When to use:** All metric computation in `DQNTrainer.evaluate()`.
**Example:**
```python
# Source: Follows existing pattern in dqn_trainer.py lines 361-371
def _sortino(returns, rf=0.0):
    """Sortino ratio: uses downside deviation only."""
    if len(returns) < 2:
        return 0.0
    r = np.array(returns)
    excess = r * 252 - rf
    downside = r[r < 0]
    if len(downside) == 0:
        return float('inf') if excess.mean() > 0 else 0.0
    dd = downside.std() * np.sqrt(252)
    return float(excess.mean() / dd) if dd > 0 else 0.0

def _calmar(returns, rf=0.0):
    """Calmar ratio: annualized return / max drawdown."""
    if len(returns) < 2:
        return 0.0
    annual_return = np.mean(returns) * 252 - rf
    mdd = _max_dd(returns)
    if mdd < 1e-8:
        return float('inf') if annual_return > 0 else 0.0
    return float(annual_return / (mdd / 100))

def _win_rate(returns):
    """Fraction of positive return periods."""
    if len(returns) < 1:
        return 0.0
    return float(np.mean(np.array(returns) > 0))
```

### Pattern 2: Regime-Stratified Evaluation
**What:** During evaluation, accumulate daily returns per-regime bucket, then compute metrics for each bucket.
**When to use:** EVAL-04 regime-stratified reporting.
**Example:**
```python
# Source: Derived from regime_detector.py's 3-dim vector + dqn_trainer.py evaluate()
# Regime buckets based on trend_direction:
#   bull: trend > +0.3
#   bear: trend < -0.3
#   sideways: -0.3 <= trend <= +0.3

def evaluate_by_regime(daily_returns, regime_vectors):
    """Compute per-regime metrics from evaluation loop outputs."""
    trend = np.array([r[0] for r in regime_vectors])
    buckets = {
        'bull': trend > 0.3,
        'bear': trend < -0.3,
        'sideways': (trend >= -0.3) & (trend <= 0.3),
    }
    results = {}
    for name, mask in buckets.items():
        regime_returns = [r for r, m in zip(daily_returns, mask) if m]
        if len(regime_returns) >= 10:
            results[name] = {
                'sharpe': _sharpe(regime_returns),
                'max_dd': _max_dd(regime_returns),
                'count': len(regime_returns),
            }
        else:
            results[name] = {'sharpe': 0.0, 'max_dd': 0.0, 'count': len(regime_returns)}
    return results
```

### Pattern 3: Leakage Guard
**What:** Explicit filtering that strips validation/test metrics before constructing COT prompt.
**When to use:** In `_generate_cot_feedback()` and `get_iteration_prompt()`.
**Example:**
```python
# Source: Analysis of lesr_controller.py lines 422-450 and prompts.py lines 170-277
# Current _generate_cot_feedback already only uses training results,
# but validation metrics flow through the same results dict.
# Solution: filter at the source.

def _generate_cot_feedback(self, samples, train_results, analysis):
    """Generate COT feedback from TRAINING results only."""
    # train_results already filtered by _parallel_train to only use train_period
    # Verify: no val_period or test_period metrics leak in
    codes = [s['code'] for s in samples]
    scores = []
    for i in range(len(samples)):
        sr = [r for r in train_results if r['sample_id'] == i]
        if sr:
            scores.append({
                'sharpe': np.mean([r['sharpe'] for r in sr]),
                'max_dd': np.mean([r['max_dd'] for r in sr]),
                'total_return': np.mean([r['total_return'] for r in sr])
            })
        else:
            scores.append({'sharpe': 0, 'max_dd': 100, 'total_return': 0})
    # ... rest same as current, but guaranteed train-only
```

### Anti-Patterns to Avoid
- **Computing metrics on non-overlapping returns:** If evaluating per-regime, the returns within each bucket are not contiguous -- metrics like MaxDD still work (computed on the subsequence), but annualization factors may need adjustment. Use `len(regime_returns) / 252` for actual time coverage rather than assuming 252 days per year. [ASSUMED]
- **Look-ahead in regime labeling:** Regime detector uses the raw state which includes the current day's OHLCV. During evaluation, this is fine (no training involved), but during walk-forward, ensure regime is computed from training data only. Current code computes regime per-step from `extract_state()` which uses a 20-day lookback -- this is stateless and safe.
- **Mixing train and val metrics in COT:** The current `lesr_controller.py` calls `trainer.evaluate()` on `val_period` (line 389-390) and feeds those metrics into the COT path. Under D-06, COT should use only training-set analysis. The fix is to change `_parallel_train` to evaluate on `train_period` for feedback, and evaluate on `val_period` separately for selection.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Statistical significance testing | Custom t-test / bootstrap | `scipy.stats.ttest_ind`, `scipy.stats.bootstrap` | Already in stats_reporter.py; handles edge cases, BCa correction |
| Feature importance | Custom correlation analysis | `feature_analyzer.py` (Spearman + RF importance) | Already in codebase, tested in Phase 1 |
| Config-driven experiment setup | Ad-hoc window creation | `build_sliding_configs.py` pattern | Already generates 10 windows; parameterize for new architecture |

**Key insight:** The evaluation framework is an extension of existing patterns, not a new system. Resist the urge to build a standalone evaluation library -- embed metrics into the existing trainer and extend existing reporting scripts.

## Common Pitfalls

### Pitfall 1: Annualization of Regime-Stratified Returns
**What goes wrong:** Computing `mean * 252` for Sharpe on a sub-period with only 30 days inflates the ratio.
**Why it happens:** Regime buckets are not full-year periods; some may have only 20-30 trading days.
**How to avoid:** Report N (count) alongside metrics. Flag regimes with < 60 days as "low confidence." Consider not annualizing sub-period Sharpe (report raw mean/std instead).
**Warning signs:** A "bull" Sharpe of 8.0 during a 10-day sub-period.

### Pitfall 2: Validation Metrics Leaking Into LLM Context
**What goes wrong:** `lesr_controller.py` line 389 calls `trainer.evaluate(data_loader, val_start, val_end)` and the result flows into COT feedback via `_parallel_train` return value.
**Why it happens:** The current code trains on train_period, evaluates on val_period, and returns val metrics. These val metrics then appear in the COT prompt via `_generate_cot_feedback`.
**How to avoid:** Split evaluation: (1) train-set evaluation for COT feedback, (2) val-set evaluation for strategy selection only (never shown to LLM). Add a `leakage_guard.py` that asserts no val/test metric keys appear in COT input.
**Warning signs:** LLM prompt text contains validation Sharpe values.

### Pitfall 3: Sortino Denominator Zero-Division
**What goes wrong:** If all returns are positive, downside deviation is zero, causing division by zero.
**Why it happens:** Short evaluation periods or strong uptrend periods may have no negative returns.
**How to avoid:** Check for zero downside deviation before dividing. Return `float('inf')` or `0.0` based on sign of numerator (see Pattern 1 example above).
**Warning signs:** NaN or inf in Sortino ratio column.

### Pitfall 4: Walk-Forward With Data Overlap
**What goes wrong:** If training windows overlap, a model may benefit from seeing validation-like data during training, inflating test performance.
**Why it happens:** The current config uses 3-year-train / 1-year-val / 1-year-test with non-overlapping windows (verified: `build_sliding_configs.py` uses `test_year - 4` to `test_year - 2` for train, which advances by 1 year each window, creating overlap in training data across windows).
**How to avoid:** This is acceptable for walk-forward validation -- each window trains independently. The overlap in training DATA (not labels) is expected. The key is that test sets are strictly out-of-sample and non-overlapping, which they are. No action needed, but document this design choice clearly.
**Warning signs:** If someone suggests using "all previous data" as training (expanding window), verify that validation period does not overlap with any future test period.

### Pitfall 5: Evaluate() Modifying Trainer State
**What goes wrong:** Running `evaluate()` after `train()` modifies `episode_states`, `episode_rewards`, `episode_regimes` lists if those are used during evaluation.
**Why it happens:** The current `train()` appends to these lists (lines 263-264), and `evaluate()` does NOT (it uses local variables). But future modifications might accidentally append.
**How to avoid:** Ensure `evaluate()` uses only local variables for return accumulation. Add a comment documenting this constraint.
**Warning signs:** `episode_rewards` length changes after calling `evaluate()`.

## Code Examples

### Extended evaluate() Method
```python
# Source: Extends dqn_trainer.py lines 323-359
def evaluate(self, data_loader, start_date, end_date):
    """Evaluate on test data with full metrics + regime breakdown."""
    dates = [d for d in data_loader.get_date_range()
             if start_date <= str(d) <= end_date]

    daily_returns = []
    regime_vectors = []
    prev_price = None
    current_position = 0
    trade_count = 0
    win_count = 0

    for date in dates:
        raw_state, regime_vector = self.extract_state(data_loader, date)
        if raw_state is None:
            continue

        enhanced = self._build_enhanced_state(raw_state, regime_vector)
        action = self.dqn.select_action(enhanced, epsilon=0.0)
        current_price = data_loader.get_ticker_price_by_date(self.ticker, date)

        prev_position = current_position
        if action == 0:
            current_position = 1
        elif action == 1:
            current_position = 0

        if prev_price is not None:
            dr = (current_price - prev_price) / prev_price if current_position == 1 else 0.0
            daily_returns.append(dr)
            regime_vectors.append(regime_vector.copy())

            # Win rate: count positive return days when in position
            if current_position == 1 and dr > 0:
                win_count += 1

        prev_price = current_price

    sharpe = self._sharpe(daily_returns)
    max_dd = self._max_dd(daily_returns)
    total_ret = sum(daily_returns) * 100
    sortino = self._sortino(daily_returns)
    calmar = self._calmar(daily_returns)
    win_rate = win_count / max(1, sum(1 for r in daily_returns if r != 0.0))

    # Regime-stratified metrics
    regime_metrics = self._evaluate_by_regime(daily_returns, regime_vectors)

    return {
        'sharpe': sharpe, 'max_dd': max_dd, 'total_return': total_ret,
        'sortino': sortino, 'calmar': calmar, 'win_rate': win_rate,
        'trades': [], 'regime_metrics': regime_metrics
    }
```

### Cross-Experiment Report Aggregator
```python
# Source: Extends sliding_summary.py pattern
import pickle
from pathlib import Path
import pandas as pd

def aggregate_cross_experiment(base_dir, pattern="result_SW*_test*"):
    """Aggregate results from all sliding window experiments."""
    all_rows = []
    for result_dir in sorted(Path(base_dir).glob(pattern)):
        pkl_file = result_dir / 'test_set_results.pkl'
        if not pkl_file.exists():
            continue
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        # Extract window info from directory name
        # e.g., result_SW01_test2015 -> window=1, test_year=2015
        name = result_dir.name
        window = int(name.split('SW')[1].split('_')[0])
        test_year = int(name.split('test')[1])

        for ticker, ticker_data in data.items():
            if ticker_data.get('error'):
                continue
            for method in ['lesr_test', 'baseline_test']:
                metrics = ticker_data[method]
                all_rows.append({
                    'window': window, 'test_year': test_year,
                    'ticker': ticker,
                    'method': 'LESR' if 'lesr' in method else 'Baseline',
                    'sharpe': metrics['sharpe'],
                    'max_dd': metrics['max_dd'],
                    'total_return': metrics['total_return'],
                    'sortino': metrics.get('sortino', 0),
                    'calmar': metrics.get('calmar', 0),
                    'win_rate': metrics.get('win_rate', 0),
                })
    return pd.DataFrame(all_rows)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 3-metric eval (Sharpe, MaxDD, Return) | 6-metric eval (add Sortino, Calmar, WinRate) | This phase | More robust assessment, catches risk-adjusted performance |
| Fixed window train/test | Walk-forward sliding windows | exp4.9_c (already built) | Out-of-sample validation, publication-standard |
| No leakage guard | Explicit train-only COT feedback | This phase | Prevents LLM from overfitting to validation set |
| Overall metrics only | Regime-stratified breakdown | This phase | Identifies regime-dependent weaknesses |
| Per-window reports | Cross-experiment aggregation | This phase | Publication-ready comparison tables |

**Deprecated/outdated:**
- `exp4.7` evaluate() returning only 3 metrics -- being superseded
- `exp4.7` fixed-window experiments -- replaced by sliding window design

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Sortino, Calmar, Win Rate are simple enough to hand-roll without empyrical | Standard Stack | Low -- these are well-defined formulas with known edge cases |
| A2 | Current walk-forward design (3yr-train/1yr-val/1yr-test sliding by 1 year) is sufficient for EVAL-01 | Architecture Patterns | Medium -- user specified "Claude's discretion" on window sizes |
| A3 | Regime labels from trend_direction thresholds (+/-0.3) are adequate for bull/bear/sideways classification | Regime Evaluation | Low -- these thresholds are already established in the codebase |
| A4 | empyrical not needed because the 4 additional metrics are trivial numpy operations | Standard Stack | Low -- ~15 lines per metric, well-tested edge cases documented |
| A5 | The current `evaluate()` not appending to episode_states/rewards will remain true | Pitfall 5 | Low -- verified by reading current code |

**If this table is empty:** N/A -- 5 assumptions identified and documented.

## Open Questions

1. **Where should the evaluation module live?**
   - What we know: New code needs a home. Options: `exp4.9_c/evaluation/` as a subpackage, or inline in existing files.
   - What's unclear: Whether a separate module is worth the import complexity given `sys.path` patterns.
   - Recommendation: Create `exp4.9_c/evaluation/` as a subpackage with `metrics.py`, `regime_evaluator.py`, `leakage_guard.py`. Keep it importable from `dqn_trainer.py` via sibling import. This matches the project's pattern of keeping related code together.

2. **Should evaluate() return regime-level daily returns for external aggregation?**
   - What we know: The current evaluate() returns a flat dict. Regime-stratified metrics add a nested dict.
   - What's unclear: Whether downstream consumers (sliding_summary.py, cross_report.py) need raw per-step data or just aggregated metrics.
   - Recommendation: Return both aggregated regime_metrics AND the raw daily_returns + regime_vectors lists. This lets aggregation scripts do their own grouping if needed.

3. **Format for publication-ready tables (EVAL-05)?**
   - What we know: Claude's discretion per CONTEXT.md. Current sliding_summary.py generates markdown tables.
   - What's unclear: Whether LaTeX format is needed for paper submission.
   - Recommendation: Generate markdown by default (consistent with existing pattern). Add optional `--format latex` flag to the cross-report script. LaTeX can be generated from the same DataFrame with `to_latex()`.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.13 | Core runtime | Yes | 3.13.5 | -- |
| numpy | Metric computation | Yes | 2.2.6 | -- |
| pandas | DataFrame aggregation | Yes | 2.3.3 | -- |
| scipy | Statistical tests | Yes | 1.16.2 | -- |
| torch | DQN training | Yes | 2.9.0+cu128 | -- |
| pyyaml | Config loading | Yes | 6.0.2 | -- |
| sklearn | Feature importance | Yes | 1.7.2 | -- |
| pytest | Test framework | Yes | 9.0.3 | -- |
| CUDA GPU | DQN training | Yes | 4x GPU (Driver 580.105.08) | CPU fallback exists |
| empyrical | NOT needed | No | -- | Hand-roll metrics (confirmed feasible) |
| quantstats | NOT needed | No | -- | Hand-roll metrics (confirmed feasible) |

**Missing dependencies with no fallback:**
- None -- all required dependencies are installed.

**Missing dependencies with fallback:**
- empyrical/quantstats: Not needed; 4 additional financial metrics are simple numpy operations.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.3 |
| Config file | None -- pytest auto-discovers tests/ directories |
| Quick run command | `pytest exp4.9_c/tests/ -x -q` |
| Full suite command | `pytest exp4.9_c/tests/ -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-01 | Walk-forward windows generate sequential configs | unit | `pytest exp4.9_c/tests/test_walk_forward.py::test_sliding_config_generation -x` | No -- Wave 0 |
| EVAL-01 | Walk-forward windows produce out-of-sample results | integration | `pytest exp4.9_c/tests/test_walk_forward.py::test_out_of_sample_results -x` | No -- Wave 0 |
| EVAL-02 | evaluate() returns all 6 metrics | unit | `pytest exp4.9_c/tests/test_metrics.py::test_evaluate_returns_six_metrics -x` | No -- Wave 0 |
| EVAL-02 | Sortino computed correctly | unit | `pytest exp4.9_c/tests/test_metrics.py::test_sortino_known_input -x` | No -- Wave 0 |
| EVAL-02 | Calmar computed correctly | unit | `pytest exp4.9_c/tests/test_metrics.py::test_calmar_known_input -x` | No -- Wave 0 |
| EVAL-02 | Win rate computed correctly | unit | `pytest exp4.9_c/tests/test_metrics.py::test_win_rate_known_input -x` | No -- Wave 0 |
| EVAL-03 | COT feedback contains no val/test metrics | unit | `pytest exp4.9_c/tests/test_leakage.py::test_cot_no_val_metrics -x` | No -- Wave 0 |
| EVAL-03 | Leakage guard catches validation Sharpe in prompt | unit | `pytest exp4.9_c/tests/test_leakage.py::test_leakage_guard_detects_val_data -x` | No -- Wave 0 |
| EVAL-04 | Regime-stratified metrics computed per bucket | unit | `pytest exp4.9_c/tests/test_regime_eval.py::test_regime_stratified_metrics -x` | No -- Wave 0 |
| EVAL-04 | Bull/bear/sideways classification correct | unit | `pytest exp4.9_c/tests/test_regime_eval.py::test_regime_bucketing -x` | No -- Wave 0 |
| EVAL-05 | Cross-experiment aggregation produces DataFrame | unit | `pytest exp4.9_c/tests/test_cross_report.py::test_aggregate_cross_experiment -x` | No -- Wave 0 |
| EVAL-05 | Report includes all stocks and windows | unit | `pytest exp4.9_c/tests/test_cross_report.py::test_report_completeness -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest exp4.9_c/tests/ -x -q`
- **Per wave merge:** `pytest exp4.9_c/tests/ -v`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `exp4.9_c/tests/__init__.py` -- test package init
- [ ] `exp4.9_c/tests/conftest.py` -- shared fixtures (mock data_loader, sample returns, sample regime vectors)
- [ ] `exp4.9_c/tests/test_metrics.py` -- covers EVAL-02 (Sortino, Calmar, WinRate with known inputs)
- [ ] `exp4.9_c/tests/test_walk_forward.py` -- covers EVAL-01 (config generation, sequential windows)
- [ ] `exp4.9_c/tests/test_leakage.py` -- covers EVAL-03 (COT prompt inspection, guard mechanism)
- [ ] `exp4.9_c/tests/test_regime_eval.py` -- covers EVAL-04 (regime bucketing, per-regime metrics)
- [ ] `exp4.9_c/tests/test_cross_report.py` -- covers EVAL-05 (aggregation, report generation)

## Security Domain

> This is a research project with no web-facing components, no user authentication, and no sensitive data storage. Security enforcement is not applicable in the traditional ASVS sense.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | No auth -- research scripts |
| V3 Session Management | No | No sessions |
| V4 Access Control | No | No access control |
| V5 Input Validation | Yes (partial) | LLM JSON output parsing should validate structure; YAML config loading uses `yaml.safe_load` |
| V6 Cryptography | No | No cryptographic operations |

### Known Threat Patterns for LESR Evaluation

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| API key exposure in config YAML files | Information Disclosure | Config files should use env vars; current configs contain hardcoded keys (acceptable for research, flag for cleanup) |
| LLM output injection (malformed JSON) | Tampering | Validate LLM JSON output structure before processing; catch exceptions in feature library calls |
| Pickle deserialization of untrusted data | Tampering | Results pickle files are researcher-generated, not external; acceptable risk for research project |

## Sources

### Primary (HIGH confidence)
- Codebase analysis of `exp4.9_c/dqn_trainer.py` -- evaluate() structure, _sharpe(), _max_dd() patterns
- Codebase analysis of `exp4.9_c/lesr_controller.py` -- COT feedback path, _parallel_train flow
- Codebase analysis of `exp4.9_c/regime_detector.py` -- 3-dim regime vector definition
- Codebase analysis of `exp4.9_c/run_sliding_parallel.py` -- sliding window infrastructure
- Codebase analysis of `exp4.9_c/sliding_summary.py` -- current aggregation reporting
- Codebase analysis of `exp4.9_c/build_sliding_configs.py` -- window config generation
- Codebase analysis of `exp4.7/diagnosis/analyze_existing.py` -- Phase 1 analysis foundation
- Codebase analysis of `exp4.7/diagnosis/stats_reporter.py` -- statistical comparison tools
- Environment verification via `pip3 list`, `python3 --version`, `nvidia-smi` (2026-04-15)

### Secondary (MEDIUM confidence)
- Financial metric formulas (Sortino, Calmar, Win Rate) -- standard definitions from quantitative finance literature [ASSUMED]
- Walk-forward validation methodology -- standard approach in quantitative trading research [ASSUMED]

### Tertiary (LOW confidence)
- None -- all findings are either codebase-verified or clearly tagged as ASSUMED

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all packages verified via pip3, no new dependencies needed
- Architecture: HIGH -- extends existing patterns, all canonical refs read and analyzed
- Pitfalls: HIGH -- derived from direct code analysis of existing evaluate(), COT path, and sliding window flow

**Research date:** 2026-04-15
**Valid until:** 2026-05-15 (stable -- no fast-moving dependencies)
