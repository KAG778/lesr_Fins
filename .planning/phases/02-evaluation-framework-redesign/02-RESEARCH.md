# Phase 2: Evaluation Framework Redesign - Research

**Researched:** 2026-04-15
**Domain:** Walk-forward validation, financial metrics, factor evaluation (IC/IR/Quantile Spread), data leakage prevention, regime-stratified evaluation, cross-experiment reporting
**Confidence:** HIGH

## Summary

Phase 2 redesigns the evaluation infrastructure for the new LESR architecture (fixed reward + structured feature library). The existing codebase in `exp4.9_c` already provides approximately 80% of the sliding window infrastructure (`run_sliding_parallel.py`, `build_sliding_configs.py`, `config_SW*.yaml`), a 3-dimensional regime detector (`regime_detector.py`), and Phase 1 diagnostic tools (`analyze_existing.py`, `stats_reporter.py`). The primary gaps are: (1) extending `DQNTrainer.evaluate()` from 3 metrics to 6 (adding Sortino, Calmar, Win Rate), (2) adding a factor evaluation module with IC, IR, and Quantile Spread per feature dimension (D-11/D-12/D-13), (3) adding regime-stratified reporting within evaluate(), (4) hardening data leakage prevention in the COT feedback path, and (5) building a cross-stock/cross-window/cross-run aggregation reporter.

The financial metric computations (Sortino, Calmar, Win Rate) and factor evaluation metrics (IC, IR, Quantile Spread) are straightforward numpy/scipy operations that should be implemented within the existing pattern rather than adding external dependencies like `empyrical` or `alphalens` (neither installed). IC uses `scipy.stats.spearmanr` (already available at scipy 1.10.1, already used in `feature_quality.py` and `feature_analyzer.py`). IR is IC_mean / IC_std computed from a rolling IC series. Quantile Spread uses `numpy.quantile` to split observations into quintile groups and computes top-minus-bottom forward return difference.

**Primary recommendation:** Create `exp4.9_c/evaluation/metrics.py` with both strategy performance metrics AND factor evaluation metrics. Extend `DQNTrainer.evaluate()` to return a `factor_metrics` dict alongside strategy metrics. Reuse Phase 1's `analyze_existing.py` as the foundation for EVAL-05 cross-experiment reporting.

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
- **D-11:** metrics.py must include factor evaluation metrics: IC (Information Coefficient, Spearman rank correlation between factor values and forward returns), IR (Information Ratio, IC mean / IC std), Quantile Spread (Top group vs Bottom group return difference). These evaluate per-feature predictive power, not overall strategy performance
- **D-12:** evaluate() must return factor_metrics per feature dimension alongside strategy performance metrics, so Phase 3 feature selection (LESR-04) has quantitative evidence
- **D-13:** Factor evaluation is a core academic contribution -- "LLM-generated features have significant IC" is more convincing than "LLM strategy has higher Sharpe"

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
| EVAL-02 | Multi-metric assessment: Sharpe, Sortino, MaxDD, Calmar, Win Rate | Section: Strategy Performance Metrics; extend DQNTrainer.evaluate() following existing _sharpe/_max_dd pattern |
| EVAL-03 | Prevent LLM data leakage in iterative optimization | Section: Leakage Prevention Audit; verify _generate_cot_feedback path, add explicit train-only filtering |
| EVAL-04 | Regime-stratified evaluation (bull/bear/sideways) | Section: Regime-Stratified Evaluation; reuse regime_detector.py, label test-period steps, compute per-regime metrics |
| EVAL-05 | Cross-stock, cross-window, cross-run comparison reports | Section: Cross-Experiment Reporting; extend sliding_summary.py + analyze_existing.py with aggregation logic |
| (D-11) | Factor evaluation metrics: IC, IR, Quantile Spread | Section: Factor Evaluation Metrics (IC/IR/Quantile Spread); scipy.stats.spearmanr for IC, rolling window for IR, numpy quantile for spread |
| (D-12) | evaluate() returns factor_metrics per feature dimension | Section: Integration With evaluate(); extract per-step feature values and forward returns during evaluation loop |
| (D-13) | Factor evaluation is core academic contribution | Section: Academic Narrative; IC/IR provide per-feature evidence that LLM selections are informative |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | 1.24.4 | Array operations, metric computation | [VERIFIED: pip list on lesr env] Already installed, used throughout codebase |
| pandas | 2.0.3 | DataFrame-based result aggregation | [VERIFIED: pip list on lesr env] Used in analyze_existing.py |
| scipy | 1.10.1 | Statistical tests, spearmanr for IC | [VERIFIED: pip list on lesr env] Used in stats_reporter.py, feature_quality.py, feature_analyzer.py |
| torch | 2.4.1 | DQN training and inference | [VERIFIED: pip list on lesr env] Core framework dependency |
| pyyaml | installed | Config file loading | [VERIFIED: in requirements.txt] Used in run_window.py |
| pytest | installed | Test framework | [VERIFIED: which pytest] Phase 1 uses this |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| sklearn | installed | RandomForest feature importance | Used in feature_analyzer.py for SHAP proxy |
| tabulate | 0.9.0 | Table formatting in reports | Used in sliding_summary.py for formatted output |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled financial metrics | empyrical (pip package) | empyrical not installed; our 4 additional metrics (Sortino, Calmar, Win Rate) are trivial numpy ops -- not worth a dependency for ~15 lines of code each [ASSUMED] |
| Hand-rolled factor evaluation | alphalens (Quantopian) | alphalens not installed, requires specific MultiIndex DataFrame format, designed for cross-sectional factor analysis across many stocks. Our use case is single-stock time-series factor evaluation on per-step features within the DQN loop -- alphalens is over-engineered for this [VERIFIED: alphalens designed for cross-sectional portfolio analysis, not RL feature evaluation] |
| Hand-rolled factor evaluation | bagel-factor | Not installed, newer package (2025), also designed for cross-sectional factor research, not RL state features [ASSUMED] |

**Installation:**
```bash
# No new packages needed -- all dependencies already installed in lesr conda env
```

**Version verification:** All versions verified via `pip list` on lesr conda environment (Python 3.8.20) on 2026-04-15.

## Architecture Patterns

### Recommended Project Structure
```
exp4.9_c/
├── evaluation/              # NEW: Evaluation framework module
│   ├── __init__.py
│   ├── metrics.py           # Strategy metrics (Sharpe, Sortino, Calmar, MaxDD, WinRate)
│   │                        # + Factor evaluation metrics (IC, IR, Quantile Spread)
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
**What:** Financial metric functions take a list/array of daily returns and return a float, matching the existing `_sharpe()` and `_max_dd()` pattern in `dqn_trainer.py`.
**When to use:** All metric computation in `DQNTrainer.evaluate()`.
**Example:**
```python
# Source: Follows existing pattern in dqn_trainer.py lines 361-371
# File: evaluation/metrics.py

import numpy as np

def sharpe(returns, rf=0.0):
    """Annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    r = np.array(returns)
    m, s = r.mean() * 252, r.std() * np.sqrt(252)
    return (m - rf) / s if s > 0 else 0.0

def sortino(returns, rf=0.0):
    """Sortino ratio: uses downside deviation only."""
    if len(returns) < 2:
        return 0.0
    r = np.array(returns)
    annualized_mean = r.mean() * 252 - rf
    downside = r[r < 0]
    if len(downside) == 0:
        return float('inf') if annualized_mean > 0 else 0.0
    dd = downside.std() * np.sqrt(252)
    return float(annualized_mean / dd) if dd > 0 else 0.0

def max_drawdown(returns):
    """Maximum drawdown as positive percentage."""
    if len(returns) < 2:
        return 0.0
    cum = np.cumsum(returns)
    peak = np.maximum.accumulate(cum)
    return abs((cum - peak).min()) * 100

def calmar(returns, rf=0.0):
    """Calmar ratio: annualized return / max drawdown."""
    if len(returns) < 2:
        return 0.0
    annual_return = np.mean(returns) * 252 - rf
    mdd = max_drawdown(returns)
    if mdd < 1e-8:
        return float('inf') if annual_return > 0 else 0.0
    return float(annual_return / (mdd / 100))

def win_rate(returns):
    """Fraction of positive return periods."""
    if len(returns) < 1:
        return 0.0
    r = np.array(returns)
    nonzero = r[r != 0.0]
    if len(nonzero) == 0:
        return 0.0
    return float(np.mean(nonzero > 0))
```

### Pattern 2: Factor Evaluation Metrics (IC / IR / Quantile Spread) -- D-11/D-12/D-13

**What:** Evaluate individual features' predictive power against forward returns. This is the core academic contribution per D-13.

**When to use:** During `evaluate()`, after the evaluation loop has collected per-step feature values and forward returns. Also during feature analysis in Phase 3's feature selection (LESR-04).

**Key design decisions:**
1. **IC uses Spearman rank correlation** (not Pearson) because factor-return relationships are often nonlinear [CITED: CONTEXT.md specifics section]
2. **IR = mean(rolling_IC) / std(rolling_IC)** where rolling IC is computed over a sliding window (default 20 days) [CITED: CONTEXT.md specifics section]
3. **Quantile Spread** splits feature values into N groups (default 5 quintiles), computes mean forward return for top and bottom groups, returns the difference [CITED: CONTEXT.md specifics section]

**Example:**
```python
# Source: evaluation/metrics.py (NEW)
# Dependencies: numpy, scipy.stats.spearmanr (already installed)

import numpy as np
from scipy.stats import spearmanr
from typing import Dict, Optional


def information_coefficient(feature_values: np.ndarray,
                            forward_returns: np.ndarray,
                            method: str = 'spearman') -> float:
    """Compute IC: rank correlation between feature values and forward returns.

    Args:
        feature_values: 1D array of feature values at each time step.
        forward_returns: 1D array of next-period returns.
        method: 'spearman' (default, preferred for nonlinear relationships).

    Returns:
        IC value (float). Range [-1, 1]. Higher absolute value = more predictive.
    """
    if len(feature_values) < 10:
        return 0.0
    # Remove NaN/inf
    mask = np.isfinite(feature_values) & np.isfinite(forward_returns)
    f, r = feature_values[mask], forward_returns[mask]
    if len(f) < 10 or np.std(f) < 1e-10 or np.std(r) < 1e-10:
        return 0.0
    corr, _ = spearmanr(f, r)
    return float(corr) if not np.isnan(corr) else 0.0


def rolling_ic(feature_values: np.ndarray,
               forward_returns: np.ndarray,
               window: int = 20) -> np.ndarray:
    """Compute rolling IC over a sliding window.

    Args:
        feature_values: 1D array.
        forward_returns: 1D array, same length.
        window: rolling window size (default 20 trading days).

    Returns:
        1D array of IC values. First (window-1) entries are NaN.
    """
    n = len(feature_values)
    ic_series = np.full(n, np.nan)
    for i in range(window - 1, n):
        f_window = feature_values[i - window + 1:i + 1]
        r_window = forward_returns[i - window + 1:i + 1]
        mask = np.isfinite(f_window) & np.isfinite(r_window)
        f_clean, r_clean = f_window[mask], r_window[mask]
        if len(f_clean) >= 10 and np.std(f_clean) > 1e-10 and np.std(r_clean) > 1e-10:
            corr, _ = spearmanr(f_clean, r_clean)
            ic_series[i] = corr if not np.isnan(corr) else 0.0
    return ic_series


def information_ratio(rolling_ic_series: np.ndarray) -> float:
    """IR = mean(IC) / std(IC) over the rolling IC series.

    Args:
        rolling_ic_series: output of rolling_ic(), may contain NaN.

    Returns:
        IR value (float). Higher = more consistent predictive power.
    """
    clean = rolling_ic_series[~np.isnan(rolling_ic_series)]
    if len(clean) < 5:
        return 0.0
    ic_mean = np.mean(clean)
    ic_std = np.std(clean)
    return float(ic_mean / ic_std) if ic_std > 1e-10 else 0.0


def quantile_spread(feature_values: np.ndarray,
                    forward_returns: np.ndarray,
                    n_quantiles: int = 5) -> float:
    """Quantile Spread: mean return of top group minus bottom group.

    Higher positive spread means the feature effectively separates
    high-return from low-return observations.

    Args:
        feature_values: 1D array of feature values.
        forward_returns: 1D array of next-period returns, same length.
        n_quantiles: number of groups (default 5 for quintiles).

    Returns:
        Spread value (float). Positive = feature is predictive.
    """
    if len(feature_values) < n_quantiles * 2:
        return 0.0
    mask = np.isfinite(feature_values) & np.isfinite(forward_returns)
    f, r = feature_values[mask], forward_returns[mask]
    if len(f) < n_quantiles * 2:
        return 0.0
    # Compute quantile boundaries
    quantile_edges = np.quantile(f, np.linspace(0, 1, n_quantiles + 1))
    # Top group: feature >= top quantile edge
    top_mask = f >= quantile_edges[-2]
    # Bottom group: feature < bottom quantile edge
    bottom_mask = f < quantile_edges[1]
    if top_mask.sum() < 2 or bottom_mask.sum() < 2:
        return 0.0
    top_return = np.mean(r[top_mask])
    bottom_return = np.mean(r[bottom_mask])
    return float(top_return - bottom_return)


def compute_factor_metrics(feature_matrix: np.ndarray,
                           forward_returns: np.ndarray,
                           feature_names: Optional[list] = None,
                           rolling_window: int = 20,
                           n_quantiles: int = 5) -> Dict:
    """Compute factor evaluation metrics for all features.

    This is the main entry point called by evaluate().

    Args:
        feature_matrix: 2D array (T, F) where T=time steps, F=features.
        forward_returns: 1D array (T,) of next-period returns.
        feature_names: optional list of feature names (length F).
        rolling_window: window for rolling IC (default 20).
        n_quantiles: groups for quantile spread (default 5).

    Returns:
        Dict with:
            per_feature: list of dicts, each with ic, ir, quantile_spread
            aggregate: dict with mean_ic, mean_ir, mean_spread, num_significant_ic
    """
    n_features = feature_matrix.shape[1]
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(n_features)]

    per_feature = []
    ics = []
    irs = []
    spreads = []

    for i in range(n_features):
        feat_vals = feature_matrix[:, i]

        ic_val = information_coefficient(feat_vals, forward_returns)
        ric = rolling_ic(feat_vals, forward_returns, window=rolling_window)
        ir_val = information_ratio(ric)
        qs_val = quantile_spread(feat_vals, forward_returns, n_quantiles)

        per_feature.append({
            'name': feature_names[i],
            'index': i,
            'ic': ic_val,
            'ir': ir_val,
            'quantile_spread': qs_val,
            'ic_significant': abs(ic_val) > 0.03,  # rough threshold
        })
        ics.append(ic_val)
        irs.append(ir_val)
        spreads.append(qs_val)

    aggregate = {
        'mean_ic': float(np.mean(ics)) if ics else 0.0,
        'mean_abs_ic': float(np.mean(np.abs(ics))) if ics else 0.0,
        'mean_ir': float(np.mean(irs)) if irs else 0.0,
        'mean_spread': float(np.mean(spreads)) if spreads else 0.0,
        'num_significant_ic': sum(1 for p in per_feature if p['ic_significant']),
        'num_features': n_features,
    }

    return {
        'per_feature': per_feature,
        'aggregate': aggregate,
    }
```

### Pattern 3: Integration of Factor Metrics Into evaluate()

**What:** During the evaluation loop in `DQNTrainer.evaluate()`, collect per-step feature values and compute forward returns alongside strategy metrics. After the loop, call `compute_factor_metrics()` on the collected data.

**When to use:** Every call to `evaluate()`. The feature values at each step are the LLM-generated features (indices 123+ in the enhanced state).

**Example:**
```python
# Source: Extension of dqn_trainer.py evaluate() method

def evaluate(self, data_loader, start_date, end_date):
    """Evaluate with strategy metrics + factor metrics + regime breakdown."""
    dates = [d for d in data_loader.get_date_range()
             if start_date <= str(d) <= end_date]

    daily_returns = []
    forward_returns = []      # NEW: for factor evaluation
    feature_values = []       # NEW: per-step feature values (indices 123+)
    regime_vectors = []
    prev_price = None
    current_position = 0

    for i, date in enumerate(dates):
        raw_state, regime_vector = self.extract_state(data_loader, date)
        if raw_state is None:
            continue

        enhanced = self._build_enhanced_state(raw_state, regime_vector)
        action = self.dqn.select_action(enhanced, epsilon=0.0)
        current_price = data_loader.get_ticker_price_by_date(self.ticker, date)

        if action == 0:
            current_position = 1
        elif action == 1:
            current_position = 0

        if prev_price is not None:
            dr = (current_price - prev_price) / prev_price if current_position == 1 else 0.0
            daily_returns.append(dr)
            regime_vectors.append(regime_vector.copy())

            # Collect feature values for factor evaluation
            if len(enhanced) > 123:
                feature_values.append(enhanced[123:].copy())

            # Forward return: next day's price change (for IC computation)
            if i < len(dates) - 1:
                next_price = data_loader.get_ticker_price_by_date(
                    self.ticker, dates[min(i+1, len(dates)-1)])
                fwd = (next_price - current_price) / current_price if current_price > 0 else 0.0
                forward_returns.append(fwd)

        prev_price = current_price

    # Strategy performance metrics
    strategy_metrics = {
        'sharpe': sharpe(daily_returns),
        'sortino': sortino(daily_returns),
        'max_dd': max_drawdown(daily_returns),
        'calmar': calmar(daily_returns),
        'win_rate': win_rate(daily_returns),
        'total_return': sum(daily_returns) * 100,
    }

    # Factor evaluation metrics (D-12)
    factor_metrics = {}
    if len(feature_values) > 20 and len(forward_returns) > 20:
        # Align lengths: forward_returns may be 1 shorter
        min_len = min(len(feature_values), len(forward_returns))
        feat_matrix = np.array(feature_values[:min_len])
        fwd_rets = np.array(forward_returns[:min_len])
        factor_metrics = compute_factor_metrics(feat_matrix, fwd_rets)

    # Regime-stratified metrics
    regime_metrics = evaluate_by_regime(daily_returns, regime_vectors)

    return {
        **strategy_metrics,
        'trades': [],
        'regime_metrics': regime_metrics,
        'factor_metrics': factor_metrics,
        'daily_returns': daily_returns,
        'regime_vectors': regime_vectors,
    }
```

### Pattern 4: Regime-Stratified Evaluation
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
    if not regime_vectors:
        return {}
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
                'sharpe': sharpe(regime_returns),
                'max_dd': max_drawdown(regime_returns),
                'sortino': sortino(regime_returns),
                'count': len(regime_returns),
            }
        else:
            results[name] = {
                'sharpe': 0.0, 'max_dd': 0.0, 'sortino': 0.0,
                'count': len(regime_returns)
            }
    return results
```

### Pattern 5: Leakage Guard
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
- **Computing IC with Pearson correlation:** Financial factor-return relationships are frequently nonlinear. Spearman rank correlation is the standard in factor research. [CITED: CONTEXT.md specifics section]
- **Computing metrics on non-overlapping returns for regime sub-periods:** If evaluating per-regime, the returns within each bucket are not contiguous -- metrics like MaxDD still work (computed on the subsequence), but annualization factors may need adjustment. Use `len(regime_returns) / 252` for actual time coverage rather than assuming 252 days per year. [ASSUMED]
- **Look-ahead in regime labeling:** Regime detector uses the raw state which includes the current day's OHLCV. During evaluation, this is fine (no training involved), but during walk-forward, ensure regime is computed from training data only. Current code computes regime per-step from `extract_state()` which uses a 20-day lookback -- this is stateless and safe.
- **Mixing train and val metrics in COT:** The current `lesr_controller.py` calls `trainer.evaluate()` on `val_period` (line 389-390) and feeds those metrics into the COT path. Under D-06, COT should use only training-set analysis. The fix is to change `_parallel_train` to evaluate on `train_period` for feedback, and evaluate on `val_period` separately for selection.
- **Computing factor IC on the same data used for training:** Factor metrics computed during `evaluate()` are for reporting only (test data). During training, factor metrics for COT feedback must come from training-set analysis only, matching D-06 leakage prevention.
- **Forward return alignment off-by-one:** When collecting forward returns for IC computation, the forward return at step i should be the return from step i to step i+1. If the loop processes dates sequentially, ensure `forward_returns[i]` aligns with `feature_values[i]`. The example in Pattern 3 shows the correct alignment.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Statistical significance testing | Custom t-test / bootstrap | `scipy.stats.ttest_ind`, `scipy.stats.bootstrap` | Already in stats_reporter.py; handles edge cases, BCa correction |
| Feature importance | Custom correlation analysis | `feature_analyzer.py` (Spearman + RF importance) | Already in codebase, tested in Phase 1 |
| Config-driven experiment setup | Ad-hoc window creation | `build_sliding_configs.py` pattern | Already generates 10 windows; parameterize for new architecture |
| Spearman rank correlation | Custom rank correlation | `scipy.stats.spearmanr` | Already used in feature_quality.py and feature_analyzer.py; handles ties, NaN, p-values |

**Key insight:** The evaluation framework is an extension of existing patterns, not a new system. Resist the urge to build a standalone evaluation library -- embed metrics into the existing trainer and extend existing reporting scripts. The factor evaluation metrics (IC/IR/Quantile Spread) are pure numpy/scipy operations that fit naturally into `evaluation/metrics.py` alongside the strategy performance metrics.

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

### Pitfall 3: Sortino / IR Zero-Division
**What goes wrong:** If all returns are positive, downside deviation is zero. If all rolling IC values are identical, IC std is zero. Both cause division by zero.
**Why it happens:** Short evaluation periods or strong trending periods may have no negative returns. Feature with constant IC has zero std.
**How to avoid:** Check for zero denominator before dividing. Return `float('inf')` or `0.0` based on sign of numerator (see Pattern 1 and Pattern 2 examples above).
**Warning signs:** NaN or inf in Sortino ratio or IR column.

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

### Pitfall 6: IC Computed on Too-Few Samples
**What goes wrong:** Computing Spearman correlation on < 10 samples gives unreliable IC values. With short evaluation periods and regime subsetting, feature IC might be computed on very small samples.
**Why it happens:** A 1-year test period has ~252 trading days. If features change per LLM iteration, some periods may have even fewer samples.
**How to avoid:** Minimum sample size check (>= 10) in `information_coefficient()`. Report count alongside IC. Flag low-confidence IC values.
**Warning signs:** IC values jumping wildly between iterations; IC computed on 5 samples reported as "significant."

### Pitfall 7: Feature Dimension Mismatch in factor_metrics
**What goes wrong:** If `revise_state()` returns different numbers of features across iterations (which happens in the current code -- LLM generates arbitrary dimensions), the feature_matrix passed to `compute_factor_metrics()` may have inconsistent columns.
**Why it happens:** The current architecture allows LLM to return any number of features. The new architecture (D-02) uses a fixed feature library with consistent dimensions, so this pitfall is primarily a concern during the transition period.
**How to avoid:** In the new architecture, feature dimensions are determined by the JSON selection and are consistent. During evaluate(), collect feature values from `enhanced[123:]` which always matches the current model's feature dim. Store feature_names alongside for interpretability.
**Warning signs:** `feature_matrix` shape changes between evaluate() calls on the same model.

## Code Examples

### Extended evaluate() Method With Factor Metrics
```python
# Source: Extends dqn_trainer.py lines 323-359 + adds D-11/D-12 factor evaluation

def evaluate(self, data_loader, start_date, end_date):
    """Evaluate on test data with full metrics + factor evaluation + regime breakdown."""
    dates = [d for d in data_loader.get_date_range()
             if start_date <= str(d) <= end_date]

    daily_returns = []
    forward_returns = []       # For IC computation
    feature_values = []        # Per-step feature values
    regime_vectors = []
    prev_price = None
    current_position = 0

    for i, date in enumerate(dates):
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

            # Collect features for factor evaluation
            if len(enhanced) > 123:
                feature_values.append(enhanced[123:].copy())

            # Forward return for IC
            if i + 1 < len(dates):
                next_price = data_loader.get_ticker_price_by_date(
                    self.ticker, dates[i + 1])
                if current_price > 0:
                    forward_returns.append(
                        (next_price - current_price) / current_price)

        prev_price = current_price

    # Strategy performance
    from evaluation.metrics import sharpe, sortino, max_drawdown, calmar, win_rate
    strategy_metrics = {
        'sharpe': sharpe(daily_returns),
        'sortino': sortino(daily_returns),
        'max_dd': max_drawdown(daily_returns),
        'calmar': calmar(daily_returns),
        'win_rate': win_rate(daily_returns),
        'total_return': sum(daily_returns) * 100,
    }

    # Factor evaluation (D-11/D-12)
    from evaluation.metrics import compute_factor_metrics
    factor_metrics = {}
    if len(feature_values) > 20 and len(forward_returns) > 20:
        min_len = min(len(feature_values), len(forward_returns))
        feat_matrix = np.array(feature_values[:min_len])
        fwd_rets = np.array(forward_returns[:min_len])
        factor_metrics = compute_factor_metrics(feat_matrix, fwd_rets)

    # Regime-stratified metrics
    from evaluation.regime_evaluator import evaluate_by_regime
    regime_metrics = evaluate_by_regime(daily_returns, regime_vectors)

    return {
        **strategy_metrics,
        'trades': [],
        'regime_metrics': regime_metrics,
        'factor_metrics': factor_metrics,
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
        name = result_dir.name
        window = int(name.split('SW')[1].split('_')[0])
        test_year = int(name.split('test')[1])

        for ticker, ticker_data in data.items():
            if ticker_data.get('error'):
                continue
            for method in ['lesr_test', 'baseline_test']:
                metrics = ticker_data[method]
                row = {
                    'window': window, 'test_year': test_year,
                    'ticker': ticker,
                    'method': 'LESR' if 'lesr' in method else 'Baseline',
                    'sharpe': metrics['sharpe'],
                    'max_dd': metrics['max_dd'],
                    'total_return': metrics['total_return'],
                    'sortino': metrics.get('sortino', 0),
                    'calmar': metrics.get('calmar', 0),
                    'win_rate': metrics.get('win_rate', 0),
                }
                # Include factor metrics if available
                fm = metrics.get('factor_metrics', {})
                if fm and 'aggregate' in fm:
                    row['mean_ic'] = fm['aggregate'].get('mean_ic', 0)
                    row['mean_ir'] = fm['aggregate'].get('mean_ir', 0)
                    row['mean_spread'] = fm['aggregate'].get('mean_spread', 0)
                all_rows.append(row)
    return pd.DataFrame(all_rows)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 3-metric eval (Sharpe, MaxDD, Return) | 6-metric eval (add Sortino, Calmar, WinRate) | This phase | More robust assessment, catches risk-adjusted performance |
| No factor evaluation | IC/IR/Quantile Spread per feature | This phase (D-11/D-12) | Per-feature predictive power, academic contribution (D-13) |
| Fixed window train/test | Walk-forward sliding windows | exp4.9_c (already built) | Out-of-sample validation, publication-standard |
| No leakage guard | Explicit train-only COT feedback | This phase | Prevents LLM from overfitting to validation set |
| Overall metrics only | Regime-stratified breakdown | This phase | Identifies regime-dependent weaknesses |
| Per-window reports | Cross-experiment aggregation | This phase | Publication-ready comparison tables |

**Deprecated/outdated:**
- `exp4.7` evaluate() returning only 3 metrics -- being superseded
- `exp4.7` fixed-window experiments -- replaced by sliding window design
- `feature_quality.py` correlation-only analysis -- superseded by IC/IR/Quantile Spread which are standard in factor research

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Sortino, Calmar, Win Rate are simple enough to hand-roll without empyrical | Standard Stack | Low -- these are well-defined formulas with known edge cases |
| A2 | Current walk-forward design (3yr-train/1yr-val/1yr-test sliding by 1 year) is sufficient for EVAL-01 | Architecture Patterns | Medium -- user specified "Claude's discretion" on window sizes |
| A3 | Regime labels from trend_direction thresholds (+/-0.3) are adequate for bull/bear/sideways classification | Regime Evaluation | Low -- these thresholds are already established in the codebase |
| A4 | empyrical/alphalens not needed because financial metrics and factor metrics are trivial numpy/scipy operations | Standard Stack | Low -- ~15 lines per strategy metric, ~40 lines for IC/IR/spread, well-tested edge cases documented |
| A5 | The current `evaluate()` not appending to episode_states/rewards will remain true | Pitfall 5 | Low -- verified by reading current code |
| A6 | Spearman rank correlation is the correct IC method for financial factor evaluation (vs Pearson) | Factor Evaluation | Low -- this is standard practice in quantitative finance factor research [CITED: CONTEXT.md specifics section, multiple web search results confirm] |
| A7 | Rolling IC window of 20 days is a reasonable default | Factor Evaluation | Medium -- standard in factor research but may need tuning per stock/period. Exposed as parameter. |
| A8 | IC significance threshold of |IC| > 0.03 is reasonable for "significant" flag | Factor Evaluation | Medium -- this is a rough heuristic. In practice, significance should be determined by the spearmanr p-value, but the threshold flag is for quick visual scanning in reports. |

## Open Questions

1. **Should factor metrics be computed during training (for COT feedback) or only during evaluation?**
   - What we know: D-12 says evaluate() returns factor_metrics. D-06 says COT must use train-only data.
   - What's unclear: Whether COT feedback should include per-feature IC values so LLM knows which features are predictive.
   - Recommendation: Yes, compute factor metrics on training data for COT feedback. This gives LLM precise per-feature signal ("RSI(14) has IC=0.15, Momentum(10) has IC=0.02 -- focus on RSI-like indicators"). This is the whole point of D-13: feedback based on factor predictive power, not just strategy Sharpe.

2. **Feature names for factor_metrics output -- how to get them?**
   - What we know: In the new architecture (D-02), LLM outputs JSON with indicator names and params. The feature library computes them. But evaluate() only sees the enhanced state array, not the names.
   - What's unclear: How to pass feature names from the feature library through to evaluate()'s output.
   - Recommendation: Store feature_names as an attribute on DQNTrainer (set during initialization when the feature library is configured). evaluate() reads self.feature_names when calling compute_factor_metrics().

3. **Format for publication-ready tables (EVAL-05)?**
   - What we know: Claude's discretion per CONTEXT.md. Current sliding_summary.py generates markdown tables.
   - What's unclear: Whether LaTeX format is needed for paper submission.
   - Recommendation: Generate markdown by default (consistent with existing pattern). Add optional `--format latex` flag to the cross-report script. LaTeX can be generated from the same DataFrame with `to_latex()`.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.8 | Core runtime | Yes | 3.8.20 | -- |
| numpy | Metric computation | Yes | 1.24.4 | -- |
| pandas | DataFrame aggregation | Yes | 2.0.3 | -- |
| scipy | Statistical tests, spearmanr | Yes | 1.10.1 | -- |
| torch | DQN training | Yes | 2.4.1 | -- |
| pyyaml | Config loading | Yes | installed | -- |
| sklearn | Feature importance | Yes | installed | -- |
| pytest | Test framework | Yes | installed | -- |
| CUDA GPU | DQN training | Yes | 4x GPU | CPU fallback exists |
| empyrical | NOT needed | No | -- | Hand-roll strategy metrics |
| alphalens | NOT needed | No | -- | Hand-roll factor evaluation metrics |

**Missing dependencies with no fallback:**
- None -- all required dependencies are installed.

**Missing dependencies with fallback:**
- empyrical/alphalens: Not needed; strategy metrics and factor evaluation metrics are simple numpy/scipy operations.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | None -- pytest auto-discovers tests/ directories |
| Quick run command | `pytest exp4.9_c/tests/ -x -q` |
| Full suite command | `pytest exp4.9_c/tests/ -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-01 | Walk-forward windows generate sequential configs | unit | `pytest exp4.9_c/tests/test_walk_forward.py::test_sliding_config_generation -x` | No -- Wave 0 |
| EVAL-01 | Walk-forward windows produce out-of-sample results | integration | `pytest exp4.9_c/tests/test_walk_forward.py::test_out_of_sample_results -x` | No -- Wave 0 |
| EVAL-02 | evaluate() returns all 6 strategy metrics | unit | `pytest exp4.9_c/tests/test_metrics.py::test_evaluate_returns_six_metrics -x` | No -- Wave 0 |
| EVAL-02 | Sortino computed correctly on known input | unit | `pytest exp4.9_c/tests/test_metrics.py::test_sortino_known_input -x` | No -- Wave 0 |
| EVAL-02 | Calmar computed correctly on known input | unit | `pytest exp4.9_c/tests/test_metrics.py::test_calmar_known_input -x` | No -- Wave 0 |
| EVAL-02 | Win rate computed correctly on known input | unit | `pytest exp4.9_c/tests/test_metrics.py::test_win_rate_known_input -x` | No -- Wave 0 |
| D-11 | IC computed correctly with Spearman correlation | unit | `pytest exp4.9_c/tests/test_factor_metrics.py::test_ic_spearman -x` | No -- Wave 0 |
| D-11 | Rolling IC produces correct-length series | unit | `pytest exp4.9_c/tests/test_factor_metrics.py::test_rolling_ic_length -x` | No -- Wave 0 |
| D-11 | IR computed from rolling IC series | unit | `pytest exp4.9_c/tests/test_factor_metrics.py::test_information_ratio -x` | No -- Wave 0 |
| D-11 | Quantile spread top-minus-bottom return | unit | `pytest exp4.9_c/tests/test_factor_metrics.py::test_quantile_spread -x` | No -- Wave 0 |
| D-12 | compute_factor_metrics returns per-feature and aggregate | unit | `pytest exp4.9_c/tests/test_factor_metrics.py::test_compute_factor_metrics -x` | No -- Wave 0 |
| D-12 | evaluate() returns factor_metrics dict | unit | `pytest exp4.9_c/tests/test_metrics.py::test_evaluate_returns_factor_metrics -x` | No -- Wave 0 |
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
- [ ] `exp4.9_c/tests/conftest.py` -- shared fixtures (mock data_loader, sample returns, sample regime vectors, synthetic feature matrix + forward returns)
- [ ] `exp4.9_c/tests/test_metrics.py` -- covers EVAL-02 (Sortino, Calmar, WinRate with known inputs) + D-12 (evaluate returns factor_metrics)
- [ ] `exp4.9_c/tests/test_factor_metrics.py` -- covers D-11 (IC, IR, Quantile Spread with known inputs, edge cases)
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
| V5 Input Validation | Yes (partial) | LLM JSON output parsing should validate structure; YAML config loading should use `yaml.safe_load` |
| V6 Cryptography | No | No cryptographic operations |

### Known Threat Patterns for LESR Evaluation

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| API key exposure in config YAML files | Information Disclosure | Config files should use env vars; current configs contain hardcoded keys (acceptable for research, flag for cleanup) |
| LLM output injection (malformed JSON) | Tampering | Validate LLM JSON output structure before processing; catch exceptions in feature library calls |
| Pickle deserialization of untrusted data | Tampering | Results pickle files are researcher-generated, not external; acceptable risk for research project |

## Sources

### Primary (HIGH confidence)
- Codebase analysis of `exp4.9_c/dqn_trainer.py` -- evaluate() structure, _sharpe(), _max_dd() patterns, state layout [120 raw + 3 regime + N features]
- Codebase analysis of `exp4.9_c/lesr_controller.py` -- COT feedback path, _parallel_train flow, _generate_cot_feedback
- Codebase analysis of `exp4.9_c/regime_detector.py` -- 3-dim regime vector definition
- Codebase analysis of `exp4.9_c/run_sliding_parallel.py` -- sliding window infrastructure
- Codebase analysis of `exp4.9_c/sliding_summary.py` -- current aggregation reporting
- Codebase analysis of `exp4.9_c/build_sliding_configs.py` -- window config generation
- Codebase analysis of `exp4.9_c/feature_analyzer.py` -- existing Spearman correlation usage for feature analysis
- Codebase analysis of `exp4.7/diagnosis/analyze_existing.py` -- Phase 1 analysis foundation
- Codebase analysis of `exp4.7/diagnosis/stats_reporter.py` -- statistical comparison tools
- Codebase analysis of `exp4.7/diagnosis/feature_quality.py` -- existing Spearman correlation + information ratio computation
- Environment verification via `pip list`, `python --version` on lesr conda env (2026-04-15)
- Functional verification of `scipy.stats.spearmanr` on lesr env (2026-04-15)
- Functional verification of `numpy.quantile` for quantile spread computation on lesr env (2026-04-15)

### Secondary (MEDIUM confidence)
- CONTEXT.md specifics section for IC/IR/Quantile Spread function signatures and design decisions [CITED: CONTEXT.md lines 106-109]
- Financial metric formulas (Sortino, Calmar, Win Rate) -- standard definitions from quantitative finance literature [ASSUMED]
- Walk-forward validation methodology -- standard approach in quantitative trading research [ASSUMED]
- IC methodology -- Spearman rank correlation is the standard method for Information Coefficient in factor research [CITED: PyQuant News, Quant Science, multiple search results confirm]

### Tertiary (LOW confidence)
- None -- all findings are either codebase-verified or clearly tagged as ASSUMED

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all packages verified via pip list on lesr env, no new dependencies needed
- Architecture: HIGH -- extends existing patterns, all canonical refs read and analyzed
- Factor evaluation metrics: HIGH -- scipy.stats.spearmanr verified working, numpy.quantile verified working, formulas are standard in quantitative finance
- Pitfalls: HIGH -- derived from direct code analysis of existing evaluate(), COT path, sliding window flow, and factor metric computation edge cases

**Research date:** 2026-04-15
**Valid until:** 2026-05-15 (stable -- no fast-moving dependencies)
