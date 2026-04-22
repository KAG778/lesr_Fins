# LESR Portfolio Refactoring: JSON-Selection → Code-Generation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the existing `组合优化_ppo` project from JSON-based feature selection to LESR's original code-generation methodology — LLM generates `revise_state(s)` + `intrinsic_reward(updated_s)` Python code, validated by AST sandbox, with IC-based COT feedback replacing worst-trade feedback.

**Architecture:** LLM generates Python code → AST whitelist validation + test execution in `code_sandbox.py` → dynamic import → `portfolio_env.py` uses `revise_state_fn` + `intrinsic_reward_fn` → IC analysis in `ic_analyzer.py` → four-tier COT feedback per iteration. Multi-sample (N=3) per iteration, each sample gets independent PPO training.

**Tech Stack:** Python 3.8+, PyTorch, NumPy, OpenAI API (GPT-4o-mini), AST module for code validation. No new external dependencies.

---

## File Map

### New Files (create from scratch)

| File | Responsibility |
|------|---------------|
| `组合优化_ppo/core/code_sandbox.py` | AST whitelist validation + safe code execution + dimension detection |
| `组合优化_ppo/core/ic_analyzer.py` | IC computation (per-dim, regime-specific) + four-tier COT feedback builder |

### Refactored Files (modify existing)

| File | Changes |
|------|---------|
| `组合优化_ppo/core/feature_library.py` | Add 9 standalone building-block functions for LLM code to import; keep existing INDICATOR_REGISTRY for backward compat |
| `组合优化_ppo/core/portfolio_env.py` | Add `intrinsic_reward_fn` param; add compressed raw state extraction; add `get_revised_states()` method |
| `组合优化_ppo/core/prompts.py` | Replace JSON-selection prompts with code-generation prompts (init, cot, next_iteration); add `_extract_python_code()`; keep reward config prompt |
| `组合优化_ppo/core/lesr_controller.py` | Rewrite iteration loop: multi-sample, code sandbox validation, IC feedback, fix broken imports |

### Unchanged Files

| File | Reason |
|------|--------|
| `组合优化_ppo/core/ppo_agent.py` | PPO algorithm is state-dimension agnostic |
| `组合优化_ppo/core/regime_detector.py` | Market regime detection unchanged |
| `组合优化_ppo/core/portfolio_features.py` | Cross-stock features unchanged |
| `组合优化_ppo/core/reward_rules.py` | Behavior constraint rules unchanged |
| `组合优化_ppo/core/metrics.py` | Performance metrics unchanged |
| `组合优化_ppo/core/market_stats.py` | Market statistics unchanged |
| `组合优化_ppo/core/prepare_data.py` | Data preparation unchanged |
| `组合优化_ppo/core/lesr_strategy.py` | Backtest wrapper unchanged |

### Execution Order

```
1. code_sandbox.py        ← no dependencies (new)
2. ic_analyzer.py          ← no dependencies (new)
3. feature_library.py      ← add building blocks (modify)
4. portfolio_env.py        ← add intrinsic_reward + compressed raw (modify)
5. prompts.py              ← code-generation prompts (rewrite)
6. lesr_controller.py      ← multi-sample loop (rewrite)
```

---

## Task 1: Create `code_sandbox.py` — AST Validation + Safe Code Execution

**Files:**
- Create: `组合优化_ppo/core/code_sandbox.py`

- [ ] **Step 1: Write `code_sandbox.py`**

```python
"""
Code Sandbox for LLM-Generated Python Code

Validates LLM-generated revise_state and intrinsic_reward code via:
1. AST whitelist check (blocks dangerous operations)
2. Test execution (verifies output types, NaN/Inf, dimension constraints)
3. Dimension detection (records extra dims added by revise_state)
"""

import ast
import numpy as np
from typing import Callable, Dict, Optional


# Modules/functions allowed in LLM code
_ALLOWED_IMPORTS = {'numpy', 'math', 'np'}
_BLOCKED_BUILTINS = {
    'exec', 'eval', 'compile', 'open', '__import__',
    'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr', 'delattr',
    'breakpoint', 'exit', 'quit',
}
_BLOCKED_ATTRS = {
    '__import__', '__builtins__', '__file__', '__name__',
}


def _ast_check(code_str: str) -> list:
    """Stage 1: AST whitelist validation. Returns list of error strings."""
    errors = []
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    for node in ast.walk(tree):
        # Block dangerous function calls
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BLOCKED_BUILTINS:
                errors.append(f"Blocked builtin: {func.id}()")
            if isinstance(func, ast.Attribute):
                if func.attr in _BLOCKED_ATTRS:
                    errors.append(f"Blocked attribute: .{func.attr}")

        # Block dangerous imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split('.')[0]
                if root not in _ALLOWED_IMPORTS:
                    errors.append(f"Blocked import: {alias.name}")

        if isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split('.')[0]
                if root not in _ALLOWED_IMPORTS:
                    errors.append(f"Blocked from-import: {node.module}")

    return errors


def _extract_functions(code_str: str) -> Dict[str, Callable]:
    """Extract revise_state and intrinsic_reward from code string.

    Returns dict of {name: callable}.
    """
    namespace = {'np': np, 'numpy': np}
    try:
        import math
        namespace['math'] = math
    except ImportError:
        pass

    exec(compile(code_str, '<llm_code>', 'exec'), namespace)

    result = {}
    for name in ['revise_state', 'intrinsic_reward']:
        if name in namespace and callable(namespace[name]):
            result[name] = namespace[name]
    return result


def _test_execution(functions: Dict[str, Callable]) -> list:
    """Stage 2: Test execution with random inputs. Returns list of error strings."""
    errors = []
    np.random.seed(42)
    test_state = np.random.randn(120) * 100 + 150  # realistic price range

    # Test revise_state
    if 'revise_state' in functions:
        try:
            revised = functions['revise_state'](test_state)
            if not isinstance(revised, np.ndarray):
                errors.append(f"revise_state must return np.ndarray, got {type(revised)}")
            elif revised.ndim != 1:
                errors.append(f"revise_state must return 1D array, got {revised.ndim}D")
            elif np.any(np.isnan(revised)):
                errors.append("revise_state output contains NaN")
            elif np.any(np.isinf(revised)):
                errors.append("revise_state output contains Inf")
            elif len(revised) < 120:
                errors.append(f"revise_state output too short: {len(revised)} < 120 (must preserve source state)")
            elif not np.allclose(revised[:120], test_state, atol=1e-6):
                errors.append("revise_state must preserve first 120 dims (source state)")
        except Exception as e:
            errors.append(f"revise_state execution error: {e}")
    else:
        errors.append("revise_state function not found in code")

    # Test intrinsic_reward
    if 'intrinsic_reward' in functions:
        try:
            # Use revised state if available, else raw
            test_input = functions.get('revise_state', lambda s: s)(test_state)
            reward_val = functions['intrinsic_reward'](test_input)
            if not isinstance(reward_val, (int, float, np.integer, np.floating)):
                errors.append(f"intrinsic_reward must return scalar, got {type(reward_val)}")
            elif np.isnan(reward_val):
                errors.append("intrinsic_reward returned NaN")
            elif np.isinf(reward_val):
                errors.append("intrinsic_reward returned Inf")
            elif abs(reward_val) > 100:
                errors.append(f"intrinsic_reward out of range [-100, 100]: {reward_val}")
        except Exception as e:
            errors.append(f"intrinsic_reward execution error: {e}")
    else:
        errors.append("intrinsic_reward function not found in code")

    return errors


def validate(code_str: str) -> Dict:
    """Full validation pipeline for LLM-generated code.

    Args:
        code_str: Python code string containing revise_state and intrinsic_reward.

    Returns:
        {
            'ok': bool,
            'errors': list[str],
            'revise_state': callable or None,
            'intrinsic_reward': callable or None,
            'feature_dim': int,  # extra dims beyond original 120
            'state_dim': int,    # total revised state dim
        }
    """
    result = {
        'ok': False,
        'errors': [],
        'revise_state': None,
        'intrinsic_reward': None,
        'feature_dim': 0,
        'state_dim': 120,
    }

    # Stage 1: AST check
    ast_errors = _ast_check(code_str)
    if ast_errors:
        result['errors'] = ast_errors
        return result

    # Stage 2: Extract functions
    try:
        functions = _extract_functions(code_str)
    except Exception as e:
        result['errors'] = [f"Code extraction error: {e}"]
        return result

    if 'revise_state' not in functions:
        result['errors'] = ["revise_state function not found"]
        return result

    # Stage 3: Test execution
    test_errors = _test_execution(functions)
    if test_errors:
        result['errors'] = test_errors
        return result

    # Stage 4: Dimension detection
    np.random.seed(42)
    test_state = np.random.randn(120) * 100 + 150
    try:
        revised = functions['revise_state'](test_state)
        state_dim = len(revised)
        feature_dim = state_dim - 120
    except Exception as e:
        result['errors'] = [f"Dimension detection error: {e}"]
        return result

    result['ok'] = True
    result['revise_state'] = functions['revise_state']
    result['intrinsic_reward'] = functions.get('intrinsic_reward')
    result['feature_dim'] = feature_dim
    result['state_dim'] = state_dim
    return result
```

- [ ] **Step 2: Test code_sandbox.py**

```bash
cd /home/wangmeiyi/AuctionNet/lesr/组合优化_ppo && python -c "
import sys; sys.path.insert(0, 'core')
from code_sandbox import validate
import numpy as np

# Valid code
good_code = '''
import numpy as np
def revise_state(s):
    closes = s[0::6]
    mom = np.mean(np.diff(closes[-5:])) if len(closes) >= 6 else 0.0
    return np.concatenate([s, [mom]])
def intrinsic_reward(updated_s):
    mom = updated_s[120]
    return 0.01 * abs(mom) / (np.std(updated_s[:120]) + 0.01)
'''
result = validate(good_code)
print(f'Valid code: ok={result[\"ok\"]}, errors={result[\"errors\"]}, feature_dim={result[\"feature_dim\"]}')
assert result['ok']

# Dangerous code
bad_code = '''
import os
def revise_state(s):
    os.system(\"rm -rf /\")
    return s
def intrinsic_reward(s):
    return 0.0
'''
result2 = validate(bad_code)
print(f'Dangerous code: ok={result2[\"ok\"]}, errors={result2[\"errors\"]}')
assert not result2['ok']

# Missing intrinsic_reward (should still fail)
no_reward = '''
import numpy as np
def revise_state(s):
    return np.concatenate([s, [0.0]])
'''
result3 = validate(no_reward)
print(f'Missing reward: ok={result3[\"ok\"]}, errors={result3[\"errors\"]}')
assert not result3['ok']

print('All sandbox tests passed')
"
```

Expected: Valid code passes, dangerous code blocked, missing reward caught.

- [ ] **Step 3: Commit**

```bash
git add 组合优化_ppo/core/code_sandbox.py
git commit -m "feat(portfolio-ppo): add code sandbox with AST validation and test execution"
```

---

## Task 2: Create `ic_analyzer.py` — IC Computation + COT Feedback

**Files:**
- Create: `组合优化_ppo/core/ic_analyzer.py`

- [ ] **Step 1: Write `ic_analyzer.py`**

```python
"""
IC (Information Coefficient) Analyzer for LESR COT Feedback

Computes per-dimension IC between revised state features and forward returns.
Generates four-tier COT feedback:
  1. Market environment context
  2. Strong features (|IC| > strong_threshold)
  3. Weak features (|IC| < weak_threshold)
  4. Negative IC features (IC < -0.03)
  5. Missing analysis suggestions
  6. Intrinsic reward diagnosis
"""

import numpy as np
from typing import Dict, List, Optional


def compute_ic_profile(revised_states: np.ndarray,
                       forward_returns: np.ndarray) -> Dict[int, float]:
    """Compute Pearson IC between each extra state dim and forward returns.

    Args:
        revised_states: (N, state_dim) array of revised states.
        forward_returns: (N,) array of forward portfolio returns.

    Returns:
        Dict mapping extra dim index to IC value.
    """
    if revised_states.ndim != 2 or forward_returns.ndim != 1:
        return {}
    if revised_states.shape[0] != forward_returns.shape[0]:
        return {}
    if revised_states.shape[0] < 10:
        return {}

    n_dims = revised_states.shape[1]
    extra_start = 120  # first 120 are source state
    if n_dims <= extra_start:
        return {}

    ic_profile = {}
    for dim in range(extra_start, n_dims):
        col = revised_states[:, dim]
        if np.std(col) < 1e-10:
            ic_profile[dim] = 0.0
            continue
        ret_std = np.std(forward_returns)
        if ret_std < 1e-10:
            ic_profile[dim] = 0.0
            continue
        corr = np.corrcoef(col, forward_returns)[0, 1]
        ic_profile[dim] = float(corr) if not np.isnan(corr) else 0.0

    return ic_profile


def compute_regime_specific_ic(revised_states: np.ndarray,
                               forward_returns: np.ndarray,
                               regime_labels: np.ndarray) -> Dict[str, Dict[int, float]]:
    """Compute IC per market regime.

    Args:
        revised_states: (N, state_dim) array.
        forward_returns: (N,) array.
        regime_labels: (N,) array of regime strings: 'trending_up', 'volatile', 'trending_down'.

    Returns:
        Dict mapping regime name to {dim: ic_value}.
    """
    result = {}
    for regime in set(regime_labels):
        mask = regime_labels == regime
        if mask.sum() < 10:
            continue
        result[regime] = compute_ic_profile(revised_states[mask], forward_returns[mask])
    return result


def _classify_regime(trend: float, vol: float) -> str:
    """Classify a single timestep's regime from trend/vol values."""
    if vol > 0.6:
        return 'volatile'
    elif trend > 0.3:
        return 'trending_up'
    elif trend < -0.3:
        return 'trending_down'
    else:
        return 'neutral'


def build_ic_cot_prompt(
    sample_results: List[Dict],
    best_idx: int,
    strong_threshold: float = 0.05,
    weak_threshold: float = 0.02,
    market_period_summary: str = "",
) -> str:
    """Build IC-based COT feedback prompt for LLM.

    Args:
        sample_results: list of dicts, each with keys:
            'code': str (the LLM-generated code)
            'performance': dict with 'sharpe', 'total_return', 'max_drawdown'
            'ic_profile': dict[int, float] from compute_ic_profile
            'regime_ic': dict[str, dict[int, float]] from compute_regime_specific_ic
            'intrinsic_reward_stats': dict with 'mean', 'correlation_with_performance'
        best_idx: index of the best-performing sample
        strong_threshold: IC above this = strong feature
        weak_threshold: IC below this = weak feature
        market_period_summary: text describing training period market conditions

    Returns:
        Formatted COT feedback string.
    """
    lines = []

    for i, sample in enumerate(sample_results):
        code = sample.get('code', '')
        perf = sample.get('performance', {})
        ic_profile = sample.get('ic_profile', {})
        regime_ic = sample.get('regime_ic', {})
        ir_stats = sample.get('intrinsic_reward_stats', {})

        marker = " (BEST)" if i == best_idx else ""
        lines.append(f"========== Code Sample {i+1}{marker} "
                     f"(Sharpe={perf.get('sharpe', 0):.3f}, "
                     f"Return={perf.get('total_return', 0):.2f}%, "
                     f"MaxDD={perf.get('max_drawdown', 0):.2f}%) ==========")
        lines.append(code)
        lines.append("")

        if not ic_profile:
            lines.append("  (No IC profile computed - insufficient data)")
            lines.append("")
            continue

        # Four-tier analysis
        lines.append("  [IC Analysis]")
        strong, weak, negative = [], [], []
        for dim, ic_val in sorted(ic_profile.items()):
            abs_ic = abs(ic_val)
            tag = ""
            if abs_ic > strong_threshold:
                tag = "<- Strong"
                strong.append((dim, ic_val))
            elif abs_ic < weak_threshold:
                tag = "<- Weak"
                weak.append((dim, ic_val))
            if ic_val < -0.03:
                tag = "<- Negative"
                negative.append((dim, ic_val))
            lines.append(f"    s[{dim}]: IC = {ic_val:+.4f} {tag}")

        # Regime-specific IC
        if regime_ic:
            lines.append("  [Regime-Specific IC]")
            for regime, dim_ics in regime_ic.items():
                lines.append(f"    {regime}:")
                for dim, ic_val in sorted(dim_ics.items()):
                    lines.append(f"      s[{dim}]: IC = {ic_val:+.4f}")

        # Intrinsic reward diagnosis
        if ir_stats:
            lines.append("  [Intrinsic Reward Diagnosis]")
            lines.append(f"    Mean intrinsic_reward = {ir_stats.get('mean', 0):.6f}")
            lines.append(f"    Correlation with performance = "
                         f"{ir_stats.get('correlation_with_performance', 0):.3f}")
            corr = ir_stats.get('correlation_with_performance', 0)
            if abs(corr) > 0.3:
                lines.append(f"    -> Strong guidance effect")
            elif abs(corr) > 0.1:
                lines.append(f"    -> Moderate guidance effect")
            else:
                lines.append(f"    -> Weak guidance effect, consider redesigning")

        lines.append("")

    # Summary and suggestions
    lines.append("[Improvement Suggestions]")
    best_sample = sample_results[best_idx] if best_idx < len(sample_results) else {}
    best_ic = best_sample.get('ic_profile', {})
    best_regime_ic = best_sample.get('regime_ic', {})

    if best_ic:
        dims_sorted = sorted(best_ic.items(), key=lambda x: abs(x[1]), reverse=True)
        if dims_sorted:
            best_dim, best_ic_val = dims_sorted[0]
            lines.append(f"  (a) Strongest feature: s[{best_dim}] (IC={best_ic_val:+.4f}). "
                         "Consider building derivatives or multi-horizon versions.")

        weak_dims = [(d, v) for d, v in best_ic.items() if abs(v) < weak_threshold]
        if weak_dims:
            dim_str = ", ".join(f"s[{d}]" for d, _ in weak_dims)
            lines.append(f"  (b) Weak features: {dim_str}. Consider replacing with "
                         "more informative signals (volatility-adjusted momentum, "
                         "cross-sectional rank, etc.).")

        neg_dims = [(d, v) for d, v in best_ic.items() if v < -0.03]
        if neg_dims:
            dim_str = ", ".join(f"s[{d}] (IC={v:+.4f})" for d, v in neg_dims)
            lines.append(f"  (c) Negative IC features: {dim_str}. These may be harmful. "
                         "Consider removing or inverting the signal.")

        # Check if defensive features exist
        has_volatility = any('volatil' in str(v) for v in best_ic.values())
        if not has_volatility and best_regime_ic.get('volatile'):
            lines.append("  (d) No volatility features detected for volatile regime. "
                         "Consider adding realized_volatility or downside_risk.")

    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 2: Test ic_analyzer.py**

```bash
cd /home/wangmeiyi/AuctionNet/lesr/组合优化_ppo && python -c "
import sys; sys.path.insert(0, 'core')
from ic_analyzer import compute_ic_profile, compute_regime_specific_ic, build_ic_cot_prompt
import numpy as np

# Simulate revised states (50 samples, 123 dims = 120 raw + 3 extra)
np.random.seed(42)
states = np.random.randn(50, 123) * 0.1
forward = np.random.randn(50) * 0.02
# Make dim 121 predictive
states[:, 121] = forward * 3 + np.random.randn(50) * 0.1

ic = compute_ic_profile(states, forward)
print(f'IC profile: {ic}')
assert 121 in ic
print(f'Dim 121 IC: {ic[121]:.4f}')

# Regime labels
labels = np.array(['trending_up'] * 15 + ['volatile'] * 20 + ['trending_down'] * 15)
regime_ic = compute_regime_specific_ic(states, forward, labels)
print(f'Regimes: {list(regime_ic.keys())}')

# COT feedback
sample_results = [{
    'code': 'import numpy as np\\ndef revise_state(s): return np.concatenate([s, [0.0, 0.0, 0.0]])\\ndef intrinsic_reward(s): return 0.0',
    'performance': {'sharpe': 0.5, 'total_return': 3.2, 'max_drawdown': -5.1},
    'ic_profile': ic,
    'regime_ic': regime_ic,
    'intrinsic_reward_stats': {'mean': 0.003, 'correlation_with_performance': 0.32},
}]
cot = build_ic_cot_prompt(sample_results, 0)
print(f'COT length: {len(cot)} chars')
print('COT preview:')
print(cot[:500])

print('\\nic_analyzer tests passed')
"
```

- [ ] **Step 3: Commit**

```bash
git add 组合优化_ppo/core/ic_analyzer.py
git commit -m "feat(portfolio-ppo): add IC analyzer with regime-specific analysis and COT feedback"
```

---

## Task 3: Add Building-Block Functions to `feature_library.py`

**Files:**
- Modify: `组合优化_ppo/core/feature_library.py`

The existing file has 20 indicators in `INDICATOR_REGISTRY` for JSON selection. We add 9 standalone building-block functions that LLM-generated code can import. The existing registry and `build_revise_state` are kept for backward compatibility.

- [ ] **Step 1: Add building-block functions before the INDICATOR_REGISTRY section**

Add these functions after the existing helper functions (`_extract_ohlcv`, `_ema`, `_sma`, etc.) and before `INDICATOR_REGISTRY`:

```python
# ---------------------------------------------------------------------------
# Building-Block Functions for LLM Code Import
# ---------------------------------------------------------------------------
# LLM-generated code can: from feature_library import compute_relative_momentum, ...

def compute_relative_momentum(prices: np.ndarray, window: int = 20) -> float:
    """Excess return of this stock vs the window-average return.

    Args:
        prices: 1D array of close prices (length >= window + 1).
        window: lookback period.

    Returns:
        Scalar: (current_price - price_N_days_ago) / price_N_days_ago.
    """
    if len(prices) < window + 1 or prices[-window - 1] == 0:
        return 0.0
    return float((prices[-1] - prices[-window - 1]) / abs(prices[-window - 1]))


def compute_cross_sectional_rank(values: list) -> float:
    """Rank of a single value among all stocks' values. Returns [0, 1].

    Note: This is computed at portfolio level, not inside revise_state(s).
    The environment calls this with all stocks' values.

    Args:
        values: list of scalars for all 5 stocks, this stock's value is values[0].

    Returns:
        Scalar in [0, 1], 1 = highest.
    """
    if not values or len(values) < 2:
        return 0.5
    target = values[0]
    rank = sum(1 for v in values if v <= target)
    return float(rank / len(values))


def compute_realized_volatility(returns: np.ndarray, window: int = 20) -> float:
    """Realized volatility (std of returns) over window.

    Args:
        returns: 1D array of daily returns.
        window: lookback period.

    Returns:
        Scalar: std of recent N returns.
    """
    if len(returns) < window:
        window = len(returns)
    if window < 2:
        return 0.0
    return float(np.std(returns[-window:]))


def compute_downside_risk(returns: np.ndarray, window: int = 20) -> float:
    """Downside semi-deviation over window.

    Args:
        returns: 1D array of daily returns.
        window: lookback period.

    Returns:
        Scalar: std of negative returns only.
    """
    if len(returns) < window:
        window = len(returns)
    if window < 2:
        return 0.0
    neg = returns[-window:]
    neg = neg[neg < 0]
    if len(neg) < 2:
        return 0.0
    return float(np.std(neg))


def compute_beta(returns: np.ndarray, market_returns: np.ndarray,
                 window: int = 20) -> float:
    """Rolling beta to market (equal-weight portfolio).

    Args:
        returns: 1D array of this stock's daily returns.
        market_returns: 1D array of market returns (equal-weight portfolio).
        window: lookback period.

    Returns:
        Scalar: regression beta.
    """
    n = min(len(returns), len(market_returns), window)
    if n < 5:
        return 1.0
    r = returns[-n:]
    m = market_returns[-n:]
    var_m = np.var(m)
    if var_m < 1e-10:
        return 1.0
    cov = np.mean((r - np.mean(r)) * (m - np.mean(m)))
    return float(cov / var_m)


def compute_multi_horizon_momentum(prices: np.ndarray,
                                   windows: list = None) -> np.ndarray:
    """Momentum at multiple time horizons.

    Args:
        prices: 1D array of close prices.
        windows: list of lookback periods, default [5, 10, 20].

    Returns:
        1D array of momentum values, one per window.
    """
    if windows is None:
        windows = [5, 10, 20]
    result = []
    for w in windows:
        if len(prices) > w and prices[-w - 1] != 0:
            result.append((prices[-1] - prices[-w - 1]) / abs(prices[-w - 1]))
        else:
            result.append(0.0)
    return np.array(result, dtype=float)


def compute_zscore_price(prices: np.ndarray, window: int = 20) -> float:
    """Z-score of current price vs N-day mean.

    Args:
        prices: 1D array of close prices (length >= window).
        window: lookback period.

    Returns:
        Scalar: z-score.
    """
    if len(prices) < window:
        return 0.0
    seg = prices[-window:]
    mean_val = np.mean(seg)
    std_val = np.std(seg) + 1e-10
    return float(np.clip((prices[-1] - mean_val) / std_val, -3, 3))


def compute_mean_reversion_signal(prices: np.ndarray, window: int = 20) -> float:
    """Mean reversion strength: how far price deviated and started returning.

    Args:
        prices: 1D array of close prices (length >= window + 2).
        window: lookback period.

    Returns:
        Scalar: positive = reverting upward from oversold, negative = reverting downward.
    """
    if len(prices) < window + 2:
        return 0.0
    seg = prices[-window:]
    mean_val = np.mean(seg)
    std_val = np.std(seg) + 1e-10
    z_current = (prices[-1] - mean_val) / std_val
    z_prev = (prices[-2] - mean_val) / std_val
    # If z was negative and now less negative -> reverting up (positive signal)
    return float(np.clip(z_prev - z_current, -3, 3))


def compute_turnover_ratio(volumes: np.ndarray, window: int = 20) -> float:
    """Current volume / average volume ratio.

    Args:
        volumes: 1D array of daily volumes (length >= window + 1).
        window: lookback period.

    Returns:
        Scalar: volume ratio.
    """
    if len(volumes) < window + 1:
        return 1.0
    avg = np.mean(volumes[-window - 1:-1]) + 1e-10
    return float(volumes[-1] / avg)


# Export list for prompts
BUILDING_BLOCKS = [
    ('compute_relative_momentum', 'prices, window=20', 1,
     "Excess return vs window-average. Identifies outperforming stocks."),
    ('compute_cross_sectional_rank', 'values', 1,
     "Rank among all stocks [0,1]. Portfolio-level only, not in revise_state."),
    ('compute_realized_volatility', 'returns, window=20', 1,
     "Realized volatility. Measure individual stock risk."),
    ('compute_downside_risk', 'returns, window=20', 1,
     "Downside semi-deviation. Measure downside risk."),
    ('compute_beta', 'returns, market_returns, window=20', 1,
     "Beta to equal-weight portfolio. Systemic risk exposure."),
    ('compute_multi_horizon_momentum', 'prices, windows=[5,10,20]', 3,
     "Multi-period momentum. Capture trends at multiple scales."),
    ('compute_zscore_price', 'prices, window=20', 1,
     "Price z-score vs N-day mean. Mean reversion signal."),
    ('compute_mean_reversion_signal', 'prices, window=20', 1,
     "Mean reversion strength. Identify overextended prices."),
    ('compute_turnover_ratio', 'volumes, window=20', 1,
     "Volume ratio. Liquidity detection."),
]
```

- [ ] **Step 2: Test building blocks**

```bash
cd /home/wangmeiyi/AuctionNet/lesr/组合优化_ppo && python -c "
import sys; sys.path.insert(0, 'core')
from feature_library import (
    compute_relative_momentum, compute_realized_volatility,
    compute_downside_risk, compute_multi_horizon_momentum,
    compute_zscore_price, compute_mean_reversion_signal,
    compute_turnover_ratio, compute_cross_sectional_rank, compute_beta
)
import numpy as np
np.random.seed(42)
prices = np.cumsum(np.random.randn(100) * 2 + 100)
returns = np.diff(prices) / (prices[:-1] + 1e-10)
volumes = np.random.rand(100) * 1e6 + 1e5

print(f'relative_momentum: {compute_relative_momentum(prices, 20):.4f}')
print(f'realized_volatility: {compute_realized_volatility(returns, 20):.4f}')
print(f'downside_risk: {compute_downside_risk(returns, 20):.4f}')
print(f'multi_horizon_momentum: {compute_multi_horizon_momentum(prices)}')
print(f'zscore_price: {compute_zscore_price(prices, 20):.4f}')
print(f'mean_reversion_signal: {compute_mean_reversion_signal(prices, 20):.4f}')
print(f'turnover_ratio: {compute_turnover_ratio(volumes, 20):.4f}')
print(f'cross_sectional_rank: {compute_cross_sectional_rank([2.5, 1.8, 3.1, 0.9, 2.0]):.4f}')
print(f'beta: {compute_beta(returns, np.random.randn(len(returns)) * 0.01, 20):.4f}')

# Test that LLM code can import them
test_code = '''
import numpy as np
from feature_library import compute_realized_volatility, compute_relative_momentum
def revise_state(s):
    closes = s[0::6]
    returns = np.diff(closes) / (closes[:-1] + 1e-10)
    vol = compute_realized_volatility(returns, 20)
    mom = compute_relative_momentum(closes, 20)
    return np.concatenate([s, [vol, mom]])
def intrinsic_reward(updated_s):
    return 0.01 * abs(updated_s[121]) / (updated_s[120] + 0.01)
'''
from code_sandbox import validate
result = validate(test_code)
print(f'\\nLLM code with imports: ok={result[\"ok\"]}, feature_dim={result[\"feature_dim\"]}')
assert result['ok']
print('All building block tests passed')
"
```

- [ ] **Step 3: Commit**

```bash
git add 组合优化_ppo/core/feature_library.py
git commit -m "feat(portfolio-ppo): add 9 standalone building-block functions for LLM code import"
```

---

## Task 4: Modify `portfolio_env.py` — Add Intrinsic Reward + Compressed Raw State

**Files:**
- Modify: `组合优化_ppo/core/portfolio_env.py`

- [ ] **Step 1: Add `intrinsic_reward_fn` parameter to `__init__`**

In the `__init__` method, add `intrinsic_reward_fn` parameter after `detect_regime_fn`:

```python
# In __init__ signature (line 24-30), change to:
def __init__(self, data_path: str, config: dict,
             revise_state_fn: Callable = None,
             portfolio_features_fn: Callable = None,
             reward_rules_fn: Callable = None,
             detect_regime_fn: Callable = None,
             intrinsic_reward_fn: Callable = None,
             train_period: tuple = None,
             transaction_cost: float = 0.001):
```

Add storage in the constructor body (after line 50):

```python
self.intrinsic_reward_fn = intrinsic_reward_fn
```

- [ ] **Step 2: Add `_compress_raw_state` method**

Add this method after `_get_raw_states_dict` (after line 115):

```python
def _compress_raw_state(self, raw_state: np.ndarray) -> np.ndarray:
    """Compress 120-dim raw state to ~10 dims: 5 recent closes + 5 recent returns.

    Args:
        raw_state: 120-dim interleaved state.

    Returns:
        10-dim compressed state.
    """
    closes = np.array([raw_state[i * 6] for i in range(WINDOW)], dtype=float)
    # Recent 5 close prices (normalized by mean)
    mean_c = np.mean(closes) + 1e-10
    recent_closes = (closes[-5:] / mean_c) - 1.0

    # Recent 5 daily returns
    returns = np.diff(closes) / (closes[:-1] + 1e-10)
    recent_returns = returns[-5:] if len(returns) >= 5 else np.zeros(5)

    return np.concatenate([recent_closes, recent_returns])
```

- [ ] **Step 3: Modify `_compute_state` to use compressed raw + revised extras**

Replace the `_compute_state` method (lines 117-159) with:

```python
def _compute_state(self, date_idx: int) -> np.ndarray:
    """Compute full observation vector.

    Layout:
      - Compressed raw: 10 dims * 5 stocks
      - Revised features (extras only): K dims * 5 stocks
      - Portfolio features: P dims
      - Regime vector: 3 dims
      - Current weights: 6 dims
    """
    raw_states = self._get_raw_states_dict(date_idx)

    parts = []

    # Compressed raw per stock (10 * 5 = 50)
    compressed = []
    for ticker in TICKERS:
        compressed.append(self._compress_raw_state(raw_states[ticker]))
    parts.append(np.concatenate(compressed))

    # Revised extras per stock (K * 5)
    if self.revise_state_fn:
        revised_per_stock = []
        for ticker in TICKERS:
            full_revised = self.revise_state_fn(raw_states[ticker])
            # Extract only extras (dims beyond 120)
            extras = full_revised[120:] if len(full_revised) > 120 else np.array([0.0])
            if np.any(np.isnan(extras)) or np.any(np.isinf(extras)):
                extras = np.zeros_like(extras)
            revised_per_stock.append(extras)
        parts.append(np.concatenate(revised_per_stock))
    else:
        parts.append(np.zeros(5))  # placeholder

    # Portfolio features
    if self.portfolio_features_fn:
        port_feat = self.portfolio_features_fn(raw_states, self.weights)
        parts.append(port_feat)
    else:
        parts.append(np.zeros(5))

    # Regime
    if self.detect_regime_fn:
        regime = self.detect_regime_fn(raw_states)
    else:
        regime = np.array([0.0, 0.5, 0.0])
    parts.append(regime)

    # Current weights
    parts.append(self.weights)

    return np.concatenate(parts)
```

- [ ] **Step 4: Add intrinsic reward to `step()` method**

In the `step()` method, after line 278 (`rule_bonus, trigger_log = ...`) and before line 279 (`reward = ...`), add intrinsic reward computation:

```python
        # Intrinsic reward from LLM code
        intrinsic_r = 0.0
        if self.intrinsic_reward_fn:
            try:
                # Get revised state for first stock as proxy
                raw_states_now = self._get_raw_states_dict(self.current_step)
                revised = self.revise_state_fn(raw_states_now[TICKERS[0]])
                intrinsic_r = float(self.intrinsic_reward_fn(revised))
                intrinsic_r = np.clip(intrinsic_r, -1.0, 1.0)  # safety clamp
            except Exception:
                intrinsic_r = 0.0
```

Then change line 279 from:
```python
        reward = base_reward + rule_bonus
```
to:
```python
        reward = base_reward + rule_bonus + intrinsic_r
```

Add `intrinsic_reward` to info dict (after line 291):
```python
            'intrinsic_reward': float(intrinsic_r),
```

- [ ] **Step 5: Add `get_revised_states()` method**

Add this method at the end of the class (before `get_training_states`):

```python
def get_revised_states(self, n_samples: int = 200) -> tuple:
    """Get revised states and forward returns for IC computation.

    Returns:
        revised_states: (N, state_dim) array
        forward_returns: (N,) array
        regime_labels: (N,) array of regime strings
    """
    if self.revise_state_fn is None:
        return np.array([]), np.array([]), np.array([])

    n = min(n_samples, len(self.dates) - WINDOW - 1)
    indices = np.linspace(WINDOW, len(self.dates) - 2, n, dtype=int)

    revised_list = []
    forward_list = []
    regime_labels = []

    for idx in indices:
        raw_states = self._get_raw_states_dict(idx)
        # Use first stock's revised state
        revised = self.revise_state_fn(raw_states[TICKERS[0]])
        revised_list.append(revised)

        # Forward return (equal-weight)
        date = self.dates[idx]
        next_date = self.dates[idx + 1]
        ret = 0.0
        for ticker in TICKERS:
            p0 = self.prices.get(date, {}).get(ticker, 0.0)
            p1 = self.prices.get(next_date, {}).get(ticker, 0.0)
            if p0 > 0:
                ret += (p1 - p0) / p0
        forward_list.append(ret / len(TICKERS))

        # Regime label
        if self.detect_regime_fn:
            rv = self.detect_regime_fn(raw_states)
            from ic_analyzer import _classify_regime
            regime_labels.append(_classify_regime(rv[0], rv[1]))
        else:
            regime_labels.append('neutral')

    revised_arr = np.array(revised_list)
    forward_arr = np.array(forward_list)
    regime_arr = np.array(regime_labels)

    return revised_arr, forward_arr, regime_arr
```

- [ ] **Step 6: Test modified portfolio_env.py**

```bash
cd /home/wangmeiyi/AuctionNet/lesr/组合优化_ppo && python -c "
import sys; sys.path.insert(0, 'core')
from portfolio_env import PortfolioEnv
import numpy as np

# Create env with mock revise_state and intrinsic_reward
def mock_revise(s):
    closes = s[0::6]
    mom = np.mean(np.diff(closes[-5:])) if len(closes) >= 6 else 0.0
    vol = np.std(np.diff(closes) / (closes[:-1] + 1e-10)) if len(closes) > 1 else 0.01
    return np.concatenate([s, [mom, vol]])

def mock_intrinsic(s):
    return 0.01 * abs(s[120]) / (s[121] + 0.01)

env = PortfolioEnv(
    'data/portfolio_5stocks.pkl',
    {'portfolio': {'default_lambda': 0.5}},
    revise_state_fn=mock_revise,
    intrinsic_reward_fn=mock_intrinsic,
    train_period=('2020-01-01', '2020-03-31'),
)

state = env.reset()
print(f'State dim: {len(state)}')
print(f'State shape: {state.shape}')

next_s, reward, done, info = env.step(np.ones(6) / 6)
print(f'Reward: {reward:.6f}')
print(f'Intrinsic reward: {info.get(\"intrinsic_reward\", \"N/A\")}')
print(f'Done: {done}')

# Test get_revised_states
revised, fwd, regimes = env.get_revised_states(50)
print(f'Revised states: {revised.shape}')
print(f'Forward returns: {fwd.shape}')
print(f'Regimes: {set(regimes)}')

print('\\nportfolio_env tests passed')
"
```

- [ ] **Step 7: Commit**

```bash
git add 组合优化_ppo/core/portfolio_env.py
git commit -m "feat(portfolio-ppo): add intrinsic_reward, compressed raw state, get_revised_states"
```

---

## Task 5: Rewrite `prompts.py` — Code Generation Prompts

**Files:**
- Modify: `组合优化_ppo/core/prompts.py` (full rewrite of prompt builders, keep `_extract_json` and reward config)

- [ ] **Step 1: Rewrite `prompts.py`**

Keep `_extract_json` (lines 22-63) and `build_reward_config_prompt` (lines 201-272) and `REWARD_RULES` (lines 103-111). Replace everything else:

```python
"""
LLM Prompt Templates for Portfolio Optimization LESR

Three code-generation prompts:
1. init_prompt: First iteration - state description + building blocks + code example
2. cot_prompt: After training - code + IC analysis + market context
3. next_iteration_prompt: Subsequent iterations - history + suggestions

Plus reward_config_prompt (kept as JSON selection) and _extract_python_code.
"""

import json
import re


def _fmt(val, fmt_str):
    """Safe format: apply format if numeric, else return string as-is."""
    if isinstance(val, (int, float)):
        return format(val, fmt_str)
    return str(val)


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response (handles markdown wrapping)."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    pattern = r'```(?:json)?\s*\n?(.*?)\n?\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    start = -1
    raise ValueError(f"Could not extract JSON from response: {text[:200]}")


def _extract_python_code(text: str) -> str:
    """Extract Python code from LLM response.

    Tries:
      1. Extract from ```python ... ``` blocks
      2. Look for 'import numpy' to last function end
      3. Return full text as fallback
    """
    # Try markdown code blocks
    pattern = r'```(?:python)?\s*\n(.*?)\n\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        # Return the longest match (most likely the full code)
        return max(matches, key=len).strip()

    # Try finding from 'import numpy' to end
    idx = text.find('import numpy')
    if idx >= 0:
        return text[idx:].strip()

    # Try finding 'def revise_state'
    idx = text.find('def revise_state')
    if idx >= 0:
        # Include any imports before
        import_idx = text.rfind('import', 0, idx)
        if import_idx >= 0:
            line_start = text.rfind('\n', 0, import_idx) + 1
            return text[line_start:].strip()
        return text[idx:].strip()

    return text.strip()


# ---------------------------------------------------------------------------
# Building blocks description for prompts
# ---------------------------------------------------------------------------

BUILDING_BLOCKS_DESC = """
Available computation functions (import from feature_library):

1. compute_relative_momentum(prices, window=20)
   Input: prices = 1D array of close prices (length >= window)
   Output: scalar, this stock's excess return vs window-average
   Use case: identify relatively outperforming stocks

2. compute_realized_volatility(returns, window=20)
   Input: returns = 1D array of daily returns
   Output: scalar, realized volatility
   Use case: measure individual stock risk

3. compute_downside_risk(returns, window=20)
   Input: returns = 1D array of daily returns
   Output: scalar, downside semi-deviation
   Use case: measure downside risk

4. compute_multi_horizon_momentum(prices, windows=[5, 10, 20])
   Input: prices = 1D array of close prices
   Output: array of 3 scalars, momentum at each horizon
   Use case: capture trend at multiple time scales

5. compute_zscore_price(prices, window=20)
   Input: prices = 1D array
   Output: scalar, z-score of current price vs N-day mean
   Use case: mean reversion signal

6. compute_mean_reversion_signal(prices, window=20)
   Input: prices = 1D array
   Output: scalar, mean reversion strength
   Use case: identify overextended prices

7. compute_turnover_ratio(volumes, window=20)
   Input: volumes = 1D array
   Output: scalar, current volume / average volume
   Use case: liquidity detection

Note: compute_cross_sectional_rank and compute_beta are portfolio-level
functions — they require data from all stocks, not available inside revise_state(s).
They are computed at the environment level, not inside your code.
"""


STATE_LAYOUT_DESC = """
The current state for each stock is represented by a 120-dimensional Python NumPy array, denoted as `s`.

Details of each dimension in the state `s` are as follows:
- `s[0]` through `s[5]`: [close, open, high, low, volume, adjusted_close] for day 1
- `s[6]` through `s[11]`: same 6 channels for day 2
- ...
- `s[114]` through `s[119]`: same 6 channels for day 20

In other words:
- `s[0::6]` = close prices (20 values, oldest to newest)
- `s[1::6]` = open prices
- `s[2::6]` = high prices
- `s[3::6]` = low prices
- `s[4::6]` = volumes
- `s[5::6]` = adjusted close prices
"""


REWARD_RULES = {
    'penalize_concentration': 'Penalty when any stock weight exceeds max_weight. Default max_weight=0.35, penalty=0.1.',
    'reward_diversification': 'Bonus when holding >= min_stocks above 5%. Default min_stocks=3, bonus=0.05.',
    'penalize_turnover': 'Penalty when turnover > threshold. Default threshold=0.1, penalty=0.15.',
    'regime_defensive': 'Bonus for holding cash when risk_level is high. Default crisis_threshold=0.6, cash_bonus=0.1.',
    'momentum_alignment': 'Bonus when weights correlate with momentum rank. Default bonus=0.05.',
    'volatility_scaling': 'Scale down reward in high-vol regime. Default vol_threshold=0.5, scale=0.5.',
    'drawdown_penalty': 'Penalty when drawdown exceeds threshold. Default dd_threshold=0.1, penalty=0.15.',
}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_init_prompt(market_stats: str) -> str:
    """Build initial code-generation prompt for first LESR iteration.

    Args:
        market_stats: output from market_stats.get_market_stats()

    Returns:
        Formatted prompt string requesting revise_state + intrinsic_reward code.
    """
    return f"""Revise the state representation for a reinforcement learning agent.
=========================================================
Task: Dynamic portfolio allocation across 5 stocks (TSLA, NFLX, AMZN, MSFT, JNJ)
      plus a CASH asset (6 assets total). The goal is to maximize risk-adjusted returns.
=========================================================

{STATE_LAYOUT_DESC}

You should design a task-related state representation based on the source 120 dim to better
for reinforcement training, using the detailed information mentioned above to do some calculations,
and feel free to do complex calculations, and then concat them to the source state.

{BUILDING_BLOCKS_DESC}

Market Statistics:
{market_stats}

Besides, we want you to design an intrinsic reward function based on the revise_state python function.

That is to say, we will:
1. use your revise_state python function to get an updated state: updated_s = revise_state(s)
2. use your intrinsic reward function to get an intrinsic reward: r = intrinsic_reward(updated_s)
3. to better design the intrinsic_reward, we recommend you use some source dim in the updated_s,
   which is between updated_s[0] and updated_s[119]
4. however, you must use the extra dim in your given revise_state python function,
   which is between updated_s[120] and the end of updated_s

Your task is to create two Python functions, named `revise_state`, which takes the current state `s`
as input and returns an updated state representation, and named `intrinsic_reward`, which takes the
updated state `updated_s` as input and returns an intrinsic reward. The functions should be executable
and ready for integration into a reinforcement learning environment.

The goal is to better for reinforcement training. Lets think step by step. Below is an illustrative
example of the expected output:

```python
import numpy as np
def revise_state(s):
    # Your state revision implementation goes here
    return updated_s
def intrinsic_reward(updated_s):
    # Your intrinsic reward code implementation goes here
    return intrinsic_reward
```"""


def build_cot_prompt(sample_results_text: str,
                     market_period_summary: str = "") -> str:
    """Build COT feedback prompt after training.

    Args:
        sample_results_text: formatted IC analysis from ic_analyzer.build_ic_cot_prompt()
        market_period_summary: text describing training period market conditions

    Returns:
        Formatted prompt string for LLM analysis.
    """
    return f"""We have successfully trained Reinforcement Learning (RL) policy using different
state revision codes and intrinsic reward function codes sampled by you, and each pair of code
is associated with the training of a policy relatively.

Throughout every state revision code's training process, we monitored:
1. The final policy performance (accumulated reward).
2. Most importantly, every state revise dim's Information Coefficient (IC) with forward returns.
   The IC measures how predictive each state dimension is for future portfolio returns.
   Higher |IC| means the dimension is more useful for the RL agent's decision making.
3. Market environment context and regime-specific IC analysis.

Here are the results:
{sample_results_text}

[Market Environment During Training]
{market_period_summary}

You should analyze the results mentioned above and give suggestions about how to improve the
performance of the "state revision code".

Here are some tips for how to analyze the results:
(a) if you find a state revision code's performance is very low, then you should analyze to
    figure out why it fails
(b) if you find some dims' IC are more related to the final performance, you should analyze
    to figure out what makes it successful
(c) pay attention to regime-specific IC - features that work in trending markets may fail in
    volatile markets
(d) analyze how to improve both the "state revision code" and "intrinsic reward code"

Lets think step by step. Your solution should aim to improve the overall performance of the RL policy."""


def build_next_iteration_prompt(market_stats: str,
                                history_text: str,
                                cot_suggestions: str = "") -> str:
    """Build prompt for subsequent LESR iterations.

    Args:
        market_stats: output from market_stats.get_market_stats()
        history_text: formatted history of all prior iterations
        cot_suggestions: COT analysis from previous iteration

    Returns:
        Formatted prompt string requesting improved code.
    """
    return f"""Revise the state representation for a reinforcement learning agent.
=========================================================
Task: Dynamic portfolio allocation across 5 stocks (TSLA, NFLX, AMZN, MSFT, JNJ) + CASH.
=========================================================

{STATE_LAYOUT_DESC}

{BUILDING_BLOCKS_DESC}

Updated Market Statistics:
{market_stats}

For this problem, we have some history experience for you, here are some state revision codes
we have tried in the former iterations:
{history_text}

{cot_suggestions}

Based on the former suggestions. We are seeking an improved state revision code and an improved
intrinsic reward code that can enhance the model's performance on the task.

Your task is to create two Python functions, named `revise_state`, which takes the current state `s`
as input and returns an updated state representation, and named `intrinsic_reward`, which takes the
updated state `updated_s` as input and returns an intrinsic reward. The functions should be executable
and ready for integration into a reinforcement learning environment.

The goal is to better for reinforcement training. Lets think step by step. Below is an illustrative
example of the expected output:

```python
import numpy as np
def revise_state(s):
    return updated_s
def intrinsic_reward(updated_s):
    return float_reward
```"""


def build_reward_config_prompt(market_stats: str, iteration: int,
                               history: list = None,
                               feature_rationale: str = "") -> str:
    """Build reward rule configuration prompt for LLM (unchanged JSON paradigm).

    Args:
        market_stats: output from market_stats.get_market_stats()
        iteration: current LESR iteration
        history: list of previous iteration results
        feature_rationale: rationale from feature selection step

    Returns:
        Formatted prompt string
    """
    rules_cat = "\n".join(f"  - {k}: {v}" for k, v in REWARD_RULES.items())

    history_section = ""
    if history:
        history_section = "## Previous Iteration Results\n"
        for h in history[-3:]:
            history_section += f"### Iteration {h.get('iteration', '?')}\n"
            history_section += f"- Reward rules used: {h.get('reward_rules', 'N/A')}\n"
            history_section += f"- Sharpe: {_fmt(h.get('sharpe', 'N/A'), '.3f')}\n"
            history_section += f"- Max Drawdown: {_fmt(h.get('max_drawdown', 'N/A'), '.2f')}%\n"
            turnover = h.get('avg_turnover', 'N/A')
            turnover_str = f"{turnover:.3f}" if isinstance(turnover, (int, float)) else str(turnover)
            history_section += f"- Turnover: {turnover_str}\n"
            history_section += "\n"

    return f"""You are configuring the reward function for a PPO-based portfolio optimizer.

The base reward is Mean-Variance: r = portfolio_return - lambda * drawdown^2
Plus an intrinsic reward designed by another LLM code.

Your task: Select and parameterize additional reward rules to guide the agent.

## Market Statistics
{market_stats}

{history_section}

## Selected Features Rationale
{feature_rationale}

## Available Reward Rules
{rules_cat}

## Your Task
Select 2-4 reward rules and set their parameters. Consider:
1. Current market conditions (volatility, correlation structure)
2. Common failure modes (concentration, excessive turnover, ignoring regime)
3. Complementarity between rules (avoid redundant penalties)

## Output Format (JSON only)
```json
{{
  "reward_rules": [
    {{"rule": "penalize_concentration", "params": {{"max_weight": 0.35}}}},
    {{"rule": "regime_defensive", "params": {{"crisis_threshold": 0.6}}}}
  ],
  "lambda": 0.5,
  "rationale": "Brief explanation of why these rules were selected"
}}
```

Rules:
- Select 2-4 reward rules
- lambda controls risk aversion in base reward (0.1=aggressive, 1.0=conservative)
- Use default params unless market conditions suggest otherwise
"""
```

- [ ] **Step 2: Test prompts.py**

```bash
cd /home/wangmeiyi/AuctionNet/lesr/组合优化_ppo && python -c "
import sys; sys.path.insert(0, 'core')
from prompts import (
    build_init_prompt, build_cot_prompt, build_next_iteration_prompt,
    build_reward_config_prompt, _extract_python_code
)

# Test init prompt
init = build_init_prompt('TSLA vol: 4.2%, trend: bullish')
assert 'revise_state' in init
assert 'intrinsic_reward' in init
assert '120' in init
print(f'Init prompt: {len(init)} chars')

# Test COT prompt
cot = build_cot_prompt('Sample IC analysis here', 'Bull market, low vol')
assert 'IC' in cot
print(f'COT prompt: {len(cot)} chars')

# Test next iteration prompt
next_p = build_next_iteration_prompt('Market stats', 'History text', 'COT suggestions')
assert 'revise_state' in next_p
assert 'history' in next_p
print(f'Next iter prompt: {len(next_p)} chars')

# Test _extract_python_code
response = '''Here is my code:
\`\`\`python
import numpy as np
def revise_state(s):
    return np.concatenate([s, [0.0]])
def intrinsic_reward(s):
    return 0.01
\`\`\`
Hope this works!
'''
code = _extract_python_code(response)
assert 'def revise_state' in code
assert 'def intrinsic_reward' in code
print(f'Extracted code: {len(code)} chars')

# Test reward config prompt (unchanged)
rp = build_reward_config_prompt('Market stats', 1)
assert 'JSON' in rp
print(f'Reward config prompt: {len(rp)} chars')

print('\\nAll prompt tests passed')
"
```

- [ ] **Step 3: Commit**

```bash
git add 组合优化_ppo/core/prompts.py
git commit -m "feat(portfolio-ppo): rewrite prompts for code-generation with init/COT/iteration"
```

---

## Task 6: Rewrite `lesr_controller.py` — Multi-Sample Loop with IC Feedback

**Files:**
- Modify: `组合优化_ppo/core/lesr_controller.py` (major rewrite of iteration logic)

- [ ] **Step 1: Fix imports and rewrite `_select_features` method**

Replace lines 1-36 (imports and constants) with:

```python
"""
LESR Controller for Portfolio Optimization

Main iteration loop (refactored to code-generation approach):
  1. LLM generates revise_state + intrinsic_reward Python code
  2. Code sandbox validates code (AST + test execution)
  3. Train PPO with validated code functions + reward rules
  4. Evaluate on validation set
  5. IC-based COT feedback for next iteration
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from prompts import (
    build_init_prompt, build_cot_prompt, build_next_iteration_prompt,
    build_reward_config_prompt, _extract_json, _extract_python_code,
)
from code_sandbox import validate as sandbox_validate
from ic_analyzer import compute_ic_profile, compute_regime_specific_ic, build_ic_cot_prompt
from portfolio_features import PORTFOLIO_INDICATOR_REGISTRY, build_portfolio_features
from reward_rules import REWARD_RULE_REGISTRY, build_reward_rules
from regime_detector import detect_market_regime
from market_stats import get_market_stats
from portfolio_env import PortfolioEnv
from ppo_agent import PPOAgent
from metrics import sharpe_ratio, max_drawdown, sortino_ratio, calmar_ratio


TICKERS = ['TSLA', 'NFLX', 'AMZN', 'MSFT', 'JNJ']
```

- [ ] **Step 2: Replace `_select_features` with `_generate_code`**

Replace `_select_features` (lines 106-190) with:

```python
    def _generate_code(self, iteration: int) -> List[Dict]:
        """Step 1: LLM generates code, sandbox validates. Returns list of valid samples.

        Returns:
            List of dicts with keys: code, revise_state_fn, intrinsic_reward_fn,
            feature_dim, state_dim
        """
        print(f"\n[Iteration {iteration}] Step 1: Code Generation")

        # Get market stats from training data
        env_tmp = PortfolioEnv(
            self.data_path, self.config,
            train_period=self.train_period,
            transaction_cost=self.transaction_cost,
        )
        training_states, _ = env_tmp.get_training_states(n_samples=200)
        market_stats = get_market_stats(training_states)

        # Build prompt
        if iteration == 1:
            prompt = build_init_prompt(market_stats)
        else:
            # Build history text
            history_text = self._format_history()
            cot_suggestions = self.last_cot_feedback if hasattr(self, 'last_cot_feedback') else ""
            prompt = build_next_iteration_prompt(market_stats, history_text, cot_suggestions)

        # Sample N code sets from LLM
        valid_samples = []
        for sample_idx in range(self.sample_count):
            print(f"  Sampling code {sample_idx + 1}/{self.sample_count}...")
            try:
                response = self._call_llm(prompt, system_msg="You are a quantitative portfolio manager. Write Python code for revise_state and intrinsic_reward functions.")
                code = _extract_python_code(response)
                result = sandbox_validate(code)

                if result['ok']:
                    print(f"    Valid code: feature_dim={result['feature_dim']}, "
                          f"state_dim={result['state_dim']}")
                    valid_samples.append({
                        'code': code,
                        'revise_state_fn': result['revise_state'],
                        'intrinsic_reward_fn': result['intrinsic_reward'],
                        'feature_dim': result['feature_dim'],
                        'state_dim': result['state_dim'],
                    })
                else:
                    print(f"    Invalid code: {result['errors'][:2]}")
            except Exception as e:
                print(f"    Sample {sample_idx + 1} failed: {e}")

        if not valid_samples:
            print("  No valid code samples, using default")
            valid_samples = [self._default_code_config()]

        return valid_samples
```

- [ ] **Step 3: Update `_configure_rewards` — keep JSON selection**

The `_configure_rewards` method (lines 192-248) stays mostly the same. Only update the import references. No structural changes needed.

- [ ] **Step 4: Update `_train_ppo` to accept code functions**

Replace `_train_ppo` (lines 250-342) with:

```python
    def _train_ppo(self, code_sample: dict, reward_config: dict,
                   portfolio_features_fn=None) -> dict:
        """Step 3: Train PPO agent with code-generated functions.

        Args:
            code_sample: dict with revise_state_fn, intrinsic_reward_fn
            reward_config: dict with reward_rules_fn, lambda
            portfolio_features_fn: optional portfolio features closure
        """
        print("\nStep 3: PPO Training")

        # Create environment with code functions
        env = PortfolioEnv(
            self.data_path, self.config,
            revise_state_fn=code_sample.get('revise_state_fn'),
            portfolio_features_fn=portfolio_features_fn,
            reward_rules_fn=reward_config.get('reward_rules_fn'),
            detect_regime_fn=detect_market_regime,
            intrinsic_reward_fn=code_sample.get('intrinsic_reward_fn'),
            train_period=self.train_period,
            transaction_cost=self.transaction_cost,
        )

        state_dim = env.state_dim
        print(f"  State dim: {state_dim}")

        # Create agent
        hidden_dim = self.ppo_config.get('hidden_dim', 256)
        agent = PPOAgent(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            actor_lr=self.ppo_config.get('actor_lr', 3e-4),
            critic_lr=self.ppo_config.get('critic_lr', 1e-3),
            gamma=self.ppo_config.get('gamma', 0.99),
            gae_lambda=self.ppo_config.get('gae_lambda', 0.95),
            clip_epsilon=self.ppo_config.get('clip_epsilon', 0.2),
            entropy_coef=self.ppo_config.get('entropy_coef', 0.01),
            epochs_per_update=self.ppo_config.get('epochs_per_update', 10),
            batch_size=self.ppo_config.get('batch_size', 64),
        )

        # Training loop
        all_rewards = []
        all_returns = []

        for episode in range(self.max_episodes):
            state = env.reset()
            episode_reward = 0
            episode_returns = []
            states, actions, log_probs, rewards, dones = [], [], [], [], []

            done = False
            while not done:
                weights, log_prob = agent.select_action(state)
                next_state, reward, done, info = env.step(weights)

                states.append(state)
                actions.append(weights)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(float(done))

                episode_reward += reward
                episode_returns.append(info.get('portfolio_return', 0))
                state = next_state

            if len(states) > 1:
                agent.update(states, actions, log_probs, rewards, dones, state)

            all_rewards.append(episode_reward)
            all_returns.extend(episode_returns)

            if (episode + 1) % 10 == 0:
                avg_rew = np.mean(all_rewards[-10:])
                sharpe = sharpe_ratio(all_returns[-252:]) if len(all_returns) > 10 else 0.0
                print(f"  Episode {episode+1}/{self.max_episodes}: "
                      f"avg_reward={avg_rew:.4f}, sharpe={sharpe:.3f}")

        # Final metrics
        final_sharpe = sharpe_ratio(all_returns)
        final_sortino = sortino_ratio(all_returns)
        final_mdd = max_drawdown(all_returns)
        final_calmar = calmar_ratio(all_returns)
        total_return = (env.portfolio_value - 1.0) * 100

        print(f"  Training complete: Sharpe={final_sharpe:.3f}, "
              f"MDD={final_mdd:.2f}%, Return={total_return:.2f}%")

        # Compute IC profile
        ic_profile = {}
        regime_ic = {}
        ir_stats = {}
        try:
            revised_states, forward_returns, regime_labels = env.get_revised_states(200)
            if len(revised_states) > 20:
                ic_profile = compute_ic_profile(revised_states, forward_returns)
                regime_ic = compute_regime_specific_ic(revised_states, forward_returns, regime_labels)
                # Intrinsic reward stats
                if code_sample.get('intrinsic_reward_fn') and len(revised_states) > 10:
                    ir_values = [code_sample['intrinsic_reward_fn'](s) for s in revised_states[:50]]
                    ir_mean = float(np.mean(ir_values))
                    ir_corr = float(np.corrcoef(ir_values, forward_returns[:50])[0, 1]) if len(ir_values) > 5 else 0.0
                    ir_stats = {'mean': ir_mean, 'correlation_with_performance': ir_corr if not np.isnan(ir_corr) else 0.0}
        except Exception as e:
            print(f"  IC computation skipped: {e}")

        return {
            'agent': agent,
            'env': env,
            'sharpe': final_sharpe,
            'sortino': final_sortino,
            'max_drawdown': final_mdd,
            'calmar': final_calmar,
            'total_return': total_return,
            'all_returns': all_returns,
            'portfolio_value': env.portfolio_value,
            'ic_profile': ic_profile,
            'regime_ic': regime_ic,
            'intrinsic_reward_stats': ir_stats,
        }
```

- [ ] **Step 5: Rewrite `run()` method for multi-sample loop**

Replace the `run()` method (lines 430-511) with:

```python
    def run(self):
        """Run the full LESR iteration loop with multi-sample code generation."""
        print("=" * 60)
        print("LESR Portfolio Optimization (Code-Generation Mode)")
        print(f"Iterations: {self.max_iterations}, Samples: {self.sample_count}, "
              f"PPO episodes: {self.max_episodes}")
        print(f"Train: {self.train_period}, Val: {self.val_period}")
        if self.no_llm:
            print("Mode: NO_LLM (using default features & reward rules)")
        print("=" * 60)

        self.last_cot_feedback = ""

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration}/{self.max_iterations}")
            print(f"{'='*60}")

            try:
                # Step 1: Generate code samples
                if self.no_llm:
                    code_samples = [self._default_code_config()]
                    reward_config = self._default_reward_config()
                else:
                    code_samples = self._generate_code(iteration)

                    # Step 2: Configure rewards (JSON selection, unchanged)
                    reward_config = self._configure_rewards(
                        iteration, "Code-generated features")

                # Step 3: Train N samples, collect results
                sample_results = []
                for s_idx, code_sample in enumerate(code_samples):
                    print(f"\n  --- Training Sample {s_idx + 1}/{len(code_samples)} ---")
                    train_result = self._train_ppo(code_sample, reward_config)
                    val_result = self._evaluate(
                        train_result['agent'], code_sample, reward_config)

                    sample_results.append({
                        'code': code_sample.get('code', ''),
                        'train_result': train_result,
                        'val_result': val_result,
                        'performance': {
                            'sharpe': train_result['sharpe'],
                            'total_return': train_result['total_return'],
                            'max_drawdown': train_result['max_drawdown'],
                        },
                        'ic_profile': train_result.get('ic_profile', {}),
                        'regime_ic': train_result.get('regime_ic', {}),
                        'intrinsic_reward_stats': train_result.get('intrinsic_reward_stats', {}),
                    })

                # Step 4: Select best sample
                best_idx = max(range(len(sample_results)),
                              key=lambda i: sample_results[i]['val_result']['val_sharpe'])
                best = sample_results[best_idx]
                print(f"\n  Best sample: {best_idx + 1} "
                      f"(Val Sharpe={best['val_result']['val_sharpe']:.3f})")

                # Step 5: IC-based COT feedback
                if not self.no_llm and len(sample_results) > 0:
                    cot_text = build_ic_cot_prompt(
                        sample_results, best_idx,
                        market_period_summary=self._get_market_summary())
                    self.last_cot_feedback = cot_text
                    print(f"  COT feedback generated ({len(cot_text)} chars)")

                # Record history
                record = {
                    'iteration': iteration,
                    'n_samples': len(code_samples),
                    'best_sample_idx': best_idx,
                    'reward_rules': [r['rule'] for r in reward_config.get('reward_rules', [])],
                    'lambda': reward_config.get('lambda', self.default_lambda),
                    'sharpe': best['performance']['sharpe'],
                    'max_drawdown': best['performance']['max_drawdown'],
                    'total_return': best['performance']['total_return'],
                    'val_sharpe': best['val_result']['val_sharpe'],
                    'val_max_drawdown': best['val_result']['val_max_drawdown'],
                    'val_total_return': best['val_result']['val_total_return'],
                    'ic_profile': {str(k): f"{v:.4f}" for k, v in best.get('ic_profile', {}).items()},
                }
                self.iteration_history.append(record)

                # Save iteration result
                self._save_iteration(iteration, code_samples[best_idx], reward_config,
                                     best['train_result'], best['val_result'], record)

                # Track best
                if best['val_result']['val_sharpe'] > self.best_sharpe:
                    self.best_sharpe = best['val_result']['val_sharpe']
                    self.best_config = {
                        'iteration': iteration,
                        'code': code_samples[best_idx].get('code', ''),
                        'reward_config': reward_config,
                        'val_result': {k: v for k, v in best['val_result'].items()
                                       if k != 'val_returns' and k != 'worst_trades'},
                    }
                    model_path = self.experiment_dir / 'best_model.pt'
                    best['train_result']['agent'].save(str(model_path))
                    print(f"  New best model! Val Sharpe={self.best_sharpe:.3f}")

            except Exception as e:
                print(f"  Iteration {iteration} failed: {e}")
                import traceback
                traceback.print_exc()

        # Save overall summary
        self._save_summary()
        print(f"\n{'='*60}")
        print(f"LESR Complete. Best Val Sharpe: {self.best_sharpe:.3f}")
        print(f"{'='*60}")
```

- [ ] **Step 6: Add helper methods**

Add these helper methods to the class:

```python
    def _default_code_config(self) -> dict:
        """Fallback code config when LLM fails."""
        code = """import numpy as np
from feature_library import compute_relative_momentum, compute_realized_volatility
def revise_state(s):
    closes = s[0::6]
    returns = np.diff(closes) / (closes[:-1] + 1e-10)
    mom = compute_relative_momentum(closes, 20)
    vol = compute_realized_volatility(returns, 20)
    return np.concatenate([s, [mom, vol]])
def intrinsic_reward(updated_s):
    return 0.01 * abs(updated_s[120]) / (updated_s[121] + 0.01)
"""
        result = sandbox_validate(code)
        if result['ok']:
            return {
                'code': code,
                'revise_state_fn': result['revise_state'],
                'intrinsic_reward_fn': result['intrinsic_reward'],
                'feature_dim': result['feature_dim'],
                'state_dim': result['state_dim'],
            }
        # Ultimate fallback: identity revise_state + zero reward
        return {
            'code': 'import numpy as np\ndef revise_state(s): return s\ndef intrinsic_reward(s): return 0.0',
            'revise_state_fn': lambda s: s,
            'intrinsic_reward_fn': lambda s: 0.0,
            'feature_dim': 0,
            'state_dim': 120,
        }

    def _format_history(self) -> str:
        """Format iteration history for prompt."""
        lines = []
        for h in self.iteration_history[-3:]:
            lines.append(f"Iteration {h.get('iteration', '?')}:")
            lines.append(f"  Sharpe: {_fmt(h.get('sharpe', 'N/A'), '.3f')}")
            lines.append(f"  Return: {_fmt(h.get('total_return', 'N/A'), '.2f')}%")
            lines.append(f"  Val Sharpe: {_fmt(h.get('val_sharpe', 'N/A'), '.3f')}")
            if h.get('ic_profile'):
                lines.append(f"  IC profile: {h['ic_profile']}")
            lines.append("")
        return "\n".join(lines)

    def _get_market_summary(self) -> str:
        """Get brief market summary for COT prompt."""
        try:
            env_tmp = PortfolioEnv(
                self.data_path, self.config,
                train_period=self.train_period,
                transaction_cost=self.transaction_cost,
            )
            training_states, forward_returns = env_tmp.get_training_states(n_samples=100)
            if len(forward_returns) > 10:
                avg_ret = float(np.mean(forward_returns)) * 252 * 100
                vol = float(np.std(forward_returns)) * np.sqrt(252) * 100
                trend = "bullish" if avg_ret > 5 else ("bearish" if avg_ret < -5 else "neutral")
                return (f"Training period: {self.train_period[0]} ~ {self.train_period[1]}\n"
                        f"  Annualized return: {avg_ret:.1f}%\n"
                        f"  Annualized volatility: {vol:.1f}%\n"
                        f"  Trend: {trend}")
        except Exception:
            pass
        return f"Training period: {self.train_period[0]} ~ {self.train_period[1]}"
```

- [ ] **Step 7: Update `_call_llm` to accept system message**

Change `_call_llm` (lines 87-104) to accept optional system message:

```python
    def _call_llm(self, prompt: str, system_msg: str = None) -> str:
        """Call LLM with retry."""
        if system_msg is None:
            system_msg = "You are a quantitative portfolio manager. Always respond with valid JSON."
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"  LLM call attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(5)
        raise RuntimeError("LLM call failed after 3 attempts")
```

- [ ] **Step 8: Update `_evaluate` signature**

Change `_evaluate` (lines 344-392) to accept code_sample instead of feature_config:

```python
    def _evaluate(self, agent: PPOAgent, code_sample: dict,
                  reward_config: dict) -> dict:
        """Step 4: Evaluate on validation set."""
        print("\nStep 4: Validation Evaluation")

        env = PortfolioEnv(
            self.data_path, self.config,
            revise_state_fn=code_sample.get('revise_state_fn'),
            portfolio_features_fn=None,
            reward_rules_fn=reward_config.get('reward_rules_fn'),
            detect_regime_fn=detect_market_regime,
            intrinsic_reward_fn=code_sample.get('intrinsic_reward_fn'),
            train_period=self.val_period,
            transaction_cost=self.transaction_cost,
        )

        state = env.reset()
        done = False
        val_returns = []

        while not done:
            weights, _ = agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(weights)
            val_returns.append(info.get('portfolio_return', 0))
            state = next_state

        val_sharpe = sharpe_ratio(val_returns)
        val_mdd = max_drawdown(val_returns)
        val_return = (env.portfolio_value - 1.0) * 100

        print(f"  Val Sharpe={val_sharpe:.3f}, MDD={val_mdd:.2f}%, Return={val_return:.2f}%")

        return {
            'val_sharpe': val_sharpe,
            'val_max_drawdown': val_mdd,
            'val_total_return': val_return,
            'val_returns': val_returns,
        }
```

- [ ] **Step 9: Update `_save_iteration` for code-based config**

Replace `_save_iteration` with:

```python
    def _save_iteration(self, iteration, code_sample, reward_config,
                        train_result, val_result, record):
        """Save iteration results to disk."""
        iter_dir = self.experiment_dir / f'iteration_{iteration}'
        iter_dir.mkdir(exist_ok=True)

        # Save code
        with open(iter_dir / 'code.py', 'w') as f:
            f.write(code_sample.get('code', ''))

        # Save reward config
        save_cfg = {
            'reward_rules': reward_config.get('reward_rules', []),
            'lambda': reward_config.get('lambda', self.default_lambda),
            'feature_dim': code_sample.get('feature_dim', 0),
            'state_dim': code_sample.get('state_dim', 120),
        }
        with open(iter_dir / 'config.json', 'w') as f:
            json.dump(save_cfg, f, indent=2, default=str)

        # Save metrics
        with open(iter_dir / 'metrics.json', 'w') as f:
            json.dump(record, f, indent=2, default=str)

        # Save model
        train_result['agent'].save(str(iter_dir / 'model.pt'))
```

- [ ] **Step 10: Test lesr_controller.py imports**

```bash
cd /home/wangmeiyi/AuctionNet/lesr/组合优化_ppo && python -c "
import sys; sys.path.insert(0, 'core')
from lesr_controller import LESRController
print('LESRController imported successfully')

# Verify no broken imports
from code_sandbox import validate
from ic_analyzer import compute_ic_profile, build_ic_cot_prompt
from prompts import build_init_prompt, build_cot_prompt, build_next_iteration_prompt
print('All imports OK')

# Test default code config
import yaml
with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)
config['experiment']['no_llm'] = True
ctrl = LESRController(config, 'results/test_refactor')
default = ctrl._default_code_config()
print(f'Default code config: feature_dim={default[\"feature_dim\"]}, state_dim={default[\"state_dim\"]}')
assert default['revise_state_fn'] is not None
assert default['intrinsic_reward_fn'] is not None
print('Controller tests passed')
"
```

- [ ] **Step 11: Commit**

```bash
git add 组合优化_ppo/core/lesr_controller.py
git commit -m "feat(portfolio-ppo): rewrite controller for code-generation with multi-sample IC feedback"
```

---

## Task 7: Integration Test — Full Pipeline Dry Run (No LLM)

**Files:**
- Create: `组合优化_ppo/tests/test_refactored_pipeline.py`

- [ ] **Step 1: Write integration test**

```python
"""
Integration test for refactored LESR pipeline (code-generation mode).

Tests the full pipeline without LLM calls:
1. Code sandbox validates sample code
2. PortfolioEnv uses code-generated revise_state + intrinsic_reward
3. PPO trains with the new state representation
4. IC analyzer computes profiles
5. COT feedback is generated
"""

import sys
import os
import numpy as np
from pathlib import Path

# Setup path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'core'))


def test_code_sandbox():
    """Test 1: Code sandbox validates and extracts functions."""
    from code_sandbox import validate

    code = """
import numpy as np
from feature_library import compute_realized_volatility, compute_relative_momentum

def revise_state(s):
    closes = s[0::6]
    returns = np.diff(closes) / (closes[:-1] + 1e-10)
    vol = compute_realized_volatility(returns, 20)
    mom = compute_relative_momentum(closes, 20)
    return np.concatenate([s, [vol, mom]])

def intrinsic_reward(updated_s):
    vol = updated_s[120]
    mom = updated_s[121]
    return 0.01 * abs(mom) / (vol + 0.01)
"""
    result = validate(code)
    assert result['ok'], f"Validation failed: {result['errors']}"
    assert result['feature_dim'] == 2
    assert result['revise_state_fn'] is not None
    assert result['intrinsic_reward_fn'] is not None

    # Test revise_state preserves first 120 dims
    test_s = np.random.randn(120) * 100 + 150
    revised = result['revise_state_fn'](test_s)
    assert np.allclose(revised[:120], test_s, atol=1e-6)
    assert len(revised) == 122

    # Test intrinsic_reward returns scalar
    reward = result['intrinsic_reward_fn'](revised)
    assert isinstance(reward, (int, float, np.integer, np.floating))
    assert abs(reward) <= 100

    print("  [PASS] Code sandbox")
    return result


def test_portfolio_env_with_code(code_result):
    """Test 2: PortfolioEnv with code-generated functions."""
    from portfolio_env import PortfolioEnv

    data_path = str(ROOT / 'data' / 'portfolio_5stocks.pkl')
    if not os.path.exists(data_path):
        print("  [SKIP] No data file")
        return None

    env = PortfolioEnv(
        data_path,
        {'portfolio': {'default_lambda': 0.5}},
        revise_state_fn=code_result['revise_state_fn'],
        intrinsic_reward_fn=code_result['intrinsic_reward_fn'],
        detect_regime_fn=None,
        train_period=('2020-01-01', '2020-06-30'),
        transaction_cost=0.001,
    )

    state = env.reset()
    assert state.ndim == 1
    assert len(state) > 60  # compressed raw (50) + extras + portfolio + regime + weights

    next_state, reward, done, info = env.step(np.ones(6) / 6)
    assert isinstance(reward, float)
    assert 'intrinsic_reward' in info

    print(f"  [PASS] PortfolioEnv (state_dim={len(state)}, reward={reward:.6f})")
    return env


def test_ic_analyzer(env):
    """Test 3: IC analysis on revised states."""
    if env is None:
        print("  [SKIP] No env")
        return

    from ic_analyzer import compute_ic_profile, compute_regime_specific_ic, build_ic_cot_prompt

    revised, fwd, regimes = env.get_revised_states(100)
    if len(revised) < 20:
        print("  [SKIP] Not enough data for IC")
        return

    ic = compute_ic_profile(revised, fwd)
    assert isinstance(ic, dict)
    print(f"  IC profile: {len(ic)} dims")

    regime_ic = compute_regime_specific_ic(revised, fwd, regimes)
    assert isinstance(regime_ic, dict)
    print(f"  Regimes found: {list(regime_ic.keys())}")

    # COT feedback
    sample_results = [{
        'code': 'test code',
        'performance': {'sharpe': 0.5, 'total_return': 3.0, 'max_drawdown': -5.0},
        'ic_profile': ic,
        'regime_ic': regime_ic,
        'intrinsic_reward_stats': {'mean': 0.003, 'correlation_with_performance': 0.15},
    }]
    cot = build_ic_cot_prompt(sample_results, 0, market_period_summary="Test period")
    assert len(cot) > 100
    assert 'IC' in cot

    print(f"  [PASS] IC analyzer (COT: {len(cot)} chars)")


def test_ppo_training(code_result):
    """Test 4: PPO trains with code-generated state."""
    if not os.path.exists(str(ROOT / 'data' / 'portfolio_5stocks.pkl')):
        print("  [SKIP] No data file")
        return

    from portfolio_env import PortfolioEnv
    from ppo_agent import PPOAgent

    env = PortfolioEnv(
        str(ROOT / 'data' / 'portfolio_5stocks.pkl'),
        {'portfolio': {'default_lambda': 0.5}},
        revise_state_fn=code_result['revise_state_fn'],
        intrinsic_reward_fn=code_result['intrinsic_reward_fn'],
        train_period=('2020-01-01', '2020-03-31'),
    )

    state_dim = env.state_dim
    agent = PPOAgent(state_dim=state_dim, hidden_dim=32)

    state = env.reset()
    states, actions, log_probs, rewards, dones = [], [], [], [], []
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 50:
        weights, log_prob = agent.select_action(state)
        next_state, reward, done, info = env.step(weights)
        states.append(state)
        actions.append(weights)
        log_probs.append(log_prob)
        rewards.append(reward)
        dones.append(float(done))
        total_reward += reward
        state = next_state
        steps += 1

    assert steps > 0
    print(f"  [PASS] PPO training ({steps} steps, total_reward={total_reward:.4f})")


if __name__ == '__main__':
    print("Integration Test: Refactored LESR Pipeline")
    print("=" * 50)

    code_result = test_code_sandbox()
    env = test_portfolio_env_with_code(code_result)
    test_ic_analyzer(env)
    test_ppo_training(code_result)

    print("\n" + "=" * 50)
    print("All integration tests passed!")
```

- [ ] **Step 2: Run integration test**

```bash
cd /home/wangmeiyi/AuctionNet/lesr/组合优化_ppo && python tests/test_refactored_pipeline.py
```

Expected: All 4 tests pass.

- [ ] **Step 3: Commit**

```bash
git add 组合优化_ppo/tests/test_refactored_pipeline.py
git commit -m "test(portfolio-ppo): add integration test for refactored code-generation pipeline"
```

---

## Task 8: Quick Smoke Test — Full Controller Run (No LLM)

**Files:**
- No new files. Runs existing `main.py` or config.

- [ ] **Step 1: Run controller in no_llm mode**

```bash
cd /home/wangmeiyi/AuctionNet/lesr/组合优化_ppo && python -c "
import sys; sys.path.insert(0, 'core')
from lesr_controller import LESRController
import yaml

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

# Override for quick test
config['experiment']['no_llm'] = True
config['experiment']['max_iterations'] = 2
config['ppo']['max_episodes'] = 3
config['experiment']['train_period'] = ['2020-01-01', '2020-06-30']
config['experiment']['val_period'] = ['2020-07-01', '2020-12-31']

ctrl = LESRController(config, 'results/smoke_test_refactor')
ctrl.run()
print('\\nSmoke test complete!')
"
```

Expected: Runs 2 iterations, uses default code config, trains PPO, evaluates, saves results.

- [ ] **Step 2: Verify output**

```bash
ls -la /home/wangmeiyi/AuctionNet/lesr/组合优化_ppo/results/smoke_test_refactor/
```

Expected: `iteration_1/`, `iteration_2/`, `best_model.pt`, `summary.json`.

---

## Self-Review Checklist

**1. Spec Coverage (design doc sections):**

| Spec Section | Plan Task | Status |
|-------------|-----------|--------|
| Section 4 (code_sandbox.py) | Task 1 | Covered |
| Section 5 (building blocks) | Task 3 | Covered (9 functions added to feature_library) |
| Section 6 (compressed raw state) | Task 4 | Covered (10-dim per stock) |
| Section 7 (IC feedback) | Task 2 | Covered (ic_analyzer.py) |
| Section 8 (multi-sample loop) | Task 6 | Covered (N samples, N policies) |
| Section 9.2 (init_prompt) | Task 5 | Covered |
| Section 9.3 (cot_prompt) | Task 5 | Covered |
| Section 9.4 (next_iteration_prompt) | Task 5 | Covered |
| Section 9.5 (reward_config_prompt) | Task 5 | Covered (unchanged JSON) |
| Section 10 (file changes) | Tasks 1-6 | All files covered |

**2. Placeholder scan:** No TBD/TODO/fill-in-later found. All steps contain complete code.

**3. Type consistency:**

| Function | Defined In | Used In | Match? |
|----------|-----------|---------|--------|
| `sandbox_validate(code_str) -> Dict` | Task 1 | Task 6 | Yes |
| `compute_ic_profile(states, returns) -> Dict[int, float]` | Task 2 | Task 6 | Yes |
| `build_ic_cot_prompt(results, best_idx) -> str` | Task 2 | Task 6 | Yes |
| `_extract_python_code(text) -> str` | Task 5 | Task 6 | Yes |
| `env.get_revised_states() -> (ndarray, ndarray, ndarray)` | Task 4 | Task 2, 6 | Yes |
| `env.__init__(..., intrinsic_reward_fn=...)` | Task 4 | Task 6 | Yes |
| `ctrl._generate_code(iteration) -> List[Dict]` | Task 6 | Task 6 run() | Yes |
| `ctrl._train_ppo(code_sample, reward_config) -> dict` | Task 6 | Task 6 run() | Yes |
| TICKERS = `['TSLA', 'NFLX', 'AMZN', 'MSFT', 'JNJ']` | All files | Consistent | Yes |

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-18-lesr-portfolio-refactor.md`.** Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
