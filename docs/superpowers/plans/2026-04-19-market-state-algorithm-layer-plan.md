# Market State to Algorithm Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move market state features from LLM-prompt-only to algorithm layer (PPO observation), add LLM market_strategy generation, and enable portfolio_features during LESR training.

**Architecture:** Extend regime_detector from 3-dim to 6-dim market state. Enable portfolio_features in LESR training. Add market_strategy function to LLM generation + sandbox validation. Update prompts to tell LLM market features are already handled. market_strategy outputs risk_scale that scales base_reward only.

**Tech Stack:** Python, NumPy, existing LESR framework (code_sandbox, prompts, portfolio_env, lesr_controller, regime_detector, portfolio_features)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `core/regime_detector.py` | Modify | Extend from 3-dim to 6-dim market state |
| `core/portfolio_env.py` | Modify | Accept market_strategy_fn, update state assembly & reward, update regime references from 3→6 dim |
| `core/code_sandbox.py` | Modify | Add market_strategy validation alongside revise_state + intrinsic_reward |
| `core/prompts.py` | Modify | Add market_strategy template, add market state description, update code examples |
| `core/lesr_controller.py` | Modify | Pass portfolio_features_fn, handle market_strategy generation & validation |
| `core/ic_analyzer.py` | Modify | Update _classify_regime for new 6-dim layout |
| `core/reward_rules.py` | Modify | Update regime_vector index references from 3→6 dim |

---

## Task 1: Extend regime_detector.py from 3-dim to 6-dim

**Files:**
- Modify: `组合优化_ppo_策略迁移_v1_市场优化/core/regime_detector.py`

- [ ] **Step 1: Update module docstring and detect_market_regime to return 6 dims**

Replace the entire file content with the updated version that computes 6 dimensions:

```python
"""
Market-Level State Detector for Portfolio Optimization

Computes 6-dimensional market state vector from equal-weight portfolio of all 5 stocks.
  [0] trend_direction:    [-1, +1]   5-day MA vs overall mean
  [1] volatility_level:   [0, 1]     recent vol z-score vs historical
  [2] risk_level:         [0, 1]     max drawdown depth in recent 10 days
  [3] avg_cross_corr:     [-1, 1]    mean of 5x5 pairwise return correlation
  [4] market_breadth:     [0, 1]     fraction of stocks with positive 5d return
  [5] volatility_ratio:   [0, 2]     recent 5d vol / recent 20d vol

Input: dict of 5 raw states {ticker: 120d_array}
"""

import numpy as np

TICKERS = ['TSLA', 'NFLX', 'AMZN', 'MSFT', 'JNJ']


def _extract_closes(s: np.ndarray) -> np.ndarray:
    n = len(s) // 6
    return np.array([s[i * 6] for i in range(n)], dtype=float)


def detect_market_regime(raw_states: dict) -> np.ndarray:
    """Compute 6-dim market state from equal-weight portfolio.

    Args:
        raw_states: dict {ticker: 120d_array}

    Returns:
        6-dim numpy array [trend, vol_level, risk, avg_corr, breadth, vol_ratio]
    """
    all_closes = []
    for ticker in TICKERS:
        if ticker in raw_states:
            all_closes.append(_extract_closes(raw_states[ticker]))

    if not all_closes:
        return np.array([0.0, 0.5, 0.0, 0.0, 0.5, 1.0])

    min_len = min(len(c) for c in all_closes)
    if min_len < 5:
        return np.array([0.0, 0.5, 0.0, 0.0, 0.5, 1.0])

    aligned = np.array([c[:min_len] for c in all_closes])
    port_closes = np.mean(aligned, axis=0)

    trend = _trend_direction(port_closes)
    volatility = _volatility_level(port_closes)
    risk = _risk_level(port_closes)
    avg_corr = _avg_cross_correlation(aligned)
    breadth = _market_breadth(aligned)
    vol_ratio = _volatility_ratio(aligned)

    return np.array([trend, volatility, risk, avg_corr, breadth, vol_ratio], dtype=float)


def _trend_direction(closes: np.ndarray) -> float:
    if len(closes) < 5 or np.std(closes) < 1e-8:
        return 0.0
    ma5 = np.mean(closes[-5:])
    ma_all = np.mean(closes)
    trend = (ma5 - ma_all) / (np.mean(closes) * 0.05 + 1e-8)
    return float(np.clip(trend, -1, 1))


def _volatility_level(closes: np.ndarray) -> float:
    if len(closes) < 5:
        return 0.5
    returns = np.diff(closes) / (closes[:-1] + 1e-8)
    recent_vol = np.std(returns[-5:])
    hist_std = np.std(returns) * 0.5 + 1e-10
    z = (recent_vol - np.std(returns)) / hist_std
    return float(np.clip((z + 1) / 3, 0, 1))


def _risk_level(closes: np.ndarray) -> float:
    if len(closes) < 3:
        return 0.0
    window = closes[-min(10, len(closes)):]
    recent_high = np.max(window)
    current = closes[-1]
    dd = (recent_high - current) / (recent_high + 1e-8)
    return float(np.clip(dd / 0.15, 0, 1))


def _avg_cross_correlation(aligned_closes: np.ndarray) -> float:
    """Mean of pairwise return correlations across all stocks."""
    n_stocks = aligned_closes.shape[0]
    if n_stocks < 2:
        return 0.0
    all_returns = []
    for i in range(n_stocks):
        c = aligned_closes[i]
        if len(c) < 2:
            return 0.0
        all_returns.append(np.diff(c) / (c[:-1] + 1e-10))
    pair_corrs = []
    for i in range(n_stocks):
        for j in range(i + 1, n_stocks):
            r1, r2 = all_returns[i], all_returns[j]
            n = min(len(r1), len(r2))
            if n < 5:
                continue
            s1, s2 = np.std(r1[-n:]), np.std(r2[-n:])
            if s1 < 1e-10 or s2 < 1e-10:
                continue
            corr = float(np.mean((r1[-n:] - np.mean(r1[-n:])) * (r2[-n:] - np.mean(r2[-n:]))) / (s1 * s2))
            pair_corrs.append(corr)
    if not pair_corrs:
        return 0.0
    return float(np.clip(np.mean(pair_corrs), -1, 1))


def _market_breadth(aligned_closes: np.ndarray) -> float:
    """Fraction of stocks with positive 5-day return."""
    n_stocks = aligned_closes.shape[0]
    positive = 0
    for i in range(n_stocks):
        c = aligned_closes[i]
        if len(c) < 6:
            continue
        ret_5d = (c[-1] - c[-6]) / (c[-6] + 1e-10)
        if ret_5d > 0:
            positive += 1
    return float(positive / n_stocks)


def _volatility_ratio(aligned_closes: np.ndarray) -> float:
    """Ratio of recent 5-day vol to recent 20-day vol for equal-weight portfolio."""
    port_closes = np.mean(aligned_closes, axis=0)
    if len(port_closes) < 21:
        return 1.0
    returns = np.diff(port_closes) / (port_closes[:-1] + 1e-10)
    if len(returns) < 20:
        return 1.0
    vol_5 = np.std(returns[-5:], ddof=1) + 1e-10
    vol_20 = np.std(returns[-20:], ddof=1) + 1e-10
    return float(np.clip(vol_5 / vol_20, 0, 3))
```

- [ ] **Step 2: Verify regime_detector works standalone**

Run: `cd /home/wangmeiyi/AuctionNet/lesr/组合优化_ppo_策略迁移_v1_市场优化 && python -c "import sys; sys.path.insert(0,'core'); from regime_detector import detect_market_regime; import numpy as np; s = {t: np.random.randn(120)*100+150 for t in ['TSLA','NFLX','AMZN','MSFT','JNJ']}; r = detect_market_regime(s); print(f'dim={len(r)}, values={r}'); assert len(r)==6, f'Expected 6, got {len(r)}'"`

- [ ] **Step 3: Commit**

```bash
git add "组合优化_ppo_策略迁移_v1_市场优化/core/regime_detector.py"
git commit -m "feat(regime): extend market state from 3-dim to 6-dim (add corr, breadth, vol_ratio)"
```

---

## Task 2: Update reward_rules.py for 6-dim regime vector

**Files:**
- Modify: `组合优化_ppo_策略迁移_v1_市场优化/core/reward_rules.py`

- [ ] **Step 1: Update rule_regime_defensive and rule_volatility_scaling to use named access**

The two reward rules that use regime_vector currently hardcode indices `regime_vector[1]` and `regime_vector[2]`. These indices haven't changed (volatility_level is still [1], risk_level is still [2]), so the existing code is actually correct. But we should verify.

No changes needed — indices [1] (volatility_level) and [2] (risk_level) are unchanged in the new 6-dim layout. The new dims [3],[4],[5] are additive.

- [ ] **Step 1: Commit (no-op, verified compatible)**

No changes needed for this file.

---

## Task 3: Update ic_analyzer.py for 6-dim regime

**Files:**
- Modify: `组合优化_ppo_策略迁移_v1_市场优化/core/ic_analyzer.py`

- [ ] **Step 1: Update _classify_regime to accept regime vector of any length**

The function `_classify_regime(trend, vol)` at line 226 is called from `portfolio_env.py` line 439 as `_classify_regime(rv[0], rv[1])` — it already extracts scalar values. No change needed to `_classify_regime` itself.

But in `portfolio_env.py` line 438-439, the regime_labels computation accesses `rv[0]` and `rv[1]` which are still correct (trend_direction is index 0, volatility_level is index 1). No change needed.

No changes needed for this file.

---

## Task 4: Update portfolio_env.py for market_strategy and 6-dim regime

**Files:**
- Modify: `组合优化_ppo_策略迁移_v1_市场优化/core/portfolio_env.py`

- [ ] **Step 1: Add market_strategy_fn parameter to __init__**

At line 43-50, add `market_strategy_fn` parameter:

```python
    def __init__(self, data_path: str, config: dict,
                 revise_state_fn: Callable = None,
                 portfolio_features_fn: Callable = None,
                 reward_rules_fn: Callable = None,
                 detect_regime_fn: Callable = None,
                 intrinsic_reward_fn: Callable = None,
                 market_strategy_fn: Callable = None,
                 train_period: tuple = None,
                 transaction_cost: float = 0.001):
```

And at the end of the init body (after line 72), add:

```python
        self.market_strategy_fn = market_strategy_fn
```

- [ ] **Step 2: Update module docstring state layout comment**

At lines 12-18, update the state layout comment:

```python
"""
Portfolio Environment for RL Training

State layout:
  - Compressed raw: 10 dims * 5 stocks = 50 dims
  - revise_state extras: K dims * 5 stocks
  - Portfolio features: P dims
  - Market state vector: 6 dims (trend/vol/risk/corr/breadth/vol_ratio)
  - Current weights: 6 dims

Action: 6-dim target weights (via Dirichlet, auto-normalized to sum=1)
Reward: base_mean-variance * risk_scale + reward rules + intrinsic reward
"""
```

- [ ] **Step 3: Update _compute_state default regime fallback**

At line 222-224, change the default regime from 3-dim to 6-dim:

```python
        # Market state (6-dim)
        if self.detect_regime_fn:
            regime = self.detect_regime_fn(raw_states)
        else:
            regime = np.array([0.0, 0.5, 0.0, 0.0, 0.5, 1.0])
        parts.append(regime)
```

- [ ] **Step 4: Update step() to apply market_strategy risk_scale**

In the `step()` method, after computing `base_reward` (line 330) and before computing `rule_bonus` (line 333), add market_strategy computation. Then apply risk_scale to base_reward in the final reward line (line 373).

Replace the reward computation block (lines 325-373) with:

```python
        # Compute reward
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value

        # Base reward: mean-variance
        lambda_mv = self.config.get('portfolio', {}).get('default_lambda', 0.5)
        base_reward = net_return - lambda_mv * (current_drawdown ** 2)

        # Market strategy risk_scale
        risk_scale = 1.0
        if self.market_strategy_fn:
            try:
                raw_states_now = self._get_raw_states_dict(self.current_step)
                market_state = self.detect_regime_fn(raw_states_now) if self.detect_regime_fn else np.array([0.0, 0.5, 0.0, 0.0, 0.5, 1.0])
                risk_scale = float(self.market_strategy_fn(market_state, self.weights))
                risk_scale = float(np.clip(risk_scale, 0.3, 2.0))
            except Exception:
                risk_scale = 1.0

        # Additional reward rules
        rule_bonus = 0.0
        trigger_log = {}
        if self.reward_rules_fn:
            raw_states = self._get_raw_states_dict(self.current_step)
            regime = self.detect_regime_fn(raw_states) if self.detect_regime_fn else None
            port_feats = {}
            if self.portfolio_features_fn:
                pf_raw = self.portfolio_features_fn(raw_states, self.weights)
                port_feats['raw'] = pf_raw

            # Compute momentum_rank for momentum_alignment rule
            try:
                from core.portfolio_features import compute_momentum_rank
                port_feats['momentum_rank'] = compute_momentum_rank(raw_states, current_weights=self.weights)
            except Exception:
                pass

            rule_bonus, trigger_log = self.reward_rules_fn(
                weights=self.weights,
                prev_weights=prev_weights,
                regime_vector=regime,
                portfolio_features=port_feats,
                base_reward=base_reward,
                current_drawdown=current_drawdown,
            )

        # Intrinsic reward from LLM code (averaged across all tickers)
        intrinsic_r = 0.0
        if self.intrinsic_reward_fn and self.revise_state_fn:
            try:
                raw_states_now = self._get_raw_states_dict(self.current_step)
                ir_values = []
                for ticker in TICKERS:
                    revised = self.revise_state_fn(raw_states_now[ticker])
                    ir_values.append(float(self.intrinsic_reward_fn(revised)))
                intrinsic_r = float(np.mean(ir_values))
                intrinsic_r = np.clip(intrinsic_r, -1.0, 1.0)
            except Exception:
                intrinsic_r = 0.0

        reward = (base_reward * risk_scale) + rule_bonus + intrinsic_r
```

- [ ] **Step 5: Update info dict to include risk_scale**

In the info dict (around line 378), add:

```python
        info = {
            'portfolio_return': net_return,
            'transaction_cost': tc_cost,
            'turnover': turnover,
            'drawdown': current_drawdown,
            'portfolio_value': self.portfolio_value,
            'weights': self.weights.copy(),
            'trigger_log': trigger_log,
            'intrinsic_reward': float(intrinsic_r),
            'risk_scale': float(risk_scale),
        }
```

- [ ] **Step 6: Verify no other 3-dim regime assumptions exist in the file**

Check `get_revised_states` method — line 438-439 uses `rv[0]` and `rv[1]` which are still correct indices. No change needed.

- [ ] **Step 7: Commit**

```bash
git add "组合优化_ppo_策略迁移_v1_市场优化/core/portfolio_env.py"
git commit -m "feat(env): add market_strategy_fn, update regime to 6-dim, risk_scale in reward"
```

---

## Task 5: Update code_sandbox.py for market_strategy validation

**Files:**
- Modify: `组合优化_ppo_策略迁移_v1_市场优化/core/code_sandbox.py`

- [ ] **Step 1: Update _extract_functions to also extract market_strategy**

At line 93-97, update the extraction loop to include `market_strategy`:

```python
    result = {}
    for name in ['revise_state', 'intrinsic_reward', 'market_strategy']:
        if name in namespace and callable(namespace[name]):
            result[name] = namespace[name]
    return result
```

- [ ] **Step 2: Update _test_execution to validate market_strategy**

After the intrinsic_reward validation block (after line 147), add:

```python
    if 'market_strategy' in functions:
        try:
            test_market_state = np.random.randn(6)
            test_weights = np.random.dirichlet(np.ones(6))
            ms_val = functions['market_strategy'](test_market_state, test_weights)
            if not isinstance(ms_val, (int, float, np.integer, np.floating)):
                errors.append(f"market_strategy must return scalar, got {type(ms_val)}")
            elif np.isnan(float(ms_val)):
                errors.append("market_strategy returned NaN")
            elif np.isinf(float(ms_val)):
                errors.append("market_strategy returned Inf")
            elif float(ms_val) < 0.3 or float(ms_val) > 2.0:
                errors.append(f"market_strategy out of range [0.3, 2.0]: {ms_val}")
        except Exception as e:
            errors.append(f"market_strategy execution error: {e}")
```

Note: market_strategy is OPTIONAL. If not present in code, no error is raised (unlike revise_state and intrinsic_reward which are required).

- [ ] **Step 3: Update validate() return to include market_strategy**

At line 152-212, update the result dict and final assembly:

```python
def validate(code_str: str) -> Dict:
    """Full validation pipeline for LLM-generated code.

    Returns:
        {
            'ok': bool,
            'errors': list[str],
            'revise_state': callable or None,
            'intrinsic_reward': callable or None,
            'market_strategy': callable or None,
            'feature_dim': int,
            'state_dim': int,
        }
    """
    result = {
        'ok': False,
        'errors': [],
        'revise_state': None,
        'intrinsic_reward': None,
        'market_strategy': None,
        'feature_dim': 0,
        'state_dim': 120,
    }

    ast_errors = _ast_check(code_str)
    if ast_errors:
        result['errors'] = ast_errors
        return result

    try:
        functions = _extract_functions(code_str)
    except Exception as e:
        result['errors'] = [f"Code extraction error: {e}"]
        return result

    if 'revise_state' not in functions:
        result['errors'] = ["revise_state function not found"]
        return result

    test_errors = _test_execution(functions)
    if test_errors:
        result['errors'] = test_errors
        return result

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
    result['market_strategy'] = functions.get('market_strategy')
    result['feature_dim'] = feature_dim
    result['state_dim'] = state_dim
    return result
```

- [ ] **Step 4: Commit**

```bash
git add "组合优化_ppo_策略迁移_v1_市场优化/core/code_sandbox.py"
git commit -m "feat(sandbox): add market_strategy validation (optional, input 6+6 dims, output [0.3,2.0])"
```

---

## Task 6: Update prompts.py for market_strategy and market state awareness

**Files:**
- Modify: `组合优化_ppo_策略迁移_v1_市场优化/core/prompts.py`

- [ ] **Step 1: Add MARKET_STATE_DESC constant**

After STATE_LAYOUT_DESC (around line 146), add:

```python
MARKET_STATE_DESC = """
IMPORTANT: Market state is already computed in the algorithm layer and always available to the RL agent.
You do NOT need to recompute market-level indicators in revise_state. The 6-dim market state vector is:
  [0] trend_direction:    [-1, +1]   market trend
  [1] volatility_level:   [0, 1]     recent volatility regime
  [2] risk_level:         [0, 1]     drawdown risk
  [3] avg_cross_corr:     [-1, 1]    overall cross-stock correlation
  [4] market_breadth:     [0, 1]     fraction of stocks rising
  [5] volatility_ratio:   [0, 3]     short-term/long-term vol ratio (>1 means rising vol)

Focus your revise_state on per-stock features. For market-level strategy, use market_strategy below.
"""
```

- [ ] **Step 2: Update build_init_prompt to include market_strategy and market state description**

Replace `build_init_prompt` function (lines 164-217) with:

```python
def build_init_prompt(market_stats: str) -> str:
    """Build initial code-generation prompt for first LESR iteration."""
    return f"""Revise the state representation for a reinforcement learning agent.
=========================================================
Task: Dynamic portfolio allocation across 5 stocks (TSLA, NFLX, AMZN, MSFT, JNJ)
      plus a CASH asset (6 assets total). The goal is to maximize risk-adjusted returns.
=========================================================

{STATE_LAYOUT_DESC}

{MARKET_STATE_DESC}

You should design a task-related state representation based on the source 120 dim to better
for reinforcement training, using the detailed information mentioned above to do some calculations,
and feel free to do complex calculations, and then concat them to the source state.

{BUILDING_BLOCKS_DESC}

Market Statistics:
{market_stats}

Besides, we want you to design two functions:

1. intrinsic_reward(updated_s) — based on the revise_state output
   - We recommend you use some source dim (updated_s[0]~updated_s[119])
   - You MUST also use extra dims (updated_s[120]~end) from your revise_state

2. market_strategy(market_state, weights) — a market-level strategy function
   - market_state: 6-dim numpy array [trend_direction, volatility_level, risk_level,
     avg_cross_corr, market_breadth, volatility_ratio]
   - weights: 6-dim numpy array [w_TSLA, w_NFLX, w_AMZN, w_MSFT, w_JNJ, w_CASH]
   - Returns: risk_scale (float in [0.3, 2.0])
     * < 1.0 = conservative (reduce risk exposure)
     * > 1.0 = aggressive (increase risk exposure)
     * = 1.0 = neutral
   - This scales the base reward: total_reward = base_reward * risk_scale + ...
   - Example: in high-risk markets (risk_level>0.6), return 0.5 to be defensive

IMPORTANT for intrinsic_reward design:
- The intrinsic_reward should help the RL agent learn BETTER states, not just amplify returns.
- If the Market Strategy Guidance above indicates high risk (Defensive/Crisis regime),
  your intrinsic_reward should penalize states with high volatility or downside risk.
- A good intrinsic_reward correlates with (but is not identical to) future portfolio performance.

IMPORTANT for market_strategy design:
- Use market_state dimensions to detect regime (e.g., risk_level, volatility_level)
- Be defensive when risk_level > 0.5 or volatility_ratio > 1.5
- Be aggressive only when trend_direction > 0.3 AND volatility_level < 0.3
- Keep it simple — 3-5 conditions is enough

Your task is to create THREE Python functions: revise_state, intrinsic_reward, and market_strategy.

```python
import numpy as np
def revise_state(s):
    # Per-stock feature engineering
    return updated_s
def intrinsic_reward(updated_s):
    # Stock-level reward signal
    return float_reward
def market_strategy(market_state, weights):
    # Market-level risk scaling
    # market_state: [trend, vol, risk, corr, breadth, vol_ratio]
    risk_level = market_state[2]
    vol_ratio = market_state[5]
    if risk_level > 0.6 or vol_ratio > 1.5:
        return 0.5
    return 1.0
```"""
```

- [ ] **Step 3: Update build_next_iteration_prompt similarly**

Replace `build_next_iteration_prompt` (lines 258-300) with:

```python
def build_next_iteration_prompt(market_stats: str,
                                history_text: str,
                                cot_suggestions: str = "") -> str:
    """Build prompt for subsequent LESR iterations."""
    return f"""Revise the state representation for a reinforcement learning agent.
=========================================================
Task: Dynamic portfolio allocation across 5 stocks (TSLA, NFLX, AMZN, MSFT, JNJ) + CASH.
=========================================================

{STATE_LAYOUT_DESC}

{MARKET_STATE_DESC}

{BUILDING_BLOCKS_DESC}

Updated Market Statistics:
{market_stats}

For this problem, we have some history experience for you, here are some state revision codes
we have tried in the former iterations:
{history_text}

{cot_suggestions}

Based on the former suggestions. We are seeking an improved state revision code, an improved
intrinsic reward code, and an improved market_strategy function.

NOTE for intrinsic_reward: Follow the Market Strategy Guidance above. In high-risk regimes,
penalize volatile states; in favorable regimes, reward exploration of informative features.

Your task is to create THREE Python functions: revise_state, intrinsic_reward, and market_strategy.

```python
import numpy as np
def revise_state(s):
    return updated_s
def intrinsic_reward(updated_s):
    return float_reward
def market_strategy(market_state, weights):
    # market_state: 6-dim [trend, vol, risk, corr, breadth, vol_ratio]
    # weights: 6-dim [w_stock1..5, w_cash]
    # Returns: float in [0.3, 2.0] — risk_scale
    return 1.0
```"""
```

- [ ] **Step 4: Commit**

```bash
git add "组合优化_ppo_策略迁移_v1_市场优化/core/prompts.py"
git commit -m "feat(prompts): add market_strategy to code-gen prompts, add market state description"
```

---

## Task 7: Update lesr_controller.py — enable portfolio_features + handle market_strategy

**Files:**
- Modify: `组合优化_ppo_策略迁移_v1_市场优化/core/lesr_controller.py`

- [ ] **Step 1: Update _default_code_config to include market_strategy**

At lines 543-575, update the default code to include a market_strategy function:

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
def market_strategy(market_state, weights):
    risk_level = market_state[2]
    if risk_level > 0.6:
        return 0.5
    return 1.0
"""
        result = sandbox_validate(code)
        if result['ok']:
            return {
                'code': code,
                'revise_state_fn': result['revise_state'],
                'intrinsic_reward_fn': result['intrinsic_reward'],
                'market_strategy_fn': result.get('market_strategy'),
                'feature_dim': result['feature_dim'],
                'state_dim': result['state_dim'],
            }
        return {
            'code': 'import numpy as np\ndef revise_state(s): return s\ndef intrinsic_reward(s): return 0.0\ndef market_strategy(market_state, weights): return 1.0',
            'revise_state_fn': lambda s: s,
            'intrinsic_reward_fn': lambda s: 0.0,
            'market_strategy_fn': lambda m, w: 1.0,
            'feature_dim': 0,
            'state_dim': 120,
        }
```

- [ ] **Step 2: Update _generate_code to extract market_strategy_fn**

At lines 180-189, update the valid_samples append to include market_strategy_fn:

```python
                if result['ok']:
                    print(f"    Valid code: feature_dim={result['feature_dim']}, "
                          f"state_dim={result['state_dim']}, "
                          f"has_market_strategy={result.get('market_strategy') is not None}")
                    valid_samples.append({
                        'code': code,
                        'revise_state_fn': result['revise_state'],
                        'intrinsic_reward_fn': result['intrinsic_reward'],
                        'market_strategy_fn': result.get('market_strategy'),
                        'feature_dim': result['feature_dim'],
                        'state_dim': result['state_dim'],
                    })
```

- [ ] **Step 3: Update _train_ppo to accept and pass portfolio_features_fn and market_strategy_fn**

At line 262-263, the signature already has `portfolio_features_fn=None`. Update the PortfolioEnv instantiation at line 281-290 to pass both:

```python
        # Build default portfolio features if not provided
        if portfolio_features_fn is None:
            pf_selection = [
                {'indicator': 'momentum_rank', 'params': {'window': 20}},
                {'indicator': 'portfolio_volatility', 'params': {'window': 20}},
                {'indicator': 'rolling_correlation', 'params': {'window': 60}},
            ]
            portfolio_features_fn = build_portfolio_features(pf_selection)

        env = PortfolioEnv(
            self.data_path, env_config,
            revise_state_fn=code_sample.get('revise_state_fn'),
            portfolio_features_fn=portfolio_features_fn,
            reward_rules_fn=reward_config.get('reward_rules_fn'),
            detect_regime_fn=detect_market_regime,
            intrinsic_reward_fn=code_sample.get('intrinsic_reward_fn'),
            market_strategy_fn=code_sample.get('market_strategy_fn'),
            train_period=period,
            transaction_cost=self.transaction_cost,
        )
```

- [ ] **Step 4: Update _evaluate to pass portfolio_features_fn and market_strategy_fn**

At lines 496-505, update the PortfolioEnv instantiation:

```python
        env = PortfolioEnv(
            self.data_path, eval_config,
            revise_state_fn=code_sample.get('revise_state_fn'),
            portfolio_features_fn=None,
            reward_rules_fn=reward_config.get('reward_rules_fn'),
            detect_regime_fn=detect_market_regime,
            intrinsic_reward_fn=code_sample.get('intrinsic_reward_fn'),
            market_strategy_fn=code_sample.get('market_strategy_fn'),
            train_period=period,
            transaction_cost=self.transaction_cost,
        )
```

Note: _evaluate still passes `portfolio_features_fn=None` for evaluation consistency with the trained state_dim. This is intentional — the training env has portfolio_features, and we need to match the state_dim during evaluation.

Wait — actually this is a bug. If we train with portfolio_features but evaluate without, state_dim won't match. We need to pass the same portfolio_features_fn during evaluation too.

Update _evaluate to also build default portfolio_features:

```python
        pf_selection = [
            {'indicator': 'momentum_rank', 'params': {'window': 20}},
            {'indicator': 'portfolio_volatility', 'params': {'window': 20}},
            {'indicator': 'rolling_correlation', 'params': {'window': 60}},
        ]
        portfolio_features_fn = build_portfolio_features(pf_selection)

        env = PortfolioEnv(
            self.data_path, eval_config,
            revise_state_fn=code_sample.get('revise_state_fn'),
            portfolio_features_fn=portfolio_features_fn,
            reward_rules_fn=reward_config.get('reward_rules_fn'),
            detect_regime_fn=detect_market_regime,
            intrinsic_reward_fn=code_sample.get('intrinsic_reward_fn'),
            market_strategy_fn=code_sample.get('market_strategy_fn'),
            train_period=period,
            transaction_cost=self.transaction_cost,
        )
```

- [ ] **Step 5: Update the retrain section (V1 Step 5) in run() to also extract market_strategy_fn**

At lines 746-759, update the code transfer section:

```python
            # Recreate code_sample from best config (transfer learned features)
            best_code = self.best_config.get('code', '')
            code_sample = self._default_code_config()
            if best_code:
                try:
                    r = sandbox_validate(best_code)
                    if r['ok']:
                        code_sample = {
                            'code': best_code,
                            'revise_state_fn': r['revise_state'],
                            'intrinsic_reward_fn': r['intrinsic_reward'],
                            'market_strategy_fn': r.get('market_strategy'),
                            'feature_dim': r['feature_dim'],
                            'state_dim': r['state_dim'],
                        }
                except Exception:
                    pass
```

- [ ] **Step 6: Update _save_iteration to save market_strategy code info**

At line 821-831, add market_strategy presence to saved config:

```python
        save_cfg = {
            'reward_rules': reward_config.get('reward_rules', []),
            'lambda': reward_config.get('lambda', self.default_lambda),
            'feature_dim': code_sample.get('feature_dim', 0),
            'state_dim': code_sample.get('state_dim', 120),
            'has_market_strategy': code_sample.get('market_strategy_fn') is not None,
        }
```

- [ ] **Step 7: Commit**

```bash
git add "组合优化_ppo_策略迁移_v1_市场优化/core/lesr_controller.py"
git commit -m "feat(controller): enable portfolio_features, handle market_strategy generation & passthrough"
```

---

## Task 8: Integration test — run a quick smoke test

**Files:**
- No new files

- [ ] **Step 1: Run a quick import test**

Run: `cd /home/wangmeiyi/AuctionNet/lesr/组合优化_ppo_策略迁移_v1_市场优化 && python -c "
import sys; sys.path.insert(0,'core')
from regime_detector import detect_market_regime
from code_sandbox import validate
from portfolio_env import PortfolioEnv
from lesr_controller import LESRController
from prompts import build_init_prompt, build_next_iteration_prompt
from portfolio_features import build_portfolio_features
import numpy as np

# Test 1: regime_detector returns 6 dims
s = {t: np.random.randn(120)*100+150 for t in ['TSLA','NFLX','AMZN','MSFT','JNJ']}
r = detect_market_regime(s)
assert len(r) == 6, f'Expected 6 dims, got {len(r)}'
print(f'Test 1 PASS: regime_detector returns {len(r)} dims: {r}')

# Test 2: sandbox validates code with market_strategy
code = '''
import numpy as np
def revise_state(s):
    return np.concatenate([s, [0.0, 0.0]])
def intrinsic_reward(updated_s):
    return 0.0
def market_strategy(market_state, weights):
    return 1.0
'''
result = validate(code)
assert result['ok'], f'Sandbox failed: {result[\"errors\"]}'
assert result.get('market_strategy') is not None, 'market_strategy not extracted'
print(f'Test 2 PASS: sandbox validates code with market_strategy')

# Test 3: sandbox rejects market_strategy out of range
code_bad = '''
import numpy as np
def revise_state(s):
    return np.concatenate([s, [0.0]])
def intrinsic_reward(updated_s):
    return 0.0
def market_strategy(market_state, weights):
    return 5.0
'''
result_bad = validate(code_bad)
assert not result_bad['ok'], f'Should have failed for out-of-range market_strategy'
print(f'Test 3 PASS: sandbox rejects out-of-range market_strategy')

# Test 4: build_init_prompt mentions market_strategy
prompt = build_init_prompt('test stats')
assert 'market_strategy' in prompt, 'market_strategy not in init prompt'
assert 'MARKET' in prompt.upper(), 'market state desc not in prompt'
print(f'Test 4 PASS: init prompt includes market_strategy')

# Test 5: portfolio_features builds correctly
pf_fn = build_portfolio_features([
    {'indicator': 'momentum_rank', 'params': {'window': 20}},
    {'indicator': 'portfolio_volatility', 'params': {'window': 20}},
])
pf_result = pf_fn(s)
print(f'Test 5 PASS: portfolio_features returns {len(pf_result)} dims')

print('\\nAll integration tests PASSED!')
"`

- [ ] **Step 2: If tests fail, fix issues**

Address any import errors, dimension mismatches, or logic errors found during the smoke test.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "fix: address integration test issues for market-state-algorithm-layer"
```

---

## Self-Review

**1. Spec coverage check:**
- 6-dim market state (trend, vol, risk, corr, breadth, vol_ratio) → Task 1
- Enable portfolio_features in LESR training → Task 7 (Step 3)
- market_strategy function generation → Tasks 5, 6, 7
- market_strategy outputs risk_scale only, affects reward only → Task 4 (Step 4)
- Sandbox validation for market_strategy → Task 5
- Prompt updates → Task 6
- Backward compatibility (market_strategy defaults to 1.0) → Task 7 (Step 1)

**2. Placeholder scan:** No TBDs, TODOs, or "fill in later" patterns found.

**3. Type consistency check:**
- `detect_market_regime` returns 6-dim `np.ndarray` — consistent across Task 1 (definition), Task 4 (usage in portfolio_env), Task 7 (lesr_controller)
- `market_strategy_fn` signature: `(market_state: ndarray, weights: ndarray) -> float` — consistent across Task 4 (portfolio_env), Task 5 (sandbox), Task 6 (prompts), Task 7 (controller)
- `validate()` return dict includes `'market_strategy'` key — consistent across Task 5 (definition), Task 7 (consumption)
- Default portfolio_features selection is the same list in Task 7 Steps 3 and 4
