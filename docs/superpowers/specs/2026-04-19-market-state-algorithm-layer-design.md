# Design: Market State to Algorithm Layer + LLM Dual-Modal Generation

**Date:** 2026-04-19
**Status:** Approved
**Target:** `组合优化_ppo_策略迁移_v1_市场优化/`

## Problem Statement

Current architecture has a misalignment of responsibilities:

1. **Market statistics** (per-stock volatility, correlation matrix, sector profiles) are computed in `market_stats.py` but only injected as text into LLM prompts — they never enter the RL observation space.
2. **Regime detection** provides only 3 dimensions (trend, volatility, risk) — insufficient for PPO to make nuanced market-aware decisions.
3. **LLM-generated `revise_state(s)`** only receives single-stock 120-dim data — it is blind to cross-stock and market-level context.
4. **Portfolio features** (`portfolio_features.py`) are never enabled during LESR training (`portfolio_features_fn=None`).

As a result, when PPO is retrained, it cannot leverage rich market information that the LLM could "see" in its prompt but chose not to encode into features.

## Design Decision

**Principle:** Market state belongs in the algorithm layer (fixed, always computed, PPO directly observes it). LLM focuses on generating generalizable strategies (stock-level features + market-level strategy), not redundant market indicators.

## Architecture: Layered Design

### Layer 1: Algorithm Layer (Fixed, PPO Observes Directly)

```
Observation vector:
  [compressed_raw(50d) | revise_state_extras(K×5) | portfolio_features(P) | market_state(6d) | weights(6d)]
```

#### 1a. Market State — 6 Dimensions (New/Extended)

Replaces the current 3-dim regime vector. Computed by extending `regime_detector.py`.

| # | Feature | Dim | Calculation | Description |
|---|---------|-----|-------------|-------------|
| 1 | trend_direction | 1 | 5-day MA vs overall mean, clipped [-1, 1] | Market trend (existing) |
| 2 | volatility_level | 1 | Recent vol z-score vs historical, clipped [0, 1] | Volatility regime (existing) |
| 3 | risk_level | 1 | Max drawdown depth in recent 10 days, clipped [0, 1] | Drawdown risk (existing) |
| 4 | avg_cross_correlation | 1 | Mean of 5×5 correlation matrix | Overall cross-stock coupling |
| 5 | market_breadth | 1 | Count of stocks with positive 5-day return / total | How many stocks are rising |
| 6 | volatility_ratio | 1 | Recent 5-day vol / recent 20-day vol | Short-term vol change rate |

**Total: 6 dimensions** (up from 3).

#### 1b. Portfolio Features — Enable in LESR Training

Currently `portfolio_features_fn=None` is passed in `lesr_controller._train_ppo()`. Change to construct and pass a portfolio features closure.

Available features from `portfolio_features.py`:
- momentum_rank, rolling_correlation, relative_strength, portfolio_volatility
- return_dispersion, mean_reversion_score, liquidity_pressure, concentration_index

Default selection for initial enablement: `momentum_rank`, `portfolio_volatility`, `rolling_correlation`.

#### 1c. Compressed Raw State & Weights

No changes. 50d compressed raw + 6d current weights remain as-is.

### Layer 2: LLM Generation Layer (Dynamic, Per-Iteration)

LLM now generates **three functions** (previously two):

#### 2a. `revise_state(s)` — Stock-Level Features (Unchanged)

- Input: 120-dim single-stock raw state
- Output: array of length >= 120 (dims 0-119 preserved, dims 120+ are extra features)
- Can import building blocks from `feature_library.py`
- This channel remains exactly as before.

#### 2b. `intrinsic_reward(updated_s)` — Stock-Level Reward (Unchanged)

- Input: full revised state (120 original + extras)
- Output: scalar reward
- This channel remains exactly as before.

#### 2c. `market_strategy(market_state, weights)` — Market-Level Strategy (New)

- Input:
  - `market_state`: 6-dim numpy array from algorithm layer
  - `weights`: 6-dim numpy array (current portfolio allocation)
- Output: **`risk_scale`** — a single float in range [0.3, 2.0]
  - Values < 1.0: conservative (reduce risk exposure)
  - Values > 1.0: aggressive (increase risk exposure)
  - Value = 1.0: neutral

**Why only risk_scale?**
- Safety: direct action modification (allocation_bias) could produce unreasonable weights.
- PPO autonomy: PPO's core value is learning optimal actions; LLM should guide via reward, not override decisions.
- Simplicity: single scalar is easy to validate and interpret.

### Reward Integration

```python
# In portfolio_env.step():
strategy = self.market_strategy_fn(market_state, self.weights)
risk_scale = np.clip(strategy, 0.3, 2.0)

total_reward = (base_reward * risk_scale) + rule_bonus + intrinsic_r
```

### Sandbox Validation for market_strategy

Add validation in `code_sandbox.py`:
1. AST whitelist check (same as revise_state)
2. Test execution with random 6-dim + 6-dim input
3. Verify output is a single float
4. Verify output is in [0.3, 2.0] range (clip if slightly outside)
5. Verify no NaN/Inf

## Prompt Changes

### What changes in `prompts.py`:

1. **Market state description** added to all prompts:
   - List the 6 market features with their meanings and typical ranges
   - Explicitly tell LLM: "Market state is already computed in the algorithm layer. Do not recompute market-level indicators in revise_state."

2. **New function template** for `market_strategy`:
   ```
   def market_strategy(market_state, weights):
       """
       market_state: 6-dim array [trend, vol_level, risk, avg_corr, breadth, vol_ratio]
       weights: 6-dim array [w_stock1, ..., w_stock5, w_cash]
       Returns: risk_scale (float in [0.3, 2.0])
       """
       # LLM fills this in
       ...
   ```

3. **COT feedback** includes market_strategy analysis:
   - How risk_scale correlated with actual market regime
   - Whether risk_scale improved risk-adjusted returns
   - Suggestions for next iteration

## State Dimension Summary

```
Before:
  [compressed_raw(50) | extras(K×5) | regime(3) | weights(6)]
  Total: 59 + K×5

After:
  [compressed_raw(50) | extras(K×5) | portfolio_features(P) | market_state(6) | weights(6)]
  Total: 62 + K×5 + P
```

## Implementation Priority

1. **Phase 1:** Extend `regime_detector.py` to return 6-dim market state
2. **Phase 2:** Enable `portfolio_features` in LESR training
3. **Phase 3:** Add `market_strategy` to LLM generation + sandbox validation
4. **Phase 4:** Update prompts and COT feedback
5. **Phase 5:** Integration testing with backward compatibility (market_strategy defaults to `lambda m, w: 1.0`)

## Files to Modify

| File | Changes |
|------|---------|
| `core/regime_detector.py` | Extend from 3-dim to 6-dim market state |
| `core/portfolio_env.py` | Update `_compute_state()` for new market_state dims + portfolio_features; update `step()` for risk_scale reward |
| `core/lesr_controller.py` | Pass portfolio_features_fn; handle market_strategy generation |
| `core/prompts.py` | Add market state description; add market_strategy template; update COT |
| `core/code_sandbox.py` | Add validation for market_strategy function |
| `core/ic_analyzer.py` | Include market_strategy analysis in COT feedback |

## Backward Compatibility

- `market_strategy_fn` defaults to `lambda m, w: 1.0` (neutral risk scale)
- If `regime_detector` returns 3-dim (old format), pad with 3 zeros
- Old saved models will have different state_dim; document this as a breaking change
