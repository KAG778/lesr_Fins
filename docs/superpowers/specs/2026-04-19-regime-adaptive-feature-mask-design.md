# Regime-Adaptive Feature Mask Design

**Date:** 2026-04-19
**Scope:** `portfolio_env.py` only — single file, ~15 lines changed

## Problem

LESR-generated features help in bear/volatile markets (risk signals, defensive indicators) but hurt in bull markets — the extra dimensions become noise that PPO overfits to, reducing Sharpe ratio.

V2 experiment results show LESR wins 3/5 windows, losing only in bull markets (W2, W5).

## Design

Add a regime-based attenuation mask in `PortfolioEnv._compute_state()`. LESR extra features are scaled by a coefficient determined by market regime, without changing state dimensions or network architecture.

### Attenuation Rules

| Regime | Condition | Coefficient | Rationale |
|--------|-----------|-------------|-----------|
| Bull | `trend > 0.5 and vol < 0.3` | 0.3 | Reduce noise from defensive features |
| Bear | `trend < -0.3` | 1.0 | Full LESR features (proven effective) |
| Volatile/Neutral | otherwise | 0.7 | Mild attenuation |

### Implementation

**File:** `portfolio_env.py`, method `_compute_state()`

**Change:** Move regime computation before LESR extras, then multiply extras by coefficient.

```python
# Before (lines 186-211):
# 1. compressed raw
# 2. LESR extras (full weight always)
# 3. portfolio features
# 4. regime
# 5. weights

# After:
# 1. compressed raw
# 2. regime (moved up — compute early for mask)
# 3. LESR extras × regime_coefficient
# 4. portfolio features
# 5. regime (keep in state vector as before)
# 6. weights
```

Key: regime appears **twice** in state — once for mask computation (not in state), once as features (in state, unchanged). Actually simpler: compute regime once, use it for mask, then append to parts as before.

### What Does NOT Change

- `state_dim` — same dimensions, only values are scaled
- PPO network architecture
- LLM code generation prompts
- `revise_state_fn` / `intrinsic_reward_fn` — unchanged
- Config format — no new fields needed (coefficients hardcoded, can be made configurable later)

### Config (Optional)

```yaml
portfolio:
  regime_mask:
    bull_coefficient: 0.3
    bear_coefficient: 1.0
    neutral_coefficient: 0.7
```

Defaults used if not specified.

## Testing

Run V2 experiment with regime mask enabled on same 5 windows. Expected: W2/W5 (bull) Sharpe improves, W3/W4 (bear) stays similar.
