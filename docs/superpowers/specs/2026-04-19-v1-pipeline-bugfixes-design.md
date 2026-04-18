# V1 Pipeline Bugfixes Design

Date: 2026-04-19

## Summary

Fix 3 bugs in the LESR v1 pipeline (`组合优化_ppo_策略迁移_v1/`) that cause incorrect COT feedback and wasted LLM reward configuration.

## Bugs and Fixes

### Fix 1: Lambda Not Passed to Environment

**File**: `core/lesr_controller.py` — `_train_ppo()` method

**Problem**: LLM selects `lambda` in reward config (`reward_config['lambda']`), but `PortfolioEnv` reads `config['portfolio']['default_lambda']` which is always 0.5. The LLM's choice has zero effect.

**Fix**: In `_train_ppo()`, inject `reward_config['lambda']` into a copy of config before creating `PortfolioEnv`:

```python
lam = reward_config.get('lambda', self.default_lambda)
env_config = dict(self.config)
env_config['portfolio'] = dict(self.config.get('portfolio', {}))
env_config['portfolio']['default_lambda'] = lam
env = PortfolioEnv(self.data_path, env_config, ...)
```

Apply the same pattern in `_evaluate()`.

### Fix 2: IC/SHAP/Intrinsic Reward Only Uses TSLA

**Files**: `core/portfolio_env.py`, `core/ic_analyzer.py`

**Problem**:
- `get_revised_states()` only samples `TICKERS[0]` (TSLA) revised states
- `step()` intrinsic_reward only uses TSLA's revised state
- IC is computed as "TSLA features vs equal-weight portfolio returns" — mismatch

**Fix**:

#### 2a. `get_revised_states()` — return per-ticker revised states

Change from returning `(revised_states, forward_returns, regime_labels)` where revised_states only has TSLA, to returning a dict:

```python
return {
    'revised_states_per_ticker': {ticker: np.ndarray},  # 5 arrays of (N, state_dim)
    'forward_returns': np.ndarray,                       # (N,) equal-weight portfolio returns
    'regime_labels': np.ndarray,                         # (N,)
}
```

Each ticker gets its own revised states array. Forward returns remain the same (equal-weight portfolio).

#### 2b. `step()` — intrinsic_reward averages over all 5 tickers

```python
if self.intrinsic_reward_fn:
    raw_states_now = self._get_raw_states_dict(self.current_step)
    ir_values = []
    for ticker in TICKERS:
        revised = self.revise_state_fn(raw_states_now[ticker])
        ir_values.append(float(self.intrinsic_reward_fn(revised)))
    intrinsic_r = float(np.mean(ir_values))
    intrinsic_r = np.clip(intrinsic_r, -1.0, 1.0)
```

#### 2c. `compute_ic_profile_ensemble()` — new function in `ic_analyzer.py`

Accepts a dict of per-ticker revised states and forward returns. Computes IC per ticker, then averages:

```python
def compute_ic_profile_ensemble(
    revised_states_per_ticker: Dict[str, np.ndarray],
    forward_returns: np.ndarray,
) -> Dict[int, float]:
    """Compute IC averaged across all tickers."""
    all_ics = []
    for ticker, states in revised_states_per_ticker.items():
        ic = compute_ic_profile(states, forward_returns)
        if ic:
            all_ics.append(ic)

    if not all_ics:
        return {}

    # Average IC across tickers for each dimension
    all_dims = set()
    for ic in all_ics:
        all_dims.update(ic.keys())

    return {dim: float(np.mean([ic.get(dim, 0.0) for ic in all_ics]))
            for dim in sorted(all_dims)}
```

Same pattern for `compute_regime_specific_ic_ensemble()`.

#### 2d. Caller changes in `lesr_controller.py`

Update `_train_ppo()` to use the new dict return and ensemble IC functions:

```python
result = env.get_revised_states(300)
revised_per_ticker = result['revised_states_per_ticker']
forward_returns = result['forward_returns']
regime_labels = result['regime_labels']

if len(next(iter(revised_per_ticker.values()))) > 20:
    ic_profile = compute_ic_profile_ensemble(revised_per_ticker, forward_returns)
    regime_ic = compute_regime_specific_ic_ensemble(revised_per_ticker, forward_returns, regime_labels)
```

Intrinsic reward stats similarly averages over tickers.

### Fix 3: SHAP extra_start/extra_end Bounds

**Files**: `core/ic_analyzer.py`, `core/lesr_controller.py`

**Problem**: `compute_critic_shap()` uses hardcoded `extra_start=50`. The actual state layout is `compressed(50) + extras(K*5) + portfolio(P) + regime(3) + weights(6)`. SHAP treats portfolio features, regime, and weights as LLM-generated features, polluting the analysis.

**Fix**:

#### 3a. Add `extra_end` parameter to `compute_critic_shap()`

```python
def compute_critic_shap(critic, env_states, extra_start=50, extra_end=None, device='cpu'):
    # If extra_end not specified, use all dims after extra_start
    if extra_end is None:
        extra_end = env_states.shape[1]
    # Only compute SHAP for dims in [extra_start, extra_end)
    for dim in range(extra_start, extra_end):
        shap_profile[dim] = float(np.mean(np.abs(shap_arr[:, dim])))
```

#### 3b. Compute correct extra_end in `_train_ppo()`

The extras section spans from index 50 to `50 + feature_dim * 5`:

```python
feature_dim = code_sample.get('feature_dim', 0)
extra_end = 50 + feature_dim * 5
shap_profile = compute_critic_shap(
    agent.critic, env_states, extra_start=50, extra_end=extra_end, device=device)
```

## Files Changed

| File | Change |
|------|--------|
| `core/lesr_controller.py` | Fix lambda injection (Fix 1), update IC/SHAP callers (Fix 2d), compute extra_end (Fix 3b) |
| `core/portfolio_env.py` | Update `get_revised_states()` return format (Fix 2a), fix `step()` intrinsic_reward (Fix 2b) |
| `core/ic_analyzer.py` | Add `compute_ic_profile_ensemble()`, `compute_regime_specific_ic_ensemble()` (Fix 2c), add `extra_end` param (Fix 3a) |

## What NOT Changed

- PPO agent, prompts, feature library, reward rules — untouched
- Baseline comparison logic — unchanged (Baseline2 remains the primary comparison)
- Config files — no changes needed
- Tests — updated to match new function signatures only
