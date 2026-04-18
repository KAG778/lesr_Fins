# V1 Pipeline Bugfixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 3 bugs that cause incorrect COT feedback and wasted LLM reward configuration in the LESR v1 pipeline.

**Architecture:** Three targeted fixes across `ic_analyzer.py`, `portfolio_env.py`, and `lesr_controller.py`. Each fix is independent but they must be applied in order (ic_analyzer first, since controller depends on it).

**Tech Stack:** Python, NumPy, PyTorch, SHAP

**Spec:** `docs/superpowers/specs/2026-04-19-v1-pipeline-bugfixes-design.md`

---

## File Structure

| File | Change |
|------|--------|
| `组合优化_ppo_策略迁移_v1/core/ic_analyzer.py` | Add `extra_end` param to `compute_critic_shap`, add `compute_ic_profile_ensemble`, `compute_regime_specific_ic_ensemble` |
| `组合优化_ppo_策略迁移_v1/core/portfolio_env.py` | Fix `step()` intrinsic_reward to average all 5 tickers, fix `get_revised_states()` to return per-ticker dict |
| `组合优化_ppo_策略迁移_v1/core/lesr_controller.py` | Inject lambda into env config, update IC/SHAP callers to use ensemble functions, compute correct `extra_end` |

---

### Task 1: Fix `compute_critic_shap` — add `extra_end` parameter

**Files:**
- Modify: `组合优化_ppo_策略迁移_v1/core/ic_analyzer.py:88-171`

- [ ] **Step 1: Add `extra_end` parameter to `compute_critic_shap`**

Change the function signature at line 88 from:

```python
def compute_critic_shap(critic, env_states: np.ndarray,
                        extra_start: int = 50,
                        device: str = 'cpu') -> Dict[int, float]:
```

to:

```python
def compute_critic_shap(critic, env_states: np.ndarray,
                        extra_start: int = 50,
                        extra_end: int = None,
                        device: str = 'cpu') -> Dict[int, float]:
```

Then at line 166-169, change:

```python
    # Only report SHAP for extra dims (LLM-generated features)
    shap_profile = {}
    for dim in range(extra_start, n_dims):
        shap_profile[dim] = float(np.mean(np.abs(shap_arr[:, dim])))
```

to:

```python
    # Only report SHAP for extra dims (LLM-generated features)
    end = extra_end if extra_end is not None else n_dims
    shap_profile = {}
    for dim in range(extra_start, end):
        shap_profile[dim] = float(np.mean(np.abs(shap_arr[:, dim])))
```

Also update the guard at line 127 from:

```python
    if n_dims <= extra_start:
        return {}
```

to:

```python
    effective_end = extra_end if extra_end is not None else n_dims
    if effective_end <= extra_start:
        return {}
```

- [ ] **Step 2: Commit**

```bash
git add 组合优化_ppo_策略迁移_v1/core/ic_analyzer.py
git commit -m "fix(ic-analyzer): add extra_end param to compute_critic_shap for accurate SHAP bounds"
```

---

### Task 2: Add ensemble IC functions to `ic_analyzer.py`

**Files:**
- Modify: `组合优化_ppo_策略迁移_v1/core/ic_analyzer.py` (append after `compute_regime_specific_ic`)

- [ ] **Step 1: Add `compute_ic_profile_ensemble`**

Insert after the `compute_regime_specific_ic` function (after line 85):

```python
def compute_ic_profile_ensemble(revised_states_per_ticker: Dict[str, np.ndarray],
                                forward_returns: np.ndarray) -> Dict[int, float]:
    """Compute IC averaged across all tickers.

    对每只股票分别计算 IC，然后取平均值。
    这避免了只用单只股票（如 TSLA）导致的有偏估计。
    """
    all_ics = []
    for ticker, states in revised_states_per_ticker.items():
        ic = compute_ic_profile(states, forward_returns)
        if ic:
            all_ics.append(ic)

    if not all_ics:
        return {}

    all_dims = set()
    for ic in all_ics:
        all_dims.update(ic.keys())

    return {dim: float(np.mean([ic.get(dim, 0.0) for ic in all_ics]))
            for dim in sorted(all_dims)}


def compute_regime_specific_ic_ensemble(revised_states_per_ticker: Dict[str, np.ndarray],
                                        forward_returns: np.ndarray,
                                        regime_labels: np.ndarray) -> Dict[str, Dict[int, float]]:
    """Compute regime-specific IC averaged across all tickers.

    按市场状态分类后，对每只股票分别计算 IC，然后取平均值。
    """
    per_ticker_regime_ics = []
    for ticker, states in revised_states_per_ticker.items():
        ric = compute_regime_specific_ic(states, forward_returns, regime_labels)
        if ric:
            per_ticker_regime_ics.append(ric)

    if not per_ticker_regime_ics:
        return {}

    all_regimes = set()
    for ric in per_ticker_regime_ics:
        all_regimes.update(ric.keys())

    result = {}
    for regime in all_regimes:
        regime_ics = [ric.get(regime, {}) for ric in per_ticker_regime_ics]
        all_dims = set()
        for ic in regime_ics:
            all_dims.update(ic.keys())
        result[regime] = {dim: float(np.mean([ic.get(dim, 0.0) for ic in regime_ics]))
                          for dim in sorted(all_dims)}

    return result
```

- [ ] **Step 2: Commit**

```bash
git add 组合优化_ppo_策略迁移_v1/core/ic_analyzer.py
git commit -m "feat(ic-analyzer): add ensemble IC functions for multi-ticker averaging"
```

---

### Task 3: Fix `step()` intrinsic_reward to average all 5 tickers

**Files:**
- Modify: `组合优化_ppo_策略迁移_v1/core/portfolio_env.py:345-354`

- [ ] **Step 1: Replace single-ticker intrinsic_reward with 5-ticker average**

Change lines 345-354 from:

```python
        # Intrinsic reward from LLM code
        intrinsic_r = 0.0
        if self.intrinsic_reward_fn:
            try:
                raw_states_now = self._get_raw_states_dict(self.current_step)
                revised = self.revise_state_fn(raw_states_now[TICKERS[0]])
                intrinsic_r = float(self.intrinsic_reward_fn(revised))
                intrinsic_r = np.clip(intrinsic_r, -1.0, 1.0)
            except Exception:
                intrinsic_r = 0.0
```

to:

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add 组合优化_ppo_策略迁移_v1/core/portfolio_env.py
git commit -m "fix(portfolio-env): average intrinsic_reward across all 5 tickers instead of TSLA only"
```

---

### Task 4: Fix `get_revised_states()` to return per-ticker dict

**Files:**
- Modify: `组合优化_ppo_策略迁移_v1/core/portfolio_env.py:374-414`

- [ ] **Step 1: Rewrite `get_revised_states()` to return per-ticker data**

Replace the entire method (lines 374-414) with:

```python
    def get_revised_states(self, n_samples: int = 200) -> dict:
        """Get revised states per ticker and forward returns for IC computation.

        获取每只股票的修订状态和远期收益，用于 IC 计算。
        返回 dict:
          - 'revised_states_per_ticker': {ticker: (N, state_dim)}
          - 'forward_returns': (N,)
          - 'regime_labels': (N,)
        """
        if self.revise_state_fn is None:
            return {
                'revised_states_per_ticker': {},
                'forward_returns': np.array([]),
                'regime_labels': np.array([]),
            }

        n = min(n_samples, len(self.dates) - WINDOW - 1)
        if n < 10:
            return {
                'revised_states_per_ticker': {},
                'forward_returns': np.array([]),
                'regime_labels': np.array([]),
            }
        indices = np.linspace(WINDOW, len(self.dates) - 2, n, dtype=int)

        revised_per_ticker = {t: [] for t in TICKERS}
        forward_list = []
        regime_labels = []

        for idx in indices:
            raw_states = self._get_raw_states_dict(idx)
            for ticker in TICKERS:
                revised = self.revise_state_fn(raw_states[ticker])
                revised_per_ticker[ticker].append(revised)

            date = self.dates[idx]
            next_date = self.dates[idx + 1]
            ret = 0.0
            for ticker in TICKERS:
                p0 = self.prices.get(date, {}).get(ticker, 0.0)
                p1 = self.prices.get(next_date, {}).get(ticker, 0.0)
                if p0 > 0:
                    ret += (p1 - p0) / p0
            forward_list.append(ret / len(TICKERS))

            if self.detect_regime_fn:
                rv = self.detect_regime_fn(raw_states)
                from ic_analyzer import _classify_regime
                regime_labels.append(_classify_regime(rv[0], rv[1]))
            else:
                regime_labels.append('neutral')

        revised_states_per_ticker = {
            t: np.array(revised_per_ticker[t]) for t in TICKERS
        }
        return {
            'revised_states_per_ticker': revised_states_per_ticker,
            'forward_returns': np.array(forward_list),
            'regime_labels': np.array(regime_labels),
        }
```

- [ ] **Step 2: Commit**

```bash
git add 组合优化_ppo_策略迁移_v1/core/portfolio_env.py
git commit -m "fix(portfolio-env): return per-ticker revised states instead of TSLA-only"
```

---

### Task 5: Fix `lesr_controller.py` — lambda injection, ensemble IC, SHAP bounds

**Files:**
- Modify: `组合优化_ppo_策略迁移_v1/core/lesr_controller.py`

- [ ] **Step 1: Update imports at line 42**

Change:

```python
from ic_analyzer import compute_ic_profile, compute_regime_specific_ic, compute_critic_shap, build_ic_cot_prompt
```

to:

```python
from ic_analyzer import (compute_ic_profile, compute_regime_specific_ic,
                         compute_ic_profile_ensemble, compute_regime_specific_ic_ensemble,
                         compute_critic_shap, build_ic_cot_prompt)
```

- [ ] **Step 2: Inject lambda into env config in `_train_ppo()`**

At lines 270-282, insert lambda injection before creating the env. Change:

```python
        period = override_period if override_period else self.train_period
        print(f"\nStep 3: PPO Training (period: {period[0]} ~ {period[1]})")

        env = PortfolioEnv(
            self.data_path, self.config,
```

to:

```python
        period = override_period if override_period else self.train_period
        print(f"\nStep 3: PPO Training (period: {period[0]} ~ {period[1]})")

        # Inject LLM-selected lambda into env config
        lam = reward_config.get('lambda', self.default_lambda)
        env_config = dict(self.config)
        env_config['portfolio'] = dict(self.config.get('portfolio', {}))
        env_config['portfolio']['default_lambda'] = lam

        env = PortfolioEnv(
            self.data_path, env_config,
```

- [ ] **Step 3: Update IC/SHAP callers in `_train_ppo()` (lines 363-392)**

Replace the IC/SHAP block (lines 363-392) with:

```python
        # Compute IC profile and Critic SHAP
        ic_profile = {}
        shap_profile = {}
        regime_ic = {}
        ir_stats = {}
        try:
            revised_result = env.get_revised_states(300)
            revised_per_ticker = revised_result['revised_states_per_ticker']
            forward_returns = revised_result['forward_returns']
            regime_labels = revised_result['regime_labels']

            if revised_per_ticker and len(next(iter(revised_per_ticker.values()))) > 20:
                ic_profile = compute_ic_profile_ensemble(revised_per_ticker, forward_returns)
                regime_ic = compute_regime_specific_ic_ensemble(
                    revised_per_ticker, forward_returns, regime_labels)

                # Critic SHAP: what does the trained policy actually use?
                device = str(agent.device)
                n_env_samples = min(100, len(env.dates) - 22)
                env_state_indices = np.linspace(20, len(env.dates) - 2,
                                                n_env_samples, dtype=int)
                env_states = np.array([env._compute_state(idx) for idx in env_state_indices])
                # Extra dims: only LLM-generated features [50 : 50 + feature_dim * 5]
                feature_dim = code_sample.get('feature_dim', 0)
                extra_end = 50 + feature_dim * 5
                shap_profile = compute_critic_shap(
                    agent.critic, env_states, extra_start=50, extra_end=extra_end,
                    device=device)
                if shap_profile:
                    print(f"  SHAP computed for {len(shap_profile)} extra dims")

                if code_sample.get('intrinsic_reward_fn'):
                    # Average intrinsic reward stats across tickers
                    first_ticker_states = next(iter(revised_per_ticker.values()))
                    if len(first_ticker_states) > 10:
                        all_ir_values = []
                        for ticker, states in revised_per_ticker.items():
                            all_ir_values.extend(
                                [code_sample['intrinsic_reward_fn'](s) for s in states[:50]])
                        ir_mean = float(np.mean(all_ir_values))
                        ir_corr = float(np.corrcoef(
                            all_ir_values[:len(forward_returns)],
                            np.tile(forward_returns[:50], len(TICKERS))[:len(all_ir_values)]
                        )[0, 1]) if len(all_ir_values) > 5 else 0.0
                        ir_stats = {
                            'mean': ir_mean,
                            'correlation_with_performance': ir_corr if not np.isnan(ir_corr) else 0.0,
                        }
```

- [ ] **Step 4: Commit**

```bash
git add 组合优化_ppo_策略迁移_v1/core/lesr_controller.py
git commit -m "fix(lesr-controller): inject lambda, use ensemble IC, fix SHAP bounds"
```

---

### Task 6: Fix `_evaluate()` lambda injection

**Files:**
- Modify: `组合优化_ppo_策略迁移_v1/core/lesr_controller.py:440-449`

- [ ] **Step 1: Inject lambda in `_evaluate()`**

Change lines 440-449 from:

```python
        env = PortfolioEnv(
            self.data_path, self.config,
            revise_state_fn=code_sample.get('revise_state_fn'),
            portfolio_features_fn=None,
            reward_rules_fn=reward_config.get('reward_rules_fn'),
            detect_regime_fn=detect_market_regime,
            intrinsic_reward_fn=code_sample.get('intrinsic_reward_fn'),
            train_period=period,
            transaction_cost=self.transaction_cost,
        )
```

to:

```python
        lam = reward_config.get('lambda', self.default_lambda)
        eval_config = dict(self.config)
        eval_config['portfolio'] = dict(self.config.get('portfolio', {}))
        eval_config['portfolio']['default_lambda'] = lam

        env = PortfolioEnv(
            self.data_path, eval_config,
            revise_state_fn=code_sample.get('revise_state_fn'),
            portfolio_features_fn=None,
            reward_rules_fn=reward_config.get('reward_rules_fn'),
            detect_regime_fn=detect_market_regime,
            intrinsic_reward_fn=code_sample.get('intrinsic_reward_fn'),
            train_period=period,
            transaction_cost=self.transaction_cost,
        )
```

- [ ] **Step 2: Commit**

```bash
git add 组合优化_ppo_策略迁移_v1/core/lesr_controller.py
git commit -m "fix(lesr-controller): inject lambda into _evaluate env config"
```
