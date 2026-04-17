# LESR Portfolio Optimization: Structural Refactoring Design

> Date: 2026-04-17
> Status: Approved
> Approach: B (Structural Refactoring)

## 1. Background

The current `组合优化_ppo` project diverges from the original LESR methodology in several key ways:

- LLM selects features from a catalog (JSON) instead of generating Python code
- Original 120-dim raw price state is discarded entirely, replaced by a few indicators
- No intrinsic reward — replaced by reward rules (behavior constraints)
- Single-sample per iteration, no multi-sample comparison
- Uses worst-trade COT instead of principled diagnostic feedback (Lipschitz/IC)

This redesign aligns the portfolio optimization system with LESR's core methodology while adapting to the financial domain.

## 2. Design Decisions

| Dimension | Decision | Rationale |
|-----------|----------|-----------|
| LLM role | Feature library + free computation | Balance creativity (code generation) with safety (registered functions only) |
| Feedback signal | IC (Information Coefficient) | Financial domain standard; measures predictive power of each state dimension |
| State construction | Compressed raw + LLM features | Preserve original information in compact form; concatenate LLM-added dimensions |
| Reward design | intrinsic_reward + reward_rules | LLM designs exploration bonus; catalog rules constrain behavior |
| Multi-sampling | N samples per iteration, compare results | Richer feedback signal for LLM; identifies which features work |
| RL algorithm | PPO with Dirichlet (unchanged) | Appropriate for simplex-constrained portfolio weights |

## 3. Architecture

```
LLM generates code (revise_state + intrinsic_reward)
  |
  v
code_sandbox.py (AST whitelist + test execution)
  |
  v
Save as .py file, dynamic import
  |
  v
portfolio_env.py:
  state = [compressed_raw(10) * 5 stocks, revised_features(K) * 5, portfolio_features(P), regime(3), weights(6)]
  reward = base_reward + intrinsic_reward(revised_state) + reward_rules_bonus
  |
  v
ppo_agent.py: PPO training (multi-sample x multi-policy)
  |
  v
ic_analyzer.py: Compute IC -> COT feedback -> next iteration
```

## 4. New Module: code_sandbox.py

### Purpose
Execute LLM-generated Python code safely. Validate before use.

### Validation Pipeline
1. **AST whitelist check**
   - Allow: `numpy`, `math`, basic operators, `if/else`, `for/while`
   - Allow: calls to registered functions from `feature_library`
   - Block: `import os/sys/subprocess`, `open()`, `exec()`, `eval()`, `__import__`

2. **Test execution**
   - Input: random 120-dim array (simulating one stock's raw state)
   - Check: output is 1D numpy array, no NaN/Inf
   - Check: `revise_state` output starts with original `s` (first 120 dims preserved)
   - Check: `intrinsic_reward` returns scalar in [-100, 100]

3. **Dimension detection**
   - Record new state dim = `len(revise_state(s))`
   - Record extra dims = new state dim - 120

### Registered Building Blocks
Exported from `feature_library.py` as standalone functions:

```python
compute_relative_momentum(prices, window)
compute_cross_sectional_rank(momentum_values)
compute_realized_volatility(returns, window)
compute_downside_risk(returns, window)
compute_beta(returns, market_returns, window)
compute_multi_horizon_momentum(prices, windows)
compute_zscore_price(prices, window)
compute_mean_reversion_signal(prices, window)
compute_turnover_ratio(volumes, window)
# ... all features from the redesigned library
```

### LLM Code Example
```python
import numpy as np
from feature_library import compute_realized_volatility, compute_relative_momentum

def revise_state(s):
    closes = s[0::6]  # extract close prices from interleaved array
    returns = np.diff(closes) / (closes[:-1] + 1e-10)

    vol = compute_realized_volatility(returns, 20)
    mom = compute_relative_momentum(closes, 20)

    return np.concatenate([s, [vol, mom]])  # 120 + 2 = 122 dims

def intrinsic_reward(updated_s):
    vol = updated_s[120]  # first extra dim
    mom = updated_s[121]  # second extra dim
    # Encourage exploring states with clear momentum signals
    return 0.01 * abs(mom) / (vol + 0.01)
```

## 5. Redesigned Feature Library

### A. Stock-Level Features (per stock, ~12 building blocks)

| Category | Function | Dims | Description |
|----------|----------|------|-------------|
| Relative Momentum | `compute_relative_momentum(prices, window=20)` | 1 | Excess return vs equal-weight basket |
| Rank Signal | `compute_cross_sectional_rank(values)` | 1 | Cross-sectional rank [0,1]. **Note: requires all stocks' values, computed at portfolio level** |
| Risk | `compute_realized_volatility(returns, window=20)` | 1 | Realized volatility |
| Risk | `compute_downside_risk(returns, window=20)` | 1 | Downside semi-deviation |
| Risk | `compute_beta(returns, market_returns, window=20)` | 1 | Beta to equal-weight portfolio |
| Momentum | `compute_multi_horizon_momentum(prices, windows=[5,10,20])` | 3 | Multi-period momentum |
| Mean Reversion | `compute_zscore_price(prices, window=20)` | 1 | Price z-score vs N-day mean |
| Mean Reversion | `compute_mean_reversion_signal(prices, window=20)` | 1 | Mean reversion strength |
| Liquidity | `compute_turnover_ratio(volumes, window=20)` | 1 | Volume change ratio |

### B. Portfolio-Level Features (cross-stock, ~6 indicators)

| Function | Dims | Description |
|----------|------|-------------|
| `compute_momentum_rank(raw_states)` | 5 | Rank stocks by momentum (existing) |
| `compute_portfolio_volatility(raw_states, window=20)` | 1 | Equal-weight portfolio vol (existing) |
| `compute_return_dispersion(raw_states)` | 1 | Cross-sectional return std (existing) |
| `compute_correlation_breadth(raw_states, window=20)` | 1 | Average pairwise correlation |
| `compute_concentration_hhi(weights)` | 1 | Herfindahl concentration index |
| `compute_risk_budget_deviation(weights, raw_states)` | 1 | Actual vs equal risk allocation |

### Key Design Principle
All features answer the question: "How much weight should THIS stock get RELATIVE TO others?" — not "Should I buy/sell this stock?"

## 6. State Representation

### Compressed Raw State
From 120 dims to ~10 dims per stock:
- Recent 5 close prices (5 dims)
- Recent 5 daily returns (5 dims)

### Full State Vector
```
Per stock (5 stocks):
  compressed_raw[i]        ~10 dims
  revised_features[i]       K dims (LLM-added, varies per iteration)

Portfolio features:         P dims (~10)
Regime vector:              3 dims
Current weights:            6 dims

Total: ~5*(10+K) + 10 + 3 + 6 ≈ 50-80 dims (manageable for PPO)
```

### Environment Changes (portfolio_env.py)
- `_compute_state()`: concatenate compressed_raw + revised extras + portfolio + regime + weights
- `step()`: add intrinsic_reward to base reward
- `revise_state(s)` receives full 120-dim raw state per stock; environment extracts only extra dims for state

## 7. IC Feedback Module (ic_analyzer.py)

### IC Computation
```python
def compute_ic_profile(revised_states, forward_returns):
    """Compute Pearson correlation between each extra state dim and forward returns."""
    extra_dims_start = 120  # source dim
    ic_per_dim = {}
    for dim in range(extra_dims_start, revised_states.shape[1]):
        corr = np.corrcoef(revised_states[:, dim], forward_returns)[0, 1]
        ic_per_dim[dim] = corr if not np.isnan(corr) else 0.0
    return ic_per_dim

def compute_regime_specific_ic(revised_states, forward_returns, regime_labels):
    """Compute IC per market regime (trending up / volatile / trending down)."""
    regime_ics = {}
    for regime in set(regime_labels):
        mask = regime_labels == regime
        regime_ics[regime] = compute_ic_profile(revised_states[mask], forward_returns[mask])
    return regime_ics
```

### Four-Tier Analysis (aligned with LESR COT structure)
1. **Market environment context**: Describe training period market conditions
2. **Strong features** (|IC| > 0.05): Keep/strengthen. Include regime-specific IC.
3. **Weak features** (|IC| < 0.02): Improve or replace.
4. **Negative IC features** (IC < -0.03): Potentially harmful.
5. **Missing analysis**: Feature categories not yet used.
6. **Intrinsic reward diagnosis**: Effectiveness of the exploration bonus.

### COT Feedback Format (with market analysis)
```
========== Code Sample 1 (Sharpe=1.23) ==========
{完整代码}

【Market Environment Overview】
Training period: 2018-01 ~ 2021-12
  Overall trend: Bullish (cumulative gain 85%)
  Volatility regime: Low(2018) -> High(2020 COVID) -> Medium(2021)
  Correlation: Spike in Mar 2020 (panic correlation)
  Worst month: 2020-03 (portfolio drawdown -18%)

【Extra Dims IC Analysis】
  s[120] (realized_volatility): IC = 0.12 <- Strong
    -> High-vol period (2020): IC = 0.18
    -> Low-vol period (2019): IC = 0.05
  s[121] (relative_momentum): IC = 0.08
    -> Trending periods: IC = 0.12, Choppy periods: IC = -0.03
  s[122] (zscore_price): IC = -0.01 <- Weak
    -> Near-zero across all sub-periods
  s[123] (downside_risk): IC = 0.06
    -> Down-market IC = 0.15, Up-market IC = 0.02

【Intrinsic Reward Diagnosis】
  Mean intrinsic_reward = 0.003
  Correlation with final performance = 0.32
  -> Moderate guidance effect

【Improvement Suggestions】
  (a) Remove zscore_price (IC=-0.01), replace with mean_reversion_signal
  (b) realized_volatility effective across all regimes, consider derivatives
  (c) Missing defensive features for down-market, add regime-aware computation
  (d) intrinsic_reward could incorporate volatility adjustment
```

## 8. Multi-Sample Controller

### Main Loop (lesr_controller.py)
```python
for iteration in range(max_iterations):
    # 1. Sample N code sets from LLM
    valid_samples = []
    for i in range(sample_count):
        code = call_llm(code_generation_prompt)
        validation = sandbox.validate(code)
        if validation.ok:
            valid_samples.append(code)

    # 2. Train N independent policies
    results = []
    for sample in valid_samples:
        env = PortfolioEnv(
            revise_state_fn=sample.revise_state,
            intrinsic_reward_fn=sample.intrinsic_reward,
            reward_rules_fn=selected_rules,
        )
        agent = PPOAgent(state_dim=env.state_dim)
        train_result = train(agent, env, max_episodes)
        ic_profile = compute_ic_profile(env.get_revised_states(), forward_returns)
        val_result = evaluate(agent, env, val_period)
        results.append({sample, train_result, val_result, ic_profile})

    # 3. Select best + generate COT feedback
    best = max(results, key=lambda r: r.val_sharpe)
    cot_feedback = build_ic_cot_prompt(results, best, strong_threshold=0.05, weak_threshold=0.02)

    # 4. Update history for next iteration
    history.append({
        'code': best.sample.code,
        'performance': best.val_result,
        'ic_analysis': cot_feedback,
    })
```

### Combined Reward
```python
# In portfolio_env.step():
base_reward = net_return - lambda_mv * drawdown^2
intrinsic_r = intrinsic_reward_fn(revised_state)  # from LLM code
rule_bonus = reward_rules_fn(weights, ...)          # from catalog
total_reward = base_reward + intrinsic_r + rule_bonus
```

## 9. LLM Prompt Design

### 9.1 Prompt Structure (aligned with LESR's 3-prompt pattern)

LESR uses 3 separate prompts in sequence:
1. `init_prompt` → first iteration (state description + task + code example)
2. `cot_prompt` → after training (code + performance + Lipschitz analysis → LLM analyzes)
3. `next_iteration_prompt` → subsequent iterations (full history + all COT suggestions)

Our design mirrors this with IC replacing Lipschitz:

| Prompt | Timing | LESR Original | Our Version |
|--------|--------|---------------|-------------|
| `init_prompt` | Iteration 1 | State desc + task + code example | State desc + building blocks + market stats + code example |
| `cot_prompt` | After each training batch | Code + Lipschitz per dim | Code + IC per dim + market analysis |
| `next_iteration_prompt` | Iteration 2+ | Full history + COT | Full history + IC suggestions + updated market stats |
| `reward_config_prompt` | Each iteration (separate) | N/A | JSON-based reward rule selection |

### 9.2 Init Prompt (first iteration)
```
Revise the state representation for a reinforcement learning agent.
=========================================================
Task: Dynamic portfolio allocation across 5 stocks (TSLA, NFLX, AMZN, MSFT, JNJ)
      plus a CASH asset (6 assets total). The goal is to maximize risk-adjusted returns.
=========================================================

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

You should design a task-related state representation based on the source 120 dim to better
for reinforcement training, using the detailed information mentioned above to do some calculations,
and feel free to do complex calculations, and then concat them to the source state.

Available computation functions (import from feature_library):

1. compute_relative_momentum(prices, window=20)
   Input: prices = 1D array of close prices (length >= window)
   Output: scalar, this stock's excess return vs equal-weight basket
   Use case: identify relatively outperforming stocks

2. compute_cross_sectional_rank(values)
   Input: values = list of scalars for all 5 stocks
   Output: scalar in [0, 1], this stock's rank
   Note: NOT available inside revise_state(s) — requires all stocks' data.
   This is computed at the portfolio level, not inside revise_state.

3. compute_realized_volatility(returns, window=20)
   Input: returns = 1D array of daily returns
   Output: scalar, annualized realized volatility
   Use case: measure individual stock risk

4. compute_downside_risk(returns, window=20)
   Input: returns = 1D array of daily returns
   Output: scalar, downside semi-deviation
   Use case: measure downside risk

5. compute_beta(returns, market_returns, window=20)
   Input: returns = 1D array, market_returns = 1D array (equal-weight portfolio returns)
   Output: scalar, regression beta
   Use case: systemic risk exposure
   Note: market_returns must be provided by the environment, not available inside revise_state(s)

6. compute_multi_horizon_momentum(prices, windows=[5, 10, 20])
   Input: prices = 1D array of close prices
   Output: array of 3 scalars, momentum at each horizon
   Use case: capture trend at multiple time scales

7. compute_zscore_price(prices, window=20)
   Input: prices = 1D array
   Output: scalar, z-score of current price vs N-day mean
   Use case: mean reversion signal

8. compute_mean_reversion_signal(prices, window=20)
   Input: prices = 1D array
   Output: scalar, strength of mean reversion
   Use case: identify overextended prices

9. compute_turnover_ratio(volumes, window=20)
   Input: volumes = 1D array
   Output: scalar, current volume / average volume
   Use case: liquidity detection

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
```
```

### 9.3 COT Prompt (after each training batch)
```
We have successfully trained Reinforcement Learning (RL) policy using {sample_count} different
state revision codes and intrinsic reward function codes sampled by you, and each pair of code
is associated with the training of a policy relatively.

Throughout every state revision code's training process, we monitored:
1. The final policy performance (accumulated reward).
2. Most importantly, every state revise dim's Information Coefficient (IC) with forward returns.
   The IC measures how predictive each state dimension is for future portfolio returns.
   Higher |IC| means the dimension is more useful for the RL agent's decision making.
3. Market environment context and regime-specific IC analysis.

Here are the results:
{s_feedback_for_all_samples}

【Market Environment During Training】
{market_period_summary}
  Overall trend: {trend_description}
  Volatility regime: {volatility_description}
  Correlation events: {correlation_events}
  Worst period: {worst_period}

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

Lets think step by step. Your solution should aim to improve the overall performance of the RL policy.
```

### 9.4 Next Iteration Prompt (subsequent iterations)
```
Revise the state representation for a reinforcement learning agent.
=========================================================
Task: Dynamic portfolio allocation across 5 stocks (TSLA, NFLX, AMZN, MSFT, JNJ) + CASH.
=========================================================

{state_description_same_as_init}

{building_blocks_list_same_as_init}

Updated Market Statistics:
{updated_market_stats}

For this problem, we have some history experience for you, here are some state revision codes
we have tried in the former iterations:
{former_history_with_all_cot_suggestions}

Based on the former suggestions. We are seeking an improved state revision code and an improved
intrinsic reward code that can enhance the model's performance on the task.

{intrinsic_reward_instructions_same_as_init}

Output format:
```python
import numpy as np
def revise_state(s):
    return updated_s
def intrinsic_reward(updated_s):
    return float_reward
```
```

### 9.5 Reward Config Prompt (separate, kept as JSON selection)
```
You are configuring the reward function for a PPO-based portfolio optimizer.

The base reward is Mean-Variance: r = portfolio_return - lambda * drawdown^2
Plus an intrinsic reward designed by another LLM code.

Your task: Select and parameterize additional reward rules to guide the agent.

Available Reward Rules:
{reward_rules_catalog}

Select 2-4 reward rules. Output JSON:
```json
{
  "reward_rules": [...],
  "lambda": 0.5,
  "rationale": "..."
}
```
```

## 10. File Changes Summary

### New Files (2)
| File | Purpose | Est. Lines |
|------|---------|-----------|
| `core/code_sandbox.py` | AST validation + safe code execution | ~150 |
| `core/ic_analyzer.py` | IC computation + tiered COT feedback | ~150 |

### Refactored Files (4)
| File | Changes | Est. Lines Changed |
|------|---------|-------------------|
| `core/feature_library.py` | Redesign for portfolio optimization; export standalone functions | ~300 |
| `core/portfolio_env.py` | Compressed raw + concatenation; add intrinsic_reward to step() | ~150 |
| `core/prompts.py` | JSON selection -> code generation; building blocks catalog | ~200 |
| `core/lesr_controller.py` | Multi-sample loop; IC feedback; code import management | ~200 |

### Unchanged Files (7)
- `core/ppo_agent.py` — PPO algorithm
- `core/regime_detector.py` — Market regime detection
- `core/portfolio_features.py` — Cross-stock features (integrate into new feature_library)
- `core/reward_rules.py` — Behavior constraint rules
- `core/metrics.py` — Performance metrics
- `core/market_stats.py` — Market statistics
- `core/prepare_data.py` — Data preparation

### Execution Order
```
1. feature_library.py    <- depends on nothing (foundation)
2. code_sandbox.py       <- depends on feature_library (registered functions)
3. ic_analyzer.py        <- standalone
4. prompts.py            <- depends on feature_library (building blocks list)
5. portfolio_env.py      <- depends on code_sandbox (import generated code)
6. lesr_controller.py    <- depends on all above
```

## 11. Alignment with Original LESR

| LESR Feature | Original | This Design |
|-------------|----------|-------------|
| LLM generates code | Full Python | Python with registered functions only |
| revise_state | Concatenate to original | Compressed original + new features |
| intrinsic_reward | LLM-designed | LLM-designed (same) |
| Feedback signal | Lipschitz constant | IC (domain-appropriate) |
| Multi-sample | N codes, N policies | N codes, N policies (same) |
| COT feedback | Code + Lipschitz per dim | Code + IC per dim (strong/weak/negative/missing) |
| RL algorithm | TD3 | PPO with Dirichlet (domain-appropriate) |
| Training dispatch | tmux windows | Direct Python (simpler) |
