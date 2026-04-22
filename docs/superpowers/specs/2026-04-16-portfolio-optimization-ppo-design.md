# Portfolio Optimization PPO Design Spec

**Date:** 2026-04-16
**Source:** exp4.9_c (single-stock timing, DQN) → 组合优化_ppo (portfolio optimization, PPO)
**Status:** Approved

---

## 1. Overview

Transform the single-stock DQN timing system (exp4.9_c) into a 5-stock portfolio optimization system using PPO with continuous weight allocation. LLM plays a higher-level role: selecting features, portfolio-level indicators, market regime assessment, and reward shaping rules — all via JSON selection from predefined libraries.

### Core Differences from exp4.9_c

| Aspect | exp4.9_c (Source) | 组合优化_ppo (Target) |
|--------|-------------------|----------------------|
| RL Algorithm | DQN (discrete) | PPO (continuous) |
| Action Space | 3 actions (BUY/SELL/HOLD) | 6-dim weight vector → softmax |
| Stocks | 1 at a time (TSLA/NFLX/AMZN/MSFT) | 5 simultaneously (+ JNJ) |
| State Input | 120-dim single stock OHLCV | 600-dim (5 stocks × 120) + portfolio features |
| Reward | Single-stock price change + intrinsic | Mean-Variance portfolio + reward rules |
| LLM Role | Feature selection only | 4-layer: features + portfolio indicators + regime + reward rules |
| LLM Output | JSON (feature selection) | JSON (4 sections) |
| Transaction Cost | 0.1% per trade | 0.1% per traded dollar (turnover-based) |
| Rebalancing | N/A (single position) | Daily target weight rebalance |

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              组合优化_ppo System Architecture                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────────────┐                       │
│  │ Data     │───→│ PortfolioEnv     │                       │
│  │ Layer    │    │ (5 stocks OHLCV) │                       │
│  └──────────┘    └──────┬───────────┘                       │
│                         │ state, reward                      │
│  ┌──────────────┐      │                                    │
│  │ LLM Module   │──→ State Enhancement:                     │
│  │ - prompts.py │    - Per-stock features (5×N₁)            │
│  │ - feature_   │    - Portfolio features (N₂)              │
│  │   library.py │    - Market regime (3)                    │
│  │ - reward_    │    - Reward rule params                   │
│  │   rules.py   │    Called every N steps, cached otherwise │
│  └──────────────┘      │                                    │
│                        ↓                                     │
│  ┌─────────────────────────────────────┐                    │
│  │          PPO Agent                  │                    │
│  │  Actor: state → 6-dim → softmax → w│                    │
│  │  Critic: state → V(s)              │                    │
│  └─────────────────────────────────────┘                    │
│              │                                               │
│              ↓                                               │
│  ┌──────────────────┐  ┌────────────────┐                  │
│  │ Evaluation       │  │ LESR Loop      │                  │
│  │ - Portfolio Sharpe│  │ - Iterative    │                  │
│  │ - Sortino/MaxDD  │  │   optimization │                  │
│  │ - Per-stock contrib│ │ - COT feedback │                  │
│  └──────────────────┘  └────────────────┘                  │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Stock Universe

5 stocks covering different sectors and volatility profiles:

| Ticker | Sector | Daily Vol | Role in Portfolio |
|--------|--------|-----------|-------------------|
| TSLA | EV/Tech | ~4% | High-return, high-risk growth driver |
| NFLX | Streaming | ~2.5% | Medium-high vol, momentum candidate |
| AMZN | E-commerce/Cloud | ~2.2% | Large-cap tech, core holding |
| MSFT | Software | ~1.8% | Stable blue-chip, anchor stock |
| JNJ | Pharma | ~1.2% | Defensive hedge, low correlation |

Plus **cash position** (risk-free asset, weight = w_cash).

Data source: `/home/wangmeiyi/AuctionNet/lesr/data/all_sp500_prices_2000_2024_delisted_include.csv`

---

## 4. State Representation

### 4.1 State Assembly

```
Final state = concat(
    per_stock_raw:     5 × 120 = 600 dims   (each stock: 20 days × OHLCV)
    market_regime:     3 dims                [trend, volatility, risk]
    per_stock_features: 5 × N₁ dims          (LLM-selected indicators per stock)
    portfolio_features: N₂ dims              (cross-stock indicators)
    current_weights:   6 dims                [w_TSLA, w_NFLX, w_AMZN, w_MSFT, w_JNJ, w_cash]
)
```

Total state dim = 600 + 3 + 5×N₁ + N₂ + 6 = 609 + 5×N₁ + N₂

### 4.2 Raw Data Layout (per stock, unchanged from exp4.9_c)

```
s[i*6 + 0] = close
s[i*6 + 1] = open
s[i*6 + 2] = high
s[i*6 + 3] = low
s[i*6 + 4] = volume
s[i*6 + 5] = adjusted_close
for i = 0..19 (20 trading days)
```

### 4.3 Market Regime Detection

Adapted from exp4.15's `regime_detector.py`. Instead of single-stock regime, compute from **equal-weight portfolio** of all 5 stocks:

- **trend_direction** [-1, +1]: MA(5) vs MA(20) of equal-weight portfolio price
- **volatility_level** [0, 1]: Portfolio return volatility, z-scored
- **risk_level** [0, 1]: Portfolio max drawdown in recent 10 days

### 4.4 Per-Stock Features

Reuse exp4.15's 20 indicators from `INDICATOR_REGISTRY`, applied to each stock independently. LLM selects 5-10 indicators via JSON; same indicators computed for all 5 stocks.

### 4.5 Portfolio-Level Features (New)

8 cross-stock indicators in a new `PORTFOLIO_INDICATOR_REGISTRY`.

**Input format**: Portfolio indicator functions receive a dict of 5 raw states:
```python
# Unpacking the 600-dim raw state into per-stock 120-dim states
raw_states = {
    'TSLA': state[0:120],
    'NFLX': state[120:240],
    'AMZN': state[240:360],
    'MSFT': state[360:480],
    'JNJ':  state[480:600],
}
# Each function also receives current_weights (6-dim) for sector_exposure
```

| Name | Output Dims | Params | Description |
|------|-------------|--------|-------------|
| `momentum_rank` | 5 | window: 10-60 | Rank each stock by past-N return (1=best, 5=worst) |
| `rolling_correlation` | 10 | window: 20-120 | All pairwise rolling correlations (5 choose 2 = 10) |
| `relative_strength` | 5 | window: 10-60 | Each stock's return vs equal-weight basket return |
| `portfolio_volatility` | 1 | window: 10-60 | Rolling std of equal-weight portfolio returns |
| `return_dispersion` | 1 | window: 10-60 | Cross-sectional std of individual stock returns |
| `sector_exposure` | 2 | (none) | [growth_weight, defensive_weight] from current portfolio weights |
| `volume_breadth` | 1 | window: 5-20 | Fraction of stocks with above-average volume |
| `mean_reversion_score` | 5 | window: 10-60 | Z-score of each stock's current price vs N-day mean |

### 4.6 Current Weights

The agent's current portfolio allocation [w_TSLA, w_NFLX, w_AMZN, w_MSFT, w_JNJ, w_cash], included so the agent can learn to minimize unnecessary turnover.

---

## 5. Action Space

### 5.1 PPO Actor Output

Actor network outputs a 6-dimensional real vector, passed through softmax to produce valid weights:

```python
raw_output = actor_network(state)         # shape: (6,)
weights = softmax(raw_output)             # all >= 0, sum = 1
# weights = [w_TSLA, w_NFLX, w_AMZN, w_MSFT, w_JNJ, w_cash]
```

### 5.2 Execution

Each trading day:
1. Agent outputs target weights
2. Compute turnover = Σ|w_new - w_old| / 2
3. Rebalance portfolio to match target weights
4. Apply transaction cost = turnover × portfolio_value × 0.001

---

## 6. Reward Function

### 6.1 Base Reward (Mean-Variance)

```
base_reward = portfolio_return - λ × portfolio_volatility
```

Where:
- `portfolio_return` = Σ(wᵢ × rᵢ) for the 5 stocks (cash return = 0)
- `portfolio_volatility` = rolling 20-day std of portfolio returns
- `λ` = risk aversion parameter (suggested by LLM, default 0.5)

### 6.2 LLM Reward Rules

LLM selects 2-5 rules from a predefined `REWARD_RULE_REGISTRY`. Each rule adds a bonus or penalty:

| Rule | Params | Description |
|------|--------|-------------|
| `penalize_concentration` | max_weight: 0.2-0.5, penalty: 0.01-0.2 | Penalty when any weight > max_weight |
| `reward_diversification` | min_stocks: 2-5, bonus: 0.01-0.1 | Bonus when ≥ min_stocks held above 5% |
| `penalize_turnover` | threshold: 0.05-0.3, penalty: 0.01-0.2 | Penalty when daily turnover > threshold |
| `regime_defensive` | crisis_threshold: 0.5-0.8, cash_bonus: 0.01-0.15 | Bonus for high cash when risk is high |
| `momentum_alignment` | bonus: 0.01-0.1 | Bonus when weights correlate with momentum_rank |
| `volatility_scaling` | vol_threshold: 0.3-0.8, scale: 0.3-0.8 | Scale down all rewards when vol > threshold |
| `drawdown_penalty` | dd_threshold: 0.05-0.2, penalty: 0.05-0.3 | Penalty when portfolio drawdown exceeds threshold |

### 6.3 Transaction Cost

```
tx_cost = turnover × 0.001
```

Deducted from reward directly.

### 6.4 Total Reward

```
total_reward = base_reward + Σ(reward_rule_bonuses) - tx_cost
```

---

## 7. LLM Integration

### 7.1 LLM Output Format (4-Layer JSON)

```json
{
  "single_stock_features": [
    {"indicator": "RSI", "params": {"window": 14}},
    {"indicator": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}}
  ],
  "portfolio_features": [
    {"indicator": "momentum_rank", "params": {"window": 20}},
    {"indicator": "rolling_correlation", "params": {"window": 60}}
  ],
  "market_assessment": {
    "regime": "bull",
    "risk_level": 0.3,
    "suggested_lambda": 0.5,
    "rationale": "Moderate uptrend with low risk; balanced approach"
  },
  "reward_rules": [
    {"rule": "penalize_concentration", "params": {"max_weight": 0.35, "penalty": 0.1}},
    {"rule": "penalize_turnover", "params": {"threshold": 0.1, "penalty": 0.15}}
  ],
  "rationale": "Overall explanation..."
}
```

### 7.2 Prompt Design Principles

1. **Complete transparency**: LLM understands the full system pipeline (PPO, reward, execution)
2. **Stock context**: Ticker names, sectors, volatility profiles with behavioral interpretation
3. **Pre-computed statistics with interpretation**: Every number comes with a "what this means" line
4. **Predefined libraries**: Only valid names from registries, no free-form code

### 7.3 Market Statistics Injection (`get_market_stats()`)

Pre-computed from training data and injected into prompt with interpretation:

```
### Per-Stock Profile
| Ticker | Daily Vol | 20d Return | Interpretation                          |
|--------|-----------|------------|-----------------------------------------|
| TSLA   | 3.82%     | +5.2%      | High vol, strong momentum               |
|        |           |            | → Good for trend-following, but risky   |
|        |           |            | → Consider concentration limits          |
[... similar for NFLX, AMZN, MSFT, JNJ ...]

### Correlation Matrix
[full 5×5 matrix with average and interpretation]

### Key Insights (auto-generated)
- "JNJ has lowest correlation with tech stocks (0.22-0.35) → Most valuable for diversification"
- "AMZN-MSFT highest pair (0.67) → Limited diversification between them"

### Regime Summary
Trend: +0.35 → Mild uptrend → Consider momentum-aligned indicators
Volatility: 0.42 → Moderate → Standard risk management applies
Risk: 0.18 → Low → Safe to maintain equity exposure
```

### 7.4 Initial Prompt Structure

```
1. Role description
2. Task background (PPO portfolio optimization, weight output, rebalancing)
3. Stock universe (names, sectors, vol profiles)
4. System pipeline explanation (state → PPO → weights → reward)
5. Available indicators
   a. Per-stock indicators (20, same as exp4.15)
   b. Portfolio-level indicators (8 new)
   c. Reward rules (7 new)
6. Market statistics (pre-computed with interpretation)
7. Output format (4-layer JSON)
8. Selection rules (counts, diversification, parameter ranges)
```

### 7.5 COT Feedback Prompt Structure

```
1. Per-candidate results:
   a. Selected features + portfolio indicators + reward rules + regime assessment
   b. Portfolio performance (Sharpe, MaxDD, Total Return, Avg Turnover)
   c. Per-stock contribution (return, avg weight, Sharpe contribution, interpretation)
   d. Reward rule activity (trigger count, avg penalty, which stocks triggered it, interpretation)
   e. Per-indicator IC values (with "useful/marginal/useless" label)
   f. Rejected indicators with reasons

2. Summary:
   a. Best candidate and Sharpe range
   b. Negative guidance (what to avoid)
   c. Specific improvement suggestions based on data patterns

3. Interpretation examples:
   - "TSLA consumed 28% of capital for 40% of returns → High efficiency but concentration risk"
   - "penalize_turnover triggered 48% of days → Agent rebalances too aggressively"
   - "rolling_correlation IC=0.08 → Useful, keep; RSI IC=0.01 → Remove"
```

### 7.6 Iteration Prompt

Curated context (~2k tokens, same as exp4.15's D-02):
- Last iteration's selection + feedback
- Best historical selection + score
- Full indicator/rule registry (same as initial)

### 7.7 Validation Pipeline

Extended from exp4.15's 6-stage pipeline:
1. Parse JSON (reuse `_extract_json`)
2. Structure check (must have `single_stock_features`, `portfolio_features`, `reward_rules`)
3. Per-indicator validation against `INDICATOR_REGISTRY` and `PORTFOLIO_INDICATOR_REGISTRY`
4. Per-rule validation against `REWARD_RULE_REGISTRY`
5. Param clipping to registered ranges
6. Closure build + live test on sample data (NaN/Inf guard)

### 7.8 Screening and Stability

- Feature screening: IC vs forward portfolio returns, variance check, dedup (reuse exp4.15 logic)
- Stability assessment: sub-period IC analysis (reuse exp4.15 logic)
- Rule screening: track trigger frequency; rules never triggered get flagged in COT

---

## 8. PPO Agent Design

### 8.1 Network Architecture

```python
class PPOActor(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 6)  # 5 stocks + cash
        )
    
    def forward(self, x):
        return F.softmax(self.network(x), dim=-1)  # valid weight vector

class PPOCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
```

### 8.2 PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| clip_epsilon | 0.2 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| actor_lr | 3e-4 |
| critic_lr | 1e-3 |
| epochs_per_update | 10 |
| batch_size | 64 |
| rollout_length | 252 (1 trading year) |
| entropy_coef | 0.01 |

### 8.3 Training Loop

```
for iteration in range(max_iterations):   # typically 5
    1. Render prompt (initial or iteration with COT feedback)
    2. Sample N candidates from LLM (3 per round)
       - Parse JSON → validate → build closures
    3. Train PPO for each valid candidate
       - 50 episodes over training period
       - Evaluate on validation period
    4. Generate COT feedback
       - Portfolio metrics + per-stock contribution
       - Per-indicator IC + reward rule activity
       - Negative guidance + improvement suggestions
    5. Save iteration results
Select best strategy by portfolio Sharpe ratio
```

---

## 9. Evaluation Metrics

### 9.1 Portfolio-Level Metrics (reuse metrics.py)

- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio
- Total Return
- Average Turnover

### 9.2 Per-Stock Metrics (new)

- Individual contribution to portfolio return
- Average weight allocation
- Sharpe contribution (return contribution / risk contribution)

### 9.3 Reward Rule Metrics (new)

- Trigger frequency per rule
- Average penalty/bonus per trigger
- Which stocks most often trigger concentration penalties

### 9.4 Factor Metrics (adapted)

- Per-indicator IC vs forward portfolio returns
- Per-indicator IC vs individual stock returns
- Stability across sub-periods

---

## 10. File Structure

```
组合优化_ppo/
├── core/
│   ├── __init__.py
│   ├── ppo_agent.py          # PPO Actor-Critic + training (NEW)
│   ├── portfolio_env.py      # Multi-stock environment (NEW)
│   ├── feature_library.py    # Per-stock indicators (reuse from exp4.15 + extend)
│   ├── portfolio_features.py # Portfolio-level indicators (NEW)
│   ├── reward_rules.py       # Reward rule registry + computation (NEW)
│   ├── regime_detector.py    # Market-level regime (adapt from exp4.15)
│   ├── prompts.py            # LLM prompts (NEW, based on exp4.15 structure)
│   ├── lesr_controller.py    # LESR optimization loop (adapt from exp4.15)
│   ├── lesr_strategy.py      # Backtest deployment (adapt from exp4.15)
│   ├── metrics.py            # Financial metrics (reuse from exp4.15)
│   ├── prepare_data.py       # Data preparation (adapt from exp4.15)
│   └── market_stats.py       # Pre-computed stats for prompts (NEW)
├── configs/
│   └── config.yaml           # Experiment configuration
├── scripts/
│   ├── main.py               # Entry point
│   └── run_windows.sh        # Sliding window runner
├── results/                  # Per-window iteration results
├── tests/                    # Unit tests
├── api_keys_template.py
└── requirements.txt
```

---

## 11. Modules Reused from exp4.15

| Module | Action | Notes |
|--------|--------|-------|
| feature_library.py | **Reuse + extend** | Keep all 20 indicators, add portfolio registry |
| metrics.py | **Reuse as-is** | All financial metrics unchanged |
| prepare_data.py | **Reuse as-is** | Data format unchanged, just load more tickers |
| prompts.py `_extract_json()` | **Reuse as-is** | JSON parsing logic unchanged |
| prompts.py `get_cot_feedback()` | **Adapt** | Structure similar, content portfolio-specific |
| validate_selection | **Adapt** | Extend for 3 registries instead of 1 |
| screen_features | **Reuse logic** | IC computation same, target is portfolio returns |
| assess_stability | **Reuse as-is** | Sub-period IC analysis unchanged |

---

## 12. Key Design Decisions

1. **JSON mode (not code generation)**: Stability over flexibility, same as exp4.15's D-04
2. **Pre-computed stats with interpretation**: Every number in prompt has "what this means" context
3. **4-layer LLM output**: Features + portfolio indicators + regime + reward rules
4. **Predefined registries**: All indicators, portfolio features, and reward rules are curated and validated
5. **PPO with softmax**: Continuous weight output, naturally constrained to simplex
6. **Daily rebalancing with 0.1% cost**: Standard setup for portfolio optimization research
7. **Mean-Variance base reward**: Classic objective, augmented by LLM-selected rules
8. **Market-level regime**: Computed from equal-weight portfolio, not per-stock
9. **Curated iteration context**: ~2k tokens per iteration, same as exp4.15's D-02
10. **Leakage prevention**: Same filter_cot_metrics + check_prompt_for_leakage from exp4.15
