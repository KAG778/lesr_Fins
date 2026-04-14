"""
Prompts for Exp4.9_c: Regime-Aware LESR

Key design:
  revise_state(s) → returns ONLY new features (framework prepends raw+regime)
  intrinsic_reward(enhanced_state) → reads regime at [120:123], priority-based logic
"""

import numpy as np
from typing import List, Dict


INITIAL_PROMPT = """
You are a financial quantitative analysis expert, specializing in extracting trading signals from stock market data.

## Task Background

We are training a stock timing strategy using reinforcement learning (DQN). The strategy makes daily decisions:
- **BUY** (action 0): Establish a long position
- **SELL** (action 1): Close position
- **HOLD** (action 2): Maintain current position

## Available Data

The raw state is a 120-dimensional NumPy array `s` with 20 trading days of interleaved OHLCV:
- `s[i*6 + 0]`: closing price of day i (i=0..19, day 19 is most recent)
- `s[i*6 + 1]`: opening price
- `s[i*6 + 2]`: high price
- `s[i*6 + 3]`: low price
- `s[i*6 + 4]`: trading volume
- `s[i*6 + 5]`: adjusted closing price

## Market Regime Information (pre-computed, injected by framework)

Your `intrinsic_reward` function receives an `enhanced_state` where:
- `enhanced_state[0:120]` = raw state (same as `s`)
- `enhanced_state[120:123]` = regime_vector (3-dimensional):
  - `[120]` **trend_direction** [-1, +1]:
    - >+0.3: uptrend (momentum strategies favored)
    - -0.3 to +0.3: sideways (mean-reversion strategies favored)
    - <-0.3: downtrend (cautious strategies)
  - `[121]` **volatility_level** [0, 1]:
    - <0.3: calm market
    - 0.3-0.7: normal
    - >0.7: extreme volatility (reduce signal strength)
  - `[122]` **risk_level** [0, 1]:
    - <0.3: safe
    - 0.3-0.7: elevated risk (recent pullback)
    - >0.7: dangerous (recent >10% drop, strongly consider exit)
- `enhanced_state[123:]` = features you compute in revise_state

## CRITICAL: Reward Logic by Priority

Your `intrinsic_reward` MUST implement this priority chain:

**Priority 1 — RISK MANAGEMENT** (check FIRST):
  if risk_level (enhanced_state[122]) > 0.7:
    → Return STRONG NEGATIVE reward (-30 to -50) for BUY-aligned features
    → Return MILD POSITIVE reward (+5 to +10) for SELL-aligned features
  if risk_level > 0.4:
    → Return moderate negative reward for BUY signals

**Priority 2 — TREND FOLLOWING** (when risk is low):
  if |trend_direction| > 0.3 and risk_level < 0.4:
    → trend > 0.3 + upward features → positive reward
    → trend < -0.3 + downward features → positive reward (correct bearish bet)

**Priority 3 — SIDEWAYS / MEAN REVERSION**:
  if |trend_direction| < 0.3 and risk_level < 0.3:
    → Reward mean-reversion features (oversold→buy, overbought→sell)
    → Penalize breakout-chasing features

**Priority 4 — HIGH VOLATILITY** (no crisis):
  if volatility_level > 0.6 and risk_level < 0.4:
    → Reduce reward magnitude by 50% (uncertain market)

## Objective

Strategy performance is measured by:
1. **Sharpe Ratio** (maximize)
2. **Maximum Drawdown** (minimize, keep under 30%)
3. **Total Return** (maximize)

## Constraints
- Transaction cost: 0.1% per trade
- Use **relative thresholds** based on historical volatility, NOT hard-coded values

## ABSOLUTE RULES (code will be rejected if violated)

1. **NO RANDOMNESS**: `intrinsic_reward` MUST be a pure deterministic function. Do NOT use `np.random.uniform`, `np.random.normal`, `random.uniform`, `random.random`, or any other random function anywhere in the code. The reward for the same input must ALWAYS return the same value.
2. **REGIME SENSITIVITY**: `intrinsic_reward` MUST return different values for different regime inputs. If risk_level=0.9 vs risk_level=0.1, the reward MUST differ.
3. **SCALAR OUTPUT**: `intrinsic_reward` must return a single float, not an array.

## Your Task

Generate two Python functions:

### Function 1: `revise_state(raw_state)`
- **Input**: 120-dimensional raw state
- **Output**: 1D numpy array of ONLY your new computed features
  - Do NOT include the original 120d raw state
  - Do NOT include regime_vector
  - Return ONLY the features you compute
- **Requirements**:
  - Compute at least 3 features
  - Handle edge cases (division by zero, missing values)
  - Features should help DQN distinguish good entry/exit points

### Function 2: `intrinsic_reward(enhanced_state)`
- **Input**: enhanced_state (raw[0:120] + regime[120:123] + features[123:])
- **Output**: reward value in [-100, 100]
- **Requirements**:
  - Read regime_vector at indices [120:123]
  - Implement the priority chain described above
  - Return float value
  - **MUST be deterministic**: no random calls, same input always produces same output
  - **MUST be regime-sensitive**: different regimes must produce different rewards

Output format:
```python
import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    ...
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management (deterministic, NO random calls)
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if len(features) > 0:
            reward += trend_direction * features[0] * 10.0

    # Priority 3: Sideways
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5.0  # mild positive for mean-reversion

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5

    return float(np.clip(reward, -100, 100))
```

IMPORTANT: The above is a MINIMAL example. Your actual implementation should be more sophisticated, using your computed features to inform the reward. The key point is: use fixed multipliers and feature values, NEVER random numbers.

Let's think step by step.
"""


def get_financial_cot_prompt(
    codes: List[str],
    scores: List[Dict],
    importance: List[np.ndarray],
    correlations: List[np.ndarray],
    original_dim: int,
    worst_trades: List[Dict] = None,
) -> str:
    """Generate COT feedback with worst-trade analysis."""
    s_feedback = ''

    for i, (code, score) in enumerate(zip(codes, scores)):
        s_feedback += f'========== Code Candidate -- {i + 1} ==========\n'
        s_feedback += code + '\n'
        s_feedback += f'Performance:\n'
        s_feedback += f'  Sharpe Ratio: {score["sharpe"]:.3f}\n'
        s_feedback += f'  Max Drawdown: {score["max_dd"]:.2f}%\n'
        s_feedback += f'  Total Return: {score["total_return"]:.2f}%\n'

        # Feature analysis
        extra_start = original_dim + 3  # raw(120) + regime(3)
        if i < len(importance) and len(importance[i]) > extra_start:
            extra_imp = importance[i][extra_start:]
            top_3 = np.argsort(extra_imp)[-min(3, len(extra_imp)):][::-1]
            s_feedback += f'\nTop features (importance):\n'
            for idx in top_3:
                s_feedback += f'  feature_{idx}: {extra_imp[idx]:.3f}\n'

        # Worst trades analysis (NEW)
        if worst_trades and i < len(worst_trades) and worst_trades[i]:
            s_feedback += f'\nWorst trades (biggest losses):\n'
            for wt in worst_trades[i][:3]:
                s_feedback += f'  Day {wt.get("day","?")}: {wt.get("action","?")} → {wt.get("return",0)*100:.1f}%, '
                s_feedback += f'regime=[trend={wt.get("trend",0):.2f}, vol={wt.get("vol",0):.2f}, risk={wt.get("risk",0):.2f}]\n'

        s_feedback += '\n'

    cot_prompt = f"""
Training results for {len(codes)} regime-aware strategies:

{"".join(s_feedback)}

Please analyze and improve:

1. Do the features properly adapt to different regimes (trend/sideways/crisis)?
2. Does intrinsic_reward correctly implement the priority chain (risk > trend > sideways)?
3. Which features help avoid the worst trades?
4. Suggest specific new features that could prevent the largest losses.
5. CRITICAL: intrinsic_reward must be DETERMINISTIC. Use fixed multipliers (e.g. -40.0, +10.0), NOT random values. Random calls cause the code to be rejected.

Goal: Improve Sharpe ratio across ALL regimes, especially reduce crisis-period losses.
"""
    return cot_prompt


def get_iteration_prompt(
    all_iter_codes: List[List[str]],
    all_iter_cot_suggestions: List[str]
) -> str:
    """Generate iteration prompt with history."""
    history = ''
    for i in range(len(all_iter_codes)):
        history += f'\n\n=== Iteration {i + 1} ===\n'
        for j, code in enumerate(all_iter_codes[i]):
            history += f'Candidate {j + 1}:\n{code}\n'
        if i < len(all_iter_cot_suggestions):
            history += f'\nFeedback:\n{all_iter_cot_suggestions[i]}\n'

    return f"""
You are a financial quantitative analysis expert specializing in REGIME-AWARE trading strategies.

Previous iterations:
{history}

Based on this experience, generate IMPROVED functions.

Requirements:
1. `revise_state(s)` → return ONLY new features (1D numpy). Do NOT include raw state or regime.
2. `intrinsic_reward(enhanced_s)` → read regime at [120:123], implement priority chain:
   - Priority 1: risk_level > 0.7 → strong negative for BUY, mild positive for SELL
   - Priority 2: |trend| > 0.3 + low risk → reward momentum alignment
   - Priority 3: sideways → reward mean-reversion
   - Priority 4: high vol → reduce reward magnitude
3. Return value of intrinsic_reward must be in [-100, 100]
4. Use relative thresholds (historical std based), NOT hard-coded values
5. Avoid repeating ineffective features from previous iterations
6. **NO RANDOMNESS**: Do NOT use np.random, random.uniform, or any random function.
   Use fixed constants and feature values to compute reward deterministically.
   Code with random calls will be automatically rejected.

Output:
```python
import numpy as np

def revise_state(s):
    features = []
    ...
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    ...
    return reward
```
"""
