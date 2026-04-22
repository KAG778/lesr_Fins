"""
Prompt Templates for Exp4.9: Regime-Conditioned LESR

Key difference from Exp4.7:
- revise_state now receives regime_vector (5d) as second argument
- intrinsic_reward must branch on regime
- COT feedback is grouped by market regime
"""

import numpy as np
from typing import List, Dict


# ==============================================================================
# Initial Prompt Template (Regime-Aware)
# ==============================================================================

INITIAL_PROMPT = """
You are a financial quantitative analysis expert, specializing in extracting trading signals from price and volume data ADAPTED TO DIFFERENT MARKET REGIMES.

## Task Background

We are training a stock timing strategy using reinforcement learning. The strategy makes decisions at each trading day:
- **BUY**: Establish a long position
- **SELL**: Close position
- **HOLD**: Maintain current position

## Available Data

The raw state is a 120-dimensional NumPy array `s`:
- `s[0:19]`: 20 days of closing prices
- `s[20:39]`: 20 days of opening prices
- `s[40:59]`: 20 days of high prices
- `s[60:79]`: 20 days of low prices
- `s[80:99]`: 20 days of trading volume
- `s[100:119]`: 20 days of adjusted closing prices

## Market Regime Information (NEW)

Your functions also receive a `regime_vector` (5-dimensional NumPy array):
- `regime_vector[0]`: **trend_strength** [-1, 1]
  - +1: strong uptrend (short MA >> long MA)
  - -1: strong downtrend
  - 0: no clear trend (sideways)
- `regime_vector[1]`: **volatility_regime** [0, 1]
  - 0: low volatility
  - 1: extreme high volatility
- `regime_vector[2]`: **momentum_signal** [-1, 1]
  - +1: strong upward momentum
  - -1: strong downward momentum
- `regime_vector[3]`: **meanrev_signal** [-1, 1]
  - +1: price at upper Bollinger (likely to fall back)
  - -1: price at lower Bollinger (likely to bounce)
- `regime_vector[4]`: **crisis_signal** [0, 1]
  - 0: normal market
  - 1: crisis in progress (extreme drawdown + volume panic)

## CRITICAL: Regime-Conditioned Design

Your features and rewards MUST adapt to the current market regime:

### Feature Design by Regime:
- **Strong Trend** (|trend_strength| > 0.3): Focus on trend-following features (MA crossover distance, ADX, trend consistency)
- **Sideways** (|trend_strength| < 0.15): Focus on mean-reversion features (Bollinger %B, RSI neutral zone, range width)
- **High Volatility** (volatility_regime > 0.7): Include ATR-based scaling, volatility breakout signals
- **Crisis** (crisis_signal > 0.5): Include drawdown rate, max consecutive losses, defensive indicators

### Reward Design by Regime:
- **TREND_UP + momentum aligned**: POSITIVE reward (trend is your friend)
- **TREND_DOWN**: CAUTIOUS reward (reduce entry signals)
- **SIDEWAYS + meanrev opportunity**: MILD POSITIVE for counter-trend entries
- **HIGH VOLATILITY**: NEGATIVE for aggressive entries, reduce reward magnitude
- **CRISIS (signal > 0.5)**: STRONG NEGATIVE for any BUY signal (stay out!)

## Objective Function

Strategy performance is measured by:
1. **Sharpe Ratio**: Risk-adjusted return (maximize)
2. **Maximum Drawdown**: Maximum loss magnitude (minimize, keep under 30%)
3. **Total Return**: Cumulative return (maximize)

## Constraints

1. Transaction cost: 0.1% commission per trade
2. Position limit: Maximum 100% position in single stock
3. Risk limit: Maximum 5% daily loss

## Your Task

Please generate two Python functions:

### Function 1: `revise_state(raw_state, regime_vector)`
- **Input 1**: Raw state (120-dimensional NumPy array)
- **Input 2**: Regime vector (5-dimensional NumPy array)
- **Output**: Enhanced state (original 120 dims + 5 regime dims + new features)
- **Requirements**:
  - You MUST use regime_vector to decide which features to compute
  - Different regimes → different useful features
  - Always include at least 3 new features beyond the raw + regime dims
  - Handle edge cases: division by zero, missing values

Example structure:
```python
def revise_state(s, regime_vector):
    trend_strength = regime_vector[0]
    volatility_regime = regime_vector[1]
    momentum_signal = regime_vector[2]
    meanrev_signal = regime_vector[3]
    crisis_signal = regime_vector[4]
    
    enhanced = np.concatenate([s, regime_vector])  # 125 dims base
    
    # Compute features based on regime...
    new_features = []
    
    if abs(trend_strength) > 0.3:
        # Trend-following features
        ...
    else:
        # Mean-reversion features
        ...
    
    return np.concatenate([enhanced, new_features])
```

### Function 2: `intrinsic_reward(enhanced_state)`
- **Input**: Enhanced state (output from revise_state)
- **Output**: Intrinsic reward value (range: [-100, 100])
- **Requirements**:
  - regime_vector is embedded at indices [120:125] of enhanced_state
  - MUST use regime_vector to condition your reward logic
  - Different regimes → different reward criteria

Example structure:
```python
def intrinsic_reward(enhanced_s):
    regime_vector = enhanced_s[120:125]
    trend_strength = regime_vector[0]
    crisis_signal = regime_vector[4]
    
    reward = 0.0
    
    # CRITICAL: Crisis override
    if crisis_signal > 0.5:
        return -50.0  # Strong negative in crisis
    
    # Different reward logic per regime...
    ...
    
    return reward
```

**Important: Use Relative Thresholds for Volatility-Adaptive Rewards**

Different stocks have different volatility levels:
- TSLA: ~4% daily std
- MSFT: ~2% daily std

Use historical volatility (computed from prices) as thresholds, not hard-coded values.

## Output Format

Please return complete, executable Python code:

```python
import numpy as np

def revise_state(s, regime_vector):
    # Your implementation
    return enhanced_s

def intrinsic_reward(enhanced_s):
    # Your implementation
    return reward
```

Let's think step by step.
"""


# ==============================================================================
# COT Feedback Prompt (Regime-Grouped)
# ==============================================================================

def get_financial_cot_prompt(
    codes: List[str],
    scores: List[Dict],
    importance: List[np.ndarray],
    correlations: List[np.ndarray],
    original_dim: int,
    regime_metrics: List[Dict] = None
) -> str:
    """
    Generate Chain-of-Thought feedback prompt with regime-grouped analysis.

    Args:
        codes: LLM-generated code snippets
        scores: Performance metrics for each code
        importance: Feature importance for each code
        correlations: Feature correlations for each code
        original_dim: Original state dimension (120)
        regime_metrics: Per-regime performance breakdown
    """
    s_feedback = ''

    for i, (code, score) in enumerate(zip(codes, scores)):
        s_feedback += f'========== Code Candidate -- {i + 1} ==========\n'
        s_feedback += code + '\n'
        s_feedback += f'Overall Performance Metrics:\n'
        s_feedback += f'  Sharpe Ratio: {score["sharpe"]:.3f}\n'
        s_feedback += f'  Max Drawdown: {score["max_dd"]:.2f}%\n'
        s_feedback += f'  Total Return: {score["total_return"]:.2f}%\n'

        # Regime-grouped performance (NEW)
        if regime_metrics and i < len(regime_metrics):
            rm = regime_metrics[i]
            s_feedback += f'\nPerformance by Market Regime:\n'
            for regime_name, metrics in rm.items():
                if isinstance(metrics, dict):
                    s_feedback += f'  {regime_name}: Sharpe={metrics.get("sharpe", 0):.2f}, Trades={metrics.get("trades", 0)}\n'

        # Original feature analysis
        s_feedback += f'\nOriginal Feature Importance (OHLCV):\n'
        for idx in range(min(5, original_dim)):
            if i < len(importance) and idx < len(importance[i]):
                s_feedback += f'  s[{idx}]: importance={importance[i][idx]:.3f}\n'

        # New feature analysis
        if i < len(importance) and len(importance[i]) > original_dim + 5:
            extra_dim = len(importance[i]) - original_dim - 5  # Skip 5 regime dims
            if extra_dim > 0:
                s_feedback += f'\nNew Feature Importance (Top 3):\n'
                new_start = original_dim + 5
                new_importance = importance[i][new_start:]
                top_extra = np.argsort(new_importance)[-3:][::-1]
                for rank, idx in enumerate(top_extra, 1):
                    actual_idx = new_start + idx
                    corr_val = correlations[i][actual_idx] if i < len(correlations) and actual_idx < len(correlations[i]) else 0
                    s_feedback += f'  new_feature_{idx}: importance={importance[i][actual_idx]:.3f}, corr={corr_val:.3f}\n'

        s_feedback += '\n'

    cot_prompt = f"""
We trained DQN policies using {len(codes)} different regime-conditioned state representations and intrinsic reward functions.

Training Results:
{s_feedback}

Please analyze the results and provide improvement suggestions:

Analysis Points:
(a) Which regimes does each code handle well? Which regimes does it fail in?
(b) Which features are important in TREND vs SIDEWAYS vs CRISIS regimes?
(c) How can the intrinsic_reward better adapt to different regimes?
(d) Are there regime transitions the strategy misses?

Regime-Specific Design Guidance:
- **TREND regimes** (|trend_strength| > 0.3): Reward momentum-aligned entries, penalize counter-trend
- **SIDEWAYS regimes** (|trend_strength| < 0.15): Reward mean-reversion signals, penalize breakout chases
- **HIGH_VOL regimes** (volatility_regime > 0.7): Scale down all rewards, discourage aggressive positions
- **CRISIS regimes** (crisis_signal > 0.5): Strong negative reward for BUY signals, mild positive for SELL/HOLD

Goal: Improve the strategy's Sharpe ratio across ALL regimes while keeping maximum drawdown under 30%.
"""

    return cot_prompt


# ==============================================================================
# Iteration Prompt (Regime-Aware)
# ==============================================================================

def get_iteration_prompt(
    all_iter_codes: List[List[str]],
    all_iter_cot_suggestions: List[str]
) -> str:
    """
    Generate iteration prompt with historical context including regime analysis.
    """
    former_history = ''

    for i in range(len(all_iter_codes)):
        former_history += f'\n\n\nFormer Iteration: {i + 1}\n'
        for j, code in enumerate(all_iter_codes[i]):
            former_history += f'Candidate {j + 1}:\n{code}\n'
        if i < len(all_iter_cot_suggestions):
            former_history += f'\nSuggestions:\n{all_iter_cot_suggestions[i]}\n'

    iteration_prompt = f"""
You are a financial quantitative analysis expert specializing in REGIME-ADAPTIVE trading strategies.

We have completed multiple iterations of optimization. Here is the historical experience:

{former_history}

Based on the above experience and suggestions, please generate IMPROVED regime-conditioned state representation and intrinsic reward functions.

Requirements:
1. Function signature: `revise_state(s, regime_vector)` — MUST accept regime_vector as second argument
2. regime_vector is at enhanced_state[120:125] — use it in intrinsic_reward
3. Different regimes need different features and reward logic
4. Avoid repeating features that were proven ineffective
5. intrinsic_reward must be in the range [-100, 100]
6. **CRITICAL**: intrinsic_reward MUST branch on regime_vector[4] (crisis_signal)
   - crisis_signal > 0.5 → strong negative reward for entries
7. Use volatility-adaptive thresholds (historical std based)

Please return complete Python code:

```python
import numpy as np

def revise_state(s, regime_vector):
    # Must use regime_vector to condition features
    enhanced = np.concatenate([s, regime_vector])  # base 125 dims
    # Add regime-conditioned features...
    return enhanced_s

def intrinsic_reward(enhanced_s):
    # Extract regime_vector from enhanced_state[120:125]
    regime_vector = enhanced_s[120:125]
    # Branch on regime...
    return reward
```
"""

    return iteration_prompt


# ==============================================================================
# Validation Prompt
# ==============================================================================

VALIDATION_PROMPT = """
Please review the following Python code for a regime-adaptive trading strategy:

1. Does `revise_state` accept TWO arguments: (raw_state, regime_vector)?
2. Does `revise_state` include regime_vector in the output (at indices [120:125])?
3. Does `intrinsic_reward` extract regime_vector from enhanced_state[120:125]?
4. Does `intrinsic_reward` branch on crisis_signal (regime_vector[4])?
5. Does `intrinsic_reward` return a value in the range [-100, 100]?
6. Are there any obvious bugs (division by zero, missing imports, etc.)?

Code to review:
{code}

Please respond with:
- VALID: if the code passes all checks
- INVALID: if any issues found, along with specific problems
"""
