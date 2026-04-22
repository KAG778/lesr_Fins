"""
Prompt Templates for Exp4.7 Financial Trading Experiment

This module contains all prompt templates used for LLM interaction,
including initial prompt, iteration prompt, and COT feedback generation.
"""

import numpy as np
from typing import List, Dict


# ==============================================================================
# Initial Prompt Template
# ==============================================================================

INITIAL_PROMPT = """
You are a financial quantitative analysis expert, specializing in extracting trading signals from price and volume data.

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

## Financial Semantics

**Key Concepts:**

1. **Returns**: Percentage change in price
   - Daily return = (Today's close - Yesterday's close) / Yesterday's close

2. **Volatility**: Magnitude of price fluctuations
   - Calculation: Standard deviation of returns
   - Meaning: Measure of risk

3. **Trend**: Sustained directional movement of price
   - Uptrend: Short-term MA > Long-term MA
   - Downtrend: Short-term MA < Long-term MA
   - Sideways: Price oscillates within a range

4. **Momentum**: Speed and direction of price change
   - Positive momentum: Price accelerating upward
   - Negative momentum: Price accelerating downward

5. **Support/Resistance**: Price levels that are difficult to break through
   - Support: "Floor" where price tends to bounce
   - Resistance: "Ceiling" where price tends to reverse

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

### Function 1: `revise_state(raw_state)`
- **Input**: Raw state (120-dimensional NumPy array)
- **Output**: Enhanced state (original 120 dimensions + new features)
- **Feature Library** -- You MUST pick 3-5 indicators from AT LEAST 2 different categories below.
  Do NOT default to the same basic set (RSI + SMA + EMA + ATR). Aim for diverse, complementary signals.

  **Category 1 - Trend:**
    - MACD (12/26/9): EMA difference; signals trend direction changes
    - ADX (14): Average Directional Index; measures trend strength (not direction)
    - EMA crossover (5/20): Short vs long EMA; positive = uptrend
    - Price position vs SMA(20): How far price is above/below its mean

  **Category 2 - Overbought/Oversold:**
    - RSI (14): Relative Strength Index; >70 overbought, <30 oversold
    - Stochastic %K/%D (14/3/3): Compares close to recent high-low range
    - Williams %R (14): Similar to Stochastic, inverted scale (-100 to 0)
    - CCI (20): Commodity Channel Index; deviation from statistical mean

  **Category 3 - Volatility:**
    - ATR (14): Average True Range; absolute volatility measure
    - Bollinger Band width (20,2): Band width as volatility proxy
    - Historical volatility (20): Std of log returns
    - Keltner Channel: EMA-based envelope using ATR

  **Category 4 - Volume/Price:**
    - OBV: On-Balance Volume; cumulative volume flow
    - VWAP deviation: Price distance from volume-weighted average
    - Volume-price divergence: Price up but volume down (bearish signal)
    - Accumulation/Distribution line: Volume-weighted close position

  **Category 5 - Statistical:**
    - Z-score of price: (price - mean) / std; mean reversion signal
    - Hurst exponent (20): Measures long-term memory (trending vs mean-reverting)
    - Autocorrelation of returns (lag 1-5): Serial dependency in returns
    - Skewness of returns (20): Asymmetry of return distribution

  **DIVERSITY CONSTRAINT**: Do NOT combine correlated indicators from the same category
  (e.g., RSI + Stochastic, or SMA + EMA crossover). Instead, combine across categories.
  Good examples:
  - MACD (trend) + Stochastic (overbought/oversold) + Bollinger Band width (volatility) + OBV (volume)
  - ADX (trend) + RSI (overbought/oversold) + Z-score (statistical) + VWAP deviation (volume)

- **Multi-timeframe features**: Calculate indicators at 5-day, 10-day, and 20-day windows where applicable.
- **Handle edge cases**: Division by zero, missing values, insufficient data length.

### Function 2: `intrinsic_reward(enhanced_state)`
- **Input**: Enhanced state
- **Output**: Intrinsic reward value (range: [-100, 100])
- **Suggestions**:
  - Positive value: Current state is suitable for trading (clear trend, controlled risk)
  - Negative value: Current state is not suitable for trading (sideways, high risk)
  - **Must use at least one of the new feature dimensions**
  - **DETERMINISTIC REQUIREMENT**: Do NOT use np.random, random.uniform, or any random/stochastic
    function in intrinsic_reward. The reward for a given state must always produce the same value.
    Any code containing random functions will be automatically rejected.

**Important: Use Relative Thresholds for Volatility-Adaptive Rewards**

Different stocks have different volatility levels. For example:
- **TSLA**: High volatility (~4% daily std), frequent ±10% moves
- **MSFT**: Low volatility (~2% daily std), rare ±5% moves

**Bad Example** (hard-coded threshold):
```python
if recent_return < -5:  # Too harsh for high-volatility stocks!
    reward -= 50
```

**Good Example** (relative to historical volatility):
```python
# Calculate historical volatility from closing prices
returns = np.diff(closing_prices) / closing_prices[:-1] * 100
historical_vol = np.std(returns)  # e.g., TSLA~4%, MSFT~2%

# Use 2x historical volatility as threshold
threshold = 2 * historical_vol  # TSLA: ~8%, MSFT: ~4%
if recent_return < -threshold:
    reward -= 50
```

This way, the reward function automatically adapts to each stock's risk profile!

## Output Format

Please return complete, executable Python code:

```python
import numpy as np

def revise_state(s):
    # Your implementation
    return enhanced_s

def intrinsic_reward(enhanced_s):
    # Your implementation
    return reward
```

Let's think step by step.
"""


# ==============================================================================
# COT Feedback Prompt
# ==============================================================================

def get_financial_cot_prompt(
    codes: List[str],
    scores: List[Dict],
    importance: List[np.ndarray],
    correlations: List[np.ndarray],
    original_dim: int
) -> str:
    """
    Generate Chain-of-Thought feedback prompt for financial trading.

    Args:
        codes: LLM-generated code snippets
        scores: Performance metrics for each code
        importance: Feature importance for each code
        correlations: Feature correlations for each code
        original_dim: Original state dimension (120)

    Returns:
        COT feedback prompt string
    """
    s_feedback = ''

    for i, (code, score) in enumerate(zip(codes, scores)):
        s_feedback += f'========== Code Candidate -- {i + 1} ==========\n'
        s_feedback += code + '\n'
        s_feedback += f'Performance Metrics:\n'
        s_feedback += f'  Sharpe Ratio: {score["sharpe"]:.3f}\n'
        s_feedback += f'  Max Drawdown: {score["max_dd"]:.2f}%\n'
        s_feedback += f'  Total Return: {score["total_return"]:.2f}%\n'

        # Original feature analysis
        s_feedback += f'\nOriginal Feature Importance (OHLCV):\n'
        for idx in range(min(5, original_dim)):
            if i < len(importance) and idx < len(importance[i]):
                s_feedback += f'  s[{idx}]: importance={importance[i][idx]:.3f}\n'

        # New feature analysis
        if i < len(importance) and len(importance[i]) > original_dim:
            extra_dim = len(importance[i]) - original_dim
            if extra_dim > 0:
                s_feedback += f'\nNew Feature Importance (Top 3):\n'
                top_extra = np.argsort(importance[i][original_dim:])[-3:][::-1]
                for rank, idx in enumerate(top_extra, 1):
                    actual_idx = original_dim + idx
                    corr_val = correlations[i][actual_idx] if i < len(correlations) and actual_idx < len(correlations[i]) else 0
                    s_feedback += f'  new_feature_{idx}: importance={importance[i][actual_idx]:.3f}, corr={corr_val:.3f}\n'

        s_feedback += '\n'

    cot_prompt = f"""
We trained DQN policies using {len(codes)} different state representation and intrinsic reward function combinations.

Training Results:
{s_feedback}

Please analyze the results above and provide improvement suggestions:

Analysis Points:
(a) Why do certain codes perform better? Which features contribute most?
(b) What are the common problems of low-performance codes? (e.g., redundant features, overfitting, missing key signals)
(c) How to improve state representation and intrinsic reward design?

Financial Scene Special Notes:
- Trend features (momentum, moving averages) are important for timing
- Volatility features help with risk control
- Volume can confirm price trends
- intrinsic_reward should give positive values when trend is clear, negative when sideways
- **Use relative thresholds based on historical volatility** - different stocks have different risk profiles
- **intrinsic_reward must be fully deterministic** -- random functions (np.random, random.uniform, etc.) are not allowed

Feature Category Diversity Check:
Review all candidates above and check which feature categories are covered:
- Trend indicators used: [MACD / ADX / EMA crossover / Price vs SMA]
- Overbought/Oversold used: [RSI / Stochastic / Williams %R / CCI]
- Volatility used: [ATR / Bollinger Band width / Historical vol]
- Volume/Price used: [OBV / VWAP / Volume divergence / A/D line]
- Statistical used: [Z-score / Hurst / Autocorrelation / Skewness]

If all candidates use the same 1-2 categories, explicitly suggest trying indicators from
UNUSED categories. For example, if none use Volume/Price (OBV, VWAP) or Statistical (Z-score,
Hurst exponent), recommend these as promising unexplored directions.

Goal: Improve the strategy's Sharpe ratio while keeping maximum drawdown under 30%.
"""

    return cot_prompt


# ==============================================================================
# Iteration Prompt
# ==============================================================================

def get_iteration_prompt(
    all_iter_codes: List[List[str]],
    all_iter_cot_suggestions: List[str]
) -> str:
    """
    Generate iteration prompt with historical context.

    Args:
        all_iter_codes: Historical codes from previous iterations
        all_iter_cot_suggestions: Historical COT suggestions

    Returns:
        Iteration prompt string
    """
    former_history = ''

    for i in range(len(all_iter_codes)):
        former_history += f'\n\n\nFormer Iteration: {i + 1}\n'
        for j, code in enumerate(all_iter_codes[i]):
            former_history += f'Candidate {j + 1}:\n{code}\n'
        if i < len(all_iter_cot_suggestions):
            former_history += f'\nSuggestions:\n{all_iter_cot_suggestions[i]}\n'

    iteration_prompt = f"""
You are a financial quantitative analysis expert.

We have completed multiple iterations of optimization. Here is the historical experience:

{former_history}

Based on the above experience and suggestions, please generate improved state representation and intrinsic reward functions.

Requirements:
1. Avoid repeating features that have been proven ineffective
2. Preserve and improve effective features
3. Try new feature combinations
4. intrinsic_reward must be in the range [-100, 100]
5. Focus on features that show high correlation with returns in previous iterations
6. **Use features from DIFFERENT categories than previous iterations.** The feature library has 5 categories:
   Trend (MACD, ADX, EMA crossover), Overbought/Oversold (RSI, Stochastic, Williams %R, CCI),
   Volatility (ATR, Bollinger Band width, Historical vol), Volume/Price (OBV, VWAP, Accumulation/Distribution),
   Statistical (Z-score, Hurst exponent, Autocorrelation, Skewness).
   Check the historical code above -- if previous iterations used RSI+MACD+ATR, try indicators from UNUSED categories.
7. **Use volatility-adaptive thresholds in intrinsic_reward** - calculate historical volatility and use multiples of it (e.g., 2x std) instead of hard-coded values like -5%
8. **intrinsic_reward must be DETERMINISTIC** -- do NOT use np.random, random.uniform, or any stochastic function. The reward must always return the same value for the same state.

Please return complete Python code:

```python
import numpy as np

def revise_state(s):
    # Your implementation
    return enhanced_s

def intrinsic_reward(enhanced_s):
    # Your implementation
    return reward
```
"""

    return iteration_prompt


# ==============================================================================
# Validation Prompt (for checking generated code)
# ==============================================================================

VALIDATION_PROMPT = """
Please review the following Python code for a financial trading strategy:

1. Does `revise_state` return an array with at least 120 dimensions?
2. Does `intrinsic_reward` return a value in the range [-100, 100]?
3. Are there any obvious bugs (division by zero, missing imports, etc.)?
4. Are the computed features meaningful for financial trading?

Code to review:
{code}

Please respond with:
- VALID: if the code passes all checks
- INVALID: if any issues found, along with specific problems
"""
