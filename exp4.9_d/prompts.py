"""
Prompt Templates for exp4.9_d

Changes from exp4.7:
- D2: LLM only generates intrinsic_reward, not revise_state
- Enhanced state structure is pre-computed and described in prompt
- A1: Ticker-specific information included
"""

import numpy as np
from typing import List, Dict, Optional


# ==============================================================================
# Stock Profile Generation (A1)
# ==============================================================================

def generate_stock_profile(
    ticker: str,
    train_data_loader=None,
    train_start: str = None,
    train_end: str = None,
) -> str:
    """Generate stock profile from training period price/volume data only."""
    if train_data_loader is not None and train_start is not None:
        dates = [
            d for d in train_data_loader.get_date_range()
            if train_start <= str(d) <= train_end
        ]
        prices_list = []
        volumes_list = []
        for d in dates:
            daily_data = train_data_loader.get_data_by_date(d)
            if ticker in daily_data.get('price', {}):
                price_dict = daily_data['price'][ticker]
                if isinstance(price_dict, dict):
                    prices_list.append(price_dict.get('close', 0))
                    volumes_list.append(price_dict.get('volume', 0))
                else:
                    prices_list.append(price_dict)
                    volumes_list.append(0)

        prices_arr = np.array(prices_list)
        volumes_arr = np.array(volumes_list)

        if len(prices_arr) < 2:
            return f"## Target Stock: {ticker}\nInsufficient training data."

        returns = np.diff(prices_arr) / prices_arr[:-1]
        daily_vol = np.std(returns) * 100
        total_ret = (prices_arr[-1] - prices_arr[0]) / prices_arr[0] * 100
        avg_vol = np.mean(volumes_arr)

        return f"""## Target Stock: {ticker}
- Training period daily volatility: {daily_vol:.2f}%
- Training period total return: {total_ret:.2f}%
- Training period avg daily volume: {avg_vol:.0f}

Optimize the reward function specifically for this stock's risk and return profile."""

    return f"## Target Stock: {ticker}"


# ==============================================================================
# Initial Prompt Template (D2: only intrinsic_reward)
# ==============================================================================

INITIAL_PROMPT_TEMPLATE = """
You are a financial quantitative analysis expert.

## Task Background

We are training a stock timing strategy using reinforcement learning (DQN). The strategy makes decisions at each trading day:
- **BUY**: Establish a long position (only when currently not holding)
- **SELL**: Close position (only when currently holding)
- **HOLD**: Maintain current position

{stock_profile}

## Your Task

Design an **intrinsic reward function** `intrinsic_reward(enhanced_state)` that helps the DQN make better trading decisions.

The state representation is **already pre-computed** — you do NOT need to write feature engineering code. The enhanced state is a 151-dimensional array:

### Enhanced State Structure

| Index | Feature | Description |
|-------|---------|-------------|
| s[0:19] | Close prices | 20 days closing prices |
| s[20:39] | Open prices | 20 days opening prices |
| s[40:59] | High prices | 20 days high prices |
| s[60:79] | Low prices | 20 days low prices |
| s[80:99] | Volume | 20 days trading volume |
| s[100:119] | Adj close | 20 days adjusted closing prices |
| **s[120:122]** | **SMA** | 5-day, 10-day Simple Moving Average |
| **s[122:124]** | **SMA + Deviation** | 20-day SMA, Price/SMA20 deviation |
| **s[124:126]** | **EMA** | 5-day EMA, 20-day EMA |
| **s[126:128]** | **MA Cross** | SMA5-SMA20 ratio, Price/SMA10 deviation |
| **s[128:131]** | **RSI** | 5-day, 10-day, 14-day RSI |
| **s[131:134]** | **MACD** | MACD line, signal line, histogram |
| **s[134]** | **Momentum** | 10-day rate of change |
| **s[135:137]** | **Volatility** | 5-day, 20-day historical volatility |
| **s[137]** | **ATR** | 14-day Average True Range |
| **s[138:140]** | **Vol Metrics** | Vol ratio (5d/20d), vol change |
| **s[140]** | **OBV Trend** | On-Balance Volume trend |
| **s[141]** | **VP Corr** | Volume-price correlation |
| **s[142:144]** | **Volume** | Volume ratio (5d/20d), relative volume |
| **s[144]** | **Regime Vol** | Volatility regime ratio |
| **s[145]** | **Trend R²** | Trend strength (R² of regression) |
| **s[146]** | **Price Pos** | Price position in 20-day range [0,1] |
| **s[147]** | **Regime Vol V** | Volume regime ratio |
| **s[148]** | **Trend Slope** | Normalized trend slope |
| **s[149]** | **BB Pos** | Bollinger Band position [0,1] |
| **s[150]** | **Position** | **1.0 = holding stock, 0.0 = not holding** |

## Reward Design Guidelines

1. **Use position flag** (s[150]) to differentiate buy vs sell signals:
   - When position = 0 (not holding): give positive reward for clear BUY opportunities (strong uptrend, oversold bounce)
   - When position = 1 (holding): give positive reward for HOLD during uptrend, and encourage SELL when trend weakens

2. **Use relative thresholds** — calculate historical volatility from s[135:137] and use multiples of it instead of hard-coded values

3. **Use regime features** (s[144:149]) to adapt behavior:
   - High volatility ratio (>2) = extreme market → be cautious
   - Strong trend R² (>0.8) = clear direction → follow trend
   - High BB position (>0.8) = overbought → consider selling

4. **Avoid both extremes**:
   - Don't always give positive reward (DQN learns nothing)
   - Don't make reward always zero (DQN ignores it)

## Objective

The strategy is measured by **Sharpe Ratio** (risk-adjusted return), with 0.1% commission per trade.

## Output Format

Please return a **single Python function**:

```python
import numpy as np

def intrinsic_reward(enhanced_state):
    # enhanced_state is 151-dimensional (see table above)
    # s = enhanced_state
    # Return a value in [-100, 100]
    return reward
```

Think step by step about which features matter most and how to combine them.
"""


def get_initial_prompt(
    ticker: str = None,
    train_data_loader=None,
    train_start: str = None,
    train_end: str = None,
    stock_profile: str = ""
) -> str:
    """Generate initial prompt with optional stock-specific information."""
    if not stock_profile and ticker:
        stock_profile = generate_stock_profile(
            ticker, train_data_loader, train_start, train_end
        )
    return INITIAL_PROMPT_TEMPLATE.format(stock_profile=stock_profile)


INITIAL_PROMPT = INITIAL_PROMPT_TEMPLATE.format(stock_profile="")


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
    """Generate COT feedback prompt."""
    s_feedback = ''

    for i, (code, score) in enumerate(zip(codes, scores)):
        s_feedback += f'========== Code Candidate -- {i + 1} ==========\n'
        s_feedback += code + '\n'
        s_feedback += f'Performance Metrics:\n'
        s_feedback += f'  Sharpe Ratio: {score["sharpe"]:.3f}\n'
        s_feedback += f'  Max Drawdown: {score["max_dd"]:.2f}%\n'
        s_feedback += f'  Total Return: {score["total_return"]:.2f}%\n'

        if 'num_trades' in score:
            s_feedback += f'  Number of Trades: {score["num_trades"]}\n'

        if i < len(importance) and len(importance[i]) > original_dim:
            extra_dim = len(importance[i]) - original_dim
            if extra_dim > 0:
                s_feedback += f'\nNew Feature Importance (Top 5 of {extra_dim}):\n'
                top_extra = np.argsort(importance[i][original_dim:])[-min(5, extra_dim):][::-1]
                for rank, idx in enumerate(top_extra, 1):
                    actual_idx = original_dim + idx
                    corr_val = correlations[i][actual_idx] if i < len(correlations) and actual_idx < len(correlations[i]) else 0
                    s_feedback += f'  feature_{idx} (s[{actual_idx}]): importance={importance[i][actual_idx]:.3f}, corr={corr_val:.3f}\n'

        s_feedback += '\n'

    cot_prompt = f"""
We trained DQN policies using {len(codes)} different intrinsic reward functions.

Training Results:
{s_feedback}

Please analyze the results and provide improvement suggestions:

Analysis Points:
(a) Which reward functions work best? What patterns do they share?
(b) What are common problems of low-performance reward functions?
(c) How should the reward function use position flag (s[150]) to differentiate buy/sell?
(d) Are there too many or too few trades? How to adjust the reward to control frequency?

Key reminders:
- The reward function receives a 151-dim state (120 OHLCV + 30 pre-computed features + 1 position flag)
- Use relative thresholds based on volatility features (s[135:137])
- Use regime features (s[144:149]) to adapt to market conditions
- The position flag s[150] tells whether currently holding (1.0) or not (0.0)

Goal: Improve Sharpe ratio while keeping max drawdown under 30%.
"""
    return cot_prompt


# ==============================================================================
# Iteration Prompt
# ==============================================================================

def get_iteration_prompt(
    all_iter_codes: List[List[str]],
    all_iter_cot_suggestions: List[str],
    ticker: str = None,
    stock_profile: str = ""
) -> str:
    """Generate iteration prompt with historical context."""
    former_history = ''

    for i in range(len(all_iter_codes)):
        former_history += f'\n\n\nFormer Iteration: {i + 1}\n'
        for j, code in enumerate(all_iter_codes[i]):
            former_history += f'Candidate {j + 1}:\n{code}\n'
        if i < len(all_iter_cot_suggestions):
            former_history += f'\nSuggestions:\n{all_iter_cot_suggestions[i]}\n'

    profile_section = f"\n{stock_profile}\n" if stock_profile else ""

    iteration_prompt = f"""
You are a financial quantitative analysis expert.
{profile_section}
We have completed multiple iterations of reward function optimization. Here is the history:

{former_history}

Based on the above, please design an improved intrinsic_reward function.

Requirements:
1. The function receives a 151-dim state (120 OHLCV + 30 features + 1 position flag at s[150])
2. Use position flag s[150] to differentiate buy signals (position=0) vs sell signals (position=1)
3. Use relative thresholds based on volatility features
4. Use regime features to adapt to market conditions
5. Return value must be in [-100, 100]
6. Avoid patterns that led to poor performance in previous iterations

```python
import numpy as np

def intrinsic_reward(enhanced_state):
    # enhanced_state is 151-dimensional
    return reward
```
"""
    return iteration_prompt


# ==============================================================================
# Validation Prompt
# ==============================================================================

VALIDATION_PROMPT = """
Please review the following Python code:

1. Does the function accept a single argument (the enhanced state array)?
2. Does it return a value in the range [-100, 100]?
3. Are there any obvious bugs (division by zero, missing imports, etc.)?
4. Does it use the position flag (enhanced_state[150])?

Code to review:
{code}

Respond with:
- VALID: if the code passes all checks
- INVALID: if any issues found, along with specific problems
"""
