"""
Prompt Templates for Exp4.9_b Financial Trading Experiment

Changes from exp4.7:
- A1: Ticker-specific information (volatility, return, volume from training data)
- A2: Market regime detection features required
- A3: Require 20-30 new features (was ~5-7)
- B1: Position flag explanation for intrinsic_reward
"""

import numpy as np
from typing import List, Dict, Optional


# ==============================================================================
# Stock Profile Generation (A1: ticker-specific info from price/volume only)
# ==============================================================================

def generate_stock_profile(
    ticker: str,
    train_data_loader=None,
    train_start: str = None,
    train_end: str = None,
    prices: np.ndarray = None,
    volumes: np.ndarray = None
) -> str:
    """
    Generate stock profile from training period price/volume data only.

    Args:
        ticker: Stock ticker symbol
        train_data_loader: Data loader instance (optional)
        train_start: Training start date
        train_end: Training end date
        prices: Pre-extracted closing prices (alternative to data_loader)
        volumes: Pre-extracted volumes (alternative to data_loader)

    Returns:
        Stock profile string for prompt
    """
    if prices is not None and len(prices) > 0:
        returns = np.diff(prices) / prices[:-1]
        daily_vol = np.std(returns) * 100
        total_ret = (prices[-1] - prices[0]) / prices[0] * 100
        avg_vol = np.mean(volumes) if volumes is not None and len(volumes) > 0 else 0
    elif train_data_loader is not None and train_start is not None:
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
            return f"## Target Stock: {ticker}\nInsufficient training data for profile."

        returns = np.diff(prices_arr) / prices_arr[:-1]
        daily_vol = np.std(returns) * 100
        total_ret = (prices_arr[-1] - prices_arr[0]) / prices_arr[0] * 100
        avg_vol = np.mean(volumes_arr)
    else:
        return f"## Target Stock: {ticker}"

    return f"""## Target Stock: {ticker}
- Training period daily volatility: {daily_vol:.2f}%
- Training period total return: {total_ret:.2f}%
- Training period avg daily volume: {avg_vol:.0f}

Optimize features specifically for this stock's risk and return profile.
Consider the stock's volatility level when designing thresholds."""


# ==============================================================================
# Initial Prompt Template
# ==============================================================================

INITIAL_PROMPT_TEMPLATE = """
You are a financial quantitative analysis expert, specializing in extracting trading signals from price and volume data.

## Task Background

We are training a stock timing strategy using reinforcement learning (DQN). The strategy makes decisions at each trading day:
- **BUY**: Establish a long position (only when currently not holding)
- **SELL**: Close position (only when currently holding)
- **HOLD**: Maintain current position

{stock_profile}

## Available Data

The raw state is a 120-dimensional NumPy array `s`:
- `s[0:19]`: 20 days of closing prices
- `s[20:39]`: 20 days of opening prices
- `s[40:59]`: 20 days of high prices
- `s[60:79]`: 20 days of low prices
- `s[80:99]`: 20 days of trading volume
- `s[100:119]`: 20 days of adjusted closing prices

## State Vector Structure (IMPORTANT)

The full state vector passed to the DQN has this structure:
- `s[0:119]`: 120 dims of raw OHLCV data
- `s[120:???]`: Your generated features (see requirements below)
- `s[-1]`: **Position flag** — 1.0 = currently holding stock, 0.0 = not holding

**Note**: The position flag is automatically appended by the system. Your `revise_state` function should NOT include it. But your `intrinsic_reward` function receives the full state including the position flag at `enhanced_s[-1]`.

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

5. **Market Regime**: The current market state
   - Strong trend: Clear directional movement, high R²
   - Sideways/Choppy: No clear direction, mean-reverting
   - Extreme volatility: Unusual price swings (crash, V-shaped recovery)

## Objective Function

Strategy performance is measured by:
1. **Sharpe Ratio**: Risk-adjusted return (maximize)
2. **Maximum Drawdown**: Maximum loss magnitude (minimize, keep under 30%)
3. **Total Return**: Cumulative return (maximize)

## Constraints

1. Transaction cost: 0.1% commission per trade (automatically applied by the system)
2. Position limit: Maximum 100% position in single stock
3. Risk limit: Maximum 5% daily loss

## Your Task

Please generate two Python functions:

### Function 1: `revise_state(raw_state)`
- **Input**: Raw state (120-dimensional NumPy array)
- **Output**: Enhanced state (original 120 dimensions + **20-30 new features**)
- **Required feature categories** (generate at least 20 features total):

**A. Multi-timeframe Trend Indicators (~6-8 features):**
  - 5-day, 10-day, 20-day SMA and/or EMA
  - Short MA vs Long MA differences
  - Price relative to moving averages

**B. Momentum Indicators (~4-6 features):**
  - RSI at multiple timeframes (5-day, 10-day, 14-day)
  - MACD line, signal line, histogram
  - Rate of change (momentum)

**C. Volatility Indicators (~3-4 features):**
  - Historical volatility (5-day, 20-day)
  - ATR (Average True Range)
  - Volatility ratio (short-term / long-term)

**D. Volume-Price Relationship (~3-4 features):**
  - OBV (On-Balance Volume) trend
  - Volume-price correlation
  - Volume ratio (recent vs historical average)

**E. Market Regime Detection (~4 features, REQUIRED):**
  - `volatility_ratio`: 5-day volatility / 20-day volatility (>2.0 = extreme)
  - `trend_strength`: Linear regression R² of closing prices (near 1 = strong trend)
  - `price_position`: Current price position within 20-day range [0, 1]
  - `volume_ratio_regime`: 5-day average volume / 20-day average volume (>2.0 = unusual activity)

- Handle edge cases: division by zero, missing values, insufficient data, etc.

### Function 2: `intrinsic_reward(enhanced_state)`
- **Input**: Enhanced state (your features + position flag at `enhanced_s[-1]`)
- **Output**: Intrinsic reward value (range: [-100, 100])
- **Must use at least one of the new feature dimensions**
- **Should use the position flag** (`enhanced_s[-1]`) to differentiate:
  - When position = 0 (not holding): give positive reward for clear BUY signals (strong uptrend, oversold bounce)
  - When position = 1 (holding): give positive reward for HOLD during uptrend, SELL signals when trend weakens
  - Penalize uncertain/choppy market conditions

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
    # Your implementation — must return 140-150 dimensional array (120 original + 20-30 new)
    return enhanced_s

def intrinsic_reward(enhanced_s):
    # enhanced_s[-1] is the position flag (1.0=holding, 0.0=not holding)
    # Your implementation
    return reward
```

Let's think step by step.
"""


def get_initial_prompt(
    ticker: str = None,
    train_data_loader=None,
    train_start: str = None,
    train_end: str = None,
    stock_profile: str = ""
) -> str:
    """
    Generate initial prompt with optional stock-specific information.

    Args:
        ticker: Stock ticker symbol (for profile generation)
        train_data_loader: Data loader instance (for profile generation)
        train_start: Training start date
        train_end: Training end date
        stock_profile: Pre-generated stock profile string

    Returns:
        Complete initial prompt string
    """
    if not stock_profile and ticker:
        stock_profile = generate_stock_profile(
            ticker, train_data_loader, train_start, train_end
        )

    return INITIAL_PROMPT_TEMPLATE.format(stock_profile=stock_profile)


# Keep backward compatibility
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

        # Trade count if available
        if 'num_trades' in score:
            s_feedback += f'  Number of Trades: {score["num_trades"]}\n'

        # Original feature analysis
        s_feedback += f'\nOriginal Feature Importance (OHLCV):\n'
        for idx in range(min(5, original_dim)):
            if i < len(importance) and idx < len(importance[i]):
                s_feedback += f'  s[{idx}]: importance={importance[i][idx]:.3f}\n'

        # New feature analysis
        if i < len(importance) and len(importance[i]) > original_dim:
            extra_dim = len(importance[i]) - original_dim
            if extra_dim > 0:
                s_feedback += f'\nNew Feature Importance (Top 5 of {extra_dim} total):\n'
                top_extra = np.argsort(importance[i][original_dim:])[-min(5, extra_dim):][::-1]
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
(d) Are there too many or too few trades? How to adjust the reward function to control trade frequency?

Financial Scene Special Notes:
- Trend features (momentum, moving averages) are important for timing
- Volatility features help with risk control
- Volume can confirm price trends
- Market regime features help identify when to trade vs. when to stay out
- intrinsic_reward should use the position flag to differentiate buy/sell signals
- **Use relative thresholds based on historical volatility** - different stocks have different risk profiles
- **Avoid both under-trading (0 trades) and over-trading (too many noisy trades)**

Goal: Improve the strategy's Sharpe ratio while keeping maximum drawdown under 30%.
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
    """
    Generate iteration prompt with historical context.

    Args:
        all_iter_codes: Historical codes from previous iterations
        all_iter_cot_suggestions: Historical COT suggestions
        ticker: Stock ticker (for profile)
        stock_profile: Pre-generated stock profile

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

    profile_section = f"\n{stock_profile}\n" if stock_profile else ""

    iteration_prompt = f"""
You are a financial quantitative analysis expert.
{profile_section}
We have completed multiple iterations of optimization. Here is the historical experience:

{former_history}

Based on the above experience and suggestions, please generate improved state representation and intrinsic reward functions.

Requirements:
1. Avoid repeating features that have been proven ineffective
2. Preserve and improve effective features
3. Try new feature combinations
4. Generate 20-30 new features (not 5-7) covering: trend, momentum, volatility, volume-price, and regime detection
5. intrinsic_reward must be in the range [-100, 100]
6. intrinsic_reward should use the position flag (enhanced_s[-1]) to differentiate buy vs sell signals
7. Focus on features that show high correlation with returns in previous iterations
8. **Use volatility-adaptive thresholds in intrinsic_reward** - calculate historical volatility and use multiples of it (e.g., 2x std) instead of hard-coded values like -5%
9. Include market regime detection features (volatility_ratio, trend_strength, price_position, volume_ratio_regime)

Please return complete Python code:

```python
import numpy as np

def revise_state(s):
    # Must return 140-150 dimensional array (120 original + 20-30 new features)
    return enhanced_s

def intrinsic_reward(enhanced_s):
    # enhanced_s[-1] is the position flag (1.0=holding, 0.0=not holding)
    return reward
```
"""

    return iteration_prompt


# ==============================================================================
# Validation Prompt (for checking generated code)
# ==============================================================================

VALIDATION_PROMPT = """
Please review the following Python code for a financial trading strategy:

1. Does `revise_state` return an array with at least 140 dimensions? (120 original + 20+ new features)
2. Does `intrinsic_reward` return a value in the range [-100, 100]?
3. Are there any obvious bugs (division by zero, missing imports, etc.)?
4. Are the computed features meaningful for financial trading?
5. Does the code include market regime detection features?

Code to review:
{code}

Please respond with:
- VALID: if the code passes all checks
- INVALID: if any issues found, along with specific problems
"""
