import numpy as np

def revise_state(s):
    # s: 120d raw state consisting of 20 days of OHLCV interleaved data
    closing_prices = s[::6]  # Closing prices (index 0, 6, 12, ...)
    trading_volumes = s[4::6]  # Trading volumes (index 4, 10, 16, ...)
    
    # Feature 1: Price Momentum (last closing price - the price 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) >= 6 else 0
    
    # Feature 2: Average Trading Volume over the last 20 days
    avg_volume = np.mean(trading_volumes) if len(trading_volumes) > 0 else 0
    
    # Feature 3: Relative Strength Index (RSI) calculation
    def calculate_rsi(prices, period=14):
        if len(prices) < period:
            return 0
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gain[-period:]) if period > 0 else 0
        avg_loss = np.mean(loss[-period:]) if period > 0 else 0
        
        if avg_loss == 0:
            return 100  # Avoid division by zero, RSI will be max
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = calculate_rsi(closing_prices)

    return np.array([price_momentum, avg_volume, rsi])
```

### Function 2: `intrinsic_reward(enhanced_state)`

In this function, we will follow the priority chain to compute the reward based on the regime vector and the computed features. The reward will vary based on the trend direction, risk level, and the actions taken by the agent.

Here is the implementation:

```python
def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY signals
        reward -= 50  # Strong penalty for risky BUY
        reward += 10  # Mild positive for SELL signals
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= 20  # Moderate penalty for risky BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 10  # Positive reward for upward features
        elif trend_direction < -0.3:
            reward += 10  # Positive reward for downward features
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features
        reward += 5  # Reward for mean-reversion features (oversold/buy, overbought/sell)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% in high volatility

    # Ensure the reward is within the bounds [-100, 100]
    return np.clip(reward, -100, 100)