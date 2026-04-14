import numpy as np

def revise_state(s):
    # s: 120-dimensional raw state
    closing_prices = s[0::6]  # Extract closing prices (every 6th element starting from index 0)
    volumes = s[4::6]         # Extract trading volumes (every 6th element starting from index 4)

    # Feature 1: Price Change (percentage change from the previous day)
    price_change = np.diff(closing_prices) / closing_prices[:-1]  # Calculate daily returns
    price_change = np.insert(price_change, 0, 0)  # Insert 0 for the first day (no previous day)
    
    # Feature 2: Average Volume (over the last 20 days)
    avg_volume = np.mean(volumes)  # Calculate average trading volume

    # Feature 3: Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.mean(gain[-14:])  # Average gain over the last 14 days
    avg_loss = np.mean(loss[-14:])  # Average loss over the last 14 days
    
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))  # RSI formula

    features = [price_change[-1], avg_volume, rsi]  # Return last price change, avg volume, and last RSI
    return np.array(features)
```

### Function 2: `intrinsic_reward(enhanced_state)`

In the `intrinsic_reward` function, we will implement the reward logic based on the priority chain described in the task. We will check the `risk_level`, `trend_direction`, and `volatility_level` to compute the reward.

Here's how we will implement it:

```python
def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]  # trend direction
    volatility_level = regime[1]  # volatility level
    risk_level = regime[2]  # risk level

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward = -40  # STRONG NEGATIVE reward for BUY-aligned features
        return reward
    elif risk_level > 0.4:
        reward = -10  # moderate negative reward for BUY signals
        return reward

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 10  # positive reward for bullish features
        else:
            reward += 10  # positive reward for bearish features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward