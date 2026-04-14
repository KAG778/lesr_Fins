import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    opening_prices = s[1:120:6]  # Extract opening prices
    high_prices = s[2:120:6]     # Extract high prices
    low_prices = s[3:120:6]      # Extract low prices
    volumes = s[4:120:6]         # Extract trading volumes

    # Feature 1: Price Momentum (relative change in closing price)
    price_momentum = (closing_prices[-1] - closing_prices[-5]) / closing_prices[-5] if closing_prices[-5] != 0 else 0

    # Feature 2: Volume Change (relative change in volume)
    average_volume = np.mean(volumes[-5:]) if np.mean(volumes[-5:]) != 0 else 1  # Prevent division by zero
    volume_change = (volumes[-1] - average_volume) / average_volume

    # Feature 3: Price Range (high - low)
    price_range = high_prices[-1] - low_prices[-1]

    features = [price_momentum, volume_change, price_range]
    return np.array(features)
```

### Function 2: `intrinsic_reward(enhanced_state)`

In this function, we will compute the reward based on the provided priority chain using the regime vector. We will examine the `trend_direction`, `volatility_level`, and `risk_level` to determine the reward.

The implementation is as follows:

```python
def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
        return reward  # Early return for high risk
    elif risk_level > 0.4:
        reward -= 10  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 10  # Reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += 10  # Reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward mean-reversion features (oversold→buy, overbought→sell)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% (uncertain market)

    return reward