import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV data)
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract volumes

    # Handle edge cases by ensuring there are enough days to compute features
    if len(closing_prices) < 5 or len(volumes) < 5:
        return np.array([])  # Not enough data to compute features

    # Feature 1: Price Momentum (current closing price minus closing price 5 days ago)
    price_momentum = closing_prices[0] - closing_prices[5]  # Current - 5 days ago

    # Feature 2: Price Change Percentage (percentage change over the last 5 days)
    price_change_percentage = (closing_prices[0] - closing_prices[5]) / closing_prices[5] if closing_prices[5] != 0 else 0

    # Feature 3: Volume Change (current volume minus average volume of the last 5 days)
    average_volume = np.mean(volumes[1:6])  # Average of the last 5 days
    volume_change = volumes[0] - average_volume  # Current volume - average volume

    features = [price_momentum, price_change_percentage, volume_change]
    return np.array(features)
```

### Function 2: `intrinsic_reward(enhanced_state)`

This function will implement the reward logic based on the priority chain. We will check the risk level first, then evaluate the trend direction, followed by the volatility level, and finally apply the appropriate rewards or penalties.

Let's implement this function:

```python
def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0

    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:
        # STRONG NEGATIVE reward for BUY-aligned features
        reward -= 40  # Strongly discourage buying
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= 20  # Caution for buying in elevated risk

    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += 10  # Positive reward for bullish trend
        else:  # Downtrend
            reward += 10  # Positive reward for bearish trend

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features (based on computed features)
        # Assuming features[0] is the price momentum, we can check if it's oversold/overbought
        price_momentum = enhanced_state[123][0]  # Example: the first feature is price momentum
        if price_momentum < 0:  # Potentially oversold
            reward += 10  # Reward buying
        else:  # Potentially overbought
            reward += 10  # Reward selling

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return min(max(reward, -100), 100)  # Ensure reward is within [-100, 100]