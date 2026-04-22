import numpy as np

def revise_state(s):
    # s: 120d raw state, representing 20 days of OHLCV data
    closing_prices = s[0::6]  # Extract the closing prices
    volumes = s[4::6]  # Extract the trading volumes

    # Feature 1: Daily Price Change
    price_change = np.diff(closing_prices, prepend=closing_prices[0])  # Daily change
    daily_price_change_ratio = price_change / closing_prices  # Relative change

    # Feature 2: Volume Change (current volume / average of last 5 days)
    avg_volume = np.convolve(volumes, np.ones(5)/5, mode='valid')  # Average volume over last 5 days
    volume_change_ratio = np.array([volumes[i] / avg_volume[i-4] if i >= 4 else np.nan for i in range(len(volumes))])
    volume_change_ratio = np.nan_to_num(volume_change_ratio)  # Handle NaN values (replace with 0)

    # Feature 3: Simple Moving Average (SMA) of closing prices (5-day SMA)
    sma = np.convolve(closing_prices, np.ones(5)/5, mode='valid')
    sma = np.concatenate((np.full(4, np.nan), sma))  # Pad the beginning with NaNs
    sma = np.nan_to_num(sma)  # Handle NaN values

    features = [daily_price_change_ratio[-1], volume_change_ratio[-1], sma[-1]]  # Use the latest value of each feature
    return np.array(features)
```

### Function 2: `intrinsic_reward(enhanced_state)`

In this function, we will implement the reward logic based on the priority chain provided in the task. We'll first evaluate the risk level, and then depending on the trend direction and volatility, we'll compute the rewards accordingly.

Here's how you can implement this:

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
        reward += np.random.uniform(-50, -30)  # STRONG NEGATIVE reward for BUY-aligned features
    elif risk_level > 0.4:
        reward += np.random.uniform(-10, -5)  # Moderate negative reward for BUY signals

    # If risk is low, we proceed to trend following
    if risk_level < 0.4:
        if abs(trend_direction) > 0.3:
            if trend_direction > 0:  # Uptrend
                reward += 10  # Positive reward for BUY-aligned features
            else:  # Downtrend
                reward += 10  # Positive reward for SELL-aligned features
        
        # Priority 3 — SIDEWAYS / MEAN REVERSION
        if abs(trend_direction) < 0.3:
            # Here we would incorporate mean-reversion logic based on features, which is not defined in the task.
            reward += 5  # Placeholder for mean-reversion reward (to be defined further)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)