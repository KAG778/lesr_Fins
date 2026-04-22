import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV)
    
    # Extract closing prices from the raw state
    closing_prices = s[0::6]  # Extract every 6th element starting from index 0
    
    # Feature 1: Daily Price Change Percentage
    price_change_pct = np.diff(closing_prices) / closing_prices[:-1]  # Daily percentage change
    price_change_pct = np.concatenate(([0], price_change_pct))  # Prepend a 0 for alignment
    
    # Feature 2: 5-Day Moving Average
    moving_average = np.convolve(closing_prices, np.ones(5)/5, mode='valid')
    moving_average = np.concatenate(([np.nan]*4, moving_average))  # Prepend NaN for alignment

    # Feature 3: Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.convolve(gain, np.ones(14)/14, mode='valid')
    avg_loss = np.convolve(loss, np.ones(14)/14, mode='valid')
    
    rs = avg_gain / avg_loss if avg_loss[-1] != 0 else 0  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate(([np.nan]*13, rsi))  # Prepend NaN for alignment

    # Combine features into a single array, while handling NaNs
    features = np.nan_to_num(np.array([price_change_pct, moving_average, rsi]).flatten(), nan=0.0)
    
    return features
```

### Function 2: `intrinsic_reward(enhanced_state)`

This function will implement the reward logic according to the specified priority chain. It will check the `risk_level`, `trend_direction`, and `volatility_level` and return the appropriate reward based on the conditions provided.

Here's the implementation:

```python
def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0  # Initialize reward

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += np.random.uniform(-50, -30)  # STRONG NEGATIVE reward for BUY-aligned features
    elif risk_level > 0.4:
        reward += np.random.uniform(-20, -10)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += np.random.uniform(10, 20)  # Positive reward for upward features
        elif trend_direction < -0.3:
            reward += np.random.uniform(10, 20)  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward mean-reversion features
        reward -= np.random.uniform(5, 15)  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Clip the reward to the range [-100, 100]