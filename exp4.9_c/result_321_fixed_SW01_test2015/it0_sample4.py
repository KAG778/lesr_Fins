import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Calculate new features:
    closing_prices = s[::6]  # Extracting closing prices
    volumes = s[4::6]        # Extracting volumes
    
    # Feature 1: Price Momentum (current - previous)
    price_momentum = closing_prices[1:] - closing_prices[:-1]
    price_momentum = np.concatenate(([0], price_momentum))  # Pad with 0 for alignment
    
    # Feature 2: Volume Change (current - previous) / previous
    volume_change = np.zeros_like(volumes)
    for i in range(1, len(volumes)):
        if volumes[i-1] != 0:
            volume_change[i] = (volumes[i] - volumes[i-1]) / volumes[i-1]
    
    # Feature 3: Price Range (high - low) / close
    highs = s[2::6]
    lows = s[3::6]
    price_range = (highs - lows) / closing_prices
    
    # Combine features into a single array
    features = np.concatenate([price_momentum, volume_change, price_range])
    
    # Handle edge cases (like NaN values)
    features = np.nan_to_num(features)  # Replace NaNs with 0
    
    return np.array(features)
```

### Function 2: `intrinsic_reward(enhanced_state)`

In this function, we will implement the reward logic according to the specified priority chain. We will read the regime vector and based on its values, we will compute the reward.

Here's how we can implement this logic:

```python
def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_state[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features.size > 0:
            if trend_direction > 0:  # Uptrend
                reward += features[0] * 10.0  # Positive reward for upward features
            else:  # Downtrend
                reward += features[1] * 10.0  # Positive reward for downward features

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features.size > 0:
            reward += 5.0  # Mild positive for mean-reversion features

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))