import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Extract closing prices and volumes from the raw state
    closing_prices = s[0:120:6]  # Closing prices (day 0, 6, 12, ..., 114)
    volumes = s[4:120:6]         # Trading volumes (day 0, 6, 12, ..., 114)

    # Calculate features
    features = []

    # Feature 1: Price Momentum (latest closing price - closing price 5 days ago)
    if len(closing_prices) > 5:
        price_momentum = closing_prices[0] - closing_prices[5]
        features.append(price_momentum)
    else:
        features.append(0.0)  # Default if not enough data

    # Feature 2: Price Change Percentage (latest closing price / previous closing price - 1)
    if len(closing_prices) > 1:
        price_change_percentage = (closing_prices[0] / closing_prices[1]) - 1
        features.append(price_change_percentage)
    else:
        features.append(0.0)  # Default if not enough data

    # Feature 3: Volume Change (latest volume - volume 5 days ago)
    if len(volumes) > 5:
        volume_change = volumes[0] - volumes[5]
        features.append(volume_change)
    else:
        features.append(0.0)  # Default if not enough data

    return np.array(features)
```

### Function 2: `intrinsic_reward(enhanced_state)`

In this function, we will implement the reward logic according to the stated priorities. We will use the computed features from the `revise_state` function to inform the reward based on the current market regime.

```python
def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = features
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
        if len(features) > 0:
            if trend_direction > 0:
                # Reward for positive trend and upward features
                reward += features[0] * 10.0  # Price Momentum
            else:
                # Reward for negative trend and downward features
                reward += -features[0] * 10.0  # Price Momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Assuming that features[1] represents Price Change Percentage
        if features[1] < 0:  # Oversold condition
            reward += 5.0  # Mild positive for potential buy
        elif features[1] > 0:  # Overbought condition
            reward += -5.0  # Mild negative for potential sell

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))