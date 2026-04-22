import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    
    # Feature 1: Price Momentum (using 5-day momentum)
    momentum = closing_prices[-1] - closing_prices[-6]  # Current closing - closing price 5 days ago
    
    # Feature 2: Volatility (standard deviation of last 5 closing prices)
    if len(closing_prices) >= 5:
        volatility = np.std(closing_prices[-5:])  # Standard deviation of last 5 days
    else:
        volatility = 0.0  # Handle edge case

    # Feature 3: Volume Change (percentage change from the previous day)
    if volumes[-2] != 0:
        volume_change = (volumes[-1] - volumes[-2]) / volumes[-2]
    else:
        volume_change = 0.0  # Handle edge case

    features = [momentum, volatility, volume_change]
    
    return np.array(features)
```

### Function 2: `intrinsic_reward(enhanced_state)`

This function will calculate the reward based on the priority chain provided. We will read the regime vector to determine the market conditions and then apply the reward logic accordingly.

Here’s the implementation:

```python
def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY aligned features
        reward = -40  # Example value in the range -30 to -50
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward = -20  # Example value in a moderate range

    # Only evaluate further if risk is low
    if risk_level < 0.4:
        # Priority 2 — TREND FOLLOWING
        if abs(trend_direction) > 0.3:
            if trend_direction > 0.3:
                reward += 10  # Reward for positive trend
            elif trend_direction < -0.3:
                reward += 10  # Reward for negative trend

        # Priority 3 — SIDEWAYS / MEAN REVERSION
        elif abs(trend_direction) < 0.3:
            # Assume we have access to features like 'oversold' or 'overbought'
            # Here we can check if the last features indicate mean reversion
            # Placeholder for mean-reversion checks
            reward += 5  # Small reward for mean-reversion

        # Priority 4 — HIGH VOLATILITY
        if volatility_level > 0.6:
            reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)