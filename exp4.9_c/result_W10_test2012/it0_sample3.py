import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices (indices 0, 6, 12, ..., 114)
    
    # Feature 1: Price Momentum - percentage change over the last 5 days
    price_momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0
    
    # Feature 2: Average Volume - average volume over the last 5 days
    volumes = s[4::6]  # Extract trading volumes (indices 4, 10, 16, ..., 114)
    average_volume = np.mean(volumes[-5:]) if len(volumes[-5:]) > 0 else 0
    
    # Feature 3: Price Volatility - standard deviation of the closing prices over the last 5 days
    price_volatility = np.std(closing_prices[-5:]) if len(closing_prices[-5:]) > 0 else 0
    
    features = [price_momentum, average_volume, price_volatility]
    return np.array(features)
```

### Function 2: `intrinsic_reward`
In this function, we will implement the reward logic based on the priority chain. The function will analyze the `enhanced_state` and calculate the reward based on the specified criteria.

Here's the implementation:

```python
def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0
    
    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:
        # Strongly negative reward for BUY-aligned features
        reward += -40  # Example strong negative reward
        # MILD POSITIVE reward for SELL-aligned features
        reward += 7  # Example mild positive reward
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -15  # Example moderate negative reward

    # Priority 2: TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += 10  # Positive reward for upward features
        else:  # Downtrend
            reward += 10  # Positive reward for downward features

    # Priority 3: SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features (oversold→buy, overbought→sell)
        reward += 5  # Example reward for mean-reversion features
        # Penalize breakout-chasing features
        reward += -5  # Example penalty for chasing breakouts

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward by 50% in high volatility

    # Ensure the reward is within the specified range [-100, 100]
    return max(-100, min(100, reward))