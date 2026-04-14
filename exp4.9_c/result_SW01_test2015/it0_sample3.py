import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices (indices 0, 6, 12, ..., 114)
    volumes = s[4::6]  # Extract volumes (indices 4, 10, 16, ..., 114)
    
    # Feature 1: Price Momentum (current - previous 5 days)
    price_momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0
    
    # Feature 2: Normalized Volume (current volume / average volume of last 5 days)
    avg_volume = np.mean(volumes[-5:]) if len(volumes[-5:]) > 0 else 1  # Avoid division by zero
    normalized_volume = volumes[-1] / avg_volume if avg_volume != 0 else 0
    
    # Feature 3: Volatility Indicator (standard deviation of last 5 closing prices)
    volatility_indicator = np.std(closing_prices[-5:]) / np.mean(closing_prices[-5:]) if np.mean(closing_prices[-5:]) != 0 else 0
    
    # Compile features into an array
    features = [price_momentum, normalized_volume, volatility_indicator]
    
    return np.array(features)
```

### Function 2: `intrinsic_reward(enhanced_state)`

In this function, we will implement the reward logic based on the provided priority chain. We'll extract the regime information and then calculate the reward according to the specified rules.

```python
def intrinsic_reward(enhanced_state):
    # Extract regime vector
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
        return reward
    if risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals
    
    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += np.random.uniform(10, 20)  # Positive reward for upward features
        else:
            reward += np.random.uniform(10, 20)  # Positive reward for downward features
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Assume features are designed for mean-reversion
        reward += np.random.uniform(5, 15)  # Reward mean-reversion features
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    return reward