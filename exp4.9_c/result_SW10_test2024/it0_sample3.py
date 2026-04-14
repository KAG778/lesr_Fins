import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]  # Extract trading volumes

    # Feature 1: Price Momentum
    price_momentum = closing_prices[-1] - closing_prices[-2]  # Latest closing price - previous closing price

    # Feature 2: Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = (delta[delta > 0]).sum() / len(delta) if len(delta) > 0 else 0
    loss = (-delta[delta < 0]).sum() / len(delta) if len(delta) > 0 else 0
    rs = gain / loss if loss > 0 else np.inf  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs)) if rs != np.inf else 100  # RSI formula, bounded to 100

    # Feature 3: Average Daily Volume Change
    volume_change = np.diff(volumes) / volumes[:-1]  # Percentage change in volume
    avg_volume_change = np.mean(volume_change) if len(volume_change) > 0 else 0  # Mean of percentage changes

    # Return the features as a numpy array
    features = [price_momentum, rsi, avg_volume_change]
    return np.array(features)
```

### Function 2: `intrinsic_reward(enhanced_state)`

Now, we will implement the `intrinsic_reward` function based on the priority chain described. We will read the regime vector values and compute the reward accordingly.

Here's the implementation:

```python
def intrinsic_reward(enhanced_state):
    # Extract regime information
    trend_direction = enhanced_state[120]
    volatility_level = enhanced_state[121]
    risk_level = enhanced_state[122]

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
        reward += np.random.uniform(5, 10)  # MILD POSITIVE reward for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 10  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 10  # Positive reward for upward features
        elif trend_direction < -0.3:
            reward += 10  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    # Ensure the reward is bounded within [-100, 100]
    return np.clip(reward, -100, 100)