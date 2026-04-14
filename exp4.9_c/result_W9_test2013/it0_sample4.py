import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract volumes
    
    # Feature 1: Percentage Change over the last 5 days
    pct_change = (closing_prices[1:] - closing_prices[:-1]) / closing_prices[:-1]
    # Handle edge cases
    pct_change = np.concatenate(([0], pct_change))  # Fill the first entry with 0, as there's no change for the first day

    # Feature 2: Average Volume over the last 5 days
    avg_volume = np.convolve(volumes, np.ones(5)/5, mode='valid')
    # Pad the beginning with zeros to match the length
    avg_volume = np.concatenate(([0, 0, 0, 0], avg_volume))  # Padding with zeros for the first four days

    # Feature 3: Relative Strength Index (RSI) calculation
    def calculate_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        rsi = np.concatenate(([0] * (period - 1), rsi))  # Pad RSI with zeros

        return rsi

    rsi = calculate_rsi(closing_prices)

    # Return the computed features
    features = np.array([pct_change[-20:], avg_volume[-20:], rsi[-20:]]).flatten()
    return features
```

### Function 2: `intrinsic_reward(enhanced_state)`

In this function, we will implement the reward logic as outlined, following the priority chain. We will extract the regime information and compute the reward based on the trading actions and the regime conditions.

```python
def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Action taken by the agent: 0 for BUY, 1 for SELL, 2 for HOLD
    action = enhanced_state[123]  # Assume this contains the action taken, for example:
    # action = ...  # This should be passed in some way, it's ambiguous in the context

    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if action == 0:  # BUY
            return np.random.uniform(-50, -30)  # Strong negative reward
        elif action == 1:  # SELL
            return np.random.uniform(5, 10)  # Mild positive reward
    elif risk_level > 0.4:
        if action == 0:  # BUY
            return np.random.uniform(-20, -10)  # Moderate negative reward

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and action == 0:  # If uptrend and action is BUY
            return np.random.uniform(10, 20)  # Positive reward
        elif trend_direction < -0.3 and action == 1:  # If downtrend and action is SELL
            return np.random.uniform(10, 20)  # Positive reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if action == 0:  # BUY in sideways
            return np.random.uniform(-10, -5)  # Penalize breakout-chasing
        elif action == 1:  # SELL in sideways
            return np.random.uniform(5, 10)  # Reward mean-reversion

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        # Reduce reward magnitude by 50%
        reward *= 0.5

    return reward