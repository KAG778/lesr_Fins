import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]          # Trading volumes
    highs = s[2::6]            # High prices
    lows = s[3::6]             # Low prices
    
    # Feature 1: Price Change Percentage
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0
    
    # Feature 2: Average Volume of the last 20 days
    avg_volume = np.mean(volumes) if len(volumes) > 0 else 0.0
    
    # Feature 3: Price Range (High - Low) of the last day
    price_range = highs[-1] - lows[-1] if highs[-1] is not None and lows[-1] is not None else 0.0
    
    features = [price_change_pct, avg_volume, price_range]
    return np.array(features)
```

### Function 2: `intrinsic_reward(enhanced_state)`

In this function, we will calculate the reward based on the provided rules and the regime vector. We will read the regime vector to determine the current market conditions and apply the reward logic accordingly. 

Here’s the implementation of the `intrinsic_reward` function:

```python
def intrinsic_reward(enhanced_state):
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:  # Dangerous
        if enhanced_state[123] > 0:  # Assuming positive feature for BUY
            reward = np.random.uniform(-50, -30)  # STRONG NEGATIVE reward for BUY
        else:  # Assuming negative feature for SELL
            reward = np.random.uniform(5, 10)  # MILD POSITIVE reward for SELL
    elif risk_level > 0.4:  # Elevated risk
        if enhanced_state[123] > 0:  # Assuming positive feature for BUY
            reward = np.random.uniform(-10, -5)  # Moderate negative reward for BUY
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            if enhanced_state[123] > 0:  # Assuming positive feature for BUY
                reward = np.random.uniform(10, 20)  # Positive reward for BUY
        else:  # Downtrend
            if enhanced_state[123] < 0:  # Assuming negative feature for SELL
                reward = np.random.uniform(10, 20)  # Positive reward for SELL
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_state[123] > 0:  # Assuming oversold signal
            reward = np.random.uniform(5, 15)  # Reward for BUY
        else:  # Assuming overbought signal
            reward = np.random.uniform(-15, -5)  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the range [-100, 100]