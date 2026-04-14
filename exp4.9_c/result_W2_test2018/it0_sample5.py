import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Get every 6th element starting from index 0 (closing prices)
    volumes = s[4::6]  # Get every 6th element starting from index 4 (volumes)
    
    # Handle edge cases
    if len(closing_prices) < 5 or len(volumes) < 5:
        return np.array([0, 0, 0])  # Return zeros if not enough data
    
    # Compute the last 5 closing prices
    last_5_closing_prices = closing_prices[-5:]
    # Price Change Ratio
    price_change_ratio = (last_5_closing_prices[-1] - last_5_closing_prices[0]) / last_5_closing_prices[0] if last_5_closing_prices[0] != 0 else 0
    
    # Compute the last 5 volumes
    last_5_volumes = volumes[-5:]
    # Volume Change Ratio
    volume_change_ratio = (last_5_volumes[-1] - np.mean(last_5_volumes[:-1])) / np.mean(last_5_volumes[:-1]) if np.mean(last_5_volumes[:-1]) != 0 else 0
    
    # Calculate RSI
    delta = np.diff(closing_prices[-14:])  # Calculate price differences for the last 14 days
    gain = np.mean(delta[delta > 0]) if len(delta[delta > 0]) > 0 else 0
    loss = -np.mean(delta[delta < 0]) if len(delta[delta < 0]) > 0 else 0
    rs = gain / loss if loss != 0 else 0  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))  # RSI formula
    
    features = [price_change_ratio, volume_change_ratio, rsi]
    return np.array(features)
```

### Step 3: Implementing `intrinsic_reward`

Now let's implement the `intrinsic_reward` function according to the specified priority chain:

```python
def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # Assuming positive price change indicates a BUY signal
            return np.random.uniform(-50, -30)  # Strong negative reward for BUY
        return np.random.uniform(5, 10)  # Mild positive reward for SELL
    elif risk_level > 0.4:
        if features[0] > 0:  # Positive price change indicates a BUY signal
            reward -= 10  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            if features[0] > 0:  # Positive price change indicates a correct BUY signal
                reward += 20  # Reward for correct trend-following
        elif trend_direction < 0:  # Downtrend
            if features[0] < 0:  # Negative price change indicates a correct SELL signal
                reward += 20  # Reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] > 70:  # RSI indicates overbought
            reward -= 15  # Penalize for buying in overbought conditions
        elif features[2] < 30:  # RSI indicates oversold
            reward += 15  # Reward for buying in oversold conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward