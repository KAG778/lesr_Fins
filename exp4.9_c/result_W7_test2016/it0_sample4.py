import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV)
    
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes

    # Feature 1: Price Change Percentage
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0

    # Feature 2: Average Volume
    average_volume = np.mean(volumes[-5:])  # Average volume of the last 5 days

    # Feature 3: RSI Calculation
    # Calculate price changes
    deltas = np.diff(closing_prices)
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    features = [price_change_pct, average_volume, rsi]
    
    return np.array(features)
```

### Step 2: Implementing `intrinsic_reward`

Now, we need to create the `intrinsic_reward` function following the priority chain specified. Let's implement it carefully.

```python
def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_state[123:]  # Our computed features
    price_change_pct = features[0]
    average_volume = features[1]
    rsi = features[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if price_change_pct > 0:  # Assuming a positive price change indicates a BUY signal
            reward = np.random.uniform(-50, -30)
        else:  # Negative price change indicates a SELL signal
            reward = np.random.uniform(5, 10)
        return reward

    if risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if price_change_pct > 0:
            reward = -10  # Arbitrary negative reward for buying in elevated risk
        return reward

    # Priority 2 — TREND FOLLOWING (when risk is low)
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and price_change_pct > 0:
            reward = 10  # Positive reward for following the trend
        elif trend_direction < -0.3 and price_change_pct < 0:
            reward = 10  # Positive reward for bearish trend
        return reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if rsi < 30:  # Oversold condition
            reward = 10  # Positive reward for buying
        elif rsi > 70:  # Overbought condition
            reward = -10  # Negative reward for selling
        return reward

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward