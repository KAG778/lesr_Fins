import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Extract closing prices
    closing_prices = s[0::6]  # Closing prices correspond to indices 0, 6, 12, ..., 114 (20 days)
    
    # 1. Price Momentum (latest closing price - closing price 5 days ago)
    price_momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0
    
    # 2. Relative Strength Index (RSI)
    def compute_rsi(prices, period=14):
        if len(prices) < period:
            return 50  # Neutral RSI
        deltas = np.diff(prices)
        gain = np.mean(deltas[deltas > 0], axis=0) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0], axis=0) if np.any(deltas < 0) else 0
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_value = compute_rsi(closing_prices)
    
    # 3. Average Daily Volume
    trading_volume = s[4::6]  # Trading volume corresponds to indices 4, 10, 16, ..., 114 (20 days)
    avg_daily_volume = np.mean(trading_volume)
    
    # Return the computed features
    features = [price_momentum, rsi_value, avg_daily_volume]
    return np.array(features)
```

### Step 2: Implementing `intrinsic_reward(enhanced_state)`

Now, we'll implement the `intrinsic_reward` function based on the priority logic provided. We will read the regime vector and calculate the reward based on the defined priorities.

Here's how we can implement `intrinsic_reward(enhanced_state)`:

```python
def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Features from revised state
    features = enhanced_state[123:]
    price_momentum = features[0]
    rsi_value = features[1]
    avg_daily_volume = features[2]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        if price_momentum > 0:  # BUY aligned
            return -40  # STRONG NEGATIVE reward for BUY-aligned features
        else:  # SELL aligned
            return 8  # MILD POSITIVE reward for SELL-aligned features
    
    if risk_level > 0.4:
        if price_momentum > 0:  # BUY aligned
            reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2: Trend Following
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and price_momentum > 0:  # Upward trend and positive momentum
            reward += 15  # Positive reward
        elif trend_direction < -0.3 and price_momentum < 0:  # Downward trend and negative momentum
            reward += 15  # Positive reward

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if rsi_value < 30:  # Oversold condition
            reward += 10  # Positive reward for buying
        elif rsi_value > 70:  # Overbought condition
            reward += 10  # Positive reward for selling
        else:
            reward -= 5  # Penalize breakout-chasing features

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]