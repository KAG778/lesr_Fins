import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    
    closing_prices = s[0::6]  # Extract closing prices (every 6th element starting from index 0)
    
    # Feature 1: Simple Moving Average (SMA) over last 5 days
    sma_5 = np.mean(closing_prices[-5:]) if len(closing_prices[-5:]) > 0 else np.nan
    
    # Feature 2: Relative Strength Index (RSI) over last 14 days
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.mean(gain[-14:]) if len(gain[-14:]) > 0 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss[-14:]) > 0 else 0
    
    rsi = 100 - (100 / (1 + (avg_gain / avg_loss))) if avg_loss != 0 else 100  # avoid division by zero
    
    # Feature 3: Price Momentum (Current Price - Price 5 days ago)
    momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else np.nan
    
    # Handle edge cases (e.g. NA values)
    features = [sma_5, rsi, momentum]
    return np.array(features)
```

### Function 2: `intrinsic_reward(enhanced_state)`

In this function, we will implement the reward logic following the priority chain. We will evaluate the risk level, trend direction, and volatility level from the regime vector and assign rewards accordingly.

```python
def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_state[123:]
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # Assuming positive SMA indicates buy signal
            reward -= np.random.uniform(30, 50)
        # Mild positive reward for SELL-aligned features
        if features[0] < 0:  # Assuming negative SMA indicates sell signal
            reward += np.random.uniform(5, 10)
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:
            reward -= np.random.uniform(10, 20)
    
    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Uptrend and bullish feature
            reward += np.random.uniform(10, 20)
        elif trend_direction < -0.3 and features[0] < 0:  # Downtrend and bearish feature
            reward += np.random.uniform(10, 20)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 30:  # Oversold condition
            reward += np.random.uniform(5, 15)
        elif features[1] > 70:  # Overbought condition
            reward += np.random.uniform(5, 15)
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    return np.clip(reward, -100, 100)  # Ensure reward stays within bounds