import numpy as np

def revise_state(raw_state):
    # s: 120d raw state
    closing_prices = raw_state[0::6]  # Extract closing prices
    recent_closing_price = closing_prices[-1]
    
    # Feature 1: Price Momentum (current price - price 5 days ago)
    if len(closing_prices) > 5:
        price_momentum = recent_closing_price - closing_prices[-6]
    else:
        price_momentum = np.nan  # Handle edge case
    
    # Feature 2: Volatility (standard deviation of the last 20 days)
    volatility = np.std(closing_prices)
    
    # Feature 3: Relative Strength Index (RSI) - calculating over the last 14 days
    gain = np.where(np.diff(closing_prices) > 0, np.diff(closing_prices), 0)
    loss = np.where(np.diff(closing_prices) < 0, -np.diff(closing_prices), 0)
    
    if len(gain) > 14 and len(loss) > 14:
        avg_gain = np.mean(gain[-14:])
        avg_loss = np.mean(loss[-14:])
        rs = avg_gain / avg_loss if avg_loss != 0 else np.nan  # Handle division by zero
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = np.nan  # Handle edge case

    features = [price_momentum, volatility, rsi]
    return np.array(features)
```

### Function 2: `intrinsic_reward(enhanced_state)`

This function will implement the reward logic based on the regime vector provided. The priority chain for the reward logic will be followed strictly.

Here's how we can implement this function:

```python
def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE for BUY-aligned
        reward += np.random.uniform(5, 10)   # MILD POSITIVE for SELL-aligned
    elif risk_level > 0.4:
        reward -= np.random.uniform(5, 15)    # Moderate negative for BUY signals
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        # Assuming features are defined as [price_momentum, volatility, rsi]
        features = enhanced_state[123:]
        if trend_direction > 0.3:  # Uptrend
            if features[0] > 0:  # Positive momentum
                reward += 10  # Positive reward for aligning with trend
        elif trend_direction < -0.3:  # Downtrend
            if features[0] < 0:  # Negative momentum
                reward += 10  # Positive reward for aligning with trend
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        features = enhanced_state[123:]
        if features[2] < 30:  # Oversold (RSI < 30)
            reward += 10  # Positive reward to BUY
        elif features[2] > 70:  # Overbought (RSI > 70)
            reward += 10  # Positive reward to SELL
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure the reward is within the specified range
    return np.clip(reward, -100, 100)