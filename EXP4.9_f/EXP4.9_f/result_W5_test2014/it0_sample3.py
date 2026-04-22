import numpy as np

def revise_state(s):
    # s: 120d raw state representing 20 days of OHLCV
    
    # Feature 1: Price momentum (percentage change from two days ago to the most recent day)
    closing_prices = s[0::6]  # Extract closing prices
    price_momentum = (closing_prices[-1] - closing_prices[-3]) / closing_prices[-3] if closing_prices[-3] != 0 else 0

    # Feature 2: Volatility (rolling standard deviation of the last 5 closing prices)
    if len(closing_prices) >= 5:
        volatility = np.std(closing_prices[-5:])
    else:
        volatility = 0

    # Feature 3: Volume change (percentage change from the previous day)
    volumes = s[4::6]  # Extract trading volumes
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0

    # Return features as a 1D numpy array
    features = [price_momentum, volatility, volume_change]
    
    return np.array(features)
```

### Function 2: `intrinsic_reward(enhanced_state)`

This function will implement the reward logic based on the priority chain provided. We will read the regime vector and apply the risk management, trend following, mean reversion, and volatility rules accordingly.

Here's the implementation:

```python
def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = new features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Start with a neutral reward
    reward = 0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # STRONG NEGATIVE reward for BUY-aligned features
        reward -= 40  # Example strong negative
        return reward  # Exit early for risk level > 0.7
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= 20  # Example moderate negative
    
    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 10  # Positive reward for bullish signals
        elif trend_direction < -0.3:  # Downtrend
            reward += 10  # Positive reward for bearish signals
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features (oversold→buy, overbought→sell)
        reward += 5  # Reward for mean-reversion
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% in high volatility
    
    return reward