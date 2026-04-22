import numpy as np

def revise_state(s):
    # s: 120d raw state containing 20 days of OHLCV data
    closing_prices = s[0::6]  # Extract closing prices from the raw state
    
    # 1. Compute Price Momentum
    if len(closing_prices) > 2:
        price_momentum = (closing_prices[-1] - closing_prices[-3]) / closing_prices[-3]
    else:
        price_momentum = 0.0  # Handle edge case
    
    # 2. Compute Price Volatility (Standard deviation of closing prices)
    price_volatility = np.std(closing_prices)
    
    # 3. Compute Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0).mean()
    loss = np.abs(np.where(delta < 0, delta, 0)).mean()
    
    # Avoid division by zero
    if loss == 0:
        rsi = 100
    else:
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
    
    # Return the computed features
    features = [price_momentum, price_volatility, rsi]
    return np.array(features)
```

### Function 2: `intrinsic_reward(enhanced_state)`

In this function, we will implement the reward logic based on the priority chain outlined in the task. We will extract the regime information and compute the reward based on the conditions specified.

Here’s the implementation:

```python
def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Extract features for reward calculation (if needed)
    # features = enhanced_state[123:]
    
    # Initialize reward
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40  # Strong negative reward for BUY-aligned features
    elif risk_level > 0.4:
        reward += -20  # Moderate negative reward for BUY signals
    
    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 10  # Positive reward for upward features
        elif trend_direction < -0.3:
            reward += 10  # Positive reward for downward features
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward mean-reversion features
        # Assuming features would indicate overbought/oversold
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    return float(np.clip(reward, -100, 100))  # Clip reward to be within [-100, 100]