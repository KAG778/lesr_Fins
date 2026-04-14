import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV)
    
    # Reshape the raw state to a 20 x 6 array for easier processing
    days = s.reshape((20, 6))
    
    # Feature 1: Price Momentum (last day closing price - closing price 5 days ago)
    price_momentum = days[-1, 0] - days[-6, 0] if len(days) > 5 else 0
    
    # Feature 2: RSI (14-day RSI calculation)
    gains = []
    losses = []
    for i in range(1, len(days)):
        change = days[i, 0] - days[i - 1, 0]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            losses.append(-change)
            gains.append(0)
    
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    # Feature 3: Volume Change (last day volume - volume 5 days ago)
    volume_change = days[-1, 4] - days[-6, 4] if len(days) > 5 else 0
    
    features = [price_momentum, rsi, volume_change]
    return np.array(features)
```

### Function 2: `intrinsic_reward(enhanced_state)`

This function will compute the reward based on the regime vector and the DQN's actions. We will follow the priority chain outlined in the task description. 

Here's how we'll implement the function:

1. Check the risk level first and apply the appropriate rewards or penalties.
2. If risk is low, check the trend direction and reward accordingly.
3. If the trend is sideways, reward mean-reversion strategies.
4. Modify the reward in high volatility conditions.

Here is the implementation:

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

    # Initialize reward
    reward = 0.0
    
    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:
        # STRONG NEGATIVE reward for BUY-aligned features
        reward += -40  # Example strong negative
        # MILD POSITIVE reward for SELL-aligned features
        reward += 8  # Example mild positive for sell
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -20  # Example moderate negative
        
    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            # Positive reward for upward features
            reward += 15 * features[0]  # Assuming price_momentum is at index 0
        else:
            # Positive reward for downward features
            reward += 15 * -features[0]  # Assuming price_momentum is at index 0

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 30:  # Assuming RSI is at index 1 and oversold
            reward += 10  # Reward for buying in oversold condition
        elif features[1] > 70:  # Overbought
            reward += -10  # Penalize for chasing breakouts

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    reward = max(-100, min(reward, 100))
    
    return reward