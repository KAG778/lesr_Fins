import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes

    # Feature 1: Moving Average (last 5 days)
    moving_average = np.mean(closing_prices[-5:])

    # Feature 2: Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    average_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    average_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    rs = average_gain / average_loss if average_loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))  # RSI formula

    # Feature 3: Volume Change (percentage change)
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0

    # Compile features into an array
    features = [moving_average, rsi, volume_change]
    
    return np.array(features)
```

### Function 2: `intrinsic_reward(enhanced_state)`

In the `intrinsic_reward` function, we will implement the reward logic based on the regime information and the computed features. 

1. **Risk Management**: Check the `risk_level` first and reward/penalize accordingly.
2. **Trend Following**: If the market is trending, reward correct alignment with the trend.
3. **Mean Reversion**: For sideways markets, reward mean-reversion signals.
4. **High Volatility**: Adjust rewards based on volatility level.

Let's implement the function:

```python
def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Extract features for reward calculation
    moving_average = enhanced_s[123]
    rsi = enhanced_s[124]
    volume_change = enhanced_s[125]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # STRONG NEGATIVE reward for BUY-aligned features
        if moving_average > enhanced_s[0]:  # Assuming BUY if moving average suggests upward trend
            reward = np.random.uniform(-50, -30)  # Strong negative for risky buy
        else:
            reward = np.random.uniform(5, 10)  # Mild positive for sell
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if moving_average > enhanced_s[0]:
            reward = -10  # Penalize buy in elevated risk

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            if moving_average > enhanced_s[0]:
                reward += 10  # Reward for correctly betting on upward trend
        elif trend_direction < -0.3:
            if moving_average < enhanced_s[0]:
                reward += 10  # Reward for correctly betting on downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if rsi < 30:  # Oversold condition
            reward += 10  # Reward buying
        elif rsi > 70:  # Overbought condition
            reward += 10  # Reward selling

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward