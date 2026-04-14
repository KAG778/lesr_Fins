import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices (indices 0, 6, 12, ..., 114)
    opening_prices = s[1::6]  # Extract opening prices (indices 1, 7, 13, ..., 115)
    high_prices = s[2::6]     # Extract high prices (indices 2, 8, 14, ..., 116)
    low_prices = s[3::6]      # Extract low prices (indices 3, 9, 15, ..., 117)
    volumes = s[4::6]         # Extract trading volumes (indices 4, 10, 16, ..., 118)

    # Feature 1: Recent Price Change (percentage change from opening price to closing price)
    recent_price_change = np.nan_to_num((closing_prices[-1] - opening_prices[-1]) / opening_prices[-1])
    
    # Feature 2: 14-day Relative Strength Index (RSI)
    delta = closing_prices[1:] - closing_prices[:-1]
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else np.mean(gain)
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else np.mean(loss)
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    # Feature 3: Recent Volatility (standard deviation of closing prices over the last 14 days)
    recent_volatility = np.std(closing_prices[-14:]) if len(closing_prices) >= 14 else np.std(closing_prices)

    # Return the computed features as a numpy array
    features = [recent_price_change, rsi, recent_volatility]
    return np.array(features)

```

### Function 2: `intrinsic_reward(enhanced_state)`

In this function, we will implement the reward logic based on the defined priority chain.

```python
def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Extract features
    recent_price_change = enhanced_s[123][0]
    rsi = enhanced_s[123][1]
    recent_volatility = enhanced_s[123][2]

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40  # STRONG NEGATIVE reward for BUY-aligned features
        if recent_price_change < 0:  # SELL-aligned features
            reward += 7  # MILD POSITIVE reward
    elif risk_level > 0.4:
        reward += -10  # moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and recent_price_change > 0:
            reward += 10  # Positive reward for upward features
        elif trend_direction < -0.3 and recent_price_change < 0:
            reward += 10  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if rsi < 30:  # Oversold
            reward += 10  # Reward for buying
        elif rsi > 70:  # Overbought
            reward += -10  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward stays within [-100, 100]
    return np.clip(reward, -100, 100)