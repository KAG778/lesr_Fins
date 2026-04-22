import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Feature 1: Price momentum (current close vs previous close)
    price_momentum = s[6] - s[0]  # Current close (day 19) - Close 19 days ago (day 0)

    # Feature 2: Average volume over the last 20 days
    avg_volume = np.mean(s[4::6])  # Average of the volume entries
    # Avoid division by zero in next feature calculation
    avg_volume = avg_volume if avg_volume != 0 else 1e-6

    # Feature 3: Relative strength index (RSI) (simplified version)
    # Calculate RSI based on closing prices
    gains = np.maximum(0, np.diff(s[0::6]))  # Daily gains
    losses = np.abs(np.minimum(0, np.diff(s[0::6])))  # Daily losses
    avg_gain = np.mean(gains[-14:])  # Average gain over last 14 days
    avg_loss = np.mean(losses[-14:])  # Average loss over last 14 days
    rs = avg_gain / avg_loss if avg_loss != 0 else 1e-6  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    features = [price_momentum, avg_volume, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_state[123:]
    price_momentum = features[0]
    avg_volume = features[1]
    rsi = features[2]

    # Initialize reward variable
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if price_momentum > 0:
            reward = np.random.uniform(-50, -30)  # Strong negative for BUY-aligned features
        else:
            reward = np.random.uniform(5, 10)  # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        if price_momentum > 0:
            reward = -10  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and price_momentum > 0:
            reward += 10  # Positive reward for bullish trend and positive momentum
        elif trend_direction < 0 and price_momentum < 0:
            reward += 10  # Positive reward for bearish trend and negative momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if rsi < 30:  # Oversold
            reward += 10  # Positive reward for potential buy
        elif rsi > 70:  # Overbought
            reward += 10  # Positive reward for potential sell
        else:
            reward -= 5  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward stays within bounds
    return np.clip(reward, -100, 100)