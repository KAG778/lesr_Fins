import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    closing_prices = s[::6]  # Extract closing prices (every 6th element starting from index 0)
    volumes = s[4::6]  # Extract trading volumes (every 6th element starting from index 4)

    # 1. Price Momentum (latest closing price - closing price 10 days ago)
    price_momentum = closing_prices[0] - closing_prices[10] if len(closing_prices) > 10 else 0

    # 2. Relative Strength Index (RSI) calculation
    # Calculate the RSI over the last 14 days (indices 6 to 19)
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices[-14:])  # Get the price changes
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / loss if loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Neutral RSI if we don't have enough data

    # 3. Volume Change (percentage change from previous day)
    volume_change = (volumes[0] - volumes[1]) / volumes[1] * 100 if volumes[1] > 0 else 0

    features = [price_momentum, rsi, volume_change]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 10  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += 10  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Assuming we have some features indicating oversold/overbought conditions
        # Here we could implement checks for RSI or other features
        reward += 5  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)