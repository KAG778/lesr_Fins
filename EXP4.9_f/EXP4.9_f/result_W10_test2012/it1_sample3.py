import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices (every 6th element starting from index 0)
    volumes = s[4::6]  # Extract trading volumes (every 6th element starting from index 4)

    # Feature 1: Price Momentum (latest closing price - closing price 10 days ago)
    price_momentum = closing_prices[0] - closing_prices[10] if len(closing_prices) > 10 else 0

    # Feature 2: Relative Strength Index (RSI) calculation over the last 14 days
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices[-14:])  # Get price changes
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / loss if loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Neutral RSI if insufficient data

    # Feature 3: Volume Change (percentage change from previous day)
    volume_change = (volumes[0] - volumes[1]) / volumes[1] * 100 if volumes[1] > 0 else 0

    # Feature 4: Standard Deviation of Closing Prices (last 14 days)
    if len(closing_prices) >= 14:
        price_std = np.std(closing_prices[-14:])
    else:
        price_std = 0

    # Feature 5: Average Trading Volume (last 14 days)
    if len(volumes) >= 14:
        avg_volume = np.mean(volumes[-14:])
    else:
        avg_volume = 0

    features = [price_momentum, rsi, volume_change, price_std, avg_volume]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate dynamic thresholds based on historical std
    risk_threshold_high = 0.7
    risk_threshold_moderate = 0.4

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        reward += np.random.uniform(5, 10)   # Mild positive for SELL-aligned features
    elif risk_level > risk_threshold_moderate:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_moderate:
        features = enhanced_s[123:]
        if trend_direction > 0.3 and features[0] > 0:  # Positive price momentum
            reward += np.random.uniform(15, 25)  # Reward upward momentum
        elif trend_direction < -0.3 and features[0] < 0:  # Negative price momentum
            reward += np.random.uniform(15, 25)  # Reward downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        features = enhanced_s[123:]
        if features[1] < 30:  # Oversold condition
            reward += np.random.uniform(10, 20)  # Reward for mean reversion
        elif features[1] > 70:  # Overbought condition
            reward += np.random.uniform(10, 20)  # Also reward for mean reversion

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))  # Clip reward to be within [-100, 100]