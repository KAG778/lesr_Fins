import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Feature 1: Price Momentum (Rate of Change)
    try:
        # Calculate momentum as the percentage change between the most recent closing price and the closing price from 5 days ago
        momentum = (s[114] - s[84]) / s[84]  # s[114] is the most recent close, s[84] is the close 5 days ago
        features.append(momentum)
    except ZeroDivisionError:
        features.append(0.0)  # If division by zero, assign 0

    # Feature 2: Average Trading Volume (last 5 days)
    avg_volume = np.mean(s[4::6][-5:])  # Last 5 volumes
    features.append(avg_volume)

    # Feature 3: Bollinger Bands % (Relative Position)
    try:
        # Calculate the rolling mean and standard deviation of the closing prices over the last 5 days
        closing_prices = s[0::6]
        mean_price = np.mean(closing_prices[-5:])
        std_dev = np.std(closing_prices[-5:])
        # Calculate Bollinger Bands % (position within the bands)
        if std_dev > 0:
            bb_percent = (closing_prices[-1] - (mean_price - 2 * std_dev)) / (4 * std_dev)  # Normalize
            features.append(bb_percent)
        else:
            features.append(0.0)  # If no variation, assign 0
    except Exception:
        features.append(0.0)  # Catch-all for any unexpected issues

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
        if enhanced_state[123] > 0:  # Assuming features[0] corresponds to a BUY signal
            return -40.0  # STRONG NEGATIVE reward for BUY
        else:
            return 7.0  # MILD POSITIVE reward for SELL
    elif risk_level > 0.4:
        if enhanced_state[123] > 0:  # Assuming features[0] corresponds to a BUY signal
            reward -= 20.0  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and enhanced_state[123] > 0:  # Assuming features[0] corresponds to a BUY signal
            reward += 20.0  # Positive reward for aligned trend
        elif trend_direction < 0 and enhanced_state[123] < 0:  # Assuming features[0] corresponds to a SELL signal
            reward += 20.0  # Positive reward for aligned trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_state[123] < 0:  # Assuming features[0] corresponds to a SELL signal (overbought)
            reward += 15.0  # Reward for mean-reversion SELL
        elif enhanced_state[123] > 0:  # Assuming features[0] corresponds to a BUY signal (oversold)
            reward += 15.0  # Reward for mean-reversion BUY

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)