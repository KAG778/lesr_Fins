import numpy as np

def revise_state(s):
    features = []
    
    # Extracting closing prices and volumes
    closing_prices = s[0::6]
    volumes = s[4::6]

    # Feature 1: Average True Range (ATR) for volatility measurement
    if len(closing_prices) >= 14:
        high_prices = s[2::6]
        low_prices = s[3::6]
        tr = np.maximum(high_prices[1:], closing_prices[1:-1]) - np.minimum(low_prices[1:], closing_prices[1:-1])
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0
        features.append(atr)
    else:
        features.append(0)  # Default to 0 if not enough data

    # Feature 2: Price Momentum (current closing - closing 5 days ago)
    if len(closing_prices) >= 6:
        momentum = closing_prices[-1] - closing_prices[-6]
        features.append(momentum)
    else:
        features.append(0)

    # Feature 3: Volume Change (current vs previous day)
    if len(volumes) >= 2 and volumes[-2] != 0:
        volume_change_pct = (volumes[-1] - volumes[-2]) / volumes[-2]
    else:
        volume_change_pct = 0
    features.append(volume_change_pct)

    # Feature 4: RSI (Relative Strength Index)
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices[-14:])
        gains = np.where(deltas > 0, deltas, 0).mean()
        losses = -np.where(deltas < 0, deltas, 0).mean()
        rs = gains / losses if losses > 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Default RSI if not enough data
    features.append(rsi)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0
    
    # Adjusting thresholds based on historical data
    risk_threshold_high = 0.7
    risk_threshold_medium = 0.4
    trend_threshold = 0.3
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        # Strong negative reward for BUY-aligned features
        reward -= np.random.uniform(30, 50) if enhanced_s[123] > 0 else np.random.uniform(5, 10)
    elif risk_level > risk_threshold_medium:
        # Moderate negative reward for BUY signals
        reward -= 15 if enhanced_s[123] > 0 else 0

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        if trend_direction > trend_threshold:
            reward += 20  # Positive reward for upward trend
        elif trend_direction < -trend_threshold:
            reward += 20  # Positive reward for downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if enhanced_s[123] < 0:  # Assuming negative features indicate selling
            reward += 15  # Reward mean-reversion features
        else:
            reward -= 10  # Penalizing breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    return float(np.clip(reward, -100, 100))