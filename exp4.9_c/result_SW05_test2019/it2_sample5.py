import numpy as np

def revise_state(s):
    # Extract closing prices and volumes from raw state
    closing_prices = s[0:120:6]  # every 6th element starting from index 0
    volumes = s[4:120:6]          # every 6th element starting from index 4 (volumes)

    features = []

    # 1. Z-Score of Price Momentum (current close vs historical mean)
    if len(closing_prices) > 5:
        recent_momentum = closing_prices[0] - closing_prices[5]
        historical_mean = np.mean(closing_prices[-20:])
        historical_std = np.std(closing_prices[-20:])
        z_momentum = (recent_momentum - historical_mean) / historical_std if historical_std > 0 else 0
    else:
        z_momentum = 0
    features.append(z_momentum)

    # 2. Z-Score of Volume (current volume vs historical mean)
    if len(volumes) > 5:
        current_volume = volumes[0]
        historical_volume_mean = np.mean(volumes[-20:])
        historical_volume_std = np.std(volumes[-20:])
        z_volume = (current_volume - historical_volume_mean) / historical_volume_std if historical_volume_std > 0 else 0
    else:
        z_volume = 0
    features.append(z_volume)

    # 3. Directional Movement Index (DMI) for trend strength
    if len(closing_prices) > 14:
        highs = s[2:120:6]
        lows = s[3:120:6]
        up_moves = np.maximum(highs[1:] - highs[:-1], 0)
        down_moves = np.maximum(lows[:-1] - lows[1:], 0)
        plus_dm = np.sum(up_moves)
        minus_dm = np.sum(down_moves)
        tr = np.maximum(highs[1:] - lows[1:], np.maximum(highs[1:] - closing_prices[:-1], closing_prices[:-1] - lows[1:]))
        atr = np.mean(tr[-14:]) if len(tr) > 14 else 0
        dmi = (plus_dm - minus_dm) / atr if atr != 0 else 0
    else:
        dmi = 0
    features.append(dmi)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate relative thresholds based on historical std deviation
    historical_std = np.std(enhanced_s[0:120])  # Use raw state for std calculation
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(40, 60)  # Strong negative for BUY
        reward += 10  # Mild positive for SELL
    elif risk_level > risk_threshold_medium:
        reward -= 10  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        if trend_direction > 0:  # Uptrend
            reward += np.random.uniform(10, 30)  # Positive reward for momentum alignment
        else:  # Downtrend
            reward += np.random.uniform(10, 30)  # Positive reward for momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) <= trend_threshold and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds