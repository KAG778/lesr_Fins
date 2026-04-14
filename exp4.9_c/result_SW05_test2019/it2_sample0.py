import numpy as np

def revise_state(s):
    # Extract closing prices and volumes from raw state
    closing_prices = s[0:120:6]  # every 6th element starting from index 0
    volumes = s[4:120:6]          # every 6th element starting from index 4 (volumes)

    features = []

    # 1. Z-Score of Price Momentum
    recent_momentum = closing_prices[0] - np.mean(closing_prices[1:6]) if len(closing_prices) > 6 else 0
    price_momentum_history = closing_prices[-20:]  # Last 20 closing prices for historical context
    historical_mean = np.mean(price_momentum_history)
    historical_std = np.std(price_momentum_history)
    z_momentum = (recent_momentum - historical_mean) / historical_std if historical_std > 0 else 0
    features.append(z_momentum)

    # 2. Z-Score of Volume
    current_volume = volumes[0] if len(volumes) > 0 else 1
    average_volume = np.mean(volumes[-20:]) if len(volumes) > 20 else current_volume
    volume_std = np.std(volumes[-20:]) if len(volumes) > 20 else 1
    z_volume = (current_volume - average_volume) / volume_std if volume_std > 0 else 0
    features.append(z_volume)

    # 3. Average True Range (ATR) for Volatility
    true_ranges = []
    for i in range(1, len(closing_prices)):
        high = s[2 + i * 6]  # high_prices
        low = s[3 + i * 6]   # low_prices
        close_prev = closing_prices[i-1]
        true_ranges.append(max(high - low, abs(high - close_prev), abs(low - close_prev)))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) > 14 else 0
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical thresholds for risk and trend
    historical_std = np.std(enhanced_s[0:120])  # Using the raw state for std calculation
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_moderate = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY
        reward += 10 * (1 - risk_level)  # Mild positive reward for SELL based on lower risk
    elif risk_level > risk_threshold_moderate:
        reward -= 10  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_moderate:
        if trend_direction > 0:  # Uptrend
            reward += 20  # Positive reward for upward momentum
        else:  # Downtrend
            reward += 20  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) <= trend_threshold and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds