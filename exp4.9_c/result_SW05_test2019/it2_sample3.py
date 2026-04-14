import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes

    features = []

    # 1. Price Momentum (current close - average close of the last 5 days)
    if len(closing_prices) > 5:
        momentum = closing_prices[0] - np.mean(closing_prices[1:6])
    else:
        momentum = 0
    features.append(momentum)

    # 2. Average True Range (ATR) for volatility measurement
    true_ranges = np.maximum(closing_prices[1:] - closing_prices[:-1], 
                              np.maximum(closing_prices[1:] - closing_prices[:-1], 
                                         closing_prices[:-1] - closing_prices[1:]))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) > 13 else 0
    features.append(atr)

    # 3. Z-score of Volume (how current volume compares to historical average)
    historical_volumes = volumes[-20:]  # Historical context for volumes
    avg_volume = np.mean(historical_volumes) if len(historical_volumes) > 0 else 1  # Avoid division by zero
    current_volume = volumes[0]  # Current volume
    z_score_volume = (current_volume - avg_volume) / np.std(historical_volumes) if np.std(historical_volumes) > 0 else 0
    features.append(z_score_volume)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate thresholds based on historical standard deviations
    historical_std = np.std(enhanced_s[0:120])  # Using raw state for std calculation
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY
        reward += np.random.uniform(0, 10)    # Mild positive reward for SELL
    elif risk_level > risk_threshold_medium:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        if trend_direction > 0:  # Uptrend
            reward += np.random.uniform(10, 30)  # Positive reward for upward momentum
        else:  # Downtrend
            reward += np.random.uniform(10, 30)  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) <= trend_threshold and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds