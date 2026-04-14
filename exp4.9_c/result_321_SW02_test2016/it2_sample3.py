import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices for the last 20 days
    closing_prices = s[0:120:6]
    volumes = s[4:120:6]
    
    # Feature 1: Z-score of Returns (20 days)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    mean_return = np.mean(daily_returns)
    std_return = np.std(daily_returns)
    z_score_returns = (daily_returns[-1] - mean_return) / std_return if std_return != 0 else 0
    features.append(z_score_returns)

    # Feature 2: Bollinger Bands (20-day period)
    if len(closing_prices) >= 20:
        moving_avg = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        current_position = (closing_prices[-1] - moving_avg) / rolling_std if rolling_std != 0 else 0
        features.append(current_position)
    else:
        features.append(0.0)  # Neutral position when not enough data

    # Feature 3: Average True Range (ATR)
    high_prices = s[1:120:6]  # High prices for 20 days
    low_prices = s[2:120:6]   # Low prices for 20 days
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:],
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]),
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # 14-day ATR
    features.append(atr)

    # Feature 4: Volume Momentum (current vs. historical average)
    avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else 0
    current_volume = volumes[-1] if len(volumes) > 0 else 0
    volume_momentum = (current_volume - avg_volume) / avg_volume if avg_volume > 0 else 0
    features.append(volume_momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Relative thresholds based on historical data (could be calculated dynamically)
    risk_threshold = np.std([0.7, 0.4, 0.3])  # Example calculation; adjust based on historical performance
    trend_threshold = 0.3  # Threshold for trend sensitivity, can also be adjusted based on historical data

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= 50  # Strong negative penalty for buying in high risk
        reward += 10 * (1 - risk_level)  # Mild positive for selling
        return np.clip(reward, -100, 100)  # Early exit to avoid further penalties

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < 0.4:
        if trend_direction > 0:
            reward += 25  # Strong positive for bullish alignment
        elif trend_direction < 0:
            reward += 25  # Strong positive for bearish alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]