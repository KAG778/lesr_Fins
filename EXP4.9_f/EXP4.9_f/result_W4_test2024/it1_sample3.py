import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices for 20 days
    volumes = s[4:120:6]         # Trading volumes for 20 days

    # Feature 1: Price Change Percentage (last day closing to previous closing)
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0

    # Feature 2: Average Daily Volume Change (percentage change in volume)
    avg_volume_change = np.mean(np.diff(volumes) / volumes[:-1]) if len(volumes) > 1 and np.all(volumes[:-1] != 0) else 0

    # Feature 3: Average True Range (ATR) for volatility measurement
    high_low = np.max(closing_prices) - np.min(closing_prices)
    atr = high_low / len(closing_prices) if len(closing_prices) > 0 else 0

    # Feature 4: Recent Price Relative to 50-day Moving Average
    if len(closing_prices) >= 50:
        moving_avg_50 = np.mean(closing_prices[-50:])
        price_relative_to_ma50 = (closing_prices[-1] - moving_avg_50) / moving_avg_50
    else:
        price_relative_to_ma50 = 0

    features = [price_change_pct, avg_volume_change, atr, price_relative_to_ma50]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(enhanced_s[123:])  # Use features to calculate a standard deviation
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Initialize reward
    reward = 0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 40  # STRONG NEGATIVE reward for BUY-aligned features
    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        if trend_direction > 0:
            reward += 20  # Positive reward for buy aligned features in uptrend
        else:
            reward += 20  # Positive reward for sell aligned features in downtrend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]