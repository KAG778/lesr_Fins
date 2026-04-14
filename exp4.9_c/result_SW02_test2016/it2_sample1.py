import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract volumes

    if len(closing_prices) < 14 or len(volumes) < 14:
        return np.zeros(4)  # Return zeros if there are not enough data points

    # Feature 1: Volatility (Standard Deviation of closing prices over the last 14 days)
    volatility = np.std(closing_prices[-14:])

    # Feature 2: Volume Change Percentage
    recent_volume = volumes[-1]
    previous_volume = volumes[-2]
    volume_change_percentage = ((recent_volume - previous_volume) / previous_volume) * 100 if previous_volume != 0 else 0

    # Feature 3: Price Distance from Upper Bollinger Band
    sma = np.mean(closing_prices[-20:])  # 20-day SMA
    std_dev = np.std(closing_prices[-20:])  # 20-day standard deviation
    upper_band = sma + (2 * std_dev)
    price_distance_to_upper_band = (closing_prices[-1] - upper_band) / std_dev if std_dev != 0 else 0

    # Feature 4: Rate of Change (RoC) for the last 14 days
    roc = (closing_prices[-1] - closing_prices[-15]) / closing_prices[-15] * 100 if len(closing_prices) >= 15 and closing_prices[-15] != 0 else 0

    features = [volatility, volume_change_percentage, price_distance_to_upper_band, roc]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate relative thresholds based on historical analysis of risk level
    historical_risk = np.std(enhanced_s[123:])  # Assuming features are in the context of risk
    risk_threshold = historical_risk * 0.7  # Example relative threshold

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= 50  # Strong negative for BUY signals
        reward += 10   # Mild positive for SELL signals
    elif risk_level > (risk_threshold / 2):
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < (risk_threshold / 2):
        reward += 30 if trend_direction > 0 else 25  # Positive reward for correct trend following

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < (risk_threshold / 2):
        reward += 20  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < (risk_threshold / 2):
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    return np.clip(reward, -100, 100)