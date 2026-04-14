import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes

    features = []

    # Feature 1: 14-day Average True Range (ATR) for volatility measurement
    def calculate_atr(high_prices, low_prices, period=14):
        true_ranges = np.maximum(high_prices[1:] - low_prices[1:],
                                  np.maximum(abs(high_prices[1:] - closing_prices[:-1]),
                                             abs(low_prices[1:] - closing_prices[:-1])))
        return np.mean(true_ranges[-period:]) if len(true_ranges) >= period else 0

    high_prices = s[2::6]
    low_prices = s[3::6]
    atr_value = calculate_atr(high_prices, low_prices)
    features.append(atr_value)

    # Feature 2: Recent Price Change Percentage (last day)
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if len(closing_prices) >= 2 and closing_prices[-2] != 0 else 0
    features.append(price_change_pct)

    # Feature 3: Volume Change (compared to the average of the last 5 days)
    avg_volume_last_5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0
    volume_change_pct = (volumes[-1] - avg_volume_last_5) / avg_volume_last_5 if avg_volume_last_5 != 0 else 0
    features.append(volume_change_pct)

    # Feature 4: Z-Score of Closing Prices (relative to the last 20 days)
    if len(closing_prices) >= 20:
        mean_price = np.mean(closing_prices[-20:])
        std_price = np.std(closing_prices[-20:])
        z_score = (closing_prices[-1] - mean_price) / std_price if std_price != 0 else 0
    else:
        z_score = 0
    features.append(z_score)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Calculate dynamic thresholds based on historical standard deviation
    historical_std = np.std(enhanced_s[123:]) if len(enhanced_s[123:]) > 0 else 1  # Avoid division by zero
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_moderate = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        reward += np.random.uniform(5, 10)     # Mild positive for SELL-aligned features

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_moderate:
        if trend_direction > 0:  # Uptrend
            reward += 20  # Positive reward for upward trend
        else:  # Downtrend
            reward += 20  # Positive reward for downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) <= trend_threshold and risk_level < 0.3:
        if enhanced_s[123] < -1:  # Oversold condition
            reward += 15  # Reward for buying in mean-reversion
        elif enhanced_s[123] > 1:  # Overbought condition
            reward += -15  # Penalize for selling in mean-reversion

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]