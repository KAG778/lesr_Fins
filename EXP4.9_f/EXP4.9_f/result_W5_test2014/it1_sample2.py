import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    trading_volumes = s[4::6]  # Extract trading volumes

    # Feature 1: Bollinger Bands (20-day moving average and standard deviation)
    if len(closing_prices) > 20:
        moving_avg = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = moving_avg + (2 * std_dev)
        lower_band = moving_avg - (2 * std_dev)
        price_pos = (closing_prices[-1] - lower_band) / (upper_band - lower_band)  # Normalized position between bands
    else:
        price_pos = 0.0  # Default when insufficient data

    # Feature 2: Average True Range (ATR) for volatility measurement
    if len(closing_prices) > 1:
        high_low = np.abs(s[1::6] - s[2::6])  # High - Low
        high_close = np.abs(s[1::6] - closing_prices[:-1])  # High - Previous Close
        low_close = np.abs(s[2::6] - closing_prices[:-1])  # Low - Previous Close
        true_ranges = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = np.mean(true_ranges[-14:]) if len(true_ranges) > 14 else np.mean(true_ranges)  # 14-period ATR
    else:
        atr = 0.0  # Default when insufficient data

    # Feature 3: Distance from 10-day Moving Average
    if len(closing_prices) > 10:
        distance_from_ma = (closing_prices[-1] - np.mean(closing_prices[-10:])) / np.mean(closing_prices[-10:])
    else:
        distance_from_ma = 0.0  # Default when insufficient data

    features = [price_pos, atr, distance_from_ma]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0

    # Calculate relative thresholds using historical std deviation
    risk_threshold = 0.7 * np.std(features)  # relative risk threshold based on features
    trend_threshold = 0.3 * np.std(features)  # relative trend threshold

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= 40 if features[0] > 0 else 10  # Strong negative for BUY-aligned features, mild positive for SELL-aligned features

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold:
        if trend_direction > trend_threshold:
            reward += 10 * features[0]  # Reward aligned momentum for upward trend
        elif trend_direction < -trend_threshold:
            reward += 10 * -features[0]  # Reward aligned momentum for downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < risk_threshold:
        if features[0] < -0.05:  # Oversold condition
            reward += 10  # Reward for buying
        elif features[0] > 0.05:  # Overbought condition
            reward += 10  # Reward for selling

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Clip reward to be within [-100, 100]