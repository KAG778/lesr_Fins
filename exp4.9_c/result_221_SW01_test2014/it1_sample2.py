import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extracting closing prices
    num_days = len(closing_prices)
    
    # Feature 1: Bollinger Bands (20-day SMA + 2 Std Dev)
    if num_days >= 20:
        sma = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = sma + 2 * std_dev
        lower_band = sma - 2 * std_dev
        bollinger_band = (closing_prices[-1] - lower_band) / (upper_band - lower_band)  # Normalized position
    else:
        bollinger_band = np.nan  # Handle edge case

    # Feature 2: Average True Range (ATR)
    if num_days >= 14:
        high_prices = s[2::6]
        low_prices = s[3::6]
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]),
                                   np.abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-14:])  # 14-day ATR
    else:
        atr = np.nan

    # Feature 3: Trend Strength Indicator (TSI)
    if num_days >= 10:
        price_change = np.diff(closing_prices[-10:])
        trend_strength = np.sum(price_change) / np.std(price_change) if np.std(price_change) != 0 else 0
    else:
        trend_strength = np.nan

    # Return only the computed features, filtering out NaN values
    features = [bollinger_band, atr, trend_strength]
    features = [f if np.isfinite(f) else 0 for f in features]  # Replace NaN with 0
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Calculate thresholds based on historical volatility
    historical_volatility = np.std(enhanced_s[123:])  # Use computed features for historical volatility
    high_risk_threshold = 0.7 * historical_volatility
    low_risk_threshold = 0.3 * historical_volatility

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative for BUY-aligned features
    elif risk_level > low_risk_threshold:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0:
            reward += 20  # Positive reward for bullish features
        elif trend_direction < 0:
            reward += 20  # Positive reward for bearish features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        reward += 10  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range