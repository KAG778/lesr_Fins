import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    num_days = len(closing_prices)

    # Feature 1: 20-day Bollinger Bands (normalized)
    if num_days >= 20:
        sma = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = sma + (2 * std_dev)
        lower_band = sma - (2 * std_dev)
        bollinger_band = (closing_prices[-1] - lower_band) / (upper_band - lower_band)  # Normalize price
    else:
        bollinger_band = np.nan

    # Feature 2: 14-day Average True Range (ATR)
    if num_days >= 14:
        high_prices = s[2::6]
        low_prices = s[3::6]
        tr = np.maximum(high_prices[-14:] - low_prices[-14:],
                        np.maximum(np.abs(high_prices[-14:] - closing_prices[-15:-1]),
                                   np.abs(low_prices[-14:] - closing_prices[-15:-1])))
        atr = np.mean(tr)
    else:
        atr = np.nan

    # Feature 3: Rate of Change (ROC) over 14 days
    if num_days >= 14:
        roc = (closing_prices[-1] - closing_prices[-15]) / closing_prices[-15] * 100
    else:
        roc = np.nan

    # Feature 4: 14-day Momentum (current price - price 14 days ago)
    momentum = closing_prices[-1] - closing_prices[-15] if num_days >= 15 else 0

    # Compile features
    features = [bollinger_band, atr, roc, momentum]
    
    # Replace NaN with 0
    features = [f if np.isfinite(f) else 0 for f in features]

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate relative thresholds based on historical volatility
    historical_volatility = np.std(enhanced_s[123:]) if len(enhanced_s[123:]) > 0 else 1  # Prevent division by zero
    high_risk_threshold = 0.7 * historical_volatility
    low_risk_threshold = 0.3 * historical_volatility

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative for high risk
        reward += 10   # Mild positive for sell signals
    elif risk_level > low_risk_threshold:
        reward -= 20  # Moderate negative for buy signals

    # Priority 2 — TREND FOLLOWING
    if risk_level < low_risk_threshold and abs(trend_direction) > 0.3:
        reward += 20 * np.sign(trend_direction)  # Positive reward aligned with trend direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        reward += 10  # Reward for mean-reversion opportunities

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range