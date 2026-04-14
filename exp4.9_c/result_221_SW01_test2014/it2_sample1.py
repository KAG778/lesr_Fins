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
        price_bollinger = (closing_prices[-1] - lower_band) / (upper_band - lower_band)  # Normalize the price
    else:
        price_bollinger = 0  # Default value for insufficient data

    # Feature 2: 14-day Average True Range (ATR)
    if num_days >= 14:
        high_prices = s[2::6]
        low_prices = s[3::6]
        true_ranges = np.maximum(high_prices[-14:] - low_prices[-14:], 
                                 np.maximum(np.abs(high_prices[-14:] - closing_prices[-15:-1]),
                                            np.abs(low_prices[-14:] - closing_prices[-15:-1])))
        atr = np.mean(true_ranges)  
    else:
        atr = 0

    # Feature 3: Rate of Change (ROC) over the last 14 days
    if num_days >= 15:
        roc = (closing_prices[-1] - closing_prices[-15]) / closing_prices[-15] * 100
    else:
        roc = 0

    # Feature 4: Volume Weighted Average Price (VWAP)
    if num_days > 0:
        volumes = s[4::6]
        vwap = np.sum(closing_prices[-num_days] * volumes[-num_days]) / np.sum(volumes[-num_days])
    else:
        vwap = 0

    # Compile features
    features = [price_bollinger, atr, roc, vwap]
    
    # Ensure all features are valid numbers (replace NaN with 0)
    features = [f if np.isfinite(f) else 0 for f in features]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0
    
    # Calculate historical thresholds based on standard deviation of the last 30 days of features
    historical_volatility = np.std(enhanced_s[123:])  # Use computed features for historical volatility
    risk_threshold_high = 0.7 * historical_volatility
    risk_threshold_mid = 0.4 * historical_volatility
    trend_threshold_high = 0.3 * historical_volatility
    trend_threshold_low = -0.3 * historical_volatility

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative for high risk
    elif risk_level > risk_threshold_mid:
        reward -= 20  # Moderate negative for medium risk

    # Priority 2 — TREND FOLLOWING
    if risk_level < risk_threshold_mid:
        if abs(trend_direction) > trend_threshold_high:
            reward += 20 * np.sign(trend_direction)  # Positive reward for aligning with trend direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold_high and risk_level < risk_threshold_mid:
        z_score = (enhanced_s[123][0] - np.mean(enhanced_s[123][:14])) / np.std(enhanced_s[123][:14]) if np.std(enhanced_s[123][:14]) != 0 else 0
        if z_score < -1:  # Oversold condition
            reward += 15  # Reward for potential buy
        elif z_score > 1:  # Overbought condition
            reward -= 15  # Penalize for potential buy

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_mid:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range