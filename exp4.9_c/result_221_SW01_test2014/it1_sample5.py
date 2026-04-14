import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    num_days = len(closing_prices)
    
    # Feature 1: Bollinger Bands (20-day)
    if num_days >= 20:
        sma = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = sma + (2 * std_dev)
        lower_band = sma - (2 * std_dev)
        price_bollinger = (closing_prices[-1] - lower_band) / (upper_band - lower_band)  # Normalize the price
    else:
        price_bollinger = np.nan

    # Feature 2: Average True Range (ATR, 14-day)
    if num_days >= 14:
        high_prices = s[2::6]
        low_prices = s[3::6]
        true_ranges = np.maximum(high_prices[-14:] - low_prices[-14:], 
                                 np.abs(high_prices[-14:] - closing_prices[-15:-1]),
                                 np.abs(low_prices[-14:] - closing_prices[-15:-1]))
        atr = np.mean(true_ranges)  
    else:
        atr = np.nan

    # Feature 3: Volume Weighted Average Price (VWAP)
    if num_days > 0:
        volumes = s[4::6]
        vwap = np.sum(closing_prices[-num_days] * volumes[-num_days]) / np.sum(volumes[-num_days])
    else:
        vwap = np.nan

    # Feature 4: 20-day Exponential Moving Average (EMA)
    if num_days >= 20:
        ema = np.array([closing_prices[0]])  # Start with first price
        for price in closing_prices[1:20]:
            ema = np.append(ema, (price * (2/(20 + 1))) + (ema[-1] * (1 - (2/(20 + 1)))))
        ema_value = ema[-1]
    else:
        ema_value = np.nan

    # Return only the computed features, filtering out NaN values
    features = [price_bollinger, atr, vwap, ema_value]
    
    # Ensure all features are valid numbers (replace NaN with 0)
    features = [f if np.isfinite(f) else 0 for f in features]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Use historical thresholds to define levels
    risk_threshold_high = 0.7
    risk_threshold_mid = 0.4
    trend_threshold_high = 0.3
    trend_threshold_low = -0.3
    volatility_threshold = 0.6

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative for high risk
    elif risk_level > risk_threshold_mid:
        reward -= 20  # Moderate negative for medium risk

    # Priority 2 — TREND FOLLOWING
    if risk_level < risk_threshold_mid:
        if abs(trend_direction) > trend_threshold_high:
            reward += 20 * np.sign(trend_direction)  # Positive reward aligned with trend direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold_high and risk_level < risk_threshold_mid:
        reward += 15  # Reward for mean-reversion opportunities

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > volatility_threshold and risk_level < risk_threshold_mid:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range