import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    num_days = len(closing_prices)

    # Feature 1: 20-day Bollinger Bands normalized value
    if num_days >= 20:
        sma = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = sma + (2 * std_dev)
        lower_band = sma - (2 * std_dev)
        price_bollinger = (closing_prices[-1] - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) != 0 else 0
    else:
        price_bollinger = 0

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

    # Feature 3: 14-day Rate of Change (ROC)
    if num_days >= 14:
        roc = (closing_prices[-1] - closing_prices[-15]) / closing_prices[-15] * 100
    else:
        roc = 0

    # Feature 4: Z-score of recent daily returns for mean reversion
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] * 100 if len(closing_prices) > 1 else np.array([0])
    z_score = (daily_returns[-1] - np.mean(daily_returns[-14:])) / np.std(daily_returns[-14:]) if len(daily_returns) >= 14 and np.std(daily_returns[-14:]) != 0 else 0

    # Compile features
    features = [price_bollinger, atr, roc, z_score]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical thresholds
    historical_volatility = np.std(enhanced_s[123:])  # Use computed features for historical volatility
    high_risk_threshold = 0.7 * historical_volatility
    low_risk_threshold = 0.3 * historical_volatility

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative for high risk (buy aligned)
        reward += 10   # Mild positive for sell aligned
    elif risk_level > low_risk_threshold:
        reward -= 20  # Moderate negative for buy signals

    # Priority 2 — TREND FOLLOWING
    if risk_level < low_risk_threshold:
        if abs(trend_direction) > 0.3:
            reward += 20 * np.sign(trend_direction)  # Positive reward aligned with trend direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        z_score = enhanced_s[123][3]  # Using the z-score from revised features
        if z_score < -1:  # Oversold condition
            reward += 15  # Reward for potential buy
        elif z_score > 1:  # Overbought condition
            reward -= 15  # Penalize for potential buy

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range