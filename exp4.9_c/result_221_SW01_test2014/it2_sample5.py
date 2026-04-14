import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    num_days = len(closing_prices)

    # Feature 1: 20-day Exponential Moving Average (EMA)
    if num_days >= 20:
        ema = np.zeros(num_days)
        ema[0] = closing_prices[0]  # Start with the first price
        for i in range(1, num_days):
            ema[i] = (closing_prices[i] * (2 / (20 + 1))) + (ema[i - 1] * (1 - (2 / (20 + 1))))
        ema_value = ema[-1]
    else:
        ema_value = np.nan

    # Feature 2: Average True Range (ATR) for volatility measurement
    if num_days >= 14:
        high_prices = s[2::6]
        low_prices = s[3::6]
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                   np.abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.nan
    else:
        atr = np.nan

    # Feature 3: Rolling Z-score of recent returns for mean reversion
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] if num_days > 1 else np.array([np.nan])
    if len(daily_returns) >= 14:
        z_score = (daily_returns[-1] - np.mean(daily_returns[-14:])) / np.std(daily_returns[-14:]) if np.std(daily_returns[-14:]) != 0 else 0
    else:
        z_score = np.nan

    # Feature 4: Trend Strength Indicator (TSI) based on recent price changes
    if num_days >= 10:
        price_change = np.diff(closing_prices[-10:])
        trend_strength = np.sum(price_change) / np.std(price_change) if np.std(price_change) != 0 else 0
    else:
        trend_strength = np.nan

    # Compile features
    features = [ema_value, atr, z_score, trend_strength]
    features = [f if np.isfinite(f) else 0 for f in features]  # Replace NaN with 0
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical thresholds based on past data
    historical_volatility = np.std(enhanced_s[123:])  # Use computed features for historical volatility
    high_risk_threshold = 0.7 * historical_volatility
    low_risk_threshold = 0.4 * historical_volatility

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative for BUY-aligned features
        reward += 10   # Mild positive for SELL-aligned features
    elif risk_level > low_risk_threshold:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    if risk_level < low_risk_threshold:
        if abs(trend_direction) > 0.3:
            reward += 20 * np.sign(trend_direction)  # Positive reward aligned with trend direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        z_score = enhanced_s[123][2]  # Using the z-score from revised features
        if z_score < -1:  # Oversold condition
            reward += 15  # Reward for potential buy
        elif z_score > 1:  # Overbought condition
            reward -= 15  # Penalize for potential buy

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range