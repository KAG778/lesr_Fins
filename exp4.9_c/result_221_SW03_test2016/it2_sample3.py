import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract volumes

    # Feature 1: Exponential Moving Average (EMA) - 10 Days
    ema_10 = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else np.nan

    # Feature 2: Price Momentum (closing price difference with previous closing price)
    momentum = closing_prices[-1] - closing_prices[-2] if len(closing_prices) >= 2 else 0.0

    # Feature 3: Average True Range (ATR) for volatility
    if len(closing_prices) >= 14:
        high_prices = s[2::6]  # Extract high prices
        low_prices = s[3::6]   # Extract low prices
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                   np.abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-14:])  # 14-day ATR
    else:
        atr = 0.0
    
    # Feature 4: Mean Reversion Z-Score
    if len(closing_prices) >= 20:
        mean_price = np.mean(closing_prices[-20:])
        std_price = np.std(closing_prices[-20:])
        z_score = (closing_prices[-1] - mean_price) / std_price if std_price > 0 else 0.0
    else:
        z_score = 0.0

    # Feature 5: Volume Change Percentage (compared to average volume over last 10 days)
    avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else 1e-10
    volume_change_pct = ((volumes[-1] - avg_volume) / avg_volume) * 100

    # Return only new features
    return np.array([ema_10, momentum, atr, z_score, volume_change_pct])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate relative thresholds based on historical std
    historical_std = np.std(enhanced_s[123:])  # Using historical std of features for thresholds
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_low = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative reward for BUY-aligned features
        reward += 10   # Mild positive reward for SELL-aligned features

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_low:
        if trend_direction > trend_threshold:
            reward += 20  # Positive reward for upward momentum
        elif trend_direction < -trend_threshold:
            reward += 20  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < risk_threshold_low:
        z_score = enhanced_s[123]  # Z-Score as a feature
        if z_score < -1:  # Oversold condition
            reward += 10  # Reward potential buy
        elif z_score > 1:  # Overbought condition
            reward += 10  # Reward potential sell

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_std and risk_level < risk_threshold_low:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the bounds of [-100, 100]
    return float(np.clip(reward, -100, 100))