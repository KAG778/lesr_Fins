import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    N = len(closing_prices)

    # Feature 1: Exponential Moving Average (EMA) for trend detection
    ema_short = np.mean(closing_prices[-5:]) if N >= 5 else np.nan
    ema_long = np.mean(closing_prices[-20:]) if N >= 20 else np.nan
    ema_diff = ema_short - ema_long  # Momentum indicator

    # Feature 2: Average True Range (ATR) for volatility
    high = s[1::6]
    low = s[2::6]
    true_ranges = np.maximum(high[-1] - low[-1], np.abs(high[-1] - closing_prices[-2]), np.abs(low[-1] - closing_prices[-2]))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else np.nan  # 14-day ATR

    # Feature 3: Market Sentiment Index (MSI) based on volume changes
    volumes = s[4::6]
    volume_change = np.diff(volumes) / (volumes[:-1] + 1e-8)  # Avoid division by zero
    msi = np.mean(volume_change[-14:]) if len(volume_change) >= 14 else np.nan  # 14-day average

    # Combine features
    features = [ema_diff, atr, msi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate relative thresholds based on historical volatility
    historical_volatility = np.std(enhanced_s[123:])  # Calculate from features
    high_risk_threshold = 1.5 * historical_volatility
    low_risk_threshold = 0.5 * historical_volatility

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE for BUY
        reward += np.random.uniform(5, 10)    # MILD POSITIVE for SELL
        return max(-100, min(100, reward))  # Early return

    elif risk_level > low_risk_threshold:
        reward -= 10  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level <= low_risk_threshold:
        if trend_direction > 0.3:  # Uptrend
            reward += 20  # Positive reward for bullish momentum
        elif trend_direction < -0.3:  # Downtrend
            reward += 20  # Positive reward for bearish momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level <= low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    return max(-100, min(100, reward))