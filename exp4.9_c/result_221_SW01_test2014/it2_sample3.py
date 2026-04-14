import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    num_days = len(closing_prices)

    features = []

    # Feature 1: 20-day Exponential Moving Average (EMA)
    if num_days >= 20:
        ema = np.mean(closing_prices[-20:])
    else:
        ema = 0  # Default to 0 if not enough data
    features.append(ema)

    # Feature 2: Average True Range (ATR) for volatility measurement
    if num_days >= 14:
        high_prices = s[2::6]
        low_prices = s[3::6]
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                   np.abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-14:])
    else:
        atr = 0  # Default to 0 if not enough data
    features.append(atr)

    # Feature 3: Rate of Change (ROC) for trend detection
    if num_days >= 15:
        roc = (closing_prices[-1] - closing_prices[-15]) / closing_prices[-15] * 100
    else:
        roc = 0  # Default to 0 if not enough data
    features.append(roc)

    # Feature 4: Z-score of recent returns for mean reversion
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] * 100 if len(closing_prices) > 1 else np.array([0])
    if len(daily_returns) >= 14:
        z_score = (daily_returns[-1] - np.mean(daily_returns[-14:])) / np.std(daily_returns[-14:]) if np.std(daily_returns[-14:]) != 0 else 0
    else:
        z_score = 0  # Default to 0 if not enough data
    features.append(z_score)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical thresholds based on standard deviation of recent features
    historical_std = np.std(enhanced_s[123:])  # Use the computed features for historical volatility
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_mid = 0.4 * historical_std
    trend_threshold_high = 0.3 * historical_std
    trend_threshold_low = -0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative for high risk
        reward += 10   # Mild positive for sell signal
    elif risk_level > risk_threshold_mid:
        reward -= 20  # Moderate negative for buy signals

    # Priority 2 — TREND FOLLOWING
    if risk_level < risk_threshold_mid:
        if abs(trend_direction) > trend_threshold_high:
            reward += 20 * np.sign(trend_direction)  # Positive reward aligned with trend direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold_high and risk_level < risk_threshold_mid:
        z_score = enhanced_s[123][3]  # Z-score feature for mean reversion
        if z_score < -1:  # Oversold condition
            reward += 15  # Reward for potential buy
        elif z_score > 1:  # Overbought condition
            reward -= 15  # Penalize for potential buy

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_mid:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range