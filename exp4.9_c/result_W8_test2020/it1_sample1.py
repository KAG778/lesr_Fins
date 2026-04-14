import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0::6]  # Extract closing prices

    # 1. 10-day moving average for short-term trend
    short_term_ma = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else 0.0
    features.append(short_term_ma)

    # 2. 50-day moving average for longer-term trend
    long_term_ma = np.mean(closing_prices[-50:]) if len(closing_prices) >= 50 else 0.0
    features.append(long_term_ma)

    # 3. Average True Range (ATR) for volatility measurement
    high_prices = s[1::6]
    low_prices = s[2::6]
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0.0
    features.append(atr)

    # 4. Z-score of daily returns for mean-reversion signals
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    mean_return = np.mean(daily_returns) if len(daily_returns) > 0 else 0.0
    std_return = np.std(daily_returns) if len(daily_returns) > 0 else 1.0  # Avoid division by zero
    z_score = (daily_returns[-1] - mean_return) / std_return if std_return != 0 else 0
    features.append(z_score)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate dynamic thresholds based on historical volatility
    historical_std = np.std(enhanced_s[123:])  # Use features to calculate historical volatility
    risk_high_threshold = 0.7 * historical_std
    risk_moderate_threshold = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > risk_high_threshold:
        reward -= 50  # Strong negative for BUY-aligned features
    elif risk_level > risk_moderate_threshold:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2: Trend Following (when risk is low)
    elif abs(trend_direction) > trend_threshold and risk_level < risk_moderate_threshold:
        if trend_direction > trend_threshold:  # Uptrend
            reward += 20  # Positive reward for upward momentum
        elif trend_direction < -trend_threshold:  # Downtrend
            reward += 20  # Positive reward for downward momentum

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 10  # Positive reward for mean-reversion actions

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < risk_moderate_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Clamp reward to [-100, 100]