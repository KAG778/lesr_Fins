import numpy as np

def revise_state(s):
    closing_prices = s[0::6]
    daily_returns = (closing_prices[1:] - closing_prices[:-1]) / closing_prices[:-1]  # Daily return percentages

    # Feature 1: Z-score of Daily Returns
    mean_return = np.mean(daily_returns)
    std_return = np.std(daily_returns) if len(daily_returns) > 1 else 1e-6  # Avoid division by zero
    z_score = (daily_returns[-1] - mean_return) / std_return if std_return != 0 else 0

    # Feature 2: Average True Range (ATR) for last 14 days
    high_prices = s[2::6]
    low_prices = s[3::6]
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # 14-day ATR

    # Feature 3: Bollinger Bands Width (20-day)
    moving_avg = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    bol_upper = moving_avg + 2 * np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    bol_lower = moving_avg - 2 * np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    bollinger_width = (bol_upper - bol_lower) / moving_avg if moving_avg != 0 else 0  # Normalized width

    features = [z_score, atr, bollinger_width]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for risky BUY
        reward += 10.0 if features[0] < 0 else 0  # Reward for SELL if Z-score indicates overbought
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for risky BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 15.0  # Reward momentum alignment with Z-score

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) <= 0.3 and risk_level < 0.3:
        if features[0] < -1:  # Oversold condition (Z-score)
            reward += 10.0  # Reward for potential BUY
        elif features[0] > 1:  # Overbought condition (Z-score)
            reward += 10.0  # Reward for potential SELL

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude in uncertain markets

    return float(np.clip(reward, -100, 100))