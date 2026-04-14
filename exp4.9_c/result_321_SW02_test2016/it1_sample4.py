import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices for the last 20 days
    closing_prices = s[0:120:6]
    
    # Feature 1: Z-score of Returns (20 days)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    mean_return = np.mean(daily_returns)
    std_return = np.std(daily_returns)
    z_score_returns = (daily_returns[-1] - mean_return) / std_return if std_return != 0 else 0
    features.append(z_score_returns)

    # Feature 2: Bollinger Bands (20-day period)
    moving_avg = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else closing_prices[-1]
    rolling_std = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    upper_band = moving_avg + (2 * rolling_std)
    lower_band = moving_avg - (2 * rolling_std)
    features.append((closing_prices[-1] - moving_avg) / rolling_std if rolling_std != 0 else 0)  # Current close relative to Bollinger Bands

    # Feature 3: Average True Range (ATR)
    high_prices = s[1:120:6]  # High prices for 20 days
    low_prices = s[2:120:6]   # Low prices for 20 days
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # 14-day ATR
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate dynamic thresholds based on historical data (for illustrative purposes, we use mean and std)
    risk_threshold = 0.5  # Placeholder for historical risk level threshold (to be dynamically calculated)
    trend_threshold = 0.3  # Placeholder for historical trend threshold (to be dynamically calculated)

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= 40  # Strong negative for risky BUY signals
        reward += np.random.uniform(5, 10)  # Mild positive for SELL signals
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold:
        if trend_direction > trend_threshold:
            reward += 20  # Reward for bullish momentum
        elif trend_direction < -trend_threshold:
            reward += 20  # Reward for bearish momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold:
        reward += 10  # Reward for mean-reversion strategies
        reward -= 5  # Penalty for chasing breakouts

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]