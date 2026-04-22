import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes
    days = len(closing_prices)  # Should be 20

    # Feature 1: Recent Price Momentum (latest close vs. average of last 5 closes)
    if days >= 6:
        recent_momentum = closing_prices[-1] - np.mean(closing_prices[-6:-1])
    else:
        recent_momentum = 0

    # Feature 2: Volume Spike (current volume vs. average of last 5 days)
    if days >= 6:
        recent_volume_avg = np.mean(volumes[-6:-1])
        volume_spike = (volumes[-1] - recent_volume_avg) / (recent_volume_avg if recent_volume_avg > 0 else 1)
    else:
        volume_spike = 0

    # Feature 3: Historical Volatility (standard deviation of daily returns)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    historical_volatility = np.std(daily_returns) if len(daily_returns) > 1 else 0

    features = [recent_momentum, volume_spike, historical_volatility]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Calculate relative thresholds based on historical volatility (for dynamic risk management)
    volatility_threshold = np.std(features[2:])  # Use historical std of the third feature (volatility)
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # Assuming feature[0] relates to upward momentum
            reward = np.random.uniform(-50, -30)  # Strong negative for BUY-aligned features
        else:
            reward = np.random.uniform(5, 10)  # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        if features[0] > 0:
            reward = np.random.uniform(-20, -10)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[0] > 0:  # Uptrend aligned with positive momentum
            reward = np.random.uniform(10, 20)  # Positive reward
        elif trend_direction < 0 and features[0] < 0:  # Downtrend aligned with negative momentum
            reward = np.random.uniform(10, 20)  # Positive reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Negative momentum in a sideways market
            reward = np.random.uniform(5, 15)  # Reward mean-reversion features
        else:
            reward = np.random.uniform(-10, 0)  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * volatility_threshold and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds