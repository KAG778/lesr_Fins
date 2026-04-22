import numpy as np

def revise_state(s):
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]          # Trading volumes
    days = len(closing_prices)  # Should be 20

    features = []

    # Feature 1: Average Daily Return (over the last 20 days)
    daily_returns = np.zeros(days)
    for i in range(days):
        if closing_prices[i] > 0:  # Avoid division by zero
            daily_returns[i] = (closing_prices[i] - closing_prices[i - 1]) / closing_prices[i - 1] if i > 0 else 0
    avg_daily_return = np.mean(daily_returns) if days > 1 else 0
    features.append(avg_daily_return)

    # Feature 2: Historical Volatility (using daily returns)
    volatility = np.std(daily_returns) if days > 1 else 0
    features.append(volatility)

    # Feature 3: Rate of Change (Momentum)
    momentum = (closing_prices[0] - closing_prices[5]) / closing_prices[5] if days > 5 and closing_prices[5] > 0 else 0
    features.append(momentum)

    # New Feature 4: Volume Weighted Average Price (VWAP)
    vwap = np.sum(closing_prices * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else 0
    features.append(vwap)

    # New Feature 5: Extreme Price Movement (to identify crisis)
    extreme_movement = np.max(np.abs(daily_returns)) if days > 1 else 0
    features.append(extreme_movement)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Calculate relative thresholds based on historical data
    historical_volatility = np.std(features[1]) if features[1] != 0 else 1  # Avoid division by zero
    risk_threshold_high = 0.7 * historical_volatility
    risk_threshold_moderate = 0.4 * historical_volatility

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        if features[2] > 0:  # BUY-aligned feature
            reward = np.random.uniform(-50, -30)  # Strong negative reward for BUY
        else:  # SELL-aligned feature
            reward = np.random.uniform(5, 10)  # Mild positive reward for SELL
    elif risk_level > risk_threshold_moderate:
        if features[2] > 0:  # BUY-aligned feature
            reward = np.random.uniform(-20, -10)  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_moderate:
        if (trend_direction > 0 and features[2] > 0) or (trend_direction < 0 and features[2] < 0):
            reward += 10  # Positive reward for aligned features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 0:  # Oversold condition
            reward += 10  # Positive reward for mean-reversion buy
        else:  # Overbought condition
            reward += -10  # Penalize breakout chasing

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds