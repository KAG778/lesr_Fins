import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Closing prices every 6th element
    volumes = s[4::6]          # Trading volumes
    days = len(closing_prices)  # Should be 20

    # Feature 1: Average Daily Return over the last 20 days
    daily_returns = np.zeros(days)
    for i in range(1, days):
        if closing_prices[i - 1] > 0:  # Avoid division by zero
            daily_returns[i] = (closing_prices[i] - closing_prices[i - 1]) / closing_prices[i - 1]
    avg_daily_return = np.mean(daily_returns) if days > 1 else 0
    features.append(avg_daily_return)

    # Feature 2: Historical Volatility (standard deviation of daily returns)
    historical_volatility = np.std(daily_returns) if days > 1 else 0
    features.append(historical_volatility)

    # Feature 3: Price Momentum (current close vs. close 5 days ago)
    price_momentum = (closing_prices[0] - closing_prices[5]) / closing_prices[5] if days > 5 and closing_prices[5] > 0 else 0
    features.append(price_momentum)

    # Feature 4: Volume Momentum (current volume vs. average volume over the last 20 days)
    avg_volume = np.mean(volumes) if len(volumes) > 0 else 0
    volume_momentum = (volumes[-1] - avg_volume) / avg_volume if avg_volume > 0 else 0
    features.append(volume_momentum)

    # Feature 5: Extreme Price Movement (to identify crisis)
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

    # Calculate relative thresholds based on historical volatility
    historical_volatility = features[1]  # Historical volatility
    risk_threshold_high = 0.7 * historical_volatility
    risk_threshold_moderate = 0.4 * historical_volatility

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        if features[0] > 0:  # If avg_daily_return is aligned with BUY
            reward = np.random.uniform(-50, -30)  # Strong negative reward for BUY
        else:  # SELL-aligned feature
            reward = np.random.uniform(5, 10)  # Mild positive reward for SELL
    elif risk_level > risk_threshold_moderate:
        if features[0] > 0:  # If avg_daily_return is aligned with BUY
            reward = np.random.uniform(-20, -10)  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_moderate:
        if (trend_direction > 0 and features[0] > 0) or (trend_direction < 0 and features[0] < 0):
            reward += 10  # Positive reward for aligned features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward += 10  # Positive reward for mean-reversion buy
        else:  # Overbought condition
            reward -= 10  # Penalize breakout chasing

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds