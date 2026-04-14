import numpy as np

def revise_state(s):
    features = []
    
    # Extracting closing prices and volumes
    closing_prices = s[0::6]
    volumes = s[4::6]
    
    # Feature 1: Rate of Change (Momentum) over the last 20 days
    if len(closing_prices) >= 20 and closing_prices[19] != 0:
        momentum = (closing_prices[0] - closing_prices[19]) / closing_prices[19]
    else:
        momentum = 0
    features.append(momentum)
    
    # Feature 2: Average Volume Change (current volume vs. 20-day average)
    average_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
    last_volume = volumes[0] if len(volumes) > 0 else 0
    volume_change = (last_volume - average_volume) / average_volume if average_volume != 0 else 0
    features.append(volume_change)
    
    # Feature 3: Historical Volatility (last 20 days)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] if len(closing_prices) > 1 else []
    historical_volatility = np.std(daily_returns[-20:]) if len(daily_returns) >= 20 else 0
    features.append(historical_volatility)

    # Feature 4: Drawdown from peak to current price (last 20 days)
    if len(closing_prices) >= 20:
        peak_price = np.max(closing_prices[-20:])
        drawdown = (peak_price - closing_prices[0]) / peak_price if peak_price != 0 else 0
    else:
        drawdown = 0
    features.append(drawdown)
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0
    
    # Calculate historical standard deviation for relative thresholds
    historical_std = np.std(features)
    if historical_std == 0:  # Handle edge case
        relative_threshold_high = 0.3
        relative_threshold_low = -0.3
    else:
        relative_threshold_high = historical_std * 0.3
        relative_threshold_low = -historical_std * 0.3

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # Positive momentum
            reward -= np.random.uniform(40, 60)  # Strong negative reward
        elif features[0] < 0:  # Negative momentum
            reward += np.random.uniform(10, 20)   # Mild positive reward

    elif risk_level > 0.4:
        if features[0] > 0:
            reward -= np.random.uniform(20, 40)  # Moderate negative reward
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Uptrend and positive momentum
            reward += np.random.uniform(10, 30)
        elif trend_direction < -0.3 and features[0] < 0:  # Downtrend and negative momentum
            reward += np.random.uniform(10, 30)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 0:  # Oversold condition (z-score)
            reward += np.random.uniform(10, 20)
        elif features[2] > 0:  # Overbought condition (z-score)
            reward -= np.random.uniform(10, 20)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds