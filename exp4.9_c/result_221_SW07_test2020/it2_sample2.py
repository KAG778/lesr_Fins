import numpy as np

def revise_state(s):
    features = []
    
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes
    
    # Feature 1: Price Momentum (Rate of Change over the last 19 days)
    momentum = (closing_prices[0] - closing_prices[19]) / closing_prices[19] if len(closing_prices) > 19 and closing_prices[19] != 0 else 0
    features.append(momentum)
    
    # Feature 2: Average Volume Change (current volume vs. 20-day average)
    if len(volumes) >= 20:
        average_volume = np.mean(volumes[-20:])
        last_volume = volumes[0]
        volume_change = (last_volume - average_volume) / average_volume if average_volume != 0 else 0
    else:
        volume_change = 0
    features.append(volume_change)
    
    # Feature 3: Historical Volatility (last 20 days)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] if len(closing_prices) > 1 else []
    historical_volatility = np.std(daily_returns[-20:]) if len(daily_returns) >= 20 else 0
    features.append(historical_volatility)
    
    # Feature 4: Z-score of Returns
    if len(daily_returns) >= 20:
        mean_return = np.mean(daily_returns[-20:])
        std_return = np.std(daily_returns[-20:])
        z_score = (daily_returns[-1] - mean_return) / std_return if std_return != 0 else 0
    else:
        z_score = 0
    features.append(z_score)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0
    
    # Calculate relative thresholds based on historical standard deviation
    historical_std = np.std(features)
    if historical_std == 0:
        high_risk_threshold = 0.7
        low_risk_threshold = 0.4
    else:
        high_risk_threshold = historical_std * 0.7
        low_risk_threshold = historical_std * 0.4
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        if features[0] > 0:  # Assuming positive momentum is a BUY signal
            reward -= np.random.uniform(30, 50)  # Strong penalty
        if features[0] < 0:  # Assuming negative momentum is a SELL signal
            reward += np.random.uniform(10, 20)  # Mild reward
    
    elif risk_level > low_risk_threshold:
        if features[0] > 0:  # Assuming positive momentum is a BUY signal
            reward -= np.random.uniform(10, 20)  # Moderate penalty

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0.3 and features[0] > 0:  # Uptrend and positive momentum
            reward += np.random.uniform(10, 20)  # Reward momentum alignment
        elif trend_direction < -0.3 and features[0] < 0:  # Downtrend and negative momentum
            reward += np.random.uniform(10, 20)  # Reward momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < -1:  # Assuming Z-score indicates oversold
            reward += np.random.uniform(10, 20)  # Reward for buying
        elif features[3] > 1:  # Assuming Z-score indicates overbought
            reward -= np.random.uniform(10, 20)  # Penalize buying

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds