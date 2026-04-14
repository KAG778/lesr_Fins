import numpy as np

def revise_state(s):
    features = []
    
    # Closing prices for the last 20 days
    closing_prices = s[0::6]
    volumes = s[4::6]  # Extract trading volumes
    
    # Feature 1: Price Momentum (Rate of Change)
    momentum = (closing_prices[0] - closing_prices[19]) / closing_prices[19] if closing_prices[19] != 0 else 0
    features.append(momentum)
    
    # Feature 2: Average Volume Change (last 20 days)
    average_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
    last_volume = volumes[0]
    volume_change = (last_volume - average_volume) / average_volume if average_volume != 0 else 0
    features.append(volume_change)
    
    # Feature 3: Historical Volatility (last 20 days)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Calculate daily returns
    historical_volatility = np.std(daily_returns[-20:]) if len(daily_returns) >= 20 else 0
    features.append(historical_volatility)
    
    # Feature 4: Maximum Drawdown (last 20 days)
    max_drawdown = max(0, np.max(closing_prices[-20:]) - closing_prices[0]) / np.max(closing_prices[-20:]) if np.max(closing_prices[-20:]) != 0 else 0
    features.append(max_drawdown)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0

    # Calculate relative thresholds based on historical data
    risk_threshold = np.mean(features[2:]) + 2 * np.std(features[2:])  # Using volatility as a threshold for high risk
    drawdown_threshold = np.mean(features[3:]) + 2 * np.std(features[3:])  # Using drawdown as a threshold

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # Assuming positive momentum
            reward -= np.random.uniform(30, 50)
        # Mild positive reward for SELL-aligned features
        if features[0] < 0:  # Assuming negative momentum
            reward += np.random.uniform(5, 10)
    
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:  # Positive momentum
            reward -= np.random.uniform(10, 20)

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            if features[0] > 0:  # Positive momentum
                reward += np.random.uniform(10, 20)
        elif trend_direction < -0.3:  # Downtrend
            if features[0] < 0:  # Negative momentum
                reward += np.random.uniform(10, 20)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward += np.random.uniform(10, 20)
        elif features[0] > 0:  # Overbought condition
            reward -= np.random.uniform(10, 20)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > risk_threshold and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Clip reward to ensure it stays within bounds
    return np.clip(reward, -100, 100)