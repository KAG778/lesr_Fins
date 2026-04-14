import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract volumes
    
    # Feature 1: Price Momentum (percentage change over the last 5 days)
    price_momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] * 100 if len(closing_prices) > 5 and closing_prices[-6] != 0 else 0
    
    # Feature 2: Volume Change Percentage
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] * 100 if len(volumes) > 1 and volumes[-2] > 0 else 0

    # Feature 3: Z-score of Recent Returns
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    z_score_returns = (returns[-1] - np.mean(returns[-30:])) / (np.std(returns[-30:]) + 1e-10) if len(returns) >= 30 else 0
    
    features = [price_momentum, volume_change, z_score_returns]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Extract features
    reward = 0.0
    
    # Calculate historical standard deviation for relative thresholds
    historical_std = np.std(features) if len(features) > 0 else 1
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # Positive price momentum indicates a BUY signal
            reward = np.random.uniform(-100, -50)  # Strong negative reward
        else:
            reward = np.random.uniform(5, 15)  # Mild positive reward for SELL signals
    elif risk_level > 0.4:
        if features[0] > 0:
            reward = np.random.uniform(-30, -10)  # Moderate negative reward

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Bullish alignment
            reward += 20 * (features[0] / historical_std)  # Reward aligned with momentum
        elif trend_direction < -0.3 and features[0] < 0:  # Bearish alignment
            reward += 20 * (features[0] / historical_std)  # Reward aligned with momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < -1:  # Assuming oversold condition for Z-score of returns
            reward += 10  # Reward for buying in oversold conditions
        elif features[2] > 1:  # Assuming overbought condition for Z-score of returns
            reward += 10  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(min(reward, 100), -100)
    
    return reward