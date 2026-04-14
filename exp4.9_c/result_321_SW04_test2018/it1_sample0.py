import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns

    # Feature 1: Z-Score of Recent Returns
    mean_return = np.mean(returns[-30:]) if len(returns) >= 30 else 0
    std_return = np.std(returns[-30:]) if len(returns) >= 30 else 1e-10  # Avoid division by zero
    z_score_returns = (returns[-1] - mean_return) / std_return
    
    # Feature 2: Volatility (standard deviation of returns)
    volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0  # Recent volatility

    # Feature 3: Adaptive Moving Average (e.g., 10-period)
    adaptive_moving_average = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else closing_prices[-1]

    features = [z_score_returns, volatility, adaptive_moving_average]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0
    
    # Calculate thresholds based on historical data
    relative_thresholds = {
        "high_risk": 0.7,  # Placeholder for historical analysis
        "medium_risk": 0.4,
        "high_trend": 0.3,
        "low_trend": -0.3,
        "high_volatility": 0.6
    }

    # Priority 1 — RISK MANAGEMENT
    if risk_level > relative_thresholds["high_risk"]:
        if features[0] > 0:  # Positive Z-score implies a BUY signal
            reward -= np.random.uniform(30, 50)  # Strong negative reward
        else:  # Negative Z-score implies a SELL signal
            reward += np.random.uniform(5, 10)  # Mild positive reward
    elif risk_level > relative_thresholds["medium_risk"]:
        if features[0] > 0:
            reward -= np.random.uniform(10, 20)  # Moderate negative reward

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > relative_thresholds["high_trend"] and risk_level < relative_thresholds["medium_risk"]:
        if trend_direction > relative_thresholds["high_trend"] and features[0] > 0:  # Uptrend
            reward += 10  # Positive reward for alignment
        elif trend_direction < relative_thresholds["low_trend"] and features[0] < 0:  # Downtrend
            reward += 10  # Positive reward for alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < relative_thresholds["high_trend"] and risk_level < relative_thresholds["medium_risk"]:
        if features[0] < -1:  # Oversold condition
            reward += 10  # Reward for buying
        elif features[0] > 1:  # Overbought condition
            reward += 10  # Reward for selling

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > relative_thresholds["high_volatility"] and risk_level < relative_thresholds["medium_risk"]:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(min(reward, 100), -100)
    
    return reward