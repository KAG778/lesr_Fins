import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract volumes

    # Feature 1: Modified Z-Score of Recent Returns
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    mean_return = np.mean(returns[-30:]) if len(returns) >= 30 else 0
    std_return = np.std(returns[-30:]) if len(returns) >= 30 else 1e-10  # Avoid division by zero
    z_score_returns = (returns[-1] - mean_return) / std_return

    # Feature 2: Volume Change Percentage Over Last Two Days
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if len(volumes) > 1 and volumes[-2] > 0 else 0

    # Feature 3: Recent Drawdown (percentage from peak)
    peak_price = np.max(closing_prices) if len(closing_prices) > 0 else closing_prices[-1]
    recent_drawdown = (peak_price - closing_prices[-1]) / peak_price if peak_price != 0 else 0

    features = [z_score_returns, volume_change, recent_drawdown]
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
        if features[0] > 0:  # Positive Z-score implies a BUY signal
            reward = np.random.uniform(-100, -50)  # Strong negative reward
        else:  # Negative Z-score implies a SELL signal
            reward = np.random.uniform(5, 15)  # Mild positive reward for SELL-aligned features
    elif risk_level > 0.4:
        if features[0] > 0:
            reward = np.random.uniform(-30, -10)  # Moderate negative reward

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Upward trend & positive return
            reward += 20 * (features[0] / historical_std)  # Reward aligned with momentum
        elif trend_direction < -0.3 and features[0] < 0:  # Downward trend & negative return
            reward += 20 * (features[0] / historical_std)  # Reward aligned with momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < -0.05:  # Assuming recent drawdown indicates oversold
            reward += 10  # Reward for buying in oversold conditions
        elif features[2] > 0.05:  # Assuming recent drawdown indicates overbought
            reward += 10  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(min(reward, 100), -100)
    
    return reward