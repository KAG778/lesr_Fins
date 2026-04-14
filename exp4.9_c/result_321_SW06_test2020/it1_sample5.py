import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Closing prices
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volumes = s[4::6]  # Volumes

    # Feature 1: Recent Volatility (Standard deviation of daily returns over the last 20 days)
    if len(daily_returns) >= 20:
        recent_volatility = np.std(daily_returns[-20:])
    else:
        recent_volatility = np.std(daily_returns) if len(daily_returns) > 0 else 0
    features.append(recent_volatility)

    # Feature 2: Maximum Drawdown over the last 30 days
    if len(closing_prices) >= 30:
        peak = np.maximum.accumulate(closing_prices[-30:])
        drawdown = (peak - closing_prices[-30:]) / peak
        max_drawdown = np.max(drawdown)
    else:
        max_drawdown = 0  # Not enough data to calculate
    features.append(max_drawdown)

    # Feature 3: Average Daily Return over the last 20 days
    avg_daily_return = np.mean(daily_returns[-20:]) if len(daily_returns) >= 20 else 0
    features.append(avg_daily_return)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0

    # Calculate relative thresholds based on historical data
    # Assuming historical std and mean values are available for normalization
    historical_mean_return = 0.0  # Placeholder for historical average return
    historical_std_return = 1.0  # Placeholder for historical standard deviation of returns

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward += -40  # Strong negative reward for risky states
        if features[0] < 0:  # If volatility is low
            reward += 10  # Mild positive reward for SELL aligned with low volatility
        return np.clip(reward, -100, 100)

    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:  # Positive recent volatility
            reward += -20  # Moderate negative reward

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 15  # Reward for positive trend alignment
        elif trend_direction < 0:
            reward += 15  # Reward for negative trend alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] > 0:  # If max drawdown is significant
            reward += 20  # Reward for being cautious in sideways markets

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)