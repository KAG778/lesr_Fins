import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices
    closing_prices = s[0::6]  # Every 6th element starting from index 0
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns

    # Feature 1: Mean of the last 5 daily returns (momentum)
    momentum = np.mean(daily_returns[-5:]) if len(daily_returns) >= 5 else 0
    features.append(momentum)

    # Feature 2: Volatility (standard deviation of daily returns)
    volatility = np.std(daily_returns[-20:]) if len(daily_returns) >= 20 else 0
    features.append(volatility)

    # Feature 3: Mean Reversion Indicator (Z-score of the last 20 prices)
    mean_price = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    z_score = (closing_prices[-1] - mean_price) / np.std(closing_prices[-20:]) if np.std(closing_prices[-20:]) != 0 else 0
    features.append(z_score)

    # Feature 4: Volume Change (Current volume - Average volume of last 5 days)
    volumes = s[4::6]  # Extract trading volumes
    avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0
    volume_change = volumes[-1] - avg_volume
    features.append(volume_change)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0

    # Calculate relative thresholds based on historical volatility
    risk_threshold = 0.7 * np.std(enhanced_s[123:])  # Example usage of historical std
    trend_threshold = 0.3  # Relative thresholds for trend direction

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward += -40  # Strong negative reward for BUY-aligned features
        reward += 10    # Mild positive reward for SELL-aligned features

    # If risk is acceptable, check trend direction
    elif risk_level < risk_threshold:
        # Priority 2 — TREND FOLLOWING
        if abs(trend_direction) > trend_threshold:
            reward += 20 if trend_direction > 0 else -20  # Positive for upward, negative for downward momentum
        
        # Priority 3 — SIDEWAYS / MEAN REVERSION
        elif abs(trend_direction) < trend_threshold:
            reward += 15  # Reward for mean-reversion strategies based on Z-score

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds