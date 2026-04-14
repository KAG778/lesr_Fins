import numpy as np

def revise_state(s):
    features = []
    
    # Reshape the raw state into a 20-day view with 6 features each
    days = s.reshape((20, 6))
    closing_prices = days[:, 0]  # Extract closing prices
    volumes = days[:, 4]          # Extract trading volumes
    
    # Feature 1: Daily Return Volatility (Standard Deviation of Daily Returns)
    daily_returns = np.diff(closing_prices, prepend=closing_prices[0]) / closing_prices
    daily_return_volatility = np.std(daily_returns)
    features.append(daily_return_volatility)

    # Feature 2: Average True Range (ATR) for volatility measurement
    high_prices = days[:, 3]
    low_prices = days[:, 5]
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:],
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]),
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0
    features.append(atr)

    # Feature 3: Rate of Change (Momentum) over the last 5 days
    momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if len(closing_prices) > 6 else 0
    features.append(momentum)

    # Feature 4: Drawdown from the highest price in the last 20 days
    max_price = np.max(closing_prices)
    drawdown = (max_price - closing_prices[-1]) / max_price if max_price > 0 else 0
    features.append(drawdown)

    # Feature 5: Volume Change (Current Volume vs. Previous Volume)
    volume_change = (volumes[-1] / volumes[-2] - 1) if len(volumes) > 1 and volumes[-2] != 0 else 0
    features.append(volume_change)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Extract features
    reward = 0.0  # Initialize reward

    # Calculate relative thresholds based on historical data
    historical_std = np.std(features) if len(features) > 0 else 1  # Avoid division by zero
    avg_volatility = np.mean(features[0])  # Using the first feature (volatility) for relative threshold
    risk_threshold_high = avg_volatility + 2 * historical_std
    risk_threshold_moderate = avg_volatility + historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        # Strong negative reward for BUY-aligned signals
        reward += -50  
        # Mild positive reward for SELL-aligned signals
        reward += 10  
    elif risk_level > risk_threshold_moderate:
        # Moderate negative reward for BUY signals
        reward += -20  

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_moderate:
        if trend_direction > 0.3 and features[2] > 0:  # Uptrend and positive momentum
            reward += 15  # Positive reward for correct direction
        elif trend_direction < -0.3 and features[2] < 0:  # Downtrend and negative momentum
            reward += 15  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < risk_threshold_moderate:
        if features[3] < 0:  # Assuming feature[3] indicates a negative momentum for mean reversion
            reward += 10  # Reward mean-reversion actions
        else:
            reward -= 5  # Penalize chasing trends in a sideways market

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward