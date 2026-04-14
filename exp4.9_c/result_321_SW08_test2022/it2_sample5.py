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

    # Feature 2: Average Volume over the last 20 days
    avg_volume = np.mean(volumes)
    features.append(avg_volume)

    # Feature 3: Rate of Change (Momentum) over the last 5 days
    momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if len(closing_prices) > 6 else 0
    features.append(momentum)

    # Feature 4: Average True Range (ATR) for volatility measurement
    high_prices = days[:, 3]
    low_prices = days[:, 5]
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]),
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0
    features.append(atr)

    # Feature 5: Percentage of Current Price Relative to the Highest Price in the Last 20 Days
    max_price = np.max(closing_prices[-20:])
    price_percentage = (closing_prices[-1] / max_price) if max_price > 0 else 0
    features.append(price_percentage)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract new features
    reward = 0.0  # Initialize reward

    # Calculate relative thresholds based on historical data
    historical_volatility = np.std(features[0])  # Using daily return volatility
    historical_volume = np.std(features[1])  # Using average volume
    historical_momentum = np.std(features[2])  # Using momentum
    risk_threshold_high = 0.7 * historical_volatility
    risk_threshold_moderate = 0.4 * historical_volatility

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        # Strong negative reward for BUY-aligned features
        reward += -50  # Strong penalty for high-risk BUY
        if features[2] < 0:  # Assuming feature[2] indicates negative momentum (SELL signal)
            reward += 10  # Mild positive for selling
    elif risk_level > risk_threshold_moderate:
        # Moderate negative reward for BUY signals
        reward += -20  # Moderate penalty for elevated risk

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_moderate:
        if trend_direction > 0.3 and features[2] > 0:  # Uptrend and positive momentum
            reward += 20  # Strong positive reward for correct direction
        elif trend_direction < -0.3 and features[2] < 0:  # Downtrend and negative momentum
            reward += 20  # Strong positive reward for correct direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3:
        if risk_level < risk_threshold_moderate:
            reward += 15  # Reward for mean-reversion features
        else:
            reward -= 10  # Penalize for chasing trends in sideways market

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_volatility and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward