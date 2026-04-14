import numpy as np

def revise_state(s):
    features = []
    
    # Reshape the raw state into a 20-day view with 6 features each
    days = s.reshape((20, 6))
    closing_prices = days[:, 0]  # Closing prices

    # Feature 1: Average Daily Return
    daily_returns = np.diff(closing_prices, prepend=closing_prices[0]) / closing_prices
    avg_daily_return = np.mean(daily_returns)
    features.append(avg_daily_return)

    # Feature 2: Volatility (Standard Deviation of Daily Returns)
    volatility = np.std(daily_returns)
    features.append(volatility)

    # Feature 3: Rate of Change (Momentum)
    momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if len(closing_prices) > 6 else 0
    features.append(momentum)

    # Feature 4: Average True Range (ATR) for volatility measurement
    highs = days[:, 2]
    lows = days[:, 3]
    true_ranges = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closing_prices[:-1]),
                                                             np.abs(lows[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0
    features.append(atr)

    # Feature 5: Drawdown from the highest price in the last 20 days
    max_price = np.max(closing_prices)
    drawdown = (max_price - closing_prices[-1]) / max_price if max_price > 0 else 0
    features.append(drawdown)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract new features
    reward = 0.0  # Initialize reward

    # Calculate historical thresholds based on the features
    avg_volatility = np.mean(features[1])  # Using volatility feature for relative threshold
    std_volatility = np.std(features[1])
    risk_threshold_high = avg_volatility + 1.5 * std_volatility
    risk_threshold_moderate = avg_volatility + 0.5 * std_volatility
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward += -50  # Strong penalty for high-risk BUY
        if features[0] < 0:  # Assuming the average daily return could indicate a SELL signal
            reward += 10  # Mild positive reward for SELL
    elif risk_level > risk_threshold_moderate:
        reward += -25  # Moderate penalty for high-risk BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_moderate:
        if trend_direction > 0.3 and features[2] > 0:  # Uptrend and positive momentum
            reward += 20  # Positive reward for alignment
        elif trend_direction < -0.3 and features[2] < 0:  # Downtrend and negative momentum
            reward += 20  # Positive reward for alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3:
        if risk_level < 0.3:
            reward += 10  # Reward mean-reversion actions
        else:
            reward -= 5  # Penalize for chasing trends in a sideways market

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > (avg_volatility + 1.5 * std_volatility) and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward