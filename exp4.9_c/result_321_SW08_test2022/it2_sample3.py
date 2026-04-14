import numpy as np

def revise_state(s):
    features = []

    # Reshape the raw state into a 20-day view with 6 features each
    days = s.reshape((20, 6))
    closing_prices = days[:, 0]  # Closing prices
    volumes = days[:, 4]          # Trading volumes

    # Feature 1: Average Daily Return
    daily_returns = np.diff(closing_prices, prepend=closing_prices[0]) / closing_prices
    avg_daily_return = np.mean(daily_returns)
    features.append(avg_daily_return)

    # Feature 2: Volatility (Standard Deviation of Daily Returns)
    volatility = np.std(daily_returns)
    features.append(volatility)

    # Feature 3: Rate of Change (Momentum) over the last 5 days
    if len(closing_prices) > 5:
        momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6]  # 5-day momentum
    else:
        momentum = 0
    features.append(momentum)

    # Feature 4: Drawdown from the highest price in the last 20 days
    max_price = np.max(closing_prices)
    drawdown = (max_price - closing_prices[-1]) / max_price if max_price > 0 else 0
    features.append(drawdown)

    # Feature 5: Average True Range (ATR) for measuring volatility
    high_prices = days[:, 3]  # High prices
    low_prices = days[:, 5]    # Low prices
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:],
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]),
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Grab the features from the enhanced state
    reward = 0.0  # Initialize reward

    # Calculate historical thresholds for risk management using standard deviation
    rolling_std = np.std(features) if features.size > 0 else 1  # Avoid division by zero
    high_risk_threshold = 0.7 * rolling_std
    moderate_risk_threshold = 0.4 * rolling_std
    trend_threshold = 0.3

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        # Strong negative reward for BUY-aligned features
        if features[2] > 0:  # Assuming feature[2] indicates a BUY signal (momentum)
            reward += -50  # Strong penalty
        # Mild positive reward for SELL-aligned features
        reward += 10  # Encouragement to sell

    elif risk_level > moderate_risk_threshold:
        # Moderate negative reward for BUY signals
        if features[2] > 0:  # Assuming feature[2] indicates a BUY signal (momentum)
            reward += -25  # Moderate penalty

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < moderate_risk_threshold:
        if trend_direction > trend_threshold and features[2] > 0:  # Uptrend and positive momentum
            reward += 20  # Strong positive reward for correct direction
        elif trend_direction < -trend_threshold and features[2] < 0:  # Downtrend and negative momentum
            reward += 20  # Strong positive reward for correct direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < moderate_risk_threshold:
        reward += 15  # Reward for mean-reversion actions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * rolling_std and risk_level < moderate_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward