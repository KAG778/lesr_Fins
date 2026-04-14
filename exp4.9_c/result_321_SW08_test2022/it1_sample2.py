import numpy as np

def revise_state(s):
    features = []
    
    # Reshape the state into a 20-day view
    days = s.reshape((20, 6))
    closing_prices = days[:, 0]  # Closing prices
    
    # Feature 1: Average Daily Return
    daily_returns = np.diff(closing_prices, prepend=closing_prices[0]) / closing_prices
    avg_daily_return = np.mean(daily_returns)
    features.append(avg_daily_return)
    
    # Feature 2: Volatility (Standard Deviation of Daily Returns)
    volatility = np.std(daily_returns)
    features.append(volatility)
    
    # Feature 3: Exponential Moving Average (EMA) for 10 and 30 days
    ema_10 = np.mean(closing_prices[-10:])  # Short-term EMA
    ema_30 = np.mean(closing_prices[-30:])  # Long-term EMA
    ema_diff = ema_10 - ema_30  # EMA difference
    features.append(ema_diff)

    # Feature 4: Average True Range (ATR) for volatility measurement
    highs = days[:, 2]
    lows = days[:, 3]
    tr = np.maximum(highs[1:] - lows[1:], highs[1:] - closing_prices[:-1])
    tr = np.maximum(tr, lows[1:] - closing_prices[:-1])
    atr = np.mean(tr[-14:])  # 14-day ATR
    features.append(atr)

    # Feature 5: Momentum Indicator (Rate of Change)
    momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if len(closing_prices) > 6 else 0
    features.append(momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0  # Initialize reward

    # Calculate thresholds based on historical data (e.g., using std of previous risk levels)
    historical_risk_threshold = np.mean(features) + 1.5 * np.std(features)  # Dynamic threshold example
    high_risk_threshold = historical_risk_threshold

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        if features[4] > 0:  # Assuming feature[4] indicates momentum
            reward += -40  # Strong penalty for buying in high-risk
        else:
            reward += 10  # Mild positive for selling

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[4] > 0:  # Uptrend and positive momentum
            reward += 15  # Positive reward for alignment
        elif trend_direction < -0.3 and features[4] < 0:  # Downtrend and negative momentum
            reward += 15  # Positive reward for alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward for mean-reversion actions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward