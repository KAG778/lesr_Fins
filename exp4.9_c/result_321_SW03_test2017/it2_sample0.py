import numpy as np

def revise_state(s):
    # Extracting closing prices
    closing_prices = s[0::6]  # Every 6th element starting from index 0 (closing prices)
    
    # Feature 1: Daily Returns (percentage change)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] if len(closing_prices) > 1 else np.array([0])
    daily_returns = np.concatenate(([0], daily_returns))  # Fill first element with 0 for shape compatibility

    # Feature 2: 14-day Relative Strength Index (RSI)
    window_rsi = 14
    if len(closing_prices) < window_rsi:
        rsi = np.zeros_like(closing_prices)
    else:
        deltas = np.diff(closing_prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.convolve(gain, np.ones(window_rsi)/window_rsi, mode='valid')
        avg_loss = np.convolve(loss, np.ones(window_rsi)/window_rsi, mode='valid')
        rs = np.concatenate(([0]*window_rsi, avg_gain / (avg_loss + 1e-10)))
        rsi = 100 - (100 / (1 + rs))

    # Feature 3: 20-day Historical Volatility
    if len(closing_prices) >= 20:
        historical_volatility = np.std(daily_returns[-20:])  # Based on last 20 daily returns
    else:
        historical_volatility = 0.0

    # Feature 4: Trend Strength (the last 5-day price momentum)
    price_changes = np.diff(closing_prices)
    trend_strength = np.sum(price_changes[-5:]) if len(price_changes) >= 5 else 0

    # Feature 5: Crisis Indicator (detecting extreme drops)
    crisis_indicator = np.mean(price_changes < -0.05)  # Proportion of days with >5% drop
    
    # Return computed features
    features = np.array([daily_returns[-1], rsi[-1], historical_volatility, trend_strength, crisis_indicator])
    return features

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    daily_return = features[0]
    rsi = features[1]
    historical_volatility = features[2]
    trend_strength = features[3]
    crisis_indicator = features[4]

    # Initialize reward
    reward = 0.0

    # Calculate dynamic risk thresholds based on historical volatility
    relative_std = historical_volatility if historical_volatility > 0 else 1.0  # Prevent division by zero
    high_risk_threshold = 0.7 * relative_std
    medium_risk_threshold = 0.4 * relative_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        if daily_return > 0:  # BUY-aligned feature
            reward = np.random.uniform(-100, -50)  # Strong negative for risky BUY
        else:  # SELL-aligned feature
            reward = np.random.uniform(10, 30)  # Mild positive for SELL
    elif risk_level > medium_risk_threshold:
        if daily_return > 0:  # BUY signal
            reward = np.random.uniform(-30, -10)  # Moderate negative for risky BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < medium_risk_threshold:
        if trend_direction > 0.3 and trend_strength > 0:  # Uptrend & positive momentum
            reward += np.random.uniform(10, 20)  # Positive reward
        elif trend_direction < -0.3 and trend_strength < 0:  # Downtrend & negative momentum
            reward += np.random.uniform(10, 20)  # Positive reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if crisis_indicator > 0.1:  # Significant crisis indicator
            reward += np.random.uniform(5, 15)  # Reward for mean-reversion buying
        elif crisis_indicator < 0.1:  # Less significant drop
            reward -= np.random.uniform(5, 15)  # Penalize mean-reversion selling

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < medium_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the specified range