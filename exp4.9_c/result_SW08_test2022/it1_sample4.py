import numpy as np

def revise_state(s):
    features = []

    # Extract closing prices and volumes
    closing_prices = s[0:120:6]
    trading_volumes = s[4:120:6]

    # Feature 1: Daily Return (percentage change)
    daily_returns = np.zeros(20)  # Daily returns will be for 19 days, 0 for alignment
    for i in range(1, 20):
        if closing_prices[i] != 0:
            daily_returns[i] = (closing_prices[i] - closing_prices[i - 1]) / closing_prices[i - 1]
    features.append(np.mean(daily_returns[1:]))  # Average daily return excluding the first day

    # Feature 2: Volatility (Standard deviation of daily returns)
    volatility = np.std(daily_returns[1:])  # Exclude the first day
    features.append(volatility)

    # Feature 3: Crisis Indicator (Number of days with returns below a threshold)
    crisis_threshold = np.mean(daily_returns) - 2 * np.std(daily_returns)
    crisis_days = np.sum(daily_returns < crisis_threshold)
    features.append(crisis_days)

    # Feature 4: Volume Change (Percentage change in average volume over 5 days)
    avg_volume = np.mean(trading_volumes)
    volume_change = (trading_volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0
    features.append(volume_change)

    # Feature 5: Momentum Indicator (5-day momentum)
    momentum = np.mean(daily_returns[-5:])  # Average return of the last 5 days
    features.append(momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate relative thresholds based on historical data
    historical_returns = enhanced_s[123:]
    mean_return = np.mean(historical_returns)
    std_return = np.std(historical_returns)

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= 50  # STRONG NEGATIVE reward for BUY-aligned features
        reward += 20 if enhanced_s[123] < 0 else 0  # MILD POSITIVE reward for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 15 if enhanced_s[123] > 0 else 0  # Positive reward for upward signals
        else:
            reward += 15 if enhanced_s[123] < 0 else 0  # Positive reward for downward signals

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10 if enhanced_s[123] < 0 else 0  # Reward mean-reversion features if oversold
        reward -= 5 if enhanced_s[123] > 0 else 0  # Penalize breakout-chasing features if overbought

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds