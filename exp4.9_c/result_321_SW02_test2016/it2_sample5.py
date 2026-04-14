import numpy as np

def revise_state(s):
    features = []

    # Extract closing prices and volumes
    closing_prices = s[0:120:6]  # Closing prices for the last 20 days
    volumes = s[4:120:6]          # Trading volumes for the same period

    # Feature 1: 10-day Rate of Change (momentum)
    roc = (closing_prices[-1] - closing_prices[-11]) / closing_prices[-11] if len(closing_prices) > 10 else 0
    features.append(roc)

    # Feature 2: Bollinger Band Width (last 20 days)
    if len(closing_prices) >= 20:
        moving_avg = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        bb_width = (moving_avg + 2 * std_dev) - (moving_avg - 2 * std_dev)  # Width of the bands
    else:
        bb_width = 0
    features.append(bb_width)

    # Feature 3: Z-score of Returns (last 20 days)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    mean_return = np.mean(daily_returns)
    std_return = np.std(daily_returns)
    z_score_returns = (daily_returns[-1] - mean_return) / std_return if std_return != 0 else 0
    features.append(z_score_returns)

    # Feature 4: Volume Oscillator (current volume vs. average volume)
    avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else 1.0  # 10-day average volume
    current_volume = volumes[-1] if len(volumes) > 0 else 0
    volume_oscillator = (current_volume - avg_volume) / avg_volume if avg_volume > 0 else 0
    features.append(volume_oscillator)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate relative thresholds based on historical data
    mean_risk = 0.5  # Placeholder for historical mean risk level
    std_risk = 0.2   # Placeholder for historical std for risk level
    risk_threshold = mean_risk + 1 * std_risk

    # Reward initialization
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= 50  # Strong negative for risky BUY signals
        reward += np.random.uniform(5, 10)  # Mild positive for SELL signals
        return np.clip(reward, -100, 100)  # Early exit

    elif risk_level > mean_risk:
        reward -= 20  # Moderate negative for elevated risk

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < mean_risk:
        if trend_direction > 0:
            reward += 20  # Reward for bullish alignment
        else:
            reward += 20  # Reward for bearish alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) <= 0.3 and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion strategies
        reward -= 5   # Penalty for chasing breakouts

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]