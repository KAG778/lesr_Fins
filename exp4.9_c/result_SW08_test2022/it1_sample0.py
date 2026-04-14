import numpy as np

def revise_state(s):
    features = []

    # Feature 1: Daily Return Volatility (standard deviation of daily returns)
    closing_prices = s[0::6]  # Extract closing prices
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    daily_return_volatility = np.std(daily_returns)
    features.append(daily_return_volatility)

    # Feature 2: Momentum (10-day momentum)
    momentum = closing_prices[-1] - closing_prices[-11] if len(closing_prices) > 10 else 0
    features.append(momentum)

    # Feature 3: Average Trading Volume Change (percentage change)
    volumes = s[4::6]  # Extract trading volumes
    avg_volume_change = np.mean(np.diff(volumes) / volumes[:-1]) if len(volumes) > 1 else 0
    features.append(avg_volume_change)

    # Feature 4: Relative Strength Index (RSI) over the last 14 days
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = np.where(daily_returns < 0, -daily_returns, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Calculate relative thresholds based on historical standard deviations
    risk_threshold = 0.7
    trend_threshold = 0.3
    low_volatility_threshold = 0.6

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= np.clip(40 * (risk_level - risk_threshold) / (1 - risk_threshold), 30, 50)  # Strong penalty for BUY
        reward += np.clip(10 * (1 - risk_level), 0, 10)  # Mild reward for SELL
    elif risk_level > 0.4:
        reward -= np.clip(20 * (risk_level - 0.4) / (risk_threshold - 0.4), 10, 20)  # Moderate penalty for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < 0.4:
        if trend_direction > trend_threshold:  # Uptrend
            reward += np.clip(20 * (trend_direction - trend_threshold) / (1 - trend_threshold), 10, 20)  # Positive reward for upward momentum
        elif trend_direction < -trend_threshold:  # Downtrend
            reward += np.clip(20 * (abs(trend_direction) - trend_threshold) / (1 - trend_threshold), 10, 20)  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 15  # Reward mean-reversion features
        reward -= 5  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > low_volatility_threshold and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds