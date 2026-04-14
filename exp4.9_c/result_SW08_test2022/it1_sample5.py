import numpy as np

def revise_state(s):
    features = []

    # Feature 1: Daily Return (percentage change)
    daily_returns = np.zeros(19)  # Daily returns will be for days 1 to 19 (19 entries)
    for i in range(1, 20):
        if s[i * 6] != 0:  # Avoid division by zero
            daily_returns[i - 1] = (s[i * 6] - s[(i - 1) * 6]) / s[(i - 1) * 6]
    features.append(np.mean(daily_returns))  # Average daily return

    # Feature 2: Price Momentum (last 5 days average return)
    momentum = np.zeros(5)
    for i in range(15, 20):  # Compute momentum for the last 5 days
        if s[i * 6] != 0 and s[(i - 1) * 6] != 0:
            momentum[i - 15] = (s[i * 6] - s[(i - 1) * 6]) / s[(i - 1) * 6]
    features.append(np.mean(momentum))  # Average momentum over the last 5 days

    # Feature 3: Relative Strength Index (RSI)
    gains = []
    losses = []
    for i in range(1, 20):
        change = s[i * 6] - s[(i - 1) * 6]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(-change)

    average_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    average_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0

    if average_loss == 0:
        rsi = 100  # RSI is 100 if no losses
    else:
        rs = average_gain / average_loss
        rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    # Feature 4: Moving Average Convergence Divergence (MACD)
    short_ema = np.mean(closing_prices[-12:])  # 12-day EMA
    long_ema = np.mean(closing_prices[-26:])   # 26-day EMA
    macd = short_ema - long_ema
    features.append(macd)

    # Feature 5: Volatility (standard deviation of daily returns)
    volatility = np.std(daily_returns)
    features.append(volatility)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Setting relative thresholds based on historical data
    historical_std = np.std(enhanced_s[123:])  # Use features to set thresholds
    high_risk_threshold = 0.7 * historical_std
    medium_risk_threshold = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
        return np.clip(reward, -100, 100)
    elif risk_level > medium_risk_threshold:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < medium_risk_threshold:
        if trend_direction > 0.3:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive reward for downward features
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward mean-reversion signals
        reward -= np.random.uniform(5, 15)  # Penalize breakout-chasing signals

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < medium_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds