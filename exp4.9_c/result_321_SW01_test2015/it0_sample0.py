import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV)
    features = []

    # 1. Compute daily returns
    closing_prices = s[0::6]  # Extract closing prices (index 0, 6, 12, ...)
    daily_returns = np.diff(closing_prices) / (closing_prices[:-1] + 1e-8)  # Avoid division by zero
    features.append(np.mean(daily_returns))  # Average daily return

    # 2. Compute volatility (standard deviation of returns)
    volatility = np.std(daily_returns)  # Standard deviation of daily returns
    features.append(volatility)

    # 3. Compute Relative Strength Index (RSI) for momentum
    gain = np.where(daily_returns > 0, daily_returns, 0)
    loss = np.where(daily_returns < 0, -daily_returns, 0)
    
    avg_gain = np.mean(gain[-14:])  # Average gain over the last 14 days
    avg_loss = np.mean(loss[-14:])  # Average loss over the last 14 days

    rs = avg_gain / (avg_loss + 1e-8)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if enhanced_state[123] >= 0:  # Assuming positive feature indicates a BUY
            reward = -40  # STRONG NEGATIVE for BUY-aligned features
        else:
            reward = +7  # MILD POSITIVE for SELL-aligned features
    elif risk_level > 0.4:
        if enhanced_state[123] >= 0:  # Assuming positive feature indicates a BUY
            reward = -20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 20  # Positive reward for bullish signal
        else:  # Downtrend
            reward += 20  # Positive reward for bearish signal

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_state[123] < 0:  # Oversold condition
            reward += 15  # Reward for buying
        else:  # Overbought condition
            reward += -15  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    reward = max(-100, min(100, reward))

    return reward