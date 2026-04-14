import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Calculate features based on OHLCV data
    features = []

    # Feature 1: Daily Returns
    daily_returns = [0]  # The first day has no previous day to compare to
    for i in range(1, 20):
        prev_close = s[(i - 1) * 6 + 0]
        curr_close = s[i * 6 + 0]
        if prev_close != 0:
            daily_return = (curr_close - prev_close) / prev_close
        else:
            daily_return = 0  # Avoid division by zero
        daily_returns.append(daily_return)
    features.append(np.mean(daily_returns))  # Mean daily return over 20 days

    # Feature 2: Volatility (Standard Deviation of Daily Returns)
    volatility = np.std(daily_returns)
    features.append(volatility)

    # Feature 3: Moving Average of Closing Prices (last 5 days)
    closing_prices = [s[i * 6 + 0] for i in range(20)]
    if len(closing_prices) >= 5:
        moving_average = np.mean(closing_prices[-5:])
    else:
        moving_average = closing_prices[-1]  # Use last day's price if not enough data
    features.append(moving_average)

    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_state[123:]  # Your computed features from revise_state
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward += -40 if features[0] > 0 else 0  # Assuming features[0] is related to bullish signals
        reward += +8 if features[0] < 0 else 0  # Assuming features[0] is related to bearish signals
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -20 if features[0] > 0 else 0  # Assuming features[0] is related to bullish signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 10 * features[0]  # Positive reward for upward features
        elif trend_direction < -0.3:
            reward += 10 * -features[0]  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward for mean-reversion features
        reward += 15 if features[0] < 0 else 0  # Assuming features[0] indicates oversold
        reward -= 15 if features[0] > 0 else 0  # Assuming features[0] indicates overbought

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within limits