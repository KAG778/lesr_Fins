import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[::6]  # every 6th element starting from index 0
    num_days = len(closing_prices)
    
    features = []

    # Compute Price Momentum
    if num_days > 5:
        momentum = closing_prices[-1] - closing_prices[-6]  # current - 5 days ago
    else:
        momentum = 0  # edge case for the first few days
    features.append(momentum)

    # Compute 5-day Moving Average
    if num_days >= 5:
        moving_average = np.mean(closing_prices[-5:])  # last 5 days
    else:
        moving_average = np.mean(closing_prices)  # if less than 5 days
    features.append(moving_average)

    # Compute Relative Strength Index (RSI)
    if num_days >= 15:  # RSI calculation typically requires a larger window
        deltas = np.diff(closing_prices[-14:])  # last 14 days
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # neutral when not enough data
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
        reward += -40  # STRONG NEGATIVE reward for BUY-aligned features
    elif risk_level > 0.4:
        reward += -10  # moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # uptrend
            reward += 20  # positive reward for upward features
        elif trend_direction < -0.3:  # downtrend
            reward += 20  # positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Assuming the features include mean-reversion signals
        reward += 10  # reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds