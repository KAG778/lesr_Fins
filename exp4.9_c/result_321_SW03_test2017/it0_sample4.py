import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)

    # Extract closing prices from raw state
    closing_prices = s[0::6]  # Every 6th element starting from index 0 (closing prices)
    
    # Feature 1: Daily Returns (percentage change)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    daily_returns = np.concatenate(([0], daily_returns))  # Fill first element with 0 for shape compatibility

    # Feature 2: 14-day Relative Strength Index (RSI)
    window = 14
    if len(closing_prices) < window:
        rsi = np.zeros_like(closing_prices)
    else:
        deltas = np.diff(closing_prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.convolve(gain, np.ones(window)/window, mode='valid')
        avg_loss = np.convolve(loss, np.ones(window)/window, mode='valid')
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = np.concatenate(([0]*window, rsi))  # Prepend zeros for the first 'window' days

    # Feature 3: 20-day Moving Average (MA)
    ma = np.convolve(closing_prices, np.ones(20)/20, mode='valid')
    ma = np.concatenate(([0]*19, ma))  # Prepend zeros for the first '19' days

    # Return computed features
    features = np.array([daily_returns, rsi, ma]).T.flatten()
    return features

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_state[123:]

    reward = 0.0

    # RISK MANAGEMENT (Priority 1)
    if risk_level > 0.7:
        if features[0] > 0:  # Assuming first feature is aligned with BUY
            reward = -40  # Strong negative for risky BUY
        else:
            reward = 10  # Mild positive for SELL
    elif risk_level > 0.4:
        if features[0] > 0:  # BUY signal
            reward = -10  # Moderate negative for risky BUY

    # TREND FOLLOWING (Priority 2)
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # BUY aligned with uptrend
            reward += 20  # Positive reward for correct trend-following
        elif trend_direction < -0.3 and features[0] < 0:  # SELL aligned with downtrend
            reward += 20  # Positive reward for correct bearish bet

    # SIDEWAYS / MEAN REVERSION (Priority 3)
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Assuming first feature is SELL aligned
            reward += 15  # Reward mean-reversion selling
        elif features[0] > 0:  # Assuming first feature is BUY aligned
            reward -= 15  # Penalize breakout-chasing

    # HIGH VOLATILITY (Priority 4)
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the specified range