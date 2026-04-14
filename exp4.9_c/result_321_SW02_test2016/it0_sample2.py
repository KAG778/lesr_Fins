import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV data)
    closing_prices = s[0::6]  # Extract every 6th element starting from index 0 (closing prices)
    
    # Calculate Price Momentum
    momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else 0
    
    # Calculate 5-day Moving Average
    moving_average = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else closing_prices[-1]
    
    # Calculate Relative Strength Index (RSI)
    deltas = np.diff(closing_prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    average_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    average_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    
    rs = average_gain / average_loss if average_loss else 0
    rsi = 100 - (100 / (1 + rs)) if average_loss else 100

    # Create features array
    features = [momentum, moving_average, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # STRONG NEGATIVE reward for BUY-aligned features
        reward += -40  # Example value in the range (-30, -50)
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -10  # Example moderate penalty

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        # Assume we have some features aligned with the trend direction
        if trend_direction > 0.3:
            reward += 20  # Positive reward for upward features
        elif trend_direction < -0.3:
            reward += 20  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Penalty for breakout chasing and reward for mean reversion features
        reward += 10  # Example reward for mean-reversion signals

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Clamp reward to the range [-100, 100]
    return float(np.clip(reward, -100, 100))