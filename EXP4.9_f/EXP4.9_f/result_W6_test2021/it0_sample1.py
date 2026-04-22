import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV data)
    closing_prices = s[0::6]
    opening_prices = s[1::6]
    high_prices = s[2::6]
    low_prices = s[3::6]
    volumes = s[4::6]

    # Feature 1: Price Momentum (last day closing price - average of last N days closing prices)
    N = 5  # Lookback for average
    if len(closing_prices) >= N:
        momentum = closing_prices[-1] - np.mean(closing_prices[-N:])
    else:
        momentum = 0.0  # Handle edge case

    # Feature 2: Volatility (standard deviation of closing prices over the last N days)
    if len(closing_prices) >= N:
        volatility = np.std(closing_prices[-N:])
    else:
        volatility = 0.0  # Handle edge case

    # Feature 3: Volume Change (percentage change from the previous day)
    volume_change = np.zeros(len(volumes))
    for i in range(1, len(volumes)):
        if volumes[i-1] != 0:
            volume_change[i] = (volumes[i] - volumes[i-1]) / volumes[i-1]
        else:
            volume_change[i] = 0.0  # Handle edge case

    # Normalize volume change for the last day
    volume_change = volume_change[-1] if len(volume_change) > 0 else 0.0

    # Return the computed features
    return np.array([momentum, volatility, volume_change])

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40  # STRONG NEGATIVE reward for BUY-aligned features
        # Assuming the last action was BUY (action 0)
        if enhanced_s[123] > 0:  # Example feature indicating buy signal
            return -40
        # SELL-aligned features (action 1)
        reward += 5  # MILD POSITIVE reward for SELL-aligned features
    elif risk_level > 0.4:
        reward += -15  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and enhanced_s[123] > 0:  # Positive momentum
            reward += 10  # Positive reward for upward features
        elif trend_direction < -0.3 and enhanced_s[123] < 0:  # Negative momentum
            reward += 10  # Positive reward for bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123] < 0:  # Assuming this indicates a sell signal for overbought
            reward += 10  # Reward for mean-reversion features
        else:
            reward -= 5  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]