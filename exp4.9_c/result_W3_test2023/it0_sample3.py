import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[::6]  # Closing prices
    opening_prices = s[1::6]  # Opening prices
    trading_volumes = s[4::6]  # Trading volumes
    
    # Feature 1: Price Momentum
    price_momentum = closing_prices[0] - closing_prices[1] if len(closing_prices) > 1 else 0
    
    # Feature 2: Relative Strength Index (RSI)
    # Calculating RSI for the last 14 days (if there are enough data points)
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices[:14])
        gain = np.where(deltas > 0, deltas, 0).mean()
        loss = -np.where(deltas < 0, deltas, 0).mean()
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = np.nan  # Not enough data for RSI

    # Feature 3: Volume Change
    volume_change = (trading_volumes[0] - trading_volumes[1]) / trading_volumes[1] if trading_volumes[1] != 0 else 0
    
    features = [price_momentum, rsi, volume_change]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if enhanced_state[123] > 0:  # Assuming features[0] relates to BUY
            reward = np.random.uniform(-50, -30)  # Strong negative for buy-aligned
        else:  # Assuming features[1] relates to SELL
            reward = np.random.uniform(5, 10)  # Mild positive for sell-aligned
    elif risk_level > 0.4:
        if enhanced_state[123] > 0:  # Assuming features[0] relates to BUY
            reward = -10  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 10  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += 10  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_state[123] < 0:  # Assuming features[1] relates to oversold (for buy)
            reward += 10  # Reward mean-reversion features
        else:  # Assuming features[1] relates to overbought (for sell)
            reward -= 5  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)