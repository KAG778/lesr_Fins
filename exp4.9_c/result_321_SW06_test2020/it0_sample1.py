import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices (every 6th element starting from index 0)
    volumes = s[4::6]  # Extract volumes (every 6th element starting from index 4)

    # Feature 1: Price Change Ratio
    price_change_ratio = (closing_prices[1:] - closing_prices[:-1]) / (closing_prices[:-1] + 1e-8)  # Avoid division by zero
    avg_price_change_ratio = np.mean(price_change_ratio[-5:]) if len(price_change_ratio) > 1 else 0
    
    # Feature 2: Volume Change Ratio
    avg_volume = np.mean(volumes)
    volume_change_ratio = volumes[-1] / (avg_volume + 1e-8)  # Avoid division by zero
    
    # Feature 3: Moving Average Convergence
    short_ma = np.mean(closing_prices[-5:])  # Short-term moving average for the last 5 days
    long_ma = np.mean(closing_prices[-20:])  # Long-term moving average for the last 20 days
    ma_convergence = short_ma - long_ma
    
    features = [avg_price_change_ratio, volume_change_ratio, ma_convergence]
    return np.array(features)

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

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # Assuming positive price change ratio indicates BUY
            reward = -40  # Strong negative reward
        else:  # Assuming negative price change ratio indicates SELL
            reward = 8  # Mild positive reward
        return reward

    if risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:  # Positive price change ratio
            reward = -20  # Moderate negative reward

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[0] > 0:  # Uptrend and positive price change
            reward = 20  # Positive reward for following the trend
        elif trend_direction < 0 and features[0] < 0:  # Downtrend and negative price change
            reward = 20  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward = 15  # Reward for buying
        elif features[0] > 0:  # Overbought condition
            reward = -15  # Penalize for chasing breakout

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% (uncertain market)

    return float(reward)