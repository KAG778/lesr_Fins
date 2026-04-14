import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices (index 0, 6, 12, ..., 114)
    volumes = s[4:120:6]          # Extract volumes (index 4, 10, 16, ..., 114)
    
    # Feature 1: Price Change Percentage over the last 20 days
    price_change = (closing_prices[-1] - closing_prices[0]) / closing_prices[0] if closing_prices[0] != 0 else 0.0
    
    # Feature 2: Average volume over the last 20 days
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0.0
    
    # Feature 3: Relative Strength Index (RSI) calculation
    deltas = np.diff(closing_prices)  # Calculate the difference between consecutive days
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0.0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0.0
    rs = gain / loss if loss != 0 else 0.0
    rsi = 100 - (100 / (1 + rs))
    
    features = [price_change, average_volume, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # Assume positive feature indicates BUY
            reward = np.random.uniform(-50, -30)  # Strong negative reward for BUY
        else:
            reward = np.random.uniform(5, 10)  # Mild positive reward for SELL
        return reward
    
    if risk_level > 0.4:
        if features[0] > 0:  # Assume positive feature indicates BUY
            reward = -10  # Moderate negative reward for BUY signals
        return reward
    
    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            if features[0] > 0:  # Positive feature for BUY
                reward = 20  # Positive reward for correct bullish position
        elif trend_direction < -0.3:  # Downtrend
            if features[0] < 0:  # Positive feature for SELL
                reward = 20  # Positive reward for correct bearish position
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 30:  # Assuming RSI < 30 indicates oversold
            reward = 15  # Reward for buying in oversold condition
        elif features[2] > 70:  # Assuming RSI > 70 indicates overbought
            reward = 15  # Reward for selling in overbought condition
            
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    return reward