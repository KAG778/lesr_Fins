import numpy as np

def revise_state(s):
    # s is a 120-dimensional raw state containing OHLCV data for 20 trading days.
    closing_prices = s[::6]  # Closing prices (indices: 0, 6, 12, ..., 114)
    volumes = s[4::6]        # Trading volumes (indices: 4, 10, 16, ..., 118)
    
    features = []
    
    # Feature 1: Price Change (percentage change from the previous day)
    price_changes = np.diff(closing_prices) / closing_prices[:-1]
    price_changes = np.concatenate(([0], price_changes))  # Pad with 0 for same length
    features.append(price_changes[-1])  # Use the most recent change
    
    # Feature 2: Simple Moving Average (SMA) over the last 5 days
    sma = np.convolve(closing_prices, np.ones(5)/5, mode='valid')
    sma = np.concatenate(([0]*4, sma))  # Pad with zeros for the first few days
    features.append(sma[-1])  # Use the most recent SMA
    
    # Feature 3: Volume Change (percentage change from the previous day)
    volume_changes = np.diff(volumes) / volumes[:-1]
    volume_changes = np.concatenate(([0], volume_changes))  # Pad with 0 for same length
    features.append(volume_changes[-1])  # Use the most recent change

    return np.array(features)

def intrinsic_reward(enhanced_state):
    # Extract the regime vector
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward -= np.random.uniform(30, 50)  # Random strong penalty for buying
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(5, 15)  # Moderate penalty for buying

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 10  # Positive reward for bullish trend
        else:
            reward += 10  # Positive reward for bearish trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Positive reward for mean-reversion strategies
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)