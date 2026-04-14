import numpy as np

def revise_state(s):
    # Extracting closing prices and volumes from the raw state
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]         # Volume data
    
    # Feature 1: Price Momentum (Rate of Change, 1-day)
    price_momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    
    # Feature 2: Historical Volatility (Standard Deviation of Returns over a window)
    returns = np.diff(closing_prices)
    historical_volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0  # Last 20 returns
    
    # Feature 3: Volume Momentum (Rate of Change, 1-day)
    volume_momentum = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0
    
    # Feature 4: Average True Range (ATR) over the last 14 days
    true_ranges = np.maximum(closing_prices[1:] - closing_prices[:-1], closing_prices[1:] - np.roll(closing_prices, 1)[1:], np.roll(closing_prices, -1)[1:] - closing_prices[:-1])
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # Last 14 true ranges
    
    # Feature 5: Relative Strength Index (RSI) over a 14-day period
    deltas = np.diff(closing_prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    average_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    average_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    rsi = 100 - (100 / (1 + (average_gain / average_loss))) if average_loss != 0 else 100
    
    features = [price_momentum, historical_volatility, volume_momentum, atr, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0
    
    # Calculate thresholds based on historical data
    risk_threshold_high = 0.7  # Relative threshold for risk
    risk_threshold_moderate = 0.4  # Moderate risk threshold
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY
    elif risk_level > risk_threshold_moderate:
        reward -= np.random.uniform(5, 15)  # Mild negative reward for BUY
    
    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_moderate:
        if trend_direction > 0:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive reward for upward momentum
        else:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive reward for downward momentum
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion strategies
        reward -= np.random.uniform(5, 15)  # Penalize breakout-chasing features
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50% in high volatility
    
    # Ensure reward stays within bounds of [-100, 100]
    reward = max(-100, min(100, reward))
    
    return reward